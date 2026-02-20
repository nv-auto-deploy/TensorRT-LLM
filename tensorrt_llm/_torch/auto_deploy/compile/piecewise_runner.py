"""ADPiecewiseRunner: manages warmup → capture → replay for a single static CUDA graph segment.

Each static submodule in a piecewise-split model is wrapped in an ADPiecewiseRunner.
The runner's behavior is controlled by two class-level contexts set by the orchestrator
(PiecewiseCapturedGraph) before each split_gm forward pass:

  - `_current_phase`: determines execution mode ("warmup", "capture", or "replay")
  - `_current_num_tokens`: identifies which bucket entry to use

Phase semantics:
  1. WARMUP: Run the submodule eagerly. (Data-ptr tracking runs but is NOT relied on
     for correctness — see note on dynamic-index identification below.)
  2. CAPTURE: Capture the submodule as a CUDA graph. All non-weight tensor args are
     treated as dynamic. For those that came from a previous static runner (found in
     the _static_output_registry), we reuse the same buffer (zero-copy). Others
     (model inputs, dynamic-segment outputs) are referenced directly and refreshed
     via _prepare_replay_inputs during replay.
  3. REPLAY: Copy only dynamic inputs into the static buffers, then replay the
     captured graph.

Dynamic-index identification:
  We do NOT rely on data_ptr() change detection during warmup, because PyTorch's
  caching allocator can reuse the same address for activation tensors across warmup
  iterations, making them falsely appear "static." Instead, we mark ALL non-weight
  tensor args as dynamic. Weights/buffers are identified by matching against
  data_ptrs collected from `submodule.parameters()` and `submodule.buffers()`.

Each runner maintains entries keyed by `num_tokens`.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from ..utils.logger import ad_logger

# ---------------------------------------------------------------------------
# Cross-bucket memory optimization helpers
#
# Between bucket captures, we drop strong Python refs to activation tensors in
# the graph pool.  This lets the caching allocator reuse those blocks for the
# next bucket's capture, reducing total pool usage from sum(all_buckets) to
# max(largest_bucket).
#
# After ALL captures, we reconstruct lightweight tensors that point at the
# pool-resident memory (kept alive by the CUDAGraph objects) so replay can
# copy into static input buffers and return static outputs as before.
#
# The two ``torch._C`` APIs below are the same ones used by PyTorch's own
# ``CUDAGraphTreeManager`` (``torch/_inductor/cudagraph_trees.py``).
# ---------------------------------------------------------------------------

_CAN_RECONSTRUCT_TENSORS: bool = hasattr(
    torch._C, "_construct_storage_from_data_pointer"
) and hasattr(torch._C, "_construct_CUDA_Tensor_From_Storage_And_Metadata")


def _tensor_metadata(t: torch.Tensor) -> dict:
    """Capture enough information to reconstruct *t* later."""
    return {
        "data_ptr": t.untyped_storage().data_ptr(),
        "nbytes": t.untyped_storage().nbytes(),
        "storage_offset": t.storage_offset(),
        "size": t.shape,
        "stride": t.stride(),
        "dtype": t.dtype,
        "device": t.device,
    }


def _reconstruct_tensor(meta: dict) -> torch.Tensor:
    """Create a tensor that views *existing* graph-pool memory described by *meta*.
    The returned tensor has **no deleter** — the CUDA-graph pool keeps the
    underlying allocation alive for as long as the ``CUDAGraph`` object exists.
    """
    storage = torch._C._construct_storage_from_data_pointer(
        meta["data_ptr"], meta["device"], meta["nbytes"]
    )
    return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(meta, storage)


@dataclass
class _FinalizedState:
    """Metadata stash created by :meth:`ADPiecewiseRunner.finalize_entry`.
    Holds everything needed to reconstruct the activation tensors that were
    dropped to free graph-pool blocks between bucket captures.
    """

    input_metadata: Dict[int, dict]  # dynamic_idx → tensor metadata
    output_metadata: List[Optional[dict]]  # per-leaf tensor metadata (None for non-tensors)
    output_non_tensors: Dict[int, Any]  # idx → non-tensor leaf values
    output_spec: Any  # TreeSpec produced by tree_flatten(static_output)


@dataclass
class SegmentEntry:
    """State for a single (num_tokens) configuration of a segment."""

    cuda_graph: Optional[torch.cuda.CUDAGraph] = None
    # Static input list — each element is a direct reference to a tensor at a fixed address.
    # During replay, _prepare_replay_inputs refreshes activation buffers as needed.
    #
    # Three categories:
    # - Weight tensors: referenced directly (already at fixed addresses, never change).
    # - Activation tensors from a previous static runner's output: reused from the
    #   static output registry. During replay the previous runner's CUDA graph writes to
    #   the same address, so _prepare_replay_inputs skips the copy (zero-copy).
    # - Activation tensors from model inputs or dynamic segment outputs: referenced
    #   directly from the capture iteration. During replay, the dynamic segment produces
    #   output at a new address, so _prepare_replay_inputs copies into this buffer.
    static_inputs: Optional[List[Any]] = None
    # Indices of dynamic (activation) tensor args that need copy during replay
    dynamic_indices: Optional[Set[int]] = None
    # Static output — the output tensor(s) produced during capture.
    # During replay, the CUDA graph writes to the same addresses, so returning
    # this object gives the caller the updated data.
    static_output: Any = None
    # Tracks data_ptr() of tensor args during warmup to identify static vs dynamic
    _warmup_data_ptrs: Optional[List[Optional[int]]] = None
    # Dynamic input indices captured via registry zero-copy. These indices are
    # safe to reconstruct from metadata after finalize.
    _reconstructable_input_indices: Optional[Set[int]] = None
    # Cross-bucket memory optimization: saved metadata when strong refs are dropped.
    # Non-None between finalize_entry() and materialize_entry() calls.
    _finalized: Optional[_FinalizedState] = None


class ADPiecewiseRunner(nn.Module):
    """Wraps a static submodule and manages its CUDA graph capture/replay.

    Behavior is controlled by two class-level contexts set by the orchestrator:
      - `_current_phase`: "warmup" (eager + ptr tracking), "capture" (CUDA graph
        capture), or "replay" (graph replay / eager fallback at runtime)
      - `_current_num_tokens`: identifies which bucket entry to use

    If `num_tokens` doesn't match any pre-configured bucket, falls back to eager.
    Bucket resolution (nearest bucket >= real token count) is handled upstream by
    DualModeCapturedGraph, so the runner always sees an exact bucket value.
    """

    # Class-level contexts: the orchestrator sets these before each split_gm forward pass
    # so ALL runners in the graph use the same correct num_tokens and phase.
    _current_num_tokens: Optional[int] = None
    _current_phase: str = "replay"  # "warmup", "capture", or "replay"

    # Class-level registry of output tensors produced during CUDA graph capture.
    # Key: (num_tokens, data_ptr) -> output tensor at a fixed address.
    # During capture, a runner checks if any of its activation inputs match a
    # registered output (by data_ptr). If so, it references that buffer directly —
    # enabling zero-copy during replay (the producer's graph writes, the consumer's
    # graph reads, same address).
    # Note: runners capture in sequential order, so all registry entries are from
    # earlier runners — no need to track runner_id.
    _static_output_registry: Dict[Tuple[int, int], torch.Tensor] = {}
    # Per-bucket counter for successful registry-prune removals.
    _registry_prune_hits: Dict[int, int] = defaultdict(int)

    # Debug flag: when True, validates replay outputs against eager baseline after
    # intra-bucket finalize + materialize.  Expensive — only for development.
    _debug_validate_replay: bool = False

    @classmethod
    def set_current_num_tokens(cls, num_tokens: Optional[int]) -> None:
        """Set the current num_tokens context for all runners.

        Called by PiecewiseCapturedGraph before each forward pass through the split graph.
        """
        cls._current_num_tokens = num_tokens

    @classmethod
    def set_current_phase(cls, phase: str) -> None:
        """Set the current execution phase for all runners.

        Called by PiecewiseCapturedGraph to control warmup → capture → replay transitions.
        Valid phases: "warmup", "capture", "replay".
        """
        assert phase in ("warmup", "capture", "replay"), f"Invalid phase: {phase}"
        cls._current_phase = phase

    @classmethod
    def clear_static_output_registry(cls) -> None:
        """Clear the static output registry.

        Called when switching between different graph configurations or resetting state.
        """
        cls._static_output_registry.clear()

    @classmethod
    def reset_registry_prune_hits(cls, num_tokens: int) -> None:
        """Reset successful registry-prune hit counter for one bucket."""
        cls._registry_prune_hits[num_tokens] = 0

    @classmethod
    def get_registry_prune_hits(cls, num_tokens: int) -> int:
        """Return successful registry-prune hit count for one bucket."""
        return cls._registry_prune_hits.get(num_tokens, 0)

    def __init__(
        self,
        submodule: nn.Module,
        piecewise_num_tokens: Optional[List[int]] = None,
        graph_pool: Optional[Tuple[int, ...]] = None,
        global_weight_ptrs: Optional[Set[int]] = None,
    ):
        super().__init__()
        self.submodule = submodule
        self._graph_pool = graph_pool

        # Collect data_ptrs of all parameters and buffers in this submodule.
        # These are weight tensors with stable addresses that NEVER need copying.
        # Everything else that appears in flat_args is a cross-partition activation
        # (from a previous static runner or a dynamic segment) and must be treated
        # as dynamic for correctness during CUDA graph replay.
        self._weight_ptrs: Set[int] = set()
        for p in submodule.parameters():
            self._weight_ptrs.add(p.data_ptr())
        for b in submodule.buffers():
            self._weight_ptrs.add(b.data_ptr())
        # Include graph-level parameter/buffer pointers. Some FX partition
        # boundaries pass weight tensors that are not owned by this submodule
        # directly; without this union they are misclassified as dynamic inputs.
        if global_weight_ptrs is not None:
            self._weight_ptrs.update(global_weight_ptrs)

        # Pre-populate entries for each bucket size
        self.entries: Dict[int, SegmentEntry] = {}
        if piecewise_num_tokens:
            for nt in piecewise_num_tokens:
                self.entries[nt] = SegmentEntry()

    def _find_entry(self, num_tokens: int) -> Optional[SegmentEntry]:
        """Find the SegmentEntry for the given num_tokens.

        Expects an exact match — bucket resolution (nearest bucket >= real token count)
        is handled upstream by DualModeCapturedGraph._find_nearest_bucket before
        num_tokens reaches the runner.

        Returns None if num_tokens doesn't match any pre-configured bucket (eager fallback).
        """
        return self.entries.get(num_tokens)

    def _track_warmup_ptrs(self, entry: SegmentEntry, flat_args: List[Any]) -> None:
        """Track data_ptr() during warmup to identify static (weight) vs dynamic (activation) args.

        On the first warmup call, record all data_ptrs. On subsequent calls, mark args whose
        data_ptr changed as "dynamic" (by setting their tracked ptr to None).
        """
        if entry._warmup_data_ptrs is None:
            # First warmup: record all data_ptrs
            entry._warmup_data_ptrs = [
                a.data_ptr() if isinstance(a, torch.Tensor) else None for a in flat_args
            ]
        else:
            # Subsequent warmup: check for changes
            for i, a in enumerate(flat_args):
                if isinstance(a, torch.Tensor):
                    if (
                        entry._warmup_data_ptrs[i] is not None
                        and a.data_ptr() != entry._warmup_data_ptrs[i]
                    ):
                        # data_ptr changed → this is a dynamic (activation) tensor
                        entry._warmup_data_ptrs[i] = None

    def _identify_dynamic_indices(self, entry: SegmentEntry, flat_args: List[Any]) -> Set[int]:
        """Mark all non-weight tensor args as dynamic.

        Weight/buffer tensors (matched via _weight_ptrs) are static.
        Everything else is dynamic — the capture code will further check
        _static_output_registry for zero-copy reuse where possible.
        """
        dynamic_indices: Set[int] = set()
        for i, a in enumerate(flat_args):
            if not isinstance(a, torch.Tensor):
                continue
            if a.data_ptr() in self._weight_ptrs:
                continue  # Weight/buffer — stable address, no copy needed
            dynamic_indices.add(i)
        return dynamic_indices

    def _prepare_replay_inputs(self, entry: SegmentEntry, flat_inputs: List[Any]) -> None:
        """Refresh dynamic activation buffers before CUDA graph replay.

        For each dynamic tensor input, this copies runtime data into the captured
        static buffer unless both tensors already share the same data_ptr() (no-copy
        fast path, common for static segment chaining).

        When runtime input is smaller than the bucketed static buffer (padding case),
        copy the valid prefix and clear the padded tail. Clearing avoids stale values
        from prior warmup/capture executions leaking into downstream ops.
        """
        for idx in entry.dynamic_indices:
            new_inp = flat_inputs[idx]
            static_inp = entry.static_inputs[idx]

            if not isinstance(new_inp, torch.Tensor) or not isinstance(static_inp, torch.Tensor):
                continue

            # Fast path: no copy needed when producer already wrote into the
            # captured static buffer (segment N output -> segment N+1 input).
            if new_inp.data_ptr() == static_inp.data_ptr():
                continue

            if static_inp.shape == new_inp.shape:
                static_inp.copy_(new_inp, non_blocking=True)
            elif (
                new_inp.shape[0] < static_inp.shape[0] and new_inp.shape[1:] == static_inp.shape[1:]
            ):
                # Padded case: runtime input is smaller along dim 0.
                n = new_inp.shape[0]
                static_inp[:n].copy_(new_inp, non_blocking=True)
                static_inp[n:].zero_()
            elif (
                new_inp.ndim >= 2
                and new_inp.shape[1] < static_inp.shape[1]
                and new_inp.shape[0] == static_inp.shape[0]
            ):
                # Padded case: runtime input is smaller along dim 1
                # (e.g., [1, real, D] vs [1, bucket, D]).
                n = new_inp.shape[1]
                static_inp[:, :n].copy_(new_inp, non_blocking=True)
                static_inp[:, n:].zero_()
            else:
                # Fallback: shapes are incompatible — this is a real error
                static_inp.copy_(new_inp, non_blocking=True)

    # ------------------------------------------------------------------
    # Cross-bucket memory optimization (#1 weak refs + #3 pool reuse)
    # ------------------------------------------------------------------

    def finalize_entry(
        self,
        num_tokens: int,
        prune_registry: bool = True,
        drop_static_inputs: bool = True,
    ) -> None:
        """Drop strong refs to activation tensors, saving metadata for later reconstruction.

        Called by the orchestrator (PiecewiseCapturedGraph) *between* bucket captures
        so the CUDA caching allocator can reclaim and reuse the underlying memory blocks.

        Must be followed by :meth:`materialize_entry` before the entry is used for replay.

        No-op when the required PyTorch internal APIs are unavailable or when the
        entry is already finalized.
        """
        entry = self.entries.get(num_tokens)
        if entry is None or entry.cuda_graph is None or not _CAN_RECONSTRUCT_TENSORS:
            return

        # Guard: already finalized
        if entry._finalized is not None:
            return

        # --- activation inputs (dynamic indices only; weights stay) ---
        # Only inputs captured via registry zero-copy are reconstructable.
        # Direct dynamic inputs (model inputs / eager dynamic outputs) are not
        # guaranteed graph-owned buffers and must stay alive.
        input_metadata: Dict[int, dict] = {}
        if entry.dynamic_indices is None or entry.static_inputs is None:
            return

        if drop_static_inputs:
            reconstructable = entry._reconstructable_input_indices or set()
            for idx in entry.dynamic_indices:
                if idx not in reconstructable:
                    continue
                inp = entry.static_inputs[idx]
                if isinstance(inp, torch.Tensor):
                    input_metadata[idx] = _tensor_metadata(inp)
                    entry.static_inputs[idx] = None  # drop strong ref

        # --- output tensors ---
        flat_output, out_spec = tree_flatten(entry.static_output)
        output_metadata: List[Optional[dict]] = []
        output_non_tensors: Dict[int, Any] = {}
        for i, leaf in enumerate(flat_output):
            if isinstance(leaf, torch.Tensor):
                output_metadata.append(_tensor_metadata(leaf))
                if prune_registry:
                    # Prune registry so caching allocator can reclaim the memory.
                    # Key must match how capture populates the registry: (num_tokens, data_ptr)
                    registry_key = (num_tokens, leaf.data_ptr())
                    removed = ADPiecewiseRunner._static_output_registry.pop(registry_key, None)
                    if removed is not None:
                        ADPiecewiseRunner._registry_prune_hits[num_tokens] += 1
            else:
                output_metadata.append(None)
                output_non_tensors[i] = leaf
        entry.static_output = None  # drop strong ref

        entry._finalized = _FinalizedState(
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            output_non_tensors=output_non_tensors,
            output_spec=out_spec,
        )

    def prune_registry_for_entry(self, num_tokens: int) -> None:
        """Prune this runner's output tensors from the static output registry.

        Used by the registry-aware pruning schedule to defer registry removal
        until all downstream runner captures that may need lookup have executed.
        """
        entry = self.entries.get(num_tokens)
        if entry is None:
            return

        # Prefer finalized metadata when available (no strong output refs).
        if entry._finalized is not None:
            for meta in entry._finalized.output_metadata:
                if meta is not None:
                    registry_key = (num_tokens, meta["data_ptr"])
                    removed = ADPiecewiseRunner._static_output_registry.pop(registry_key, None)
                    if removed is not None:
                        ADPiecewiseRunner._registry_prune_hits[num_tokens] += 1
            return

        # Fallback: prune from live static output tensors.
        if entry.static_output is None:
            return
        flat_output, _ = tree_flatten(entry.static_output)
        for leaf in flat_output:
            if isinstance(leaf, torch.Tensor):
                registry_key = (num_tokens, leaf.data_ptr())
                removed = ADPiecewiseRunner._static_output_registry.pop(registry_key, None)
                if removed is not None:
                    ADPiecewiseRunner._registry_prune_hits[num_tokens] += 1

    def materialize_entry(self, num_tokens: int) -> None:
        """Replace strong refs with lightweight reconstructed pointers.

        Called after ALL runners in the bucket have finished capturing.
        The reconstructed tensors point at the same graph-pool addresses that
        were baked into the CUDA graph during capture.  The pool keeps the
        memory alive; the new tensor objects simply provide Python-level
        handles for ``copy_()`` (input refresh) and returning static output.

        This is where the actual ref-dropping happens: the original strong
        tensors are replaced by allocator-bypassing reconstructed handles,
        allowing ``empty_cache()`` to reclaim the underlying blocks.
        """
        entry = self.entries.get(num_tokens)
        if entry is None or entry._finalized is None:
            return

        state = entry._finalized

        # --- reconstruct activation inputs ---
        for idx, meta in state.input_metadata.items():
            entry.static_inputs[idx] = _reconstruct_tensor(meta)

        # --- reconstruct output ---
        flat_output: List[Any] = []
        for i, meta in enumerate(state.output_metadata):
            if meta is not None:
                flat_output.append(_reconstruct_tensor(meta))
            else:
                flat_output.append(state.output_non_tensors[i])
        entry.static_output = tree_unflatten(flat_output, state.output_spec)

        entry._finalized = None  # cleanup — entry is ready for replay

    def forward(self, *args, **kwargs) -> Any:
        # Use the class-level contexts set by the orchestrator
        num_tokens = ADPiecewiseRunner._current_num_tokens
        phase = ADPiecewiseRunner._current_phase
        entry = self._find_entry(num_tokens) if num_tokens is not None else None

        if entry is None:
            # Unknown num_tokens or exceeds all buckets — fallback to eager
            return self.submodule(*args, **kwargs)

        # Flatten inputs once (used by all phases)
        flat_args, args_spec = tree_flatten((args, kwargs))

        # --- WARMUP PHASE ---
        if phase == "warmup":
            # Track data_ptr() to distinguish weights from activations
            self._track_warmup_ptrs(entry, flat_args)
            return self.submodule(*args, **kwargs)

        # --- CAPTURE PHASE ---
        if phase == "capture":
            ad_logger.debug(f"ADPiecewiseRunner: capturing CUDA graph for num_tokens={num_tokens}")

            # Identify which args are dynamic (activations) vs static (weights)
            entry.dynamic_indices = self._identify_dynamic_indices(entry, flat_args)

            # Build static_inputs list for this entry. Every element is a direct
            # reference (no cloning) — we just need each tensor at a persistent address.
            #
            # For activation tensors, we check the static output registry to find
            # outputs from previous static runners. During replay, those runners'
            # CUDA graphs write to the same address, so _prepare_replay_inputs can skip the
            # copy (zero-copy). All other activation tensors (model inputs, dynamic
            # segment outputs) are referenced directly from this capture iteration;
            # _prepare_replay_inputs will copy new data into them during replay.
            entry.static_inputs = []
            entry._reconstructable_input_indices = set()
            num_reused = 0
            num_referenced = 0
            for i, a in enumerate(flat_args):
                if isinstance(a, torch.Tensor) and i in entry.dynamic_indices:
                    # Check if this activation is a previous static runner's output
                    # (if so, record the registry reference for zero-copy during replay)
                    prev_output = ADPiecewiseRunner._static_output_registry.get(
                        (num_tokens, a.data_ptr())
                    )
                    if prev_output is not None:
                        entry.static_inputs.append(prev_output)
                        entry._reconstructable_input_indices.add(i)
                        num_reused += 1
                    else:
                        # Model input or dynamic segment output — reference directly.
                        # During replay, _prepare_replay_inputs will copy new data into this buffer.
                        entry.static_inputs.append(a)
                else:
                    # Weight tensor — reference directly (fixed address, never changes)
                    entry.static_inputs.append(a)
                    if isinstance(a, torch.Tensor):
                        num_referenced += 1

            # Unflatten back to get the static args/kwargs
            static_args_kwargs = tree_unflatten(entry.static_inputs, args_spec)
            static_args, static_kwargs = static_args_kwargs

            # Capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool):
                output = self.submodule(*static_args, **static_kwargs)

            torch.cuda.synchronize()

            # Fallback: if no pool was provided at construction time, store the
            # auto-created pool so subsequent captures within this runner reuse it.
            if self._graph_pool is None:
                self._graph_pool = graph.pool()

            entry.cuda_graph = graph
            entry.static_output = output

            # Register outputs in the static output registry so next runners can reuse them
            flat_output, _ = tree_flatten(output)
            for out_tensor in flat_output:
                if isinstance(out_tensor, torch.Tensor):
                    ADPiecewiseRunner._static_output_registry[
                        (num_tokens, out_tensor.data_ptr())
                    ] = out_tensor

            num_dynamic = len(entry.dynamic_indices) - num_reused
            ad_logger.debug(
                f"ADPiecewiseRunner: captured graph for num_tokens={num_tokens} — "
                f"{num_dynamic} dynamic activation buffers, "
                f"{num_reused} reused from previous static segments, "
                f"{num_referenced} weight tensors (zero-copy)"
            )

            return output

        # --- REPLAY PHASE ---
        if entry._finalized is not None:
            self.materialize_entry(num_tokens)

        if (
            entry.cuda_graph is None
            or entry.static_inputs is None
            or entry.dynamic_indices is None
            or entry.static_output is None
        ):
            ad_logger.warning(
                f"ADPiecewiseRunner: missing capture state for num_tokens={num_tokens}; "
                "falling back to eager execution for this segment."
            )
            return self.submodule(*args, **kwargs)

        # Copy only dynamic inputs into static buffers.
        # _prepare_replay_inputs skips copy if input is already at static buffer address
        # (common case: segment N's output is segment N+1's input, so addresses match)
        self._prepare_replay_inputs(entry, flat_args)

        # Replay the captured graph
        entry.cuda_graph.replay()

        return entry.static_output

    @property
    def graph_pool(self):
        """Return the CUDA graph memory pool (for sharing across runners)."""
        return self._graph_pool

    @graph_pool.setter
    def graph_pool(self, pool):
        self._graph_pool = pool

    # ------------------------------------------------------------------
    # Intra-bucket finalize schedule (Phase B2)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_finalize_schedule(
        split_gm: "torch.fx.GraphModule",
    ) -> Dict[str, List[str]]:
        """Compute a safe finalize schedule from the FX graph topology.

        For each ``call_module`` node that wraps an :class:`ADPiecewiseRunner`,
        find the **last consumer** (in topological order) of that node's output.
        After the last consumer executes, the runner's output is no longer needed
        and it is safe to call :meth:`finalize_entry`.

        Returns:
            ``{trigger_node_name: [runner_submod_name, ...]}``  — for each FX
            node, the list of runner submodule names that become safe to finalize
            once that node has executed.  Most entries are empty (omitted).
        """
        # Build topo index: node.name -> int position in graph
        topo_index: Dict[str, int] = {}
        for idx, node in enumerate(split_gm.graph.nodes):
            topo_index[node.name] = idx

        # For each runner node, find the last (in topo order) direct consumer
        # of its output.  If the runner has no users (unlikely but possible for
        # dead code), we finalize immediately after the runner itself.
        runner_last_consumer: Dict[str, str] = {}  # runner_target -> trigger node name

        for node in split_gm.graph.nodes:
            if node.op != "call_module":
                continue
            submod = getattr(split_gm, node.target, None)
            if not isinstance(submod, ADPiecewiseRunner):
                continue

            if not node.users:
                # No consumers — finalize right after this node
                runner_last_consumer[node.target] = node.name
                continue

            last_user = max(node.users, key=lambda u: topo_index.get(u.name, -1))
            runner_last_consumer[node.target] = last_user.name

        # Invert: trigger_node -> [runners to finalize]
        schedule: Dict[str, List[str]] = defaultdict(list)
        for runner_target, trigger_name in runner_last_consumer.items():
            schedule[trigger_name].append(runner_target)

        return dict(schedule)

    @staticmethod
    def compute_registry_prune_schedule(
        split_gm: "torch.fx.GraphModule",
    ) -> Dict[str, List[str]]:
        """Compute when it is safe to prune registry entries for each runner.

        For a producer runner A, registry entries must remain available until all
        downstream static runners that can directly consume A's output have
        captured. We approximate this with a frontier walk:

        - Traverse users from A through non-runner nodes.
        - The first ADPiecewiseRunner encountered on each path is a potential
          capture-time lookup consumer of A.
        - Do not traverse past that runner.

        A's registry entry can be pruned right after the latest frontier runner
        executes. This is less conservative than using "latest reachable runner"
        and restores memory reuse while preserving lookup correctness.

        Returns:
            ``{trigger_node_name: [runner_submod_name, ...]}`` mapping a trigger
            node to producers whose registry entries are safe to prune after that
            node executes.
        """
        topo_index: Dict[str, int] = {}
        for idx, node in enumerate(split_gm.graph.nodes):
            topo_index[node.name] = idx

        runner_nodes: Dict[str, torch.fx.Node] = {}
        for node in split_gm.graph.nodes:
            if node.op != "call_module":
                continue
            submod = getattr(split_gm, node.target, None)
            if isinstance(submod, ADPiecewiseRunner):
                runner_nodes[node.target] = node

        prune_trigger: Dict[str, str] = {}  # runner_target -> trigger node name
        for runner_target, runner_node in runner_nodes.items():
            visited: Set[str] = set()
            stack = list(runner_node.users.keys())
            frontier_runner_nodes: List[torch.fx.Node] = []

            while stack:
                user = stack.pop()
                if user.name in visited:
                    continue
                visited.add(user.name)

                if user.op == "call_module":
                    submod = getattr(split_gm, user.target, None)
                    if isinstance(submod, ADPiecewiseRunner):
                        # First downstream runner on this path: keep as frontier
                        # consumer and stop traversing beyond it.
                        frontier_runner_nodes.append(user)
                        continue

                # Non-runner node: keep exploring along this path.
                stack.extend(user.users.keys())

            if frontier_runner_nodes:
                trigger_node = max(
                    frontier_runner_nodes,
                    key=lambda n: topo_index.get(n.name, -1),
                )
                prune_trigger[runner_target] = trigger_node.name
            else:
                # No downstream static runner capture needs this producer.
                prune_trigger[runner_target] = runner_node.name

        schedule: Dict[str, List[str]] = defaultdict(list)
        for runner_target, trigger_name in prune_trigger.items():
            schedule[trigger_name].append(runner_target)
        return dict(schedule)
