"""Compile backend with cudagraph.

1. Monolithic CUDA graph: captures entire model as one graph for decode-only.
2. Piecewise CUDA graph: splits model at dynamic ops, captures static segments
   individually. Used for prefill/mixed batches when piecewise_enabled=True.

When piecewise_enabled=True, a DualModeCapturedGraph is returned that dispatches:
  - Decode-only batches → monolithic CapturedGraph (fastest, single graph replay)
  - Prefill/mixed batches → PiecewiseCapturedGraph (per-segment replay + eager dynamic ops)
"""

import copy  # noqa: I001
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from torch.fx import GraphModule, Interpreter
from torch.fx._pytree import tree_flatten_spec
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten

from tensorrt_llm._torch.autotuner import autotune

from ...utils.cuda_graph import CudaGraphWarmUpPhase
from ...utils.logger import ad_logger
from ..compiler import CompileBackendRegistry, CompilerBackend, GetArgsKwargsForBatchSize
from ..piecewise_runner import ADPiecewiseRunner
from ..piecewise_utils import SplitInfo, split_graph_at_dynamic_ops


def _log_cuda_mem(label: str) -> None:
    """Log current CUDA memory state without side effects (no empty_cache)."""
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    resv = torch.cuda.memory_reserved()
    peak_alloc = torch.cuda.max_memory_allocated()
    frag = resv - alloc
    ad_logger.info(
        f"[CUDA MEM] {label} | "
        f"alloc: {alloc / (1 << 30):.2f}GB | "
        f"resv: {resv / (1 << 30):.2f}GB | "
        f"free: {free / (1 << 30):.2f}GB | "
        f"frag: {frag / (1 << 30):.2f}GB | "
        f"peak_alloc: {peak_alloc / (1 << 30):.2f}GB"
    )


class IntraBucketFinalizeInterpreter(Interpreter):
    """FX Interpreter that finalizes static runners as soon as their outputs
    are fully consumed during the capture pass.

    Combines Phase A (garbage_collect_values=True for FX env refs) with
    Phase B (finalize_entry for runner-held refs + registry refs).

    The finalize schedule is computed from the FX graph topology by
    :meth:`ADPiecewiseRunner.compute_finalize_schedule` — each runner is
    finalized only after its LAST consumer node has executed, which is safe
    even when skip connections exist in the graph.
    """

    def __init__(
        self,
        split_gm: GraphModule,
        finalize_schedule: Dict[str, List[str]],
        registry_prune_schedule: Dict[str, List[str]],
        num_tokens: int,
    ):
        super().__init__(split_gm, garbage_collect_values=True)
        self.finalize_schedule = finalize_schedule
        self.registry_prune_schedule = registry_prune_schedule
        self.num_tokens = num_tokens

    def run_node(self, n):
        result = super().run_node(n)

        # Drop runner-held refs as soon as execution liveness allows, but keep
        # registry entries until downstream runner captures have completed.
        runners_to_finalize = self.finalize_schedule.get(n.name, [])
        for runner_name in runners_to_finalize:
            runner = getattr(self.module, runner_name, None)
            if isinstance(runner, ADPiecewiseRunner):
                runner.finalize_entry(
                    self.num_tokens,
                    prune_registry=False,
                    drop_static_inputs=True,
                )
                ad_logger.debug(
                    f"IntraBucketFinalizeInterpreter: finalized {runner_name} "
                    f"for num_tokens={self.num_tokens} (trigger: {n.name})"
                )

        # Prune registry entries only when no downstream static runner capture
        # should need lookup for the producer anymore.
        runners_to_prune = self.registry_prune_schedule.get(n.name, [])
        for runner_name in runners_to_prune:
            runner = getattr(self.module, runner_name, None)
            if isinstance(runner, ADPiecewiseRunner):
                runner.prune_registry_for_entry(self.num_tokens)
                ad_logger.debug(
                    f"IntraBucketFinalizeInterpreter: pruned registry for {runner_name} "
                    f"num_tokens={self.num_tokens} (trigger: {n.name})"
                )

        return result


# Trivial FX ops that are metadata-only or typically no-ops — used to identify
# static segments with no meaningful GPU compute (e.g., between adjacent dynamic ops).
# NOTE: reshape, contiguous, and to *can* launch kernels in edge cases (non-contiguous
# tensors, dtype/device casts), but in practice these appear only as lightweight
# plumbing in empty partitions.
_TRIVIAL_CALL_FUNCTIONS = {operator.getitem, getattr}
_TRIVIAL_CALL_METHODS = {
    "view",
    "reshape",
    "contiguous",
    "permute",
    "transpose",
    "unsqueeze",
    "squeeze",
    "expand",
    "size",
    "dim",
    "to",
}


def _submod_has_cuda_ops(submod: nn.Module) -> bool:
    """Check if a submodule has ops beyond those in _TRIVIAL_CALL_FUNCTIONS/METHODS."""
    if not isinstance(submod, GraphModule):
        return True  # Conservative: non-FX modules assumed to have GPU ops

    for node in submod.graph.nodes:
        if node.op == "call_module":
            # nn.Module calls (Linear, LayerNorm, etc.) launch CUDA kernels
            return True
        if node.op == "call_function":
            if node.target in _TRIVIAL_CALL_FUNCTIONS:
                continue
            # Any non-trivial call_function is potentially a CUDA op
            return True
        if node.op == "call_method":
            if node.target in _TRIVIAL_CALL_METHODS:
                continue
            # Non-trivial method call — could launch a kernel
            return True

    return False


def _args_kwargs_flatten_spec(in_spec: TreeSpec, *args, **kwargs) -> List[Any]:
    """Flatten inputs according to provided in_spec."""
    all_args: PyTree = (args, kwargs)
    return tree_flatten_spec(all_args, in_spec)


def _args_kwargs_flatten(*args, **kwargs) -> Tuple[List[Any], TreeSpec]:
    """Flatten inputs and return flattened inputs together with the TreeSpec."""
    all_args: PyTree = (args, kwargs)
    return tree_flatten(all_args)


class CapturedGraph(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_batched_inputs: Optional[int] = None,  # number of batched, dynamic inputs...
    ):
        super().__init__()
        self.model = model
        self.num_batched_inputs = num_batched_inputs if num_batched_inputs is not None else 1
        self.cudagraphs: Dict[Tuple[int, ...], CUDAGraph] = {}
        self._input_buffers: List[torch.Tensor] = [
            torch.empty(0, 1) for _ in range(self.num_batched_inputs)
        ]
        self._out_buffer_flat: List[torch.Tensor] = None
        self._args_hash: Optional[Tuple[int, ...]] = None
        self._cuda_graph_mem_pool = None

        # store the in_spec and out_spec during graph capture
        self._in_spec = None
        self._out_spec = None

    def _get_hash(self, flat_args: List[Any]) -> Tuple[int, ...]:
        return tuple(hash(a) for a in flat_args)

    def _capture_one_graph(self, *args, **kwargs) -> torch.cuda.CUDAGraph:
        """Capture and return one cuda graph."""
        # warm-up and invoke autotuner
        with CudaGraphWarmUpPhase(), autotune():
            for _ in range(3):
                self.model(*args, **kwargs)

        # capture graph now
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._cuda_graph_mem_pool):
            # compute output
            out = self.model(*args, **kwargs)
            # write out into output buffer up to out batch size
            out_flat = tree_flatten_spec(out, self._out_spec)
            for o_buffer, o in zip(self._out_buffer_flat, out_flat):
                o_buffer[: o.shape[0]] = o
        torch.cuda.synchronize()
        self._cuda_graph_mem_pool = self._cuda_graph_mem_pool or graph.pool()
        return graph

    def capture_graph(self, get_args_kwargs: GetArgsKwargsForBatchSize, batch_sizes: List[int]):
        """Capture and pre-fetch the graph for desired batch sizes."""
        assert not self.cudagraphs, "Graphs already captured."

        # sort batch sizes in descending order
        batch_sizes = sorted(batch_sizes, reverse=True)

        # get args, kwargs for the first time for the largest batch size
        args, kwargs = get_args_kwargs(batch_sizes[0])

        # flatten args, kwargs for the first time and record in_spec
        all_args_flat, self._in_spec = _args_kwargs_flatten(*args, **kwargs)

        # extract the batched input tensors
        args_batched = all_args_flat[: self.num_batched_inputs]
        args_static = all_args_flat[self.num_batched_inputs :]

        # set the args hash --> this is used to compare the static inputs during graph replay
        self._args_hash = self._get_hash(args_static)

        # store the input buffers for the largest batch size
        self._input_buffers = [a.clone() for a in args_batched]

        # create new args, kwargs with the input buffers and static args
        args, kwargs = self._in_spec.unflatten(self._input_buffers + args_static)

        # capture output once with cuda_graph_max_batch_size to capture output buffers
        # store the out_spec at this point
        with CudaGraphWarmUpPhase():
            ad_logger.info(f"Warm up with max_batch_size={batch_sizes[0]} before graph capture")
            out = self.model(*args, **kwargs)
        self._out_buffer_flat, self._out_spec = tree_flatten(out)

        # capture graph now for a range of batch sizes
        for bs in batch_sizes:
            ad_logger.info(f"Capturing graph for batch size: {bs}")

            # get new args, kwargs for the current batch size
            args, kwargs = get_args_kwargs(bs)
            all_args_flat = _args_kwargs_flatten_spec(self._in_spec, *args, **kwargs)
            args_batched = all_args_flat[: self.num_batched_inputs]
            args_static = all_args_flat[self.num_batched_inputs :]

            # assert that static args match the stored hash
            assert self._args_hash == self._get_hash(args_static), (
                "Static args mismatch during capture"
            )

            # copy new inputs to input buffers
            for i, input_tensor in enumerate(args_batched):
                self._input_buffers[i][: input_tensor.shape[0]].copy_(
                    input_tensor, non_blocking=True
                )

            # setup args, kwargs
            inputs_truncated = [in_buffer[:bs] for in_buffer in self._input_buffers]
            args, kwargs = self._in_spec.unflatten(inputs_truncated + args_static)

            # capture graph for truncated inputs
            combined_shape = sum((tuple(input.shape) for input in inputs_truncated), start=())
            self.cudagraphs[combined_shape] = self._capture_one_graph(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        """Run the compiled graph."""
        # flatten args, kwargs
        all_args_flat = _args_kwargs_flatten_spec(self._in_spec, *args, **kwargs)

        # extract the batched input tensors
        args_batched = all_args_flat[: self.num_batched_inputs]
        args_static = all_args_flat[self.num_batched_inputs :]

        # check if args_static match the stored hash
        if self._args_hash != self._get_hash(args_static):
            return self.model(*args, **kwargs)

        # Calculate combined shape tuple as hash for cudagraph lookup
        combined_shape = sum((arg.shape for arg in args_batched), start=())

        # regular forward for non-matching shapes
        if combined_shape not in self.cudagraphs:
            return self.model(*args, **kwargs)

        # copy inputs to input buffers
        for i, input_tensor in enumerate(args_batched):
            self._input_buffers[i][: input_tensor.shape[0]].copy_(input_tensor, non_blocking=True)

        # run forward pass via graph
        self.cudagraphs[combined_shape].replay()

        # retrieve output from buffer, cut to batch size, and unflatten
        bs = args_batched[0].shape[0]
        out_flat = [o_b[:bs] for o_b in self._out_buffer_flat]
        return self._out_spec.unflatten(out_flat)


class PiecewiseCapturedGraph(nn.Module):
    """Manages piecewise CUDA graph capture/replay for prefill/mixed batches.

    The model is split at dynamic op boundaries (attention, SSM, conv, delta).
    Static segments are wrapped in ADPiecewiseRunner for CUDA graph capture.
    Dynamic segments run eagerly. The split_gm orchestrates the flow.
    """

    def __init__(
        self,
        model: nn.Module,
        piecewise_num_tokens: Optional[List[int]] = None,
    ):
        super().__init__()
        self.original_model = model
        self.piecewise_num_tokens = piecewise_num_tokens or []
        self.split_info: Optional[SplitInfo] = None
        self.split_gm: Optional[GraphModule] = None
        self._is_prepared = False

    def prepare(self) -> None:
        """Prepare the piecewise graph: swap to inplace ops, split, wrap static segments."""
        if self._is_prepared:
            return

        model = self.original_model
        if not isinstance(model, GraphModule):
            ad_logger.warning(
                "PiecewiseCapturedGraph: model is not a GraphModule, "
                "piecewise CUDA graph requires an FX GraphModule. "
                "Falling back to eager execution."
            )
            self._is_prepared = True
            return

        # Create a new GraphModule that shares all parameters/buffers/submodules
        # with the original (zero-copy) but has its OWN copy of the FX graph
        # (so split_graph_at_dynamic_ops mutations don't affect the original).
        gm = GraphModule(model, copy.deepcopy(model.graph))

        # Split graph at dynamic op boundaries
        self.split_info = split_graph_at_dynamic_ops(gm)
        self.split_gm = self.split_info.split_gm

        # Skip trivial submodules that have no CUDA ops (only contain getitem/reshape plumbing).
        # Capturing these as CUDA graphs produces empty graphs and triggers PyTorch warnings.
        # Create a shared pool upfront so all runners share memory allocations.
        graph_pool = torch.cuda.graph_pool_handle()
        # Graph-level parameter/buffer pointers are used by runners to avoid
        # misclassifying forwarded weight tensors as dynamic inputs.
        global_weight_ptrs = {p.data_ptr() for p in self.split_gm.parameters()}
        global_weight_ptrs.update(b.data_ptr() for b in self.split_gm.buffers())
        num_wrapped = 0
        num_skipped = 0
        for idx in self.split_info.static_submod_indices:
            submod_name = f"submod_{idx}"
            if hasattr(self.split_gm, submod_name):
                original_submod = getattr(self.split_gm, submod_name)

                if not _submod_has_cuda_ops(original_submod):
                    ad_logger.info(
                        f"PiecewiseCapturedGraph: skipping {submod_name} "
                        f"(no CUDA ops, will run eagerly)"
                    )
                    num_skipped += 1
                    continue

                runner = ADPiecewiseRunner(
                    submodule=original_submod,
                    piecewise_num_tokens=self.piecewise_num_tokens,
                    graph_pool=graph_pool,
                    global_weight_ptrs=global_weight_ptrs,
                )
                setattr(self.split_gm, submod_name, runner)
                num_wrapped += 1

        self._is_prepared = True
        ad_logger.info(
            f"PiecewiseCapturedGraph: prepared with "
            f"{self.split_info.num_submodules} submodules "
            f"({num_wrapped} wrapped for CUDA graph, {num_skipped} trivial skipped, "
            f"{len(self.split_info.dynamic_submod_indices)} dynamic eager), "
            f"piecewise_num_tokens={self.piecewise_num_tokens}"
        )

    def _get_placeholder_names(self) -> List[str]:
        """Return placeholder node names in graph order (for Interpreter.run)."""
        return [n.target for n in self.split_gm.graph.nodes if n.op == "placeholder"]

    def warmup_and_capture(
        self,
        get_args_kwargs: Callable[[int], Any],
        warmup_iters: int = 3,
    ) -> None:
        """Warmup and capture CUDA graphs for all configured num_tokens values.

        Follows the same pattern as monolithic CapturedGraph._capture_one_graph:
        the orchestrator controls the warmup → capture transition explicitly.

        The capture phase uses an IntraBucketFinalizeInterpreter that:
          - Eagerly frees FX intermediate values (garbage_collect_values=True)
          - Finalizes each static runner after its last consumer node executes
        This dramatically reduces peak graph-pool usage during capture of large
        buckets by releasing cross-submodule activation tensors as they become dead.

        Args:
            get_args_kwargs: Callable that takes num_tokens and returns (args, kwargs).
            warmup_iters: Number of eager warmup iterations before capture (default: 3,
                matching monolithic CapturedGraph._capture_one_graph).
        """
        if not self._is_prepared:
            self.prepare()

        if self.split_gm is None:
            return

        runners: List[ADPiecewiseRunner] = [
            m for m in self.split_gm.modules() if isinstance(m, ADPiecewiseRunner)
        ]

        # Compute the intra-bucket finalize schedule once from the graph topology.
        # This maps each FX node name → list of runner submodule names that become
        # safe to finalize after that node executes.
        finalize_schedule = ADPiecewiseRunner.compute_finalize_schedule(self.split_gm)
        registry_prune_schedule = ADPiecewiseRunner.compute_registry_prune_schedule(self.split_gm)
        ad_logger.info(
            f"PiecewiseCapturedGraph: computed intra-bucket finalize schedule "
            f"({sum(len(v) for v in finalize_schedule.values())} runners across "
            f"{len(finalize_schedule)} trigger nodes)"
        )
        ad_logger.info(
            f"PiecewiseCapturedGraph: computed registry-prune schedule "
            f"({sum(len(v) for v in registry_prune_schedule.values())} runners across "
            f"{len(registry_prune_schedule)} trigger nodes)"
        )

        # Placeholder names for converting kwargs → positional args for Interpreter.run()
        placeholder_names = self._get_placeholder_names()

        # Sort num_tokens in descending order (largest first for memory allocation)
        num_tokens_list = sorted(self.piecewise_num_tokens, reverse=True)
        for idx, nt in enumerate(num_tokens_list):
            ad_logger.info(f"PiecewiseCapturedGraph: warming up for num_tokens={nt}")
            args, kwargs = get_args_kwargs(nt)
            # Warm up only the largest bucket to avoid repeated eager-pass peak
            # allocations (e.g., large to_dtype temporaries) on smaller buckets.
            # Smaller buckets are captured directly using the same split graph.
            warmup_iters_nt = warmup_iters if idx == 0 else 0

            # Set the num_tokens context so ALL ADPiecewiseRunners use the correct value.
            # This is critical: in piecewise-split models, some submodules receive
            # intermediate tensors (SSM metadata, chunk indices) whose dim0 != num_tokens,
            # so inferring from arg shapes is unreliable.
            ADPiecewiseRunner.set_current_num_tokens(nt)

            # Reset peak stats so we can measure per-phase peaks
            torch.cuda.reset_peak_memory_stats()
            _log_cuda_mem(f"nt={nt} before_warmup")

            with CudaGraphWarmUpPhase():
                ADPiecewiseRunner.set_current_phase("warmup")
                for _ in range(warmup_iters_nt):
                    self.split_gm(*args, **kwargs)

                _log_cuda_mem(f"nt={nt} after_warmup")

                # Free warmup activation blocks before capture so the graph pool
                # can use the reclaimed CUDA memory instead of competing with it.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                _log_cuda_mem(f"nt={nt} after_warmup_empty_cache")
                # Reset peak stats again so capture peak is measured cleanly
                torch.cuda.reset_peak_memory_stats()

                # Capture phase: use IntraBucketFinalizeInterpreter to capture
                # CUDA graphs while aggressively freeing intermediate tensors.
                ADPiecewiseRunner.set_current_phase("capture")
                ADPiecewiseRunner.reset_registry_prune_hits(nt)

                # Convert kwargs → positional args matching placeholder order
                interp_args = []
                for name in placeholder_names:
                    if name in kwargs:
                        interp_args.append(kwargs[name])

                interp = IntraBucketFinalizeInterpreter(
                    self.split_gm,
                    finalize_schedule,
                    registry_prune_schedule,
                    nt,
                )
                interp.run(*interp_args)

            _log_cuda_mem(f"nt={nt} after_capture")
            # Debug telemetry: how many dynamic inputs are currently eligible for
            # reconstruction (registry-backed) vs pinned, and how many registry
            # entries were actually pruned during capture.
            total_dyn_inputs = 0
            total_reconstructable_inputs = 0
            total_dyn_input_bytes = 0
            total_reconstructable_input_bytes = 0
            total_pinned_input_bytes = 0
            unique_dyn_input_bytes = 0
            unique_reconstructable_input_bytes = 0
            unique_pinned_input_bytes = 0
            seen_dyn_ptrs = set()
            seen_reconstructable_ptrs = set()
            seen_pinned_ptrs = set()
            total_output_tensors = 0
            total_finalized_runners = 0
            for runner in runners:
                entry = runner.entries.get(nt)
                if entry is None:
                    continue
                if entry.dynamic_indices is not None:
                    total_dyn_inputs += len(entry.dynamic_indices)
                    reconstructable_set = entry._reconstructable_input_indices or set()
                    for idx in entry.dynamic_indices:
                        if entry.static_inputs is None or idx >= len(entry.static_inputs):
                            continue
                        inp = entry.static_inputs[idx]
                        if not isinstance(inp, torch.Tensor):
                            continue
                        nbytes = inp.untyped_storage().nbytes()
                        ptr = inp.untyped_storage().data_ptr()
                        total_dyn_input_bytes += nbytes
                        if ptr not in seen_dyn_ptrs:
                            unique_dyn_input_bytes += nbytes
                            seen_dyn_ptrs.add(ptr)

                        if idx in reconstructable_set:
                            total_reconstructable_input_bytes += nbytes
                            if ptr not in seen_reconstructable_ptrs:
                                unique_reconstructable_input_bytes += nbytes
                                seen_reconstructable_ptrs.add(ptr)
                        else:
                            total_pinned_input_bytes += nbytes
                            if ptr not in seen_pinned_ptrs:
                                unique_pinned_input_bytes += nbytes
                                seen_pinned_ptrs.add(ptr)
                total_reconstructable_inputs += len(entry._reconstructable_input_indices or set())
                if entry._finalized is not None:
                    total_finalized_runners += 1
                    total_output_tensors += sum(
                        1 for meta in entry._finalized.output_metadata if meta is not None
                    )
                elif entry.static_output is not None:
                    flat_output, _ = tree_flatten(entry.static_output)
                    total_output_tensors += sum(
                        1 for out_tensor in flat_output if isinstance(out_tensor, torch.Tensor)
                    )
            ad_logger.info(
                f"PiecewiseCapturedGraph: nt={nt} capture_stats | "
                f"dynamic_inputs={total_dyn_inputs} | "
                f"reconstructable_inputs={total_reconstructable_inputs} | "
                f"pinned_inputs={max(total_dyn_inputs - total_reconstructable_inputs, 0)} | "
                f"dynamic_input_bytes={total_dyn_input_bytes / (1 << 30):.2f}GB | "
                f"reconstructable_input_bytes={total_reconstructable_input_bytes / (1 << 30):.2f}GB | "
                f"pinned_input_bytes={total_pinned_input_bytes / (1 << 30):.2f}GB | "
                f"dynamic_input_bytes_unique={unique_dyn_input_bytes / (1 << 30):.2f}GB | "
                f"reconstructable_input_bytes_unique={unique_reconstructable_input_bytes / (1 << 30):.2f}GB | "
                f"pinned_input_bytes_unique={unique_pinned_input_bytes / (1 << 30):.2f}GB | "
                f"output_tensors={total_output_tensors} | "
                f"finalized_runners={total_finalized_runners}/{len(runners)} | "
                f"registry_prune_hits={ADPiecewiseRunner.get_registry_prune_hits(nt)}"
            )
            ad_logger.info(f"PiecewiseCapturedGraph: captured graphs for num_tokens={nt}")

            # Keep entries in finalized form during multi-bucket capture to
            # minimize live graph-pool references and maximize reuse. We defer
            # materialization for all buckets until after the capture loop.
            _log_cuda_mem(f"nt={nt} after_finalize")

            # Cross-bucket cleanup: clear registry + empty_cache so the
            # caching allocator can reuse freed blocks for the next bucket.
            if idx < len(num_tokens_list) - 1:
                ADPiecewiseRunner.clear_static_output_registry()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                _log_cuda_mem(f"nt={nt} after_cross_bucket_cleanup")

        # Reconstruct finalized entries after all buckets are captured.
        for nt in num_tokens_list:
            for runner in runners:
                runner.materialize_entry(nt)

        # Clear contexts after warmup/capture phase
        ADPiecewiseRunner.set_current_num_tokens(None)
        ADPiecewiseRunner.set_current_phase("replay")
        ADPiecewiseRunner.clear_static_output_registry()

    def forward(self, *args, num_tokens: Optional[int] = None, **kwargs) -> Any:
        """Forward pass through the piecewise graph.

        Each submodule handles its own capture/replay:
        - Static submodules (ADPiecewiseRunner): replay CUDA graph if available
        - Dynamic submodules: run eagerly

        Args:
            num_tokens: The total number of tokens in this batch. Must be provided
                by the caller (DualModeCapturedGraph) — we cannot reliably infer it
                from arg shapes because kwargs like input_ids may be [1, num_tokens]
                (shape[0]=1, not num_tokens) and the first kwarg might not be input_ids.
        """
        if self.split_gm is not None:
            # Set num_tokens context for all ADPiecewiseRunners.
            ADPiecewiseRunner.set_current_num_tokens(num_tokens)
            result = self.split_gm(*args, **kwargs)
            return result
        else:
            # Fallback: model is not a GraphModule, run eagerly
            return self.original_model(*args, **kwargs)


class DualModeCapturedGraph(nn.Module):
    """Dispatches between monolithic CG (decode) and piecewise CG (prefill/mixed).

    At runtime:
    - If batch is decode-only (num_prefill == 0) -> use monolithic CapturedGraph
    - If batch has prefill/mixed tokens and total num_tokens <= largest pre-captured
      bucket -> use PiecewiseCapturedGraph with the smallest bucket >= num_tokens
    - Otherwise -> fall back to eager

    Padding is handled upstream by SequenceInfo._padded_num_tokens: input_ids and
    position_ids are shaped to the bucket size via _shape_for_forward, while all
    metadata (batch_info_host, cu_seqlens, etc.) remains unchanged so dynamic ops
    process only real tokens. Output logits are truncated back to the real token
    count after the forward pass.
    """

    def __init__(
        self,
        monolithic: CapturedGraph,
        piecewise: PiecewiseCapturedGraph,
        batch_info_kwarg_name: str = "batch_info_host",
        batched_input_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.monolithic = monolithic
        self.piecewise = piecewise
        self.batch_info_kwarg_name = batch_info_kwarg_name
        # Names of kwargs used to infer total num_tokens
        self.batched_input_names = batched_input_names or ["input_ids", "position_ids"]

        # Sorted list of pre-captured bucket sizes for nearest-bucket lookup
        self._captured_num_tokens_sorted: List[int] = sorted(piecewise.piecewise_num_tokens)

    def _is_decode_only(self, **kwargs) -> bool:
        """Check if the current batch is decode-only using batch_info_host.

        batch_info_host = [num_prefill, num_prefill_tokens, num_decode]
        Decode-only means num_prefill == 0.
        """
        batch_info = kwargs.get(self.batch_info_kwarg_name)
        if batch_info is not None and isinstance(batch_info, torch.Tensor):
            # batch_info_host[0] = num_prefill
            num_prefill = batch_info[0].item()
            return num_prefill == 0

        # Fallback heuristic: check if first batched input has sequence dim == 1
        # (decode = 1 token per sequence)
        for name in self.batched_input_names:
            v = kwargs.get(name)
            if v is not None and isinstance(v, torch.Tensor) and v.ndim >= 2:
                return v.shape[1] == 1

        # Default to monolithic (decode) path
        return True

    def _get_num_tokens(self, **kwargs) -> int:
        """Extract total num_tokens from the batched inputs.

        For prefill/mixed with flattened layout: input_ids shape = [1, total_num_tokens]
        We use numel() which works for both [1, N] and [N] layouts.
        """
        for name in self.batched_input_names:
            v = kwargs.get(name)
            if v is not None and isinstance(v, torch.Tensor):
                return v.numel()
        return 0

    def _find_nearest_bucket(self, num_tokens: int) -> Optional[int]:
        """Find smallest captured bucket >= num_tokens, or None."""
        for bucket in self._captured_num_tokens_sorted:
            if bucket >= num_tokens:
                return bucket
        return None

    def forward(self, *args, **kwargs) -> Any:
        # NOTE: AD calls model(**named_args) so everything is in kwargs, args is empty
        if self._is_decode_only(**kwargs):
            return self.monolithic(*args, **kwargs)

        # ── PREFILL/MIXED PATH ──
        num_tokens = self._get_num_tokens(**kwargs)
        bucket = self._find_nearest_bucket(num_tokens)
        if bucket is not None:
            # Piecewise CG path -- padding is handled upstream by SequenceInfo
            return self.piecewise(*args, num_tokens=bucket, **kwargs)

        # No bucket large enough -- eager fallback
        ad_logger.debug(
            f"DualModeCapturedGraph: num_tokens={num_tokens} exceeds largest bucket "
            f"{self._captured_num_tokens_sorted[-1] if self._captured_num_tokens_sorted else 'N/A'}"
            f", falling back to eager"
        )
        return self.piecewise.original_model(*args, **kwargs)


@CompileBackendRegistry.register("torch-cudagraph")
class TorchCudagraphCompiler(CompilerBackend):
    """Compiler that uses CUDA graphs.

    Supports two modes:
    - piecewise_enabled=False (default): monolithic CG only (decode-only batches)
    - piecewise_enabled=True: dual-mode (monolithic for decode + piecewise for prefill/mixed)
    """

    def __init__(
        self,
        *args_for_init,
        cuda_graph_batch_sizes: Optional[List[int]] = None,
        num_batched_inputs: int = 1,
        get_args_kwargs_for_compile: GetArgsKwargsForBatchSize = None,
        piecewise_enabled: bool = False,
        piecewise_num_tokens: Optional[List[int]] = None,
        get_mixed_args_kwargs_for_compile: Optional[Callable[[int], Any]] = None,
        **kwargs_for_init,
    ):
        super().__init__(*args_for_init, **kwargs_for_init)
        self.num_batched_inputs = num_batched_inputs
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes or []
        self.get_args_kwargs_for_compile = get_args_kwargs_for_compile
        self.piecewise_enabled = piecewise_enabled
        self.piecewise_num_tokens = piecewise_num_tokens or []
        self.get_mixed_args_kwargs_for_compile = get_mixed_args_kwargs_for_compile

    @torch.inference_mode()
    def compile(self) -> nn.Module:
        assert self.get_args_kwargs_for_compile is not None, (
            "get_args_kwargs_for_compile must be provided"
        )

        monolithic = CapturedGraph(self.model, num_batched_inputs=self.num_batched_inputs)
        monolithic.capture_graph(self.get_args_kwargs_for_compile, self.cuda_graph_batch_sizes)

        piecewise = None
        if self.piecewise_enabled:
            ad_logger.info("TorchCudagraphCompiler: dual-mode enabled (monolithic + piecewise)")
            piecewise = PiecewiseCapturedGraph(
                model=self.model,
                piecewise_num_tokens=self.piecewise_num_tokens,
            )
            piecewise.prepare()

            if self.get_mixed_args_kwargs_for_compile is not None and self.piecewise_num_tokens:
                piecewise.warmup_and_capture(self.get_mixed_args_kwargs_for_compile)

        if piecewise is not None:
            return DualModeCapturedGraph(monolithic, piecewise)
        return monolithic
