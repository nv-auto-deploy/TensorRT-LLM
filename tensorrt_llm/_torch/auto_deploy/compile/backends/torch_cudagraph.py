"""Compile backend with cudagraph."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from torch.fx._pytree import tree_flatten_spec
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten

from tensorrt_llm._torch.autotuner import autotune

from ...utils.cuda_graph import CudaGraphWarmUpPhase
from ...utils.logger import ad_logger
from ..compiler import CompileBackendRegistry, CompilerBackend


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
        cuda_graph_batch_sizes: List[int],
        num_batched_inputs: int,  # number of batched, dynamic inputs...
    ):
        super().__init__()
        self.model = model
        self.cuda_graph_max_batch_size = max(cuda_graph_batch_sizes)
        ad_logger.info(f"Setting {self.cuda_graph_max_batch_size=}")
        self.num_batched_inputs = num_batched_inputs if num_batched_inputs is not None else 1
        self.cudagraphs: Dict[Tuple[int, ...], CUDAGraph] = {}
        self._input_buffers: List[torch.Tensor] = [
            torch.empty(0, 1) for _ in range(self.num_batched_inputs)
        ]
        self._out_buffer_flat: List[torch.Tensor] = None
        self._args_hash: Optional[Tuple[int, ...]] = None
        self.cuda_graph_batch_sizes = sorted(cuda_graph_batch_sizes, reverse=True)
        self._cuda_graph_mem_pool = None

        # store the in_spec and out_spec during graph capture
        self._in_spec = None
        self._out_spec = None

    def _get_hash(self, flat_args: List[Any]) -> Tuple[int, ...]:
        return tuple(hash(a) for a in flat_args)

    @staticmethod
    def round_up_to_closest(batch_sizes: Iterable[int], bs: int) -> Optional[int]:
        """Return closest batch size larger or equal to bs."""
        if bs > max(batch_sizes, default=0):
            return None
        return min(batch_sizes, key=lambda x: (x < bs, abs(x - bs)), default=None)

    def round_to_cuda_batch_size(self, bs: int) -> int:
        """Round batch size to the nearest cuda batch size."""
        return self.round_up_to_closest(self.cuda_graph_batch_sizes, bs)

    def _capture_one_graph(self, *args, **kwargs) -> torch.cuda.CUDAGraph:
        """Capture and return one cuda graph."""
        import os
        import time

        # Determine batch size from first batched input
        all_args_flat = _args_kwargs_flatten_spec(self._in_spec, *args, **kwargs)
        batch_size = all_args_flat[0].shape[0] if len(all_args_flat) > 0 else "unknown"

        # Enable profiling if requested via environment variable
        enable_profiling = os.environ.get("TRTLLM_PROFILE_CUDA_GRAPH_CAPTURE", "0") == "1"
        profile_dir = os.environ.get("TRTLLM_PROFILE_DIR", "./cuda_graph_profiles")

        # Check if this is a torch-compiled model to reduce warmup iterations
        # For compiled models, the first iteration does the compilation,
        # so we only need 1 warmup iteration instead of 3
        warmup_iters = getattr(self, "_warmup_iters", 3)

        capture_start_time = time.time()
        ad_logger.info(f"[AutoDeploy Backend]   -> Starting capture for batch_size={batch_size}")

        if enable_profiling:
            os.makedirs(profile_dir, exist_ok=True)
            ad_logger.info(
                f"[AutoDeploy Backend]     -> Profiling enabled for batch_size={batch_size}"
            )

        # warm-up and invoke autotuner with optional profiling
        warmup_start_time = time.time()
        if enable_profiling:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
            ) as prof:
                with CudaGraphWarmUpPhase(), autotune():
                    for _ in range(warmup_iters):
                        self.model(*args, **kwargs)
            warmup_trace = os.path.join(profile_dir, f"ad_backend_bs{batch_size}_warmup.json")
            prof.export_chrome_trace(warmup_trace)
            ad_logger.info(f"[AutoDeploy Backend]     -> Warmup trace saved: {warmup_trace}")
        else:
            with CudaGraphWarmUpPhase(), autotune():
                for _ in range(warmup_iters):
                    self.model(*args, **kwargs)

        warmup_time = time.time() - warmup_start_time
        ad_logger.info(
            f"[AutoDeploy Backend]     -> Warmup ({warmup_iters} iterations) "
            f"took {warmup_time:.4f}s"
        )

        # capture graph now
        torch.cuda.synchronize()
        actual_capture_start = time.time()
        graph = torch.cuda.CUDAGraph()

        if enable_profiling:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
            ) as prof:
                with torch.cuda.graph(graph, pool=self._cuda_graph_mem_pool):
                    # compute output
                    out = self.model(*args, **kwargs)
                    # write out into output buffer up to out batch size
                    out_flat = tree_flatten_spec(out, self._out_spec)
                    for o_buffer, o in zip(self._out_buffer_flat, out_flat):
                        o_buffer[: o.shape[0]] = o
            capture_trace = os.path.join(profile_dir, f"ad_backend_bs{batch_size}_capture.json")
            prof.export_chrome_trace(capture_trace)
            ad_logger.info(f"[AutoDeploy Backend]     -> Capture trace saved: {capture_trace}")
        else:
            with torch.cuda.graph(graph, pool=self._cuda_graph_mem_pool):
                # compute output
                out = self.model(*args, **kwargs)
                # write out into output buffer up to out batch size
                out_flat = tree_flatten_spec(out, self._out_spec)
                for o_buffer, o in zip(self._out_buffer_flat, out_flat):
                    o_buffer[: o.shape[0]] = o

        torch.cuda.synchronize()
        actual_capture_time = time.time() - actual_capture_start
        ad_logger.info(f"[AutoDeploy Backend]     -> Graph capture took {actual_capture_time:.4f}s")

        self._cuda_graph_mem_pool = self._cuda_graph_mem_pool or graph.pool()

        total_time = time.time() - capture_start_time
        ad_logger.info(
            f"[AutoDeploy Backend]   -> Completed capture for batch_size={batch_size} "
            f"in {total_time:.4f}s (warmup: {warmup_time:.4f}s, "
            f"capture: {actual_capture_time:.4f}s)"
        )

        return graph

    def capture_graph(self, *args, **kwargs):
        """Capture and pre-fetch the graph for variable batch size."""
        # check this is the first time we capture the graph
        assert not self.cudagraphs, "Graphs already captured."

        # flatten args, kwargs for the first time and record in_spec
        all_args_flat, self._in_spec = _args_kwargs_flatten(*args, **kwargs)

        # extract the batched input tensors
        args_batched = all_args_flat[: self.num_batched_inputs]
        args_static = all_args_flat[self.num_batched_inputs :]

        # set the args hash --> this is used to compare the static inputs during graph replay
        self._args_hash = self._get_hash(args_static)

        # sanity checks on the batched inputs
        msg_bs = (
            f"Input batch size exceeds maximum CUDA graph batch size. "
            f"CUDA graph max batch size: {self.cuda_graph_max_batch_size}, "
            f"but got input batch sizes: {[input.shape[0] for input in args_batched]}. "
            f"Did you intentionally set the maximal value of cuda_graph_batch_sizes lower "
            f"than the max_batch_size? It will fall back to non-CUDA graph forward pass for "
            f"batch sizes exceeding the max_batch_size."
        )
        if any(self.cuda_graph_max_batch_size < input.shape[0] for input in args_batched):
            ad_logger.info(msg_bs)

        # repeat the batched input tensors to the cuda_graph_max_batch_size
        self._input_buffers = [
            input[:1].repeat_interleave(self.cuda_graph_max_batch_size, dim=0)
            for input in args_batched
        ]

        # create new args, kwargs with the input buffers and static args
        args, kwargs = self._in_spec.unflatten(self._input_buffers + args_static)

        # capture output once with cuda_graph_max_batch_size to capture output buffers
        # store the out_spec at this point
        with CudaGraphWarmUpPhase():
            ad_logger.info(f"Warm up with {self.cuda_graph_max_batch_size=} before graph capture")
            out = self.model(*args, **kwargs)
        self._out_buffer_flat, self._out_spec = tree_flatten(out)

        # capture graph now for a range of batch sizes
        import time

        total_capture_start = time.time()
        ad_logger.info(
            f"[AutoDeploy Backend] Starting CUDA graph captures for "
            f"{len(self.cuda_graph_batch_sizes)} batch sizes"
        )

        for bs in self.cuda_graph_batch_sizes:
            ad_logger.info(f"[AutoDeploy Backend] Capturing graph for batch size: {bs}")

            # setup args, kwargs
            inputs_truncated = [in_buffer[:bs] for in_buffer in self._input_buffers]
            args, kwargs = self._in_spec.unflatten(inputs_truncated + args_static)

            # capture graph for truncated inputs
            combined_shape = sum((input.shape for input in inputs_truncated), start=())
            self.cudagraphs[combined_shape] = self._capture_one_graph(*args, **kwargs)

        total_capture_time = time.time() - total_capture_start
        avg_time = total_capture_time / len(self.cuda_graph_batch_sizes)
        ad_logger.info(
            f"[AutoDeploy Backend] Completed all CUDA graph captures in "
            f"{total_capture_time:.4f}s ({avg_time:.4f}s per batch size)"
        )

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

        # Calculate rounded-up shapes for each input
        rounded_shapes = [
            (self.round_to_cuda_batch_size(input.shape[0]),) + input.shape[1:]
            for input in args_batched
        ]
        combined_shape = sum(rounded_shapes, start=())

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
        out_flat = [o_b[:bs].detach().clone() for o_b in self._out_buffer_flat]
        return self._out_spec.unflatten(out_flat)


@CompileBackendRegistry.register("torch-cudagraph")
class TorchCudagraphCompiler(CompilerBackend):
    """Compiler that uses only CUDA graphs."""

    def __init__(
        self,
        *args_for_init,
        cuda_graph_batch_sizes: Optional[List[int]] = None,
        num_batched_inputs: int = 1,
        max_batch_size: Optional[int] = None,
        **kwargs_for_init,
    ):
        super().__init__(*args_for_init, **kwargs_for_init)

        # heuristic to identify max batch size
        assert max_batch_size or cuda_graph_batch_sizes, (
            "At least one of max_batch_size or cuda_graph_batch_sizes must be provided."
        )
        self.max_batch_size = max_batch_size or max(cuda_graph_batch_sizes)

        self.num_batched_inputs = num_batched_inputs
        if not cuda_graph_batch_sizes:
            # Use heuristic which includes commonly-used sizes like 1 and max_bs
            self.cuda_graph_batch_sizes = self._get_graph_batch_sizes(self.max_batch_size)
            ad_logger.info(f"Using heuristic cuda_graph_batch_sizes: {self.cuda_graph_batch_sizes}")
        else:
            # Sanitize user-provided sizes: clamp to [1, max_batch_size], dedupe, sort desc
            # No point capturing CUDA graphs for batch sizes larger than max_batch_size
            effective = {
                min(max(1, int(b)), int(self.max_batch_size))
                for b in cuda_graph_batch_sizes
                if isinstance(b, (int, float)) and b > 0
            }
            self.cuda_graph_batch_sizes = sorted(effective, reverse=True)

            # Log if we clamped any values
            original_values = [
                int(b) for b in cuda_graph_batch_sizes if isinstance(b, (int, float)) and b > 0
            ]
            clamped_values = [v for v in original_values if v > self.max_batch_size]
            if clamped_values:
                ad_logger.info(
                    f"Clamped CUDA graph batch sizes {clamped_values} to max_batch_size={self.max_batch_size}"
                )

            ad_logger.info(
                f"Using explicit cuda_graph_batch_sizes: requested={cuda_graph_batch_sizes}"
                f" -> effective={self.cuda_graph_batch_sizes}"
                f" (clamped to [1, {self.max_batch_size}])"
            )

    @torch.inference_mode()
    def compile(self) -> CapturedGraph:
        captured_model = CapturedGraph(
            self.model,
            cuda_graph_batch_sizes=self.cuda_graph_batch_sizes,
            num_batched_inputs=self.num_batched_inputs,
        )

        # try capturing cudagraph
        if self.args is not None or self.kwargs is not None:
            captured_model.capture_graph(*self.args, **self.kwargs)

        return captured_model

    @staticmethod
    def _get_graph_batch_sizes(
        max_bs: int, extra: Optional[List[int]] = None, multiplier: int = 128
    ) -> List[int]:
        """Heuristic to set batch sizes for graph capture."""
        # do 1, max_bs, and extra as special batch sizes
        batch_sizes = {1, max_bs, *(extra or [])}

        # add all multiples of multiplier up to max_bs
        batch_sizes.update(range(multiplier, max_bs + 1, multiplier))

        # return as sorted list
        return sorted(batch_sizes, reverse=True)
