import gc
import time
from contextlib import ContextDecorator, contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch

from .logger import ad_logger


@dataclass
class MemSnapshot:
    t: float
    device: int
    total_free_cuda: int  # bytes from cuda.mem_get_info()
    total_cuda: int  # bytes from cuda.mem_get_info()
    reserved_bytes: int  # torch.cuda.memory_stats()["reserved_bytes.all.current"]
    allocated_bytes: int  # torch.cuda.memory_stats()["allocated_bytes.all.current"]
    inactive_split_bytes: (
        int  # torch.cuda.memory_stats().get("inactive_split_bytes.all.current", 0)
    )
    fragmentation_proxy: float  # inactive_split / max(1, (reserved - allocated))
    reserved_free_bytes: int  # reserved - allocated
    peak_reserved_bytes: int  # reserved_bytes.all.peak
    peak_allocated_bytes: int  # allocated_bytes.all.peak

    raw_stats: Dict[str, Any]  # full memory_stats() (for later offline inspection)
    # NOTE: We *try* to attach memory_snapshot(); if unavailable, we set to None
    allocator_snapshot: Optional[Dict[str, Any]]


class CUDAMemFragProbe(ContextDecorator):
    """
    Usage:
        with CUDAMemFragProbe(tag="gemm_fusion", device=0) as probe:
            run_my_fusion_pass()
        # probe.before / probe.after / probe.delta pretty-printable and programmatic

    What you get:
    - `before` and `after` MemSnapshot instances (allocator + CUDA-level)
    - `delta()` dict with human-friendly bytes deltas
    - `print_report()` quick summary with fragmentation proxy:
         fragmentation_proxy = inactive_split_bytes / max(1, (reserved - allocated))
      This approximates how much of the *free-within-reserved* space is stuck in unusable splits.
    """

    def __init__(
        self, tag: str = "", device: Optional[int] = None, sync: bool = True, printer=print
    ):
        if device is None:
            device = torch.cuda.current_device()
        self.tag = tag
        self.device = device
        self.sync = sync
        self.printer = printer
        self.before: Optional[MemSnapshot] = None
        self.after: Optional[MemSnapshot] = None

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.before = self._collect()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.sync:
            torch.cuda.synchronize(self.device)
        self.after = self._collect()
        return False  # do not suppress exceptions

    # ---------- Public helpers ----------
    def delta(self) -> Dict[str, Any]:
        assert self.before and self.after, "Probe did not run correctly."
        b, a = self.before, self.after
        return {
            "tag": self.tag,
            "device": self.device,
            "reserved_bytes_delta": a.reserved_bytes - b.reserved_bytes,
            "allocated_bytes_delta": a.allocated_bytes - b.allocated_bytes,
            "inactive_split_bytes_delta": a.inactive_split_bytes - b.inactive_split_bytes,
            "reserved_free_bytes_delta": a.reserved_free_bytes - b.reserved_free_bytes,
            "total_free_cuda_delta": a.total_free_cuda - b.total_free_cuda,
            "fragmentation_proxy_before": b.fragmentation_proxy,
            "fragmentation_proxy_after": a.fragmentation_proxy,
        }

    def print_report(self):
        d = self.delta()
        human = lambda x: f"{x / 1024 / 1024:.2f} MiB"
        self.printer(f"[CUDAMemFragProbe] tag='{self.tag}' device={self.device}")
        self.printer(
            f"  reserved: {human(self.before.reserved_bytes)} -> {human(self.after.reserved_bytes)} "
            f"(Δ {human(d['reserved_bytes_delta'])})"
        )
        self.printer(
            f"  allocated: {human(self.before.allocated_bytes)} -> {human(self.after.allocated_bytes)} "
            f"(Δ {human(d['allocated_bytes_delta'])})"
        )
        self.printer(
            f"  free-within-reserved: {human(self.before.reserved_free_bytes)} -> {human(self.after.reserved_free_bytes)} "
            f"(Δ {human(d['reserved_free_bytes_delta'])})"
        )
        self.printer(
            f"  inactive_split (free but fragmented): {human(self.before.inactive_split_bytes)} -> {human(self.after.inactive_split_bytes)} "
            f"(Δ {human(d['inactive_split_bytes_delta'])})"
        )
        self.printer(
            f"  CUDA free (driver): {human(self.before.total_free_cuda)} -> {human(self.after.total_free_cuda)} "
            f"(Δ {human(d['total_free_cuda_delta'])})"
        )
        self.printer(
            f"  fragmentation_proxy: {d['fragmentation_proxy_before']:.3f} -> {d['fragmentation_proxy_after']:.3f}"
        )

    # ---------- Internals ----------
    def _collect(self) -> MemSnapshot:
        torch.cuda.set_device(self.device)
        free_cuda, total_cuda = torch.cuda.mem_get_info(self.device)
        stats = torch.cuda.memory_stats(self.device)

        reserved = int(stats.get("reserved_bytes.all.current", 0))
        allocated = int(stats.get("allocated_bytes.all.current", 0))
        inactive_split = int(stats.get("inactive_split_bytes.all.current", 0))
        peak_reserved = int(stats.get("reserved_bytes.all.peak", 0))
        peak_allocated = int(stats.get("allocated_bytes.all.peak", 0))
        reserved_free = max(0, reserved - allocated)
        frag_proxy = float(inactive_split) / float(max(1, reserved_free))

        # `memory_snapshot()` is available on CUDA builds; guard just in case.
        allocator_snapshot = None
        try:
            allocator_snapshot = torch.cuda.memory_snapshot()
        except Exception:
            allocator_snapshot = None

        return MemSnapshot(
            t=time.time(),
            device=self.device,
            total_free_cuda=int(free_cuda),
            total_cuda=int(total_cuda),
            reserved_bytes=reserved,
            allocated_bytes=allocated,
            inactive_split_bytes=inactive_split,
            fragmentation_proxy=frag_proxy,
            reserved_free_bytes=reserved_free,
            peak_reserved_bytes=peak_reserved,
            peak_allocated_bytes=peak_allocated,
            raw_stats=stats,
            allocator_snapshot=allocator_snapshot,
        )

    # Optional: expose raw dicts for logging/serialization
    def asdict(self) -> Dict[str, Any]:
        return {
            "before": asdict(self.before) if self.before else None,
            "after": asdict(self.after) if self.after else None,
            "delta": self.delta() if (self.before and self.after) else None,
        }


@contextmanager
def cuda_memory_tracker(logger=ad_logger):
    """
    Context manager to track CUDA memory allocation differences.

    Logs a warning if there is an increase in memory allocation after the
    code block, which might indicate a potential memory leak.
    """
    mem_before = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        mem_after = torch.cuda.memory_allocated()
        leaked = mem_after - mem_before
        if leaked > 0:
            logger.warning(f"Potential memory leak detected, leaked memory: {leaked} bytes")
