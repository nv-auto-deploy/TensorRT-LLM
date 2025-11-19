"""
Custom ops to enable multi-stream execution.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable, Dict, Tuple

import torch


class _Singleton(type):
    _instances: Dict[type, Any] = {}
    _lock = RLock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:  # double-checked locking
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# A singleton that holds the pointers to the cuda streams and events.
# In multi-gpu scenario, each GPU/rank has its own CudaStreamManager.
class CudaStreamManager(metaclass=_Singleton):
    AUX_STREAM_NAME = "aux"
    MAIN_STREAM_NAME = "main"

    def __init__(self) -> None:
        # In case __init__ ever gets called twice, guard against re-init
        if hasattr(self, "streams"):
            return

        self._lock = RLock()

        # Events needed for stream synchronization
        self.events: Dict[str, Any] = {
            self.AUX_STREAM_NAME: torch.cuda.Event(),
            self.MAIN_STREAM_NAME: torch.cuda.Event(),
        }

        # Streams for multi-stream execution
        self.aux_stream = torch.cuda.Stream()
        self.streams: Dict[str, Any] = {
            self.AUX_STREAM_NAME: self.aux_stream,
            self.MAIN_STREAM_NAME: torch.cuda.default_stream(),
        }


cuda_stream_manager = CudaStreamManager()


@torch.library.custom_op("auto_deploy::record_event", mutates_args=())
def record_event(stream_name: str) -> None:
    event = cuda_stream_manager.events[stream_name]
    event.record()


@torch.library.custom_op("auto_deploy::wait_event", mutates_args=())
def wait_event(event_name: str) -> None:
    event = cuda_stream_manager.events[event_name]
    event.wait()


def record_event_wrapper(
    fn: Callable,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Callable:
    torch.ops.auto_deploy.record_event(cuda_stream_manager.MAIN_STREAM_NAME)
    output = fn(*args, **kwargs)
    return output


def aux_stream_wrapper(
    fn: Callable,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Callable:
    stream_name = cuda_stream_manager.AUX_STREAM_NAME
    with torch.cuda.stream(cuda_stream_manager.streams[stream_name]):
        torch.ops.auto_deploy.wait_event(cuda_stream_manager.MAIN_STREAM_NAME)
        output = fn(*args)
        torch.ops.auto_deploy.record_event(cuda_stream_manager.AUX_STREAM_NAME)
    torch.ops.auto_deploy.wait_event(cuda_stream_manager.AUX_STREAM_NAME)
    return output
