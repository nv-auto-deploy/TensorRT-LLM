# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""2-GPU numerics for ``trtllm_dist_all_reduce`` with the ``ONESHOT_SMALL`` strategy."""

import traceback

import pytest
import torch
from torch.distributed import DistNetworkError

# MPI pool leaks a thread on shutdown — suppress the threadleak warning.
pytestmark = pytest.mark.threadleak(enabled=False)

WORLD_SIZE = 2


def _init_dist(port):
    import torch.distributed as dist

    import tensorrt_llm
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — registers custom ops
    from tensorrt_llm._torch.auto_deploy.distributed.common import initialize_or_skip
    from tensorrt_llm._utils import get_free_port, mpi_broadcast

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    if port is None:
        port = mpi_broadcast(get_free_port() if rank == 0 else None)
    initialize_or_skip(port=port)
    return rank, dist.get_world_size()


def _cleanup():
    import torch.distributed as dist

    from tensorrt_llm._torch.auto_deploy.distributed.common import cleanup

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
    cleanup()


def _exact_sum(t):
    # Exact fp32 all-rank sum via allgather (reference independent of the AR op).
    import torch.distributed as dist

    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return torch.stack([g.float() for g in gathered]).sum(dim=0)


def _worker_oneshot_small_numerics(world_size, port):
    rank, ws = _init_dist(port)
    try:
        torch.manual_seed(1234 + rank)
        ar = torch.ops.auto_deploy.trtllm_dist_all_reduce

        # numel == 4096 (one decode token) -> ONESHOT kernel engaged.
        t = torch.randn(1, 1, 4096, dtype=torch.bfloat16, device="cuda")
        exact = _exact_sum(t)
        y_small = ar(t, "ONESHOT_SMALL")
        y_nccl = ar(t, "NCCL")
        # bf16 sums: reduction orders round differently; a few bf16 ulps of slack.
        torch.testing.assert_close(y_small.float(), exact, rtol=5e-2, atol=2e-1)
        torch.testing.assert_close(y_small.float(), y_nccl.float(), rtol=5e-2, atol=2e-1)

        # > 4096 elements -> NCCL fallback path.
        for shape in ((2, 1, 4096), (7, 4096), (512, 4096)):
            big = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
            exact_big = _exact_sum(big)
            y_big = ar(big, "ONESHOT_SMALL")
            torch.testing.assert_close(y_big.float(), exact_big, rtol=5e-2, atol=2e-1)

        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        _cleanup()


def _worker_oneshot_small_cuda_graph(world_size, port):
    rank, ws = _init_dist(port)
    try:
        import torch.distributed as dist

        ar = torch.ops.auto_deploy.trtllm_dist_all_reduce
        static_in = torch.randn(1, 1, 4096, dtype=torch.bfloat16, device="cuda")

        # Eager warmup allocates the ONESHOT IPC workspace outside capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                ar(static_in, "ONESHOT_SMALL")
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured_out = ar(static_in, "ONESHOT_SMALL")

        # Two replays with fresh data: flags/lamport state must survive replay.
        for fill_seed in (7, 13):
            torch.manual_seed(fill_seed * 100 + rank)
            static_in.copy_(torch.randn_like(static_in))
            exact = _exact_sum(static_in)
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured_out.float(), exact, rtol=5e-2, atol=2e-1)

        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        _cleanup()


def _run_with_retries(worker_fn, world_size, **kwargs):
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    max_retries = 5
    last_exc = None
    for _ in range(max_retries):
        pool = MpiPoolSession(n_workers=world_size)
        try:
            return pool.submit_sync(worker_fn, port=None, world_size=world_size, **kwargs)
        except DistNetworkError as e:
            last_exc = e
            if "EADDRINUSE" not in str(e) and "address already in use" not in str(e).lower():
                raise
        finally:
            pool.shutdown()
    raise RuntimeError(f"Dist init failed after {max_retries} retries") from last_exc


@pytest.mark.skipif(torch.cuda.device_count() < WORLD_SIZE, reason="Requires >= 2 GPUs")
def test_oneshot_small_numerics_2gpu():
    results = _run_with_retries(_worker_oneshot_small_numerics, world_size=WORLD_SIZE)
    assert all(r is True for r in results), f"Unexpected worker results: {results}"


@pytest.mark.skipif(torch.cuda.device_count() < WORLD_SIZE, reason="Requires >= 2 GPUs")
def test_oneshot_small_cuda_graph_2gpu():
    results = _run_with_retries(_worker_oneshot_small_cuda_graph, world_size=WORLD_SIZE)
    assert all(r is True for r in results), f"Unexpected worker results: {results}"
