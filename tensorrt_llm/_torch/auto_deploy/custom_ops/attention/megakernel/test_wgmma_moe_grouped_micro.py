#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Grouped Hopper BF16 WGMMA microbenchmark for Gemma4 MoE expert shapes."""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

TOPK = 8
HIDDEN_SIZE = 2816
EXPERT_INTERMEDIATE = 704
FC1_ROWS = 2 * EXPERT_INTERMEDIATE
FC2_ROWS = HIDDEN_SIZE


def _repo_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "cpp/kernels/xqa/gmma.cuh").exists()
    )


def _cuda_source() -> str:
    return r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cpp/kernels/xqa/gmma.cuh"

namespace {

constexpr int kTileM = 64;
constexpr int kTileK = 16;
constexpr int kBlockThreads = 128;
constexpr int kElemsPerLdGrain = sizeof(LdGrain) / sizeof(__nv_bfloat16);
static_assert(kElemsPerLdGrain == 8);

template <int kDupN>
__global__ void grouped_dup_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ w,
    float* __restrict__ checksum,
    int hidden_size,
    int num_rows,
    int num_slots,
    int variant,
    int share_input)
{
    int const slot = blockIdx.y;
    int const row_base = blockIdx.x * kTileM;
    if (slot >= num_slots || row_base >= num_rows)
    {
        return;
    }

    static constexpr int kBCols = (kDupN < 16 ? 16 : kDupN);

    struct alignas(256) SharedStorage
    {
        Array2D<__nv_bfloat16, kTileM, kTileK> a;
        Array2D<__nv_bfloat16, kTileK, kBCols> b;
        Array2D<LdGrain, kDupN, kTileM / kElemsPerLdGrain> out;
        float reduce[kBlockThreads];
    };
    __shared__ SharedStorage smem;

    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int const warp_rank = tid / 32;
    float acc[kDupN / 8][2][2];
    bool acc_has_value = false;

    auto const desc_a_base = gmma::makeMatDesc(
                                 nullptr,
                                 0,
                                 decltype(smem.a)::rowBytes * 8,
                                 &smem.a,
                                 gmma::getSwizzleMode<false>(decltype(smem.a){}))
                                 .raw();
    auto const desc_b_base = gmma::makeMatDesc(
                                 nullptr,
                                 0,
                                 decltype(smem.b)::rowBytes * 8,
                                 &smem.b,
                                 gmma::getSwizzleMode<false>(decltype(smem.b){}))
                                 .raw();

    auto const* slot_w = w + (static_cast<int64_t>(slot) * num_rows + row_base) * hidden_size;
    auto const* slot_x = x + (share_input ? 0 : static_cast<int64_t>(slot) * hidden_size);

    for (int k0 = 0; k0 < hidden_size; k0 += kTileK)
    {
        for (int idx = tid; idx < kTileM * kTileK; idx += kBlockThreads)
        {
            int const row = idx / kTileK;
            int const col = idx % kTileK;
            int const global_row = row_base + row;
            smem.a.template at<false>(row, col)
                = global_row < num_rows ? slot_w[row * hidden_size + (k0 + col)] : __float2bfloat16(0.0f);
        }
        for (int idx = tid; idx < kTileK * kBCols; idx += kBlockThreads)
        {
            int const k = idx / kBCols;
            int const n = idx % kBCols;
            smem.b.template at<false>(k, n) = n < kDupN ? slot_x[k0 + k] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        gmma::fence();
        auto const desc_a = gmma::addAddr(desc_a_base, &smem.a(0, 0));
        auto const desc_b = gmma::addAddr(desc_b_base, &smem.b(0, 0));
        switch (variant)
        {
        case 0: gmma::mma_async_shmA<__nv_bfloat16, kDupN, false, false>(acc, desc_a, desc_b, acc_has_value); break;
        case 1: gmma::mma_async_shmA<__nv_bfloat16, kDupN, false, true>(acc, desc_a, desc_b, acc_has_value); break;
        case 2: gmma::mma_async_shmA<__nv_bfloat16, kDupN, true, false>(acc, desc_a, desc_b, acc_has_value); break;
        case 3: gmma::mma_async_shmA<__nv_bfloat16, kDupN, true, true>(acc, desc_a, desc_b, acc_has_value); break;
        default: return;
        }
        gmma::commit_group();
        gmma::wait_group<0>();
        acc_has_value = true;
        __syncthreads();
    }

    using F16Acc = Array2D<Vec<uint32_t, 2>, 1, kDupN / 8>;
    F16Acc f16_acc;
    reinterpret_cast<Vec<__nv_bfloat16, sizeof(f16_acc) / sizeof(__nv_bfloat16)>&>(f16_acc)
        = convert<__nv_bfloat16>(reinterpret_cast<Vec<float, sizeof(acc) / sizeof(float)> const&>(acc));

    uint32_t const idx_half = lane / 16;
    uint32_t const idx_in_half = lane % 16;
    uint32_t const idx_oct_inside_half = idx_in_half / 8;
    uint32_t const idx_row_inside_oct = lane % 8;
    uint32_t const warp_base_c = 16 * warp_rank;
    auto const get_dst_addr = [&](uint32_t idx_acc_core_mat) -> LdGrain*
    {
        uint32_t const dst_r = idx_acc_core_mat * 8 + idx_row_inside_oct;
        uint32_t const dst_c = (warp_base_c + 8 * idx_oct_inside_half) / kElemsPerLdGrain;
        return &smem.out.template at<true>(dst_r, dst_c);
    };
    auto const get_acc_data = [&](uint32_t idx_acc_core_mat)
    {
        return f16_acc(0, idx_acc_core_mat);
    };

    if constexpr (kDupN == 8)
    {
        auto* const dst_addr = lane < 16 ? get_dst_addr(0) : nullptr;
        stmatrix<true, 2>(dst_addr, get_acc_data(0));
    }
    else
    {
        auto* const dst_addr = get_dst_addr(idx_half);
        Vec<uint32_t, 2> const data[2] = {get_acc_data(0), get_acc_data(1)};
        stmatrix<true, 4>(dst_addr, reinterpret_cast<LdGrain const&>(data));
    }
    __syncthreads();

    float local = 0.0f;
    for (int idx = tid; idx < kTileM * kDupN; idx += kBlockThreads)
    {
        int const row = idx / kDupN;
        int const col = idx % kDupN;
        auto const grain = smem.out.template at<false>(col, row / kElemsPerLdGrain);
        auto const vals = reinterpret_cast<Vec<__nv_bfloat16, kElemsPerLdGrain> const&>(grain);
        if (row_base + row < num_rows)
        {
            local += __bfloat162float(vals[row % kElemsPerLdGrain]);
        }
    }
    smem.reduce[tid] = local;
    __syncthreads();

    for (int offset = kBlockThreads / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            smem.reduce[tid] += smem.reduce[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        checksum[slot * gridDim.x + blockIdx.x] = smem.reduce[0];
    }
}

void run_grouped_dup_into(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor checksum,
    int dup_n,
    int variant,
    bool share_input)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && checksum.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16, "w must be bf16");
    TORCH_CHECK(checksum.scalar_type() == at::kFloat, "checksum must be float32");
    TORCH_CHECK(w.dim() == 3, "w must be [slots, rows, hidden]");
    TORCH_CHECK(checksum.dim() == 2, "checksum must be [slots, tiles]");

    int const num_slots = static_cast<int>(w.size(0));
    int const num_rows = static_cast<int>(w.size(1));
    int const hidden_size = static_cast<int>(w.size(2));
    TORCH_CHECK(checksum.size(0) == num_slots, "checksum slot dimension mismatch");
    TORCH_CHECK(checksum.size(1) == (num_rows + kTileM - 1) / kTileM, "checksum tile dimension mismatch");
    TORCH_CHECK(x.dim() == 2, "x must be [slots_or_1, hidden]");
    TORCH_CHECK(x.size(1) == hidden_size, "x hidden dimension mismatch");
    TORCH_CHECK(share_input ? x.size(0) == 1 : x.size(0) == num_slots, "x slot dimension mismatch");

    auto stream = at::cuda::getDefaultCUDAStream();
    auto const* x_ptr = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>());
    auto const* w_ptr = reinterpret_cast<__nv_bfloat16 const*>(w.data_ptr<at::BFloat16>());
    auto* checksum_ptr = checksum.data_ptr<float>();
    dim3 const grid((num_rows + kTileM - 1) / kTileM, num_slots);
    dim3 const block(kBlockThreads);

    switch (dup_n)
    {
    case 8:
        grouped_dup_kernel<8><<<grid, block, 0, stream>>>(
            x_ptr, w_ptr, checksum_ptr, hidden_size, num_rows, num_slots, variant, share_input ? 1 : 0);
        break;
    case 16:
        grouped_dup_kernel<16><<<grid, block, 0, stream>>>(
            x_ptr, w_ptr, checksum_ptr, hidden_size, num_rows, num_slots, variant, share_input ? 1 : 0);
        break;
    default: TORCH_CHECK(false, "dup_n must be 8 or 16");
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run_grouped_dup_into", &run_grouped_dup_into, "Run grouped duplicated-column WGMMA microkernel");
}
"""


def _load_module():
    repo_root = _repo_root()
    src = _cuda_source()
    src_hash = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
    build_dir = Path(__file__).resolve().parent / "_build_wgmma"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=f"wgmma_moe_grouped_micro_{src_hash}",
        cpp_sources="",
        cuda_sources=src,
        extra_include_paths=[str(repo_root)],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=False,
    )


def _bench_us(fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return statistics.median(times_us)


def _run_case(mod, *, name: str, x: torch.Tensor, w: torch.Tensor, share_input: bool) -> None:
    ref = (
        torch.einsum("sk,srk->sr", x.float().expand(TOPK, -1), w.float())
        if share_input
        else torch.einsum("sk,srk->sr", x.float(), w.float())
    )
    variant_names = {
        0: "nn",
        1: "nt",
        2: "tn",
        3: "tt",
    }
    print(name)
    for dup_n in (8, 16):
        best_variant = None
        best_time = None
        for variant in range(4):
            checksum = torch.empty(
                (TOPK, (w.size(1) + 63) // 64), device="cuda", dtype=torch.float32
            )
            mod.run_grouped_dup_into(x, w, checksum, dup_n, variant, share_input)
            total = checksum.sum().item()
            expected = ref.sum().item() * dup_n
            diff = abs(total - expected)

            def run() -> None:
                mod.run_grouped_dup_into(x, w, checksum, dup_n, variant, share_input)

            us = _bench_us(run, warmup=10, iters=40)
            if best_time is None or us < best_time:
                best_time = us
                best_variant = variant
            print(
                f"  dup_n={dup_n:>2d}  variant={variant_names[variant]}  time={us:7.2f} us  "
                f"checksum_diff={diff:.4f}"
            )
        assert best_variant is not None and best_time is not None
        print(
            f"  best dup_n={dup_n:>2d} variant={variant_names[best_variant]} time={best_time:7.2f} us"
        )


def main() -> None:
    if not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        print(f"Skipping: requires Hopper+, got {torch.cuda.get_device_name()}")
        return

    mod = _load_module()
    torch.manual_seed(123)

    print("=" * 72)
    print("Gemma4 Megakernel: Grouped Hopper BF16 WGMMA MoE Microbenchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 72)

    fc1_x = torch.randn(1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    fc1_w = torch.randn(TOPK, FC1_ROWS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    _run_case(mod, name="grouped_moe_fc1", x=fc1_x, w=fc1_w, share_input=True)

    fc2_x = torch.randn(TOPK, EXPERT_INTERMEDIATE, device="cuda", dtype=torch.bfloat16)
    fc2_w = torch.randn(TOPK, FC2_ROWS, EXPERT_INTERMEDIATE, device="cuda", dtype=torch.bfloat16)
    _run_case(mod, name="grouped_moe_fc2", x=fc2_x, w=fc2_w, share_input=False)

    print("=" * 72)


if __name__ == "__main__":
    main()
