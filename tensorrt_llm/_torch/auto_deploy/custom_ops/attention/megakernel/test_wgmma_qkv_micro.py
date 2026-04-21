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

"""Standalone Hopper BF16 WGMMA microbenchmark for decode-style QKV tiles."""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

HIDDEN_SIZE = 2816
QKV_WIDTH = 8192


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
__global__ void wgmma_qkv_dup_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ w,
    float* __restrict__ checksum,
    int hidden_size,
    int num_rows,
    int variant)
{
    int const row_base = blockIdx.x * kTileM;
    if (row_base >= num_rows)
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

    for (int k0 = 0; k0 < hidden_size; k0 += kTileK)
    {
        for (int idx = tid; idx < kTileM * kTileK; idx += kBlockThreads)
        {
            int const row = idx / kTileK;
            int const col = idx % kTileK;
            int const global_row = row_base + row;
            smem.a.template at<false>(row, col)
                = global_row < num_rows ? w[global_row * hidden_size + (k0 + col)] : __float2bfloat16(0.0f);
        }
        for (int idx = tid; idx < kTileK * kBCols; idx += kBlockThreads)
        {
            int const k = idx / kBCols;
            int const n = idx % kBCols;
            smem.b.template at<false>(k, n) = n < kDupN ? x[k0 + k] : __float2bfloat16(0.0f);
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
        if (row < num_rows)
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
        checksum[blockIdx.x] = smem.reduce[0];
    }
}

template <int kDupN>
__global__ void wgmma_qkv_dup_debug_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ w,
    float* __restrict__ out,
    int hidden_size,
    int variant)
{
    int const row_base = 0;

    static constexpr int kBCols = (kDupN < 16 ? 16 : kDupN);

    struct alignas(256) SharedStorage
    {
        Array2D<__nv_bfloat16, kTileM, kTileK> a;
        Array2D<__nv_bfloat16, kTileK, kBCols> b;
        Array2D<LdGrain, kDupN, kTileM / kElemsPerLdGrain> out;
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

    for (int k0 = 0; k0 < hidden_size; k0 += kTileK)
    {
        for (int idx = tid; idx < kTileM * kTileK; idx += kBlockThreads)
        {
            int const row = idx / kTileK;
            int const col = idx % kTileK;
            smem.a.template at<false>(row, col) = w[(row_base + row) * hidden_size + (k0 + col)];
        }
        for (int idx = tid; idx < kTileK * kBCols; idx += kBlockThreads)
        {
            int const k = idx / kBCols;
            int const n = idx % kBCols;
            smem.b.template at<false>(k, n) = n < kDupN ? x[k0 + k] : __float2bfloat16(0.0f);
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

    for (int idx = tid; idx < kTileM * kDupN; idx += kBlockThreads)
    {
        int const row = idx / kDupN;
        int const col = idx % kDupN;
        auto const grain = smem.out.template at<false>(col, row / kElemsPerLdGrain);
        auto const vals = reinterpret_cast<Vec<__nv_bfloat16, kElemsPerLdGrain> const&>(grain);
        out[row * kDupN + col] = __bfloat162float(vals[row % kElemsPerLdGrain]);
    }
}

template <int kDupN>
__global__ void wgmma_qkv_dup_kernel_kmajor(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ w,
    float* __restrict__ checksum,
    int hidden_size,
    int num_rows)
{
    int const row_base = blockIdx.x * kTileM;
    if (row_base >= num_rows)
    {
        return;
    }

    static constexpr int kBCols = (kDupN < 16 ? 16 : kDupN);

    struct alignas(256) SharedStorage
    {
        Array2D<__nv_bfloat16, kTileK, kTileM> a;
        Array2D<__nv_bfloat16, kTileK, kBCols> b;
        Array2D<LdGrain, kTileM, kDupN / 8> out;
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
                                 gmma::getSwizzleMode<true>(decltype(smem.a){}))
                                 .raw();
    auto const desc_b_base = gmma::makeMatDesc(
                                 nullptr,
                                 0,
                                 decltype(smem.b)::rowBytes * 8,
                                 &smem.b,
                                 gmma::getSwizzleMode<true>(decltype(smem.b){}))
                                 .raw();

    for (int k0 = 0; k0 < hidden_size; k0 += kTileK)
    {
        for (int idx = tid; idx < kTileK * kTileM; idx += kBlockThreads)
        {
            int const k = idx / kTileM;
            int const row = idx % kTileM;
            int const global_row = row_base + row;
            smem.a.template at<true>(k, row)
                = global_row < num_rows ? w[global_row * hidden_size + (k0 + k)] : __float2bfloat16(0.0f);
        }
        for (int idx = tid; idx < kTileK * kBCols; idx += kBlockThreads)
        {
            int const k = idx / kBCols;
            int const n = idx % kBCols;
            smem.b.template at<true>(k, n) = n < kDupN ? x[k0 + k] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        gmma::fence();
        auto const desc_a = gmma::addAddr(desc_a_base, &smem.a(0, 0));
        auto const desc_b = gmma::addAddr(desc_b_base, &smem.b(0, 0));
        gmma::mma_async_shmA<__nv_bfloat16, kDupN, false, false>(acc, desc_a, desc_b, acc_has_value);
        gmma::commit_group();
        gmma::wait_group<0>();
        acc_has_value = true;
        __syncthreads();
    }

    int const idx_row = lane % 8;
#pragma unroll
    for (int n = 0; n < kDupN / 8; ++n)
    {
        for (int i = 0; i < 2; ++i)
        {
            Vec<uint32_t, 1> bf16_pair;
            Vec<float, 2> fp32_pair;
            fp32_pair[0] = acc[n][i][0];
            fp32_pair[1] = acc[n][i][1];
            reinterpret_cast<Vec<__nv_bfloat16, 2>&>(bf16_pair[0]) = convert<__nv_bfloat16>(fp32_pair);
            auto* dst = &smem.out.template at<true>(16 * warp_rank + 8 * i + idx_row, n);
            stmatrix<true, 1>(dst, bf16_pair);
        }
    }
    __syncthreads();

    float local = 0.0f;
    for (int idx = tid; idx < kTileM * (kDupN / 8); idx += kBlockThreads)
    {
        int const row = idx / (kDupN / 8);
        int const col = idx % (kDupN / 8);
        auto const grain = smem.out.template at<false>(row, col);
        auto const vals = reinterpret_cast<Vec<__nv_bfloat16, 8> const&>(grain);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int const global_col = col * 8 + i;
            if (row < num_rows && global_col < kDupN)
            {
                local += __bfloat162float(vals[i]);
            }
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
        checksum[blockIdx.x] = smem.reduce[0];
    }
}

template <int kDupN>
__global__ void stmatrix_smoke_kernel(float* __restrict__ checksum)
{
    struct alignas(256) SharedStorage
    {
        Array2D<LdGrain, kTileM, kDupN / 8> out;
        float reduce[kBlockThreads];
    };
    __shared__ SharedStorage smem;

    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int const warp_rank = tid / 32;
    int const idx_row = lane % 8;

    for (int n = 0; n < kDupN / 8; ++n)
    {
        for (int i = 0; i < 2; ++i)
        {
            Vec<uint32_t, 1> bf16_pair;
            Vec<float, 2> fp32_pair;
            fp32_pair[0] = float(1000 * warp_rank + 100 * n + 10 * i + 1);
            fp32_pair[1] = float(1000 * warp_rank + 100 * n + 10 * i + 2);
            reinterpret_cast<Vec<__nv_bfloat16, 2>&>(bf16_pair[0]) = convert<__nv_bfloat16>(fp32_pair);
            auto* dst = &smem.out.template at<true>(16 * warp_rank + 8 * i + idx_row, n);
            stmatrix<true, 1>(dst, bf16_pair);
        }
    }
    __syncthreads();

    float local = 0.0f;
    for (int idx = tid; idx < kTileM * (kDupN / 8); idx += kBlockThreads)
    {
        int const row = idx / (kDupN / 8);
        int const col = idx % (kDupN / 8);
        auto const grain = smem.out.template at<false>(row, col);
        auto const vals = reinterpret_cast<Vec<__nv_bfloat16, 8> const&>(grain);
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            local += __bfloat162float(vals[j]);
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
        checksum[0] = smem.reduce[0];
    }
}

void run_dup_into(torch::Tensor x, torch::Tensor w, torch::Tensor checksum, int dup_n, int variant)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
    TORCH_CHECK(checksum.is_cuda(), "checksum must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16, "w must be bf16");
    TORCH_CHECK(checksum.scalar_type() == at::kFloat, "checksum must be float32");
    TORCH_CHECK(x.dim() == 1, "x must be 1D");
    TORCH_CHECK(w.dim() == 2, "w must be 2D");
    TORCH_CHECK(w.size(1) == x.size(0), "weight width must match x");
    TORCH_CHECK(checksum.dim() == 1, "checksum must be 1D");
    TORCH_CHECK(checksum.size(0) == (w.size(0) + kTileM - 1) / kTileM, "checksum shape mismatch");

    auto stream = at::cuda::getDefaultCUDAStream();

    auto const* x_ptr = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>());
    auto const* w_ptr = reinterpret_cast<__nv_bfloat16 const*>(w.data_ptr<at::BFloat16>());
    auto* checksum_ptr = checksum.data_ptr<float>();
    dim3 const grid((w.size(0) + kTileM - 1) / kTileM);
    dim3 const block(kBlockThreads);

    switch (dup_n)
    {
    case 8:
        wgmma_qkv_dup_kernel<8><<<grid, block, 0, stream>>>(
            x_ptr, w_ptr, checksum_ptr, static_cast<int>(x.size(0)), static_cast<int>(w.size(0)), variant);
        break;
    case 16:
        wgmma_qkv_dup_kernel<16><<<grid, block, 0, stream>>>(
            x_ptr, w_ptr, checksum_ptr, static_cast<int>(x.size(0)), static_cast<int>(w.size(0)), variant);
        break;
    default: TORCH_CHECK(false, "dup_n must be 8 or 16");
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

void run_dup_kmajor_into(torch::Tensor x, torch::Tensor w, torch::Tensor checksum, int dup_n)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
    TORCH_CHECK(checksum.is_cuda(), "checksum must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16, "w must be bf16");
    TORCH_CHECK(checksum.scalar_type() == at::kFloat, "checksum must be float32");
    TORCH_CHECK(x.dim() == 1, "x must be 1D");
    TORCH_CHECK(w.dim() == 2, "w must be 2D");
    TORCH_CHECK(w.size(1) == x.size(0), "weight width must match x");
    TORCH_CHECK(checksum.dim() == 1, "checksum must be 1D");
    TORCH_CHECK(checksum.size(0) == (w.size(0) + kTileM - 1) / kTileM, "checksum shape mismatch");

    auto stream = at::cuda::getDefaultCUDAStream();

    auto const* x_ptr = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>());
    auto const* w_ptr = reinterpret_cast<__nv_bfloat16 const*>(w.data_ptr<at::BFloat16>());
    auto* checksum_ptr = checksum.data_ptr<float>();
    dim3 const grid((w.size(0) + kTileM - 1) / kTileM);
    dim3 const block(kBlockThreads);

    switch (dup_n)
    {
    case 8:
        wgmma_qkv_dup_kernel_kmajor<8><<<grid, block, 0, stream>>>(
            x_ptr, w_ptr, checksum_ptr, static_cast<int>(x.size(0)), static_cast<int>(w.size(0)));
        break;
    case 16:
        wgmma_qkv_dup_kernel_kmajor<16><<<grid, block, 0, stream>>>(
            x_ptr, w_ptr, checksum_ptr, static_cast<int>(x.size(0)), static_cast<int>(w.size(0)));
        break;
    default: TORCH_CHECK(false, "dup_n must be 8 or 16");
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

void run_dup_debug_into(torch::Tensor x, torch::Tensor w, torch::Tensor out, int dup_n, int variant)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16, "w must be bf16");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == kTileM && out.size(1) == dup_n, "out shape mismatch");

    auto stream = at::cuda::getDefaultCUDAStream();
    auto const* x_ptr = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>());
    auto const* w_ptr = reinterpret_cast<__nv_bfloat16 const*>(w.data_ptr<at::BFloat16>());
    auto* out_ptr = out.data_ptr<float>();
    dim3 const grid(1);
    dim3 const block(kBlockThreads);

    switch (dup_n)
    {
    case 8:
        wgmma_qkv_dup_debug_kernel<8><<<grid, block, 0, stream>>>(
            x_ptr, w_ptr, out_ptr, static_cast<int>(x.size(0)), variant);
        break;
    case 16:
        wgmma_qkv_dup_debug_kernel<16><<<grid, block, 0, stream>>>(
            x_ptr, w_ptr, out_ptr, static_cast<int>(x.size(0)), variant);
        break;
    default: TORCH_CHECK(false, "dup_n must be 8 or 16");
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA debug kernel launch failed: ", cudaGetErrorString(err));
}

void run_stmatrix_smoke_into(torch::Tensor checksum, int dup_n)
{
    TORCH_CHECK(checksum.is_cuda(), "checksum must be CUDA");
    TORCH_CHECK(checksum.scalar_type() == at::kFloat, "checksum must be float32");
    TORCH_CHECK(checksum.dim() == 1 && checksum.size(0) == 1, "checksum shape mismatch");

    auto stream = at::cuda::getDefaultCUDAStream();
    dim3 const grid(1);
    dim3 const block(kBlockThreads);

    switch (dup_n)
    {
    case 8: stmatrix_smoke_kernel<8><<<grid, block, 0, stream>>>(checksum.data_ptr<float>()); break;
    case 16: stmatrix_smoke_kernel<16><<<grid, block, 0, stream>>>(checksum.data_ptr<float>()); break;
    default: TORCH_CHECK(false, "dup_n must be 8 or 16");
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA smoke kernel launch failed: ", cudaGetErrorString(err));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run_dup_into", &run_dup_into, "Run duplicated-column WGMMA QKV microkernel");
    m.def("run_dup_kmajor_into", &run_dup_kmajor_into, "Run K-major duplicated-column WGMMA QKV microkernel");
    m.def("run_dup_debug_into", &run_dup_debug_into, "Run duplicated-column WGMMA QKV debug kernel");
    m.def("run_stmatrix_smoke_into", &run_stmatrix_smoke_into, "Run stmatrix smoke kernel");
}
"""


def _load_module():
    repo_root = _repo_root()
    src = _cuda_source()
    src_hash = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
    build_dir = Path(__file__).resolve().parent / "_build_wgmma"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=f"wgmma_qkv_micro_{src_hash}",
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


def main() -> None:
    if not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        print(f"Skipping: requires Hopper+, got {torch.cuda.get_device_name()}")
        return

    mod = _load_module()
    torch.manual_seed(42)
    x = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(QKV_WIDTH, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x.unsqueeze(0), w).squeeze(0).float()

    print("=" * 72)
    print("Gemma4 Megakernel: Hopper BF16 WGMMA QKV Microbenchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 72)
    variant_names = {
        0: "nn",
        1: "nt",
        2: "tn",
        3: "tt",
    }
    for dup_n in (8, 16):
        smoke = torch.empty(1, device="cuda", dtype=torch.float32)
        mod.run_stmatrix_smoke_into(smoke, dup_n)
        print(f"dup_n={dup_n:>2d}  stmatrix_smoke_sum={smoke.item():.4f}")
        for variant in range(4):
            checksum = torch.empty((QKV_WIDTH + 63) // 64, device="cuda", dtype=torch.float32)
            mod.run_dup_into(x, w, checksum, dup_n, variant)
            total = checksum.sum().item()
            expected = ref.sum().item() * dup_n
            diff = abs(total - expected)

            def run() -> None:
                mod.run_dup_into(x, w, checksum, dup_n, variant)

            us = _bench_us(run)
            print(
                f"dup_n={dup_n:>2d}  variant={variant_names[variant]}  time={us:7.2f} us  "
                f"checksum_diff={diff:.4f}  total={total:.4f}  expected={expected:.4f}"
            )
        try:
            checksum = torch.empty((QKV_WIDTH + 63) // 64, device="cuda", dtype=torch.float32)
            mod.run_dup_kmajor_into(x, w, checksum, dup_n)
            total = checksum.sum().item()
            expected = ref.sum().item() * dup_n
            diff = abs(total - expected)

            def run_kmajor() -> None:
                mod.run_dup_kmajor_into(x, w, checksum, dup_n)

            us = _bench_us(run_kmajor)
            print(
                f"dup_n={dup_n:>2d}  variant=kmajor_nn  time={us:7.2f} us  "
                f"checksum_diff={diff:.4f}  total={total:.4f}  expected={expected:.4f}"
            )
        except RuntimeError as exc:
            print(f"dup_n={dup_n:>2d}  variant=kmajor_nn  FAILED: {exc}")
            return
    print("=" * 72)


if __name__ == "__main__":
    main()
