/*
 * Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/fusedLayernormKernels/layernorm_param.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

// FP8 E4M3 representable range (symmetric)
constexpr float kFP8E4M3Max = 448.0f;

template <typename T>
__device__ __forceinline__ float toFloat(T x);

template <>
__device__ __forceinline__ float toFloat<half>(half x)
{
    return __half2float(x);
}

template <>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(__nv_bfloat16 x)
{
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float x);

template <>
__device__ __forceinline__ half fromFloat<half>(float x)
{
    return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float x)
{
    return __float2bfloat16(x);
}

template <typename T>
__global__ void fusedAddRMSNormFP8Kernel(GeneralFP8AddBiasResidualPreLayerNormParam<T> const param)
{
    int const row = blockIdx.x;
    if (row >= param.m)
        return;

    int const n = param.n;
    float const eps = param.layernorm_eps;
    float const scale_val = (param.scale != nullptr) ? *param.scale : 1.0f;

    T const* __restrict__ input_row = param.input + row * n;
    T const* __restrict__ residual_row = param.residual + row * n;
    T const* __restrict__ gamma = param.gamma;
    T* __restrict__ output_row = param.output + row * n;
    __nv_fp8_e4m3* __restrict__ normed_fp8_row = param.normed_output + row * n;
    T* __restrict__ hp_normed_row
        = param.high_precision_normed_output != nullptr ? param.high_precision_normed_output + row * n : nullptr;

    // Phase 1: add + compute sum of squares (for RMS)
    __shared__ float s_rms;
    float sum_sq = 0.0f;
    for (int tid = threadIdx.x; tid < n; tid += blockDim.x)
    {
        float const x = toFloat<T>(input_row[tid]);
        float const r = toFloat<T>(residual_row[tid]);
        float const add_out = x + r;
        output_row[tid] = fromFloat<T>(add_out);
        sum_sq += add_out * add_out;
    }
    sum_sq = blockReduceSum<float>(sum_sq);
    if (threadIdx.x == 0)
    {
        s_rms = sqrtf(sum_sq / static_cast<float>(n) + eps);
    }
    __syncthreads();
    float const rms = s_rms;

    // Phase 2: normalize, scale for FP8, clamp, cast; optionally write hp_normed
    for (int tid = threadIdx.x; tid < n; tid += blockDim.x)
    {
        float const add_out = toFloat<T>(output_row[tid]);
        float const normed = (add_out / rms) * toFloat<T>(gamma[tid]);
        if (hp_normed_row != nullptr)
        {
            hp_normed_row[tid] = fromFloat<T>(normed);
        }
        float const scaled = normed / scale_val;
        float const clamped = fminf(fmaxf(scaled, -kFP8E4M3Max), kFP8E4M3Max);
        normed_fp8_row[tid] = __nv_fp8_e4m3(clamped);
    }
}

template <typename T>
void invokeFusedAddRMSNormFP8(GeneralFP8AddBiasResidualPreLayerNormParam<T> const& param)
{
    dim3 grid(param.m);
    dim3 block(256);
    fusedAddRMSNormFP8Kernel<T><<<grid, block, 0, param.stream>>>(param);
}

template void invokeFusedAddRMSNormFP8<half>(GeneralFP8AddBiasResidualPreLayerNormParam<half> const& param);
template void invokeFusedAddRMSNormFP8<__nv_bfloat16>(
    GeneralFP8AddBiasResidualPreLayerNormParam<__nv_bfloat16> const& param);

} // namespace kernels
} // namespace tensorrt_llm
