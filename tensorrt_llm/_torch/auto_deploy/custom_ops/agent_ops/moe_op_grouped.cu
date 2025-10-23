///////////////////////
// Optimization Ideas:
///////////////////////
// This version introduces major performance optimizations:
//
// 1. **Grouped GEMM**: Replace sequential per-expert GEMM calls with batched grouped GEMM
//    - Use cublasGemmGroupedBatchedEx to process all experts' W1 and W2 in 2 calls instead of 2*N calls
//    - Dramatically reduces kernel launch overhead and improves GPU utilization
//    - Better scheduling and potential for parallelism across experts
//
// 2. **CUB Library Integration**: Use CUDA CUB primitives for parallel operations
//    - Replace simple prefix_sum_kernel with CUB's DeviceScan::ExclusiveSum
//    - More efficient parallel scan with better work distribution
//
// 3. **Memory Layout Optimization**: 
//    - Allocate contiguous buffers for all experts' intermediate data
//    - Better memory locality and reduced fragmentation
//    - Pre-calculate all memory offsets for each expert
//
// 4. **Reduced Synchronization**: 
//    - Single synchronization point after grouped GEMMs instead of per-expert syncs
//    - Better overlap of computation and data movement
//
// Expected improvements:
// - 2-5x speedup from grouped GEMM depending on number of experts
// - Reduced CPU overhead from fewer API calls
// - Better GPU occupancy with batched operations

///////////////////////
// Implementation:
///////////////////////

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>

using bf16 = __nv_bfloat16;

__global__ void build_expert_maps_kernel(
    const int* __restrict__ selected_experts,
    const float* __restrict__ routing_weights,
    const int* __restrict__ expert_offsets,
    int* __restrict__ expert_write_counters,
    int* __restrict__ token_indices,
    float* __restrict__ routing_gathered,
    int batch_size,
    int num_selected,
    int num_experts
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;

    if (token < batch_size) {
        #pragma unroll
        for (int s = 0; s < num_selected; ++s) {
            int expert = selected_experts[token * num_selected + s];
            if (expert >= 0 && expert < num_experts) {
                int base = expert_offsets[expert];
                int pos = atomicAdd(&expert_write_counters[expert], 1);
                int write_pos = base + pos;

                token_indices[write_pos] = token;
                routing_gathered[write_pos] = routing_weights[token * num_selected + s];
            }
        }
    }
}

// Optimized gather kernel for all experts at once
__global__ void gather_features_batched_kernel(
    const bf16* __restrict__ x,
    const int* __restrict__ token_indices,
    bf16* __restrict__ x_gathered,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ expert_counts,
    int hidden_dim,
    int batch_size,
    int num_experts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        int num_tokens = expert_counts[expert_idx];
        int offset = expert_offsets[expert_idx];
        int total = num_tokens * hidden_dim;
        
        for (int i = idx; i < total; i += stride) {
            int local_token = i / hidden_dim;
            int feat = i % hidden_dim;
            int orig_token = token_indices[offset + local_token];
            
            if (orig_token >= 0 && orig_token < batch_size) {
                x_gathered[offset * hidden_dim + i] = x[orig_token * hidden_dim + feat];
            } else {
                x_gathered[offset * hidden_dim + i] = __float2bfloat16(0.0f);
            }
        }
    }
}

// Batched activation kernel for all experts
__global__ void relu_squared_batched_kernel(
    bf16** intermediate_ptrs,
    const int* __restrict__ sizes,
    int num_experts
) {
    int expert_idx = blockIdx.y;
    if (expert_idx >= num_experts) return;
    
    int size = sizes[expert_idx];
    bf16* data = intermediate_ptrs[expert_idx];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float val = __bfloat162float(data[i]);
        val = fmaxf(val, 0.0f);
        data[i] = __float2bfloat16(val * val);
    }
}

__global__ void scatter_output_batched_kernel(
    const bf16* __restrict__ expert_outputs,
    const int* __restrict__ token_indices,
    const float* __restrict__ routing_weights,
    bf16* __restrict__ final_output,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ expert_counts,
    int hidden_dim,
    int batch_size,
    int num_experts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        int num_tokens = expert_counts[expert_idx];
        int offset = expert_offsets[expert_idx];
        int total = num_tokens * hidden_dim;
        
        for (int i = idx; i < total; i += stride) {
            int local_token = i / hidden_dim;
            int feat = i % hidden_dim;
            int orig_token = token_indices[offset + local_token];
            
            if (orig_token >= 0 && orig_token < batch_size) {
                float weight = routing_weights[offset + local_token];
                float val = __bfloat162float(expert_outputs[offset * hidden_dim + i]) * weight;
                bf16 val_bf16 = __float2bfloat16(val);
                atomicAdd(&final_output[orig_token * hidden_dim + feat], val_bf16);
            }
        }
    }
}

__global__ void count_expert_tokens_kernel(
    const int* __restrict__ selected_experts,
    int* __restrict__ expert_counts,
    int batch_size,
    int num_selected,
    int num_experts
) {
    extern __shared__ int smem_counts[];

    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        smem_counts[e] = 0;
    }
    __syncthreads();

    for (int token = blockIdx.x * blockDim.x + threadIdx.x;
         token < batch_size;
         token += gridDim.x * blockDim.x) {
        #pragma unroll
        for (int s = 0; s < num_selected; ++s) {
            int expert = selected_experts[token * num_selected + s];
            if (expert >= 0 && expert < num_experts) {
                atomicAdd(&smem_counts[expert], 1);
            }
        }
    }
    __syncthreads();

    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        if (smem_counts[e] > 0) {
            atomicAdd(&expert_counts[e], smem_counts[e]);
        }
    }
}

__global__ void zero_counters_kernel(int* __restrict__ counters, int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_experts) {
        counters[idx] = 0;
    }
}

void launch_gpu_implementation(
    void* x,
    void* selected_experts,
    void* routing_weights,
    void** w1_weights,
    void** w2_weights,
    void* output,
    int batch_size,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int num_selected,
    cudaStream_t stream
) {
    int threads = 256;
    cudaMemsetAsync(output, 0, batch_size * hidden_dim * sizeof(bf16), stream);

    // Allocate buffers
    int* expert_counts;
    int* expert_offsets;
    int* expert_write_counters;
    cudaMallocAsync(&expert_counts, num_experts * sizeof(int), stream);
    cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int), stream);
    cudaMallocAsync(&expert_write_counters, num_experts * sizeof(int), stream);

    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);

    // Count tokens per expert
    int blocks = (batch_size + threads - 1) / threads;
    size_t smem_size = num_experts * sizeof(int);
    count_expert_tokens_kernel<<<blocks, threads, smem_size, stream>>>(
        (const int*)selected_experts,
        expert_counts,
        batch_size,
        num_selected,
        num_experts
    );

    // Use CUB for prefix sum (more efficient than custom kernel)
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                    expert_counts, expert_offsets, num_experts + 1, stream);
    cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                    expert_counts, expert_offsets, num_experts + 1, stream);

    blocks = (num_experts + threads - 1) / threads;
    zero_counters_kernel<<<blocks, threads, 0, stream>>>(expert_write_counters, num_experts);

    // Build expert maps
    int total_assignments = batch_size * num_selected;
    int* token_indices_all;
    float* routing_gathered_all;
    cudaMallocAsync(&token_indices_all, total_assignments * sizeof(int), stream);
    cudaMallocAsync(&routing_gathered_all, total_assignments * sizeof(float), stream);

    blocks = (batch_size + threads - 1) / threads;
    build_expert_maps_kernel<<<blocks, threads, 0, stream>>>(
        (const int*)selected_experts,
        (const float*)routing_weights,
        expert_offsets,
        expert_write_counters,
        token_indices_all,
        routing_gathered_all,
        batch_size,
        num_selected,
        num_experts
    );

    // Copy counts and offsets to host for grouped GEMM setup
    std::vector<int> h_counts(num_experts);
    std::vector<int> h_offsets(num_experts + 1);
    cudaMemcpyAsync(h_counts.data(), expert_counts, num_experts * sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_offsets.data(), expert_offsets, (num_experts + 1) * sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Count non-empty experts
    int num_active_experts = 0;
    for (int i = 0; i < num_experts; ++i) {
        if (h_counts[i] > 0) num_active_experts++;
    }

    if (num_active_experts == 0) {
        // Cleanup and return early
        cudaFreeAsync(expert_counts, stream);
        cudaFreeAsync(expert_offsets, stream);
        cudaFreeAsync(expert_write_counters, stream);
        cudaFreeAsync(token_indices_all, stream);
        cudaFreeAsync(routing_gathered_all, stream);
        cudaFreeAsync(d_temp_storage, stream);
        return;
    }

    // Allocate contiguous buffers for all experts
    int total_tokens = h_offsets[num_experts];
    bf16* x_gathered_all;
    bf16* intermediate_all;
    bf16* expert_out_all;
    cudaMallocAsync(&x_gathered_all, total_tokens * hidden_dim * sizeof(bf16), stream);
    cudaMallocAsync(&intermediate_all, total_tokens * intermediate_dim * sizeof(bf16), stream);
    cudaMallocAsync(&expert_out_all, total_tokens * hidden_dim * sizeof(bf16), stream);

    // Prepare arrays for grouped GEMM
    std::vector<cublasOperation_t> transa_array(num_active_experts, CUBLAS_OP_T);
    std::vector<cublasOperation_t> transb_array(num_active_experts, CUBLAS_OP_N);
    std::vector<int> m_array_w1(num_active_experts);
    std::vector<int> n_array_w1(num_active_experts);
    std::vector<int> k_array_w1(num_active_experts);
    std::vector<int> lda_array_w1(num_active_experts);
    std::vector<int> ldb_array_w1(num_active_experts);
    std::vector<int> ldc_array_w1(num_active_experts);
    std::vector<float> alpha_array(num_active_experts, 1.0f);
    std::vector<float> beta_array(num_active_experts, 0.0f);
    std::vector<int> group_size(num_active_experts, 1);

    std::vector<bf16*> h_w1_ptrs(num_active_experts);
    std::vector<bf16*> h_x_gathered_ptrs(num_active_experts);
    std::vector<bf16*> h_intermediate_ptrs(num_active_experts);
    std::vector<bf16*> h_w2_ptrs(num_active_experts);
    std::vector<bf16*> h_expert_out_ptrs(num_active_experts);

    // Gather all features first
    blocks = std::min(256, (total_tokens * hidden_dim + threads - 1) / threads);
    gather_features_batched_kernel<<<blocks, threads, 0, stream>>>(
        (const bf16*)x,
        token_indices_all,
        x_gathered_all,
        expert_offsets,
        expert_counts,
        hidden_dim,
        batch_size,
        num_experts
    );

    // Setup parameters for active experts
    int active_idx = 0;
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        int num_tokens = h_counts[expert_idx];
        if (num_tokens == 0) continue;
        
        int offset = h_offsets[expert_idx];

        // W1 GEMM: intermediate[intermediate_dim x num_tokens] = W1^T[intermediate_dim x hidden_dim] * x[hidden_dim x num_tokens]
        m_array_w1[active_idx] = intermediate_dim;
        n_array_w1[active_idx] = num_tokens;
        k_array_w1[active_idx] = hidden_dim;
        lda_array_w1[active_idx] = hidden_dim;       // W1 is [hidden_dim x intermediate_dim], transposed
        ldb_array_w1[active_idx] = hidden_dim;       // x is [hidden_dim x num_tokens]
        ldc_array_w1[active_idx] = intermediate_dim; // output is [intermediate_dim x num_tokens]

        h_w1_ptrs[active_idx] = (bf16*)w1_weights[expert_idx];
        h_x_gathered_ptrs[active_idx] = x_gathered_all + offset * hidden_dim;
        h_intermediate_ptrs[active_idx] = intermediate_all + offset * intermediate_dim;

        h_w2_ptrs[active_idx] = (bf16*)w2_weights[expert_idx];
        h_expert_out_ptrs[active_idx] = expert_out_all + offset * hidden_dim;

        active_idx++;
    }

    // Allocate device arrays for pointers
    bf16** d_w1_ptrs;
    bf16** d_x_gathered_ptrs;
    bf16** d_intermediate_ptrs;
    bf16** d_w2_ptrs;
    bf16** d_expert_out_ptrs;

    cudaMallocAsync(&d_w1_ptrs, num_active_experts * sizeof(bf16*), stream);
    cudaMallocAsync(&d_x_gathered_ptrs, num_active_experts * sizeof(bf16*), stream);
    cudaMallocAsync(&d_intermediate_ptrs, num_active_experts * sizeof(bf16*), stream);
    cudaMallocAsync(&d_w2_ptrs, num_active_experts * sizeof(bf16*), stream);
    cudaMallocAsync(&d_expert_out_ptrs, num_active_experts * sizeof(bf16*), stream);

    cudaMemcpyAsync(d_w1_ptrs, h_w1_ptrs.data(), 
                    num_active_experts * sizeof(bf16*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x_gathered_ptrs, h_x_gathered_ptrs.data(), 
                    num_active_experts * sizeof(bf16*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_intermediate_ptrs, h_intermediate_ptrs.data(), 
                    num_active_experts * sizeof(bf16*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_w2_ptrs, h_w2_ptrs.data(), 
                    num_active_experts * sizeof(bf16*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_expert_out_ptrs, h_expert_out_ptrs.data(), 
                    num_active_experts * sizeof(bf16*), cudaMemcpyHostToDevice, stream);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    // ========== GROUPED GEMM for W1 ==========
    cublasStatus_t stat1 = cublasGemmGroupedBatchedEx(
        handle,
        transa_array.data(),
        transb_array.data(),
        m_array_w1.data(),
        n_array_w1.data(),
        k_array_w1.data(),
        alpha_array.data(),
        (const void* const*)d_w1_ptrs,
        CUDA_R_16BF,
        lda_array_w1.data(),
        (const void* const*)d_x_gathered_ptrs,
        CUDA_R_16BF,
        ldb_array_w1.data(),
        beta_array.data(),
        (void* const*)d_intermediate_ptrs,
        CUDA_R_16BF,
        ldc_array_w1.data(),
        num_active_experts,
        group_size.data(),
        CUBLAS_COMPUTE_32F
    );

    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasGemmGroupedBatchedEx (W1) failed with status: " << stat1 << std::endl;
    }

    // ========== Apply activation to all intermediates ==========
    std::vector<int> h_intermediate_sizes(num_active_experts);
    for (int i = 0; i < num_active_experts; ++i) {
        h_intermediate_sizes[i] = n_array_w1[i] * intermediate_dim;
    }
    
    int* d_intermediate_sizes;
    cudaMallocAsync(&d_intermediate_sizes, num_active_experts * sizeof(int), stream);
    cudaMemcpyAsync(d_intermediate_sizes, h_intermediate_sizes.data(), 
                    num_active_experts * sizeof(int), cudaMemcpyHostToDevice, stream);

    int max_intermediate_size = *std::max_element(h_intermediate_sizes.begin(), h_intermediate_sizes.end());
    dim3 act_blocks((max_intermediate_size + threads - 1) / threads, num_active_experts);
    relu_squared_batched_kernel<<<act_blocks, threads, 0, stream>>>(
        d_intermediate_ptrs,
        d_intermediate_sizes,
        num_active_experts
    );

    // ========== GROUPED GEMM for W2 ==========
    std::vector<int> m_array_w2(num_active_experts);
    std::vector<int> n_array_w2(num_active_experts);
    std::vector<int> k_array_w2(num_active_experts);
    std::vector<int> lda_array_w2(num_active_experts);
    std::vector<int> ldb_array_w2(num_active_experts);
    std::vector<int> ldc_array_w2(num_active_experts);

    for (int i = 0; i < num_active_experts; ++i) {
        // W2 GEMM: output[hidden_dim x num_tokens] = W2^T[hidden_dim x intermediate_dim] * intermediate[intermediate_dim x num_tokens]
        m_array_w2[i] = hidden_dim;
        n_array_w2[i] = n_array_w1[i];  // same num_tokens
        k_array_w2[i] = intermediate_dim;
        lda_array_w2[i] = intermediate_dim;  // W2 is [intermediate_dim x hidden_dim], transposed
        ldb_array_w2[i] = intermediate_dim;  // intermediate is [intermediate_dim x num_tokens]
        ldc_array_w2[i] = hidden_dim;        // output is [hidden_dim x num_tokens]
    }

    cublasStatus_t stat2 = cublasGemmGroupedBatchedEx(
        handle,
        transa_array.data(),
        transb_array.data(),
        m_array_w2.data(),
        n_array_w2.data(),
        k_array_w2.data(),
        alpha_array.data(),
        (const void* const*)d_w2_ptrs,
        CUDA_R_16BF,
        lda_array_w2.data(),
        (const void* const*)d_intermediate_ptrs,
        CUDA_R_16BF,
        ldb_array_w2.data(),
        beta_array.data(),
        (void* const*)d_expert_out_ptrs,
        CUDA_R_16BF,
        ldc_array_w2.data(),
        num_active_experts,
        group_size.data(),
        CUBLAS_COMPUTE_32F
    );

    if (stat2 != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasGemmGroupedBatchedEx (W2) failed with status: " << stat2 << std::endl;
    }

    // ========== Scatter outputs ==========
    blocks = std::min(256, (total_tokens * hidden_dim + threads - 1) / threads);
    scatter_output_batched_kernel<<<blocks, threads, 0, stream>>>(
        expert_out_all,
        token_indices_all,
        routing_gathered_all,
        (bf16*)output,
        expert_offsets,
        expert_counts,
        hidden_dim,
        batch_size,
        num_experts
    );

    // Cleanup
    cudaFreeAsync(expert_counts, stream);
    cudaFreeAsync(expert_offsets, stream);
    cudaFreeAsync(expert_write_counters, stream);
    cudaFreeAsync(token_indices_all, stream);
    cudaFreeAsync(routing_gathered_all, stream);
    cudaFreeAsync(x_gathered_all, stream);
    cudaFreeAsync(intermediate_all, stream);
    cudaFreeAsync(expert_out_all, stream);
    cudaFreeAsync(d_w1_ptrs, stream);
    cudaFreeAsync(d_x_gathered_ptrs, stream);
    cudaFreeAsync(d_intermediate_ptrs, stream);
    cudaFreeAsync(d_w2_ptrs, stream);
    cudaFreeAsync(d_expert_out_ptrs, stream);
    cudaFreeAsync(d_intermediate_sizes, stream);
    cudaFreeAsync(d_temp_storage, stream);

    cublasDestroy(handle);
}


