/**
 * ======================================================================================
 * Project: DSL for Sequence Alignment
 * File: tk_sw.cu
 * Architecture: NVIDIA Ampere / Hopper / Blackwell
 * Date: Dec 10, 2025
 * ======================================================================================
 * Compilation:
    sed -i 's/static __device__ inline constexpr bf16_2/static __device__ inline bf16_2/g' ThunderKittens/include/common/base_types.cuh
    sed -i 's/static __device__ inline constexpr half_2/static __device__ inline half_2/g' ThunderKittens/include/common/base_types.cuh
    sed -i 's/cuCtxCreate_v4(&contexts\[i\], 0, devices\[i\])/0/g' ThunderKittens/include/types/device/ipc.cuh
    nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_86 -std=c++20 --extended-lambda --expt-relaxed-constexpr -I./ThunderKittens/include tk_sw.cu -o tk_sw.so
 */

#include <cuda_runtime.h>
#include <cuda.h> // Include Driver API FIRST to avoid macro collisions
#include <iostream>

// 1. Monkey Patch for missing types (FP4/FP8)
struct fp4_2 { }; 
struct fp4_4 { }; 
struct fp8e4m3 { };

// 2. Fix IPC Error safely
// We define this macro AFTER <cuda.h> so it doesn't break the function declaration inside cuda.h,
// but it WILL replace the function call inside ThunderKittens/ipc.cuh.
#define cuCtxCreate_v4(...) CUDA_SUCCESS

// 3. ThunderKittens Include
#define KITTENS_HOPPER 
#include "kittens.cuh"

// 4. Inject int16_t packing support
namespace kittens {
namespace base_types {
    template<> struct packing<int16_t> {
        using unpacked_type = int16_t;
        using packed_type = uint32_t;
        static constexpr int num() { return 2; }
    };
}
}

#define NEG_INF -32000
#define BLOCK_SIZE 256

struct __align__(4) AlignResult {
    int16_t score;
    int32_t end_ref;
    int32_t end_query;
};

using namespace kittens;

// ... (Rest of the Kernel remains unchanged) ...

__global__ void __launch_bounds__(BLOCK_SIZE, 1) sw_kittens_kernel(
    int N,
    const int8_t* __restrict__ refs_flat, const int* __restrict__ ref_offsets, const int* __restrict__ ref_lens,
    const int8_t* __restrict__ queries_flat, const int* __restrict__ query_offsets, const int* __restrict__ query_lens,
    AlignResult* __restrict__ results,
    int16_t* __restrict__ global_ws_H,
    int16_t* __restrict__ global_ws_E,
    int16_t* __restrict__ global_ws_F,
    int stride, 
    int match, int mismatch, int gap_o, int gap_e
) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using RedVec = sv<int16_t, BLOCK_SIZE>; 
    RedVec &sm_s = al.allocate<RedVec>();
    using CoordVec = sv<int32_t, BLOCK_SIZE>;
    CoordVec &sm_r = al.allocate<CoordVec>();
    CoordVec &sm_q = al.allocate<CoordVec>();

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    if (bx >= N) return;

    int16_t best_s_reg = 0;
    int32_t best_r_reg = -1;
    int32_t best_q_reg = -1;

    int M = ref_lens[bx];
    int K = query_lens[bx];
    const int8_t* ref_ptr = refs_flat + ref_offsets[bx];
    const int8_t* query_ptr = queries_flat + query_offsets[bx];
    size_t ws_base = (size_t)bx * 3 * stride;

    int init_limit = 3 * stride;
    for (int k = tx; k < init_limit; k += BLOCK_SIZE) {
        global_ws_H[ws_base + k] = 0;
        global_ws_E[ws_base + k] = NEG_INF;
        global_ws_F[ws_base + k] = NEG_INF;
    }
    __syncthreads(); 

    int total_diags = M + K + 1;
    for (int d = 0; d < total_diags; ++d) {
        int curr_idx = (d % 3) * stride;
        int prev_idx = ((d + 2) % 3) * stride;
        int pprev_idx = ((d + 1) % 3) * stride;

        int i_min = max(0, d - M);
        int i_max = min(d, K); 
        int valid_len = i_max - i_min + 1;

        for (int pass_idx = 0; pass_idx < (valid_len + BLOCK_SIZE - 1) / BLOCK_SIZE; ++pass_idx) {
            int off = pass_idx * BLOCK_SIZE + tx;
            if (off < valid_len) {
                int i = i_min + off;
                int j = d - i;
                
                if (i > 0 && j > 0) {
                    int16_t h_left = global_ws_H[ws_base + prev_idx + i];
                    int16_t e_left = global_ws_E[ws_base + prev_idx + i];
                    int16_t h_up = global_ws_H[ws_base + prev_idx + (i - 1)];
                    int16_t f_up = global_ws_F[ws_base + prev_idx + (i - 1)];
                    int16_t h_diag = global_ws_H[ws_base + pprev_idx + (i - 1)];

                    int16_t e_new = max((int)(h_left + gap_o), (int)(e_left + gap_e));
                    int16_t f_new = max((int)(h_up + gap_o), (int)(f_up + gap_e));

                    int8_t val_q = query_ptr[i - 1];
                    int8_t val_r = ref_ptr[j - 1];
                    int16_t score_match = h_diag + (val_q == val_r ? match : mismatch);
                    int16_t h_new = max(0, (int)score_match);
                    h_new = max((int)h_new, (int)e_new);
                    h_new = max((int)h_new, (int)f_new);

                    global_ws_H[ws_base + curr_idx + i] = h_new;
                    global_ws_E[ws_base + curr_idx + i] = e_new;
                    global_ws_F[ws_base + curr_idx + i] = f_new;

                    if (h_new > best_s_reg) {
                        best_s_reg = h_new; best_r_reg = j; best_q_reg = i;
                    }
                }
            }
        }
        __syncthreads();
    }

    sm_s[tx] = best_s_reg;
    sm_r[tx] = best_r_reg;
    sm_q[tx] = best_q_reg;
    __syncthreads();

    if (tx == 0) {
        int16_t final_s = 0; int32_t final_r = -1; int32_t final_q = -1;
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            int16_t val = sm_s[k]; 
            if (val > final_s) { final_s = val; final_r = sm_r[k]; final_q = sm_q[k]; }
        }
        results[bx].score = final_s;
        results[bx].end_ref = (final_s > 0) ? final_r - 1 : -1;
        results[bx].end_query = (final_s > 0) ? final_q - 1 : -1;
    }
}

extern "C" {
    void run_sw_kittens(
        int N,
        intptr_t d_refs, intptr_t d_ref_offsets, intptr_t d_ref_lens,
        intptr_t d_queries, intptr_t d_query_offsets, intptr_t d_query_lens,
        intptr_t d_results,
        intptr_t d_H, intptr_t d_E, intptr_t d_F,
        int stride,
        int match, int mismatch, int gap_o, int gap_e
    ) {
        size_t shmem_size = 4096; 
        cudaFuncSetAttribute(sw_kittens_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

        sw_kittens_kernel<<<N, BLOCK_SIZE, shmem_size>>>(
            N,
            (const int8_t*)d_refs, (const int*)d_ref_offsets, (const int*)d_ref_lens,
            (const int8_t*)d_queries, (const int*)d_query_offsets, (const int*)d_query_lens,
            (AlignResult*)d_results,
            (int16_t*)d_H, (int16_t*)d_E, (int16_t*)d_F,
            stride, match, mismatch, gap_o, gap_e
        );
    }
}