/**
 * ======================================================================================
 * Project: DSL for Sequence Alignment
 * File: cuda_sw.cu
 * Architecture: NVIDIA Ampere / Hopper / Blackwell
 * Date: Dec 10, 2025
 * ======================================================================================
 * Compilation:
 * nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_120 -std=c++17 cuda_sw.cu -o cuda_sw.so
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <cstdio>

#define NEG_INF -32000
#define BLOCK_SIZE 256

struct __align__(4) AlignResult {
    int16_t score;
    int32_t end_ref;
    int32_t end_query;
};

// 1:1 Match with TK's launch bounds
__global__ void __launch_bounds__(BLOCK_SIZE, 1) sw_affine_wavefront_kernel(
    const int8_t* __restrict__ refs_flat,
    const int* __restrict__ ref_offsets,
    const int* __restrict__ ref_lens,
    const int8_t* __restrict__ queries_flat,
    const int* __restrict__ query_offsets,
    const int* __restrict__ query_lens,
    AlignResult* __restrict__ results,
    // Global Workspace
    int16_t* __restrict__ global_ws_H,
    int16_t* __restrict__ global_ws_E,
    int16_t* __restrict__ global_ws_F,
    int stride,
    int match, int mismatch, int gap_o, int gap_e
) {
    // 1. Shared Memory for Reduction (Matching TK's sv logic)
    __shared__ int16_t sm_s[BLOCK_SIZE];
    __shared__ int32_t sm_r[BLOCK_SIZE];
    __shared__ int32_t sm_q[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    
    // Safety check (Matching TK)
    // Note: TK code had `if (bx >= N) return;` but CUDA grid usually exact matches N.
    // We keep it implicitly handled by metadata fetch or explicit if needed.
    // Assuming GridDim.x == N exactly.

    // 2. Metadata Load
    int M = ref_lens[bx];
    int K = query_lens[bx];
    
    const int8_t* ref_ptr = refs_flat + ref_offsets[bx];
    const int8_t* query_ptr = queries_flat + query_offsets[bx];

    size_t ws_base = (size_t)bx * 3 * stride;

    // 3. Registers (Persistent State)
    int16_t best_s_reg = 0;
    int32_t best_r_reg = -1;
    int32_t best_q_reg = -1;

    // 4. Init Global Workspace (Parallel)
    // Matching TK's simple stride loop
    int init_limit = 3 * stride;
    for (int k = tx; k < init_limit; k += BLOCK_SIZE) {
        global_ws_H[ws_base + k] = 0;
        global_ws_E[ws_base + k] = NEG_INF;
        global_ws_F[ws_base + k] = NEG_INF;
    }
    
    __syncthreads();

    // 5. Wavefront Loop
    int total_diags = M + K + 1;

    for (int d = 0; d < total_diags; ++d) {
        int curr_idx = (d % 3) * stride;
        int prev_idx = ((d + 2) % 3) * stride;
        int pprev_idx = ((d + 1) % 3) * stride;

        // Bounds
        int i_min = max(0, d - M);
        int i_max = min(d, K); 
        int valid_len = i_max - i_min + 1;

        // Inner Loop: Intra-Diagonal Parallelism
        // Using explicit pass calculation to match TK's loop unrolling hints
        int num_passes = (valid_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int pass_idx = 0; pass_idx < num_passes; ++pass_idx) {
            int off = pass_idx * BLOCK_SIZE + tx;
            
            if (off < valid_len) {
                int i = i_min + off;
                int j = d - i;

                // STRICT ALIGNMENT: Check boundaries BEFORE access
                // This matches the logic in the latest TK kernel
                if (i > 0 && j > 0) {
                    // Dependency Fetch (Global Mem)
                    // H[i][j-1] -> prev row, index i
                    int16_t h_left = global_ws_H[ws_base + prev_idx + i];
                    int16_t e_left = global_ws_E[ws_base + prev_idx + i];
                    
                    // H[i-1][j] -> prev row, index i-1
                    int16_t h_up   = global_ws_H[ws_base + prev_idx + (i - 1)];
                    int16_t f_up   = global_ws_F[ws_base + prev_idx + (i - 1)];
                    
                    // H[i-1][j-1] -> pprev row, index i-1
                    int16_t h_diag = global_ws_H[ws_base + pprev_idx + (i - 1)];

                    // Compute
                    int16_t e_new = max((int)(h_left + gap_o), (int)(e_left + gap_e));
                    int16_t f_new = max((int)(h_up + gap_o), (int)(f_up + gap_e));
                    
                    int8_t val_q = query_ptr[i - 1];
                    int8_t val_r = ref_ptr[j - 1];
                    int16_t score_match = h_diag + (val_q == val_r ? match : mismatch);
                    
                    int16_t h_new = max(0, (int)score_match);
                    h_new = max((int)h_new, (int)e_new);
                    h_new = max((int)h_new, (int)f_new);

                    // Store Back
                    global_ws_H[ws_base + curr_idx + i] = h_new;
                    global_ws_E[ws_base + curr_idx + i] = e_new;
                    global_ws_F[ws_base + curr_idx + i] = f_new;

                    // Register Update
                    if (h_new > best_s_reg) {
                        best_s_reg = h_new;
                        best_r_reg = j;
                        best_q_reg = i;
                    }
                }
                // Else: Boundary cells stay 0/NEG_INF (Init value), no write needed.
            }
        }
        __syncthreads();
    }

    // 6. Flush to Shared Memory
    sm_s[tx] = best_s_reg;
    sm_r[tx] = best_r_reg;
    sm_q[tx] = best_q_reg;
    
    __syncthreads();

    // 7. Final Block Reduction (Thread 0 Linear Scan)
    // Matching TK reduction logic exactly
    if (tx == 0) {
        int16_t final_s = 0;
        int32_t final_r = -1;
        int32_t final_q = -1;
        
        // Unroll hint for compiler
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE; ++k) {
            int16_t val = sm_s[k];
            if (val > final_s) {
                final_s = val;
                final_r = sm_r[k];
                final_q = sm_q[k];
            }
        }
        
        results[bx].score = final_s;
        results[bx].end_ref = (final_s > 0) ? final_r - 1 : -1;
        results[bx].end_query = (final_s > 0) ? final_q - 1 : -1;
    }
}

extern "C" {
    void run_sw_cuda(
        int N,
        intptr_t d_refs, intptr_t d_ref_offsets, intptr_t d_ref_lens,
        intptr_t d_queries, intptr_t d_query_offsets, intptr_t d_query_lens,
        intptr_t d_results,
        intptr_t d_H, intptr_t d_E, intptr_t d_F,
        int stride,
        int match, int mismatch, int gap_o, int gap_e
    ) {
        sw_affine_wavefront_kernel<<<N, BLOCK_SIZE>>>(
            (const int8_t*)d_refs, (const int*)d_ref_offsets, (const int*)d_ref_lens,
            (const int8_t*)d_queries, (const int*)d_query_offsets, (const int*)d_query_lens,
            (AlignResult*)d_results,
            (int16_t*)d_H, (int16_t*)d_E, (int16_t*)d_F,
            stride, match, mismatch, gap_o, gap_e
        );
    }
}