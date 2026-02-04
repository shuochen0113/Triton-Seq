/**
 * ======================================================================================
 * Project: DSL for Sequence Alignment
 * File: cpu_sw.cpp
 * Date: Dec 10, 2025
 * Compile: g++ -shared -fPIC -O3 -march=native -fopenmp -std=c++17 cpu_sw.cpp -o cpu_sw.so
 * ======================================================================================
 */

#include <vector>
#include <algorithm>
#include <cstring>
#include <limits>
#include <omp.h>
#include <cstdint>

constexpr int16_t NEG_INF = -32000;

struct AlignResult {
    int16_t score;
    int32_t end_ref;
    int32_t end_query;
};

struct CpuThreadBuffers {
    std::vector<int16_t> h_prev;
    std::vector<int16_t> f_prev;
    void ensure_size(size_t size) {
        if (h_prev.size() < size) {
            h_prev.resize(size);
            f_prev.resize(size);
        }
    }
};

inline AlignResult sw_affine_cpu_kernel_logic(
    const int8_t* ref, int32_t M,
    const int8_t* query, int32_t N,
    CpuThreadBuffers& buf,
    int16_t match, int16_t mismatch, int16_t open, int16_t extend
) {
    buf.ensure_size(M + 1);
    int16_t* H = buf.h_prev.data();
    int16_t* F = buf.f_prev.data();

    std::memset(H, 0, (M + 1) * sizeof(int16_t));
    for(int k=0; k<=M; ++k) F[k] = NEG_INF;

    AlignResult global_max = {0, -1, -1};

    for (int32_t i = 0; i < N; ++i) {
        int8_t q_val = query[i];
        int16_t e_curr = NEG_INF;
        int16_t h_diag = 0;
        int16_t h_left = 0;

        for (int32_t j = 0; j < M; ++j) {
            int32_t col_idx = j + 1;
            int16_t h_up = H[col_idx];
            int16_t f_up = F[col_idx];
            
            int16_t f_curr = std::max((int16_t)(h_up + open), (int16_t)(f_up + extend));
            F[col_idx] = f_curr;

            if (j == 0) { e_curr = NEG_INF; h_left = 0; }
            e_curr = std::max((int16_t)(h_left + open), (int16_t)(e_curr + extend));

            int8_t r_val = ref[j];
            int16_t score_match = h_diag + (q_val == r_val ? match : mismatch);

            int16_t h_curr = 0;
            h_curr = std::max(h_curr, score_match);
            h_curr = std::max(h_curr, e_curr);
            h_curr = std::max(h_curr, f_curr);

            if (h_curr > global_max.score) {
                global_max.score = h_curr;
                global_max.end_ref = j;
                global_max.end_query = i;
            }

            h_diag = h_up;
            H[col_idx] = h_curr;
            h_left = h_curr;
        }
    }
    return global_max;
}

extern "C" {
    void run_sw_cpu(
        int N,
        const int8_t* refs_flat, const int32_t* ref_offsets, const int32_t* ref_lens,
        const int8_t* queries_flat, const int32_t* query_offsets, const int32_t* query_lens,
        AlignResult* results,
        int num_threads,
        int match, int mismatch, int gap_open, int gap_extend
    ) {
        omp_set_num_threads(num_threads);

        #pragma omp parallel
        {
            CpuThreadBuffers thread_buf;
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < N; ++i) {
                const int8_t* r_ptr = refs_flat + ref_offsets[i];
                const int8_t* q_ptr = queries_flat + query_offsets[i];
                results[i] = sw_affine_cpu_kernel_logic(
                    r_ptr, ref_lens[i],
                    q_ptr, query_lens[i],
                    thread_buf,
                    (int16_t)match, (int16_t)mismatch, (int16_t)gap_open, (int16_t)gap_extend
                );
            }
        }
    }
}