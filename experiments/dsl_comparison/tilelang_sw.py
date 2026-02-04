"""
======================================================================================
Project: DSL for Sequence Alignment
File: tilelang_sw.py
Date: Dec 10, 2025
Role: Pure Compute Mechanism (TVM/TileLang Backend)
======================================================================================
"""

import torch
import tilelang
import tilelang.language as T

# Define the constants for clarity in the script
NEG_INF = -32000

def create_sw_kernel(N, total_ref_bytes, total_query_bytes, max_len):
    
    BLOCK_SIZE = 256
    
    # Stride for workspace (aligned)
    STRIDE = 1
    while STRIDE < max_len + 2: STRIDE *= 2
    
    # Total workspace size (N blocks * 3 rows * STRIDE)
    WsSize = N * 3 * STRIDE

    @T.prim_func
    def main(
        p_refs: T.handle, p_ref_off: T.handle, p_ref_len: T.handle,
        p_queries: T.handle, p_query_off: T.handle, p_query_len: T.handle,
        p_res_s: T.handle, p_res_r: T.handle, p_res_q: T.handle,
        # Workspace: H, E, F
        p_H: T.handle, p_E: T.handle, p_F: T.handle,
        match: T.int32, mismatch: T.int32, gap_o: T.int32, gap_e: T.int32
    ):
        T.func_attr({"global_symbol": "sw_wavefront", "tir.noalias": True})
        
        # Buffer Definitions
        Refs = T.match_buffer(p_refs, (total_ref_bytes,), dtype="int8")
        RefOff = T.match_buffer(p_ref_off, (N,), dtype="int32")
        RefLen = T.match_buffer(p_ref_len, (N,), dtype="int32")
        
        Queries = T.match_buffer(p_queries, (total_query_bytes,), dtype="int8")
        QueryOff = T.match_buffer(p_query_off, (N,), dtype="int32")
        QueryLen = T.match_buffer(p_query_len, (N,), dtype="int32")
        
        ResS = T.match_buffer(p_res_s, (N,), dtype="int16")
        ResR = T.match_buffer(p_res_r, (N,), dtype="int32")
        ResQ = T.match_buffer(p_res_q, (N,), dtype="int32")
        
        # Workspace Buffers
        H_buf = T.match_buffer(p_H, (WsSize,), dtype="int16")
        E_buf = T.match_buffer(p_E, (WsSize,), dtype="int16")
        F_buf = T.match_buffer(p_F, (WsSize,), dtype="int16")

        # Kernel Launch
        with T.Kernel(N, threads=BLOCK_SIZE) as (bx):
            
            # 1. Global Shared Memory Allocation (Block Scope)
            sm_s = T.alloc_shared((BLOCK_SIZE,), "int16")
            sm_r = T.alloc_shared((BLOCK_SIZE,), "int32")
            sm_q = T.alloc_shared((BLOCK_SIZE,), "int32")
            
            for tx in T.thread_binding(0, BLOCK_SIZE, "threadIdx.x"):
                
                # 2. Init Thread-Local State in Shared Memory
                # This ensures persistence and visibility
                sm_s[tx] = T.int16(0)
                sm_r[tx] = T.int32(-1)
                sm_q[tx] = T.int32(-1)

                # Load Metadata
                M = RefLen[bx]
                K = QueryLen[bx]
                
                ref_start = RefOff[bx]
                query_start = QueryOff[bx]
                
                ws_base = bx * 3 * STRIDE
                
                # DEBUG: Print kernel start info for Block 0, Thread 0
                # if bx == 0 and tx == 0:
                #     T.evaluate(T.call_extern("int", "printf", 
                #         "DEBUG[Start] Block 0: M=%d, K=%d, RefStart=%d, QStart=%d, WSBase=%d\\n", 
                #         M, K, ref_start, query_start, ws_base))

                # 3. Init Workspace (Parallel)
                init_limit = 3 * STRIDE
                for k in T.serial((init_limit + BLOCK_SIZE - 1) // BLOCK_SIZE):
                    local_off = k * BLOCK_SIZE + tx
                    if local_off < init_limit:
                        H_buf[ws_base + local_off] = T.int16(0)
                        E_buf[ws_base + local_off] = T.int16(NEG_INF)
                        F_buf[ws_base + local_off] = T.int16(NEG_INF)
                
                T.evaluate(T.call_extern("void", "__syncthreads"))
                
                # 4. Wavefront Loop
                total_diags = M + K + 1
                
                for d in T.serial(total_diags):
                    curr_idx = (d % 3) * STRIDE
                    prev_idx = ((d + 2) % 3) * STRIDE 
                    pprev_idx = ((d + 1) % 3) * STRIDE 
                    
                    i_min = T.max(1, d - M)
                    i_max = T.min(K, d - 1)
                    L = i_max - i_min + 1
                    
                    for pass_idx in T.serial((L + BLOCK_SIZE - 1) // BLOCK_SIZE):
                        off = pass_idx * BLOCK_SIZE + tx
                        if off < L:
                            i = i_min + off
                            j = d - i
                            
                            # --- Fetch Dependencies ---
                            h_left = T.Select(j > 1, H_buf[ws_base + prev_idx + i], T.int16(0))
                            e_left = T.Select(j > 1, E_buf[ws_base + prev_idx + i], T.int16(NEG_INF))
                            
                            h_up = T.Select(i > 1, H_buf[ws_base + prev_idx + (i - 1)], T.int16(0))
                            f_up = T.Select(i > 1, F_buf[ws_base + prev_idx + (i - 1)], T.int16(NEG_INF))
                            
                            h_diag = T.Select((i > 1) and (j > 1), H_buf[ws_base + pprev_idx + (i - 1)], T.int16(0))
                            
                            # --- Compute ---
                            e_new = T.max(h_left + T.cast(gap_o, "int16"), e_left + T.cast(gap_e, "int16"))
                            f_new = T.max(h_up + T.cast(gap_o, "int16"), f_up + T.cast(gap_e, "int16"))
                            
                            val_q = Queries[query_start + (i - 1)]
                            val_r = Refs[ref_start + (j - 1)]
                            
                            score_sub = T.Select(val_q == val_r, match, mismatch)
                            score_match = h_diag + T.cast(score_sub, "int16")
                            
                            h_new = T.max(
                                T.max(T.int16(0), score_match),
                                T.max(e_new, f_new)
                            )
                            
                            # --- Store ---
                            H_buf[ws_base + curr_idx + i] = h_new
                            E_buf[ws_base + curr_idx + i] = e_new
                            F_buf[ws_base + curr_idx + i] = f_new
                            
                            # --- Update Best (Directly to Shared Memory) ---
                            # FIX: Use Shared Memory as the definitive storage for local best
                            # This bypasses any register allocation issues in deep loops
                            current_best = sm_s[tx]
                            if h_new > current_best:
                                sm_s[tx] = h_new
                                sm_r[tx] = j
                                sm_q[tx] = i
                                
                                # Optional Debug
                                # if bx == 0 and h_new > 10: # Only print significant updates
                                #      T.evaluate(T.call_extern("int", "printf", 
                                #         "DEBUG[SharedUpdate] TX=%d: New Best: %d\\n", tx, h_new))
                                
                    T.evaluate(T.call_extern("void", "__syncthreads"))
                    
                # 5. Block Reduction
                T.evaluate(T.call_extern("void", "__syncthreads"))
                
                if tx == 0:
                    final_s = T.alloc_fragment((1,), "int16")
                    final_r = T.alloc_fragment((1,), "int32")
                    final_q = T.alloc_fragment((1,), "int32")
                    
                    final_s[0] = T.int16(0)
                    final_r[0] = T.int32(-1)
                    final_q[0] = T.int32(-1)
                    
                    for k in T.serial(BLOCK_SIZE):
                        val = sm_s[k]
                        if val > final_s[0]:
                            final_s[0] = val
                            final_r[0] = sm_r[k]
                            final_q[0] = sm_q[k]
                    
                    # if bx == 0:
                    #      T.evaluate(T.call_extern("int", "printf", 
                    #         "DEBUG[Final] Block 0 Result: Score=%d, R=%d, Q=%d\\n", final_s[0], final_r[0], final_q[0]))

                    # Write Final Result
                    ResS[bx] = final_s[0]
                    # Convert 1-based DP coord to 0-based result index
                    r_out = T.Select(final_s[0] > 0, final_r[0] - 1, -1)
                    q_out = T.Select(final_s[0] > 0, final_q[0] - 1, -1)
                    
                    ResR[bx] = r_out
                    ResQ[bx] = q_out

    return main

def run_sw_tilelang(
    N, 
    ref_flat, ref_off, ref_len,
    query_flat, query_off, query_len,
    max_len,
    match, mismatch, gap_open, gap_extend
):
    t_refs = torch.from_numpy(ref_flat).to("cuda")
    t_ref_off = torch.from_numpy(ref_off).to("cuda")
    t_ref_len = torch.from_numpy(ref_len).to("cuda")
    
    t_query = torch.from_numpy(query_flat).to("cuda")
    t_query_off = torch.from_numpy(query_off).to("cuda")
    t_query_len = torch.from_numpy(query_len).to("cuda")
    
    t_res_s = torch.zeros(N, dtype=torch.int16, device="cuda")
    t_res_r = torch.zeros(N, dtype=torch.int32, device="cuda")
    t_res_q = torch.zeros(N, dtype=torch.int32, device="cuda")
    
    STRIDE = 1
    while STRIDE < max_len + 2: STRIDE *= 2
    ws_size = N * 3 * STRIDE
    
    t_H = torch.zeros(ws_size, dtype=torch.int16, device="cuda")
    t_E = torch.zeros(ws_size, dtype=torch.int16, device="cuda")
    t_F = torch.zeros(ws_size, dtype=torch.int16, device="cuda")
    
    total_ref_bytes = t_refs.shape[0]
    total_query_bytes = t_query.shape[0]
    
    prim_func = create_sw_kernel(N, total_ref_bytes, total_query_bytes, max_len)
    rt_mod = tilelang.compile(prim_func, target="cuda")
    
    rt_mod(
        t_refs, t_ref_off, t_ref_len,
        t_query, t_query_off, t_query_len,
        t_res_s, t_res_r, t_res_q,
        t_H, t_E, t_F,
        match, mismatch, gap_open, gap_extend
    )
    
    return t_res_s.cpu().numpy(), t_res_r.cpu().numpy(), t_res_q.cpu().numpy()