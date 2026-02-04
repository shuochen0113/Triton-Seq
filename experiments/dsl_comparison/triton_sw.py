"""
======================================================================================
Project: DSL for Sequence Alignment
File: triton_sw.py
Date: Dec 10, 2025
======================================================================================
"""

import torch
import triton
import triton.language as tl
import sys

@triton.jit
def sw_affine_kernel(
    q_ptr, r_ptr, q_off_ptr, r_off_ptr, q_len_ptr, r_len_ptr,
    H_ptr, E_ptr, F_ptr,
    out_score_ptr, out_r_ptr, out_q_ptr,
    MATCH: tl.constexpr, MISM: tl.constexpr, GAP_O: tl.constexpr, GAP_E: tl.constexpr,
    STRIDE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    m = tl.load(r_len_ptr + pid)
    n = tl.load(q_len_ptr + pid)
    q_start = tl.load(q_off_ptr + pid)
    r_start = tl.load(r_off_ptr + pid)
    
    curr_q_ptr = q_ptr + q_start
    curr_r_ptr = r_ptr + r_start

    base_ws_off = pid * 3 * STRIDE
    H_base = H_ptr + base_ws_off
    E_base = E_ptr + base_ws_off
    F_base = F_ptr + base_ws_off

    MIN_VAL = -32000 

    best_score = tl.full((), 0, tl.int32)
    best_r = tl.full((), -1, tl.int32)
    best_q = tl.full((), -1, tl.int32)

    total_diags = m + n + 1 
    
    for d in range(0, total_diags):
        curr_idx = (d % 3) * STRIDE
        prev_idx = ((d + 2) % 3) * STRIDE
        pprev_idx = ((d + 1) % 3) * STRIDE

        i_min = tl.maximum(1, d - m)
        i_max = tl.minimum(n, d - 1)
        L = i_max - i_min + 1
        
        diag_max_s = tl.full((BLOCK_SIZE,), MIN_VAL, tl.int32)
        diag_max_i = tl.full((BLOCK_SIZE,), -1, tl.int32)
        
        for off in range(0, L, BLOCK_SIZE):
            offs = off + tl.arange(0, BLOCK_SIZE)
            mask = offs < L
            
            i = i_min + offs
            j = d - i
            
            q_val = tl.load(curr_q_ptr + (i - 1), mask=mask, other=4).to(tl.int32)
            r_val = tl.load(curr_r_ptr + (j - 1), mask=mask, other=4).to(tl.int32)
            
            is_match = (q_val == r_val)
            sub_score = tl.where(is_match, MATCH, MISM)

            H_prev_ptr = H_base + prev_idx
            E_prev_ptr = E_base + prev_idx
            F_prev_ptr = F_base + prev_idx
            H_pprev_ptr = H_base + pprev_idx
            
            has_left = (j > 1)
            has_up   = (i > 1)
            has_diag = (i > 1) & (j > 1)

            h_left = tl.load(H_prev_ptr + i, mask=mask & has_left, other=0).to(tl.int32)
            e_left = tl.load(E_prev_ptr + i, mask=mask & has_left, other=MIN_VAL).to(tl.int32)
            
            h_up = tl.load(H_prev_ptr + (i - 1), mask=mask & has_up, other=0).to(tl.int32)
            f_up = tl.load(F_prev_ptr + (i - 1), mask=mask & has_up, other=MIN_VAL).to(tl.int32)
            
            h_diag = tl.load(H_pprev_ptr + (i - 1), mask=mask & has_diag, other=0).to(tl.int32)

            e_curr = tl.maximum(h_left + GAP_O, e_left + GAP_E)
            f_curr = tl.maximum(h_up + GAP_O, f_up + GAP_E)
            score_match = h_diag + sub_score
            
            h_curr = tl.maximum(0, score_match)
            h_curr = tl.maximum(h_curr, e_curr)
            h_curr = tl.maximum(h_curr, f_curr)

            out_idx = i
            H_curr_ptr = H_base + curr_idx + out_idx
            E_curr_ptr = E_base + curr_idx + out_idx
            F_curr_ptr = F_base + curr_idx + out_idx
            
            tl.store(H_curr_ptr, h_curr.to(tl.int16), mask=mask)
            tl.store(E_curr_ptr, e_curr.to(tl.int16), mask=mask)
            tl.store(F_curr_ptr, f_curr.to(tl.int16), mask=mask)
            
            is_new_max = h_curr > diag_max_s
            diag_max_s = tl.where(mask & is_new_max, h_curr, diag_max_s)
            diag_max_i = tl.where(mask & is_new_max, i, diag_max_i)

        iter_max_s = tl.max(diag_max_s, axis=0)
        is_winner = (diag_max_s == iter_max_s)
        winner_i = tl.max(tl.where(is_winner, diag_max_i, -1), axis=0)
        
        cand_r = d - winner_i
        cand_q = winner_i

        cond = iter_max_s > best_score
        
        best_score = tl.where(cond, iter_max_s, best_score)
        best_r = tl.where(cond, cand_r, best_r)
        best_q = tl.where(cond, cand_q, best_q)

    tl.store(out_score_ptr + pid, best_score.to(tl.int16))
    tl.store(out_r_ptr + pid, (best_r - 1).to(tl.int32))
    tl.store(out_q_ptr + pid, (best_q - 1).to(tl.int32))

def run_sw_triton(
    N, ref_flat, ref_off, ref_len,
    query_flat, query_off, query_len,
    max_len,
    match, mismatch, gap_open, gap_extend
):
    device = torch.device('cuda')
    
    t_ref = torch.from_numpy(ref_flat).to(device).to(torch.int8)
    t_ref_off = torch.from_numpy(ref_off).to(device).to(torch.int32)
    t_ref_len = torch.from_numpy(ref_len).to(device).to(torch.int32)
    
    t_query = torch.from_numpy(query_flat).to(device).to(torch.int8)
    t_query_off = torch.from_numpy(query_off).to(device).to(torch.int32)
    t_query_len = torch.from_numpy(query_len).to(device).to(torch.int32)

    STRIDE = triton.next_power_of_2(max_len)
    total_ws_elements = N * 3 * STRIDE
    
    H_buf = torch.zeros(total_ws_elements, dtype=torch.int16, device=device)
    E_buf = torch.full((total_ws_elements,), -32000, dtype=torch.int16, device=device)
    F_buf = torch.full((total_ws_elements,), -32000, dtype=torch.int16, device=device)

    out_score = torch.zeros(N, dtype=torch.int16, device=device)
    out_r = torch.zeros(N, dtype=torch.int32, device=device)
    out_q = torch.zeros(N, dtype=torch.int32, device=device)

    grid = (N,)
    BLOCK_SIZE = 256 
    
    sw_affine_kernel[grid](
        t_query, t_ref, t_query_off, t_ref_off, t_query_len, t_ref_len,
        H_buf, E_buf, F_buf,
        out_score, out_r, out_q,
        MATCH=match, MISM=mismatch, GAP_O=gap_open, GAP_E=gap_extend,
        STRIDE=STRIDE, BLOCK_SIZE=BLOCK_SIZE
    )

    return out_score.cpu().numpy(), out_r.cpu().numpy(), out_q.cpu().numpy()