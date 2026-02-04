"""
======================================================================================
Project: DSL for Sequence Alignment
File: cutile_sw.py
Date: Dec 10, 2025 (Compatible with bench.py)
Fixes (correct v1):
1. FIXED DATA CORRUPTION BUG:
   - Previously, 'c_safe_idx = 0' caused all invalid threads (across all blocks)
     to write garbage to spill_buffer[0].
   - This corrupted the DP state for Block 0 (Pair 0) and potentially others due to races.
2. Solution:
   - Pass 'dump_offset' pointing to the safety padding at the end of allocation.
   - All OOB reads/writes are redirected to this safe zone.
3. Cleaned up arithmetic to ensure no ghost matches propagate.
======================================================================================
"""

import torch
import cuda.tile as ct
import numpy as np

# Config
TILE_SIZE = 256  
NEG_INF_16 = -32000

@ct.kernel
def sw_affine_cutile_kernel(
    # --- Data Arrays ---
    refs: ct.Array, queries: ct.Array,
    ref_offsets: ct.Array, query_offsets: ct.Array,
    ref_lens: ct.Array, query_lens: ct.Array,
    spill_buffer: ct.Array,
    
    # --- Outputs ---
    out_scores: ct.Array, out_ends_ref: ct.Array, out_ends_query: ct.Array,
    
    # --- Constants ---
    stride: int,
    dump_offset: int, # NEW: Safe index for OOB access
    match_score: int, mismatch_score: int, 
    gap_open: int, gap_extend: int
):
    # [1] Task Coordinates
    bid = ct.bid(0)
    
    # Load Metadata (Rank-0)
    m = ct.load(ref_lens, index=(bid,), shape=())
    n = ct.load(query_lens, index=(bid,), shape=())
    r_start = ct.load(ref_offsets, index=(bid,), shape=())
    q_start = ct.load(query_offsets, index=(bid,), shape=())

    # [2] Constants
    c_match = ct.int16(match_score)
    c_mism = ct.int16(mismatch_score)
    c_gap_o = ct.int16(gap_open)
    c_gap_e = ct.int16(gap_extend)
    c_zero = ct.int16(0)
    c_neg_inf = ct.int16(NEG_INF_16)
    
    # Safe index: Points to the extra padding allocated at the end
    c_dump_idx = ct.int32(dump_offset)

    # Base Address
    base_addr = bid * (9 * stride)
    
    # [3] State Initialization
    best_score = ct.full((), 0, dtype=ct.int16)
    best_r = ct.full((), -1, dtype=ct.int32)
    best_q = ct.full((), -1, dtype=ct.int32)

    # [4] Wavefront Iteration
    total_diags = m + n + 1
    
    for k in range(total_diags):
        # Rotation
        curr_row = k % 3
        prev_row = (k + 2) % 3
        pprev_row = (k + 1) % 3
        
        # Row Base Addresses
        addr_curr_base = base_addr + (curr_row * 3 * stride)
        addr_prev_base = base_addr + (prev_row * 3 * stride)
        addr_pprev_base = base_addr + (pprev_row * 3 * stride)
        
        # Bounds
        i_min = ct.maximum(0, k - m)
        i_max = ct.minimum(k, n)
        valid_len = ct.maximum(0, i_max - i_min + 1)
        
        # Prev/PPrev Bounds
        prev_i_min = ct.maximum(0, (k - 1) - m)
        pprev_i_min = ct.maximum(0, (k - 2) - m)
        
        num_tiles = ct.cdiv(valid_len, TILE_SIZE).astype(ct.int32)
        
        for t in range(num_tiles):
            # --- [A] Domain ---
            tid = ct.arange(TILE_SIZE, dtype=ct.int32)
            diag_offset = t * TILE_SIZE
            
            # Mask
            active_mask = (tid + diag_offset) < valid_len
            
            # Coords
            idx_i = i_min + diag_offset + tid
            idx_j = k - idx_i
            
            # --- [B] Address Arithmetic ---
            
            # 1. Diagonal (k-2)
            idxs_pprev = (idx_i - 1) - pprev_i_min
            addrs_h_diag = addr_pprev_base + idxs_pprev
            
            # 2. Left (k-1)
            idxs_prev_left = idx_i - prev_i_min
            addrs_h_left = addr_prev_base + idxs_prev_left
            addrs_e_left = addr_prev_base + stride + idxs_prev_left
            
            # 3. Up (k-1)
            idxs_prev_up = (idx_i - 1) - prev_i_min
            addrs_h_up = addr_prev_base + idxs_prev_up
            addrs_f_up = addr_prev_base + (2 * stride) + idxs_prev_up

            # Boundary Conditions
            cond_i_gt_0 = idx_i > 0
            cond_j_gt_0 = idx_j > 0
            
            # --- [C] Memory Fetch (Gather + Clamp to Dump) ---
            # Using c_dump_idx instead of 0 ensures we don't corrupt/read Block 0's data
            
            safe_h_diag = ct.where(active_mask & cond_i_gt_0 & cond_j_gt_0, addrs_h_diag, c_dump_idx)
            safe_h_left = ct.where(active_mask & cond_j_gt_0, addrs_h_left, c_dump_idx)
            safe_e_left = ct.where(active_mask & cond_j_gt_0, addrs_e_left, c_dump_idx)
            safe_h_up   = ct.where(active_mask & cond_i_gt_0, addrs_h_up, c_dump_idx)
            safe_f_up   = ct.where(active_mask & cond_i_gt_0, addrs_f_up, c_dump_idx)
            
            h_diag = ct.gather(spill_buffer, safe_h_diag).astype(ct.int16)
            h_left = ct.gather(spill_buffer, safe_h_left).astype(ct.int16)
            e_left = ct.gather(spill_buffer, safe_e_left).astype(ct.int16)
            h_up   = ct.gather(spill_buffer, safe_h_up).astype(ct.int16)
            f_up   = ct.gather(spill_buffer, safe_f_up).astype(ct.int16)
            
            # Value Masking
            val_h_diag = ct.where(cond_i_gt_0 & cond_j_gt_0, h_diag, c_zero)
            val_h_left = ct.where(cond_j_gt_0, h_left, c_zero)
            val_e_left = ct.where(cond_j_gt_0, e_left, c_neg_inf)
            val_h_up   = ct.where(cond_i_gt_0, h_up, c_zero)
            val_f_up   = ct.where(cond_i_gt_0, f_up, c_neg_inf)
            
            # --- [D] Sequence Data ---
            # Clamp to q_start/r_start for OOB reads (safe, as we mask result)
            
            idxs_q_raw = q_start + (idx_i - 1)
            safe_idxs_q = ct.where(active_mask & cond_i_gt_0, idxs_q_raw, q_start)
            val_q = ct.gather(queries, safe_idxs_q)
            # Important: Set to unique value if invalid to prevent accidental match
            val_q = ct.where(cond_i_gt_0, val_q, ct.int8(100))
            
            idxs_r_raw = r_start + (idx_j - 1)
            safe_idxs_r = ct.where(active_mask & cond_j_gt_0, idxs_r_raw, r_start)
            val_r = ct.gather(refs, safe_idxs_r)
            val_r = ct.where(cond_j_gt_0, val_r, ct.int8(200))
            
            # --- [E] Arithmetic ---
            is_match = val_q == val_r
            score_sub = ct.where(is_match, c_match, c_mism)
            
            v_e = ct.maximum(ct.add(val_h_left, c_gap_o), ct.add(val_e_left, c_gap_e))
            v_f = ct.maximum(ct.add(val_h_up, c_gap_o), ct.add(val_f_up, c_gap_e))
            term_match = ct.add(val_h_diag, score_sub)
            v_h = ct.maximum(c_zero, ct.maximum(term_match, ct.maximum(v_e, v_f)))
            
            # --- [F] Write Back (Scatter to Dump) ---
            
            dest_cols = diag_offset + tid
            
            base_store_h = addr_curr_base + dest_cols
            base_store_e = addr_curr_base + stride + dest_cols
            base_store_f = addr_curr_base + (2 * stride) + dest_cols
            
            # Redirect invalid writes to dump_idx
            safe_store_h = ct.where(active_mask, base_store_h, c_dump_idx)
            safe_store_e = ct.where(active_mask, base_store_e, c_dump_idx)
            safe_store_f = ct.where(active_mask, base_store_f, c_dump_idx)
            
            ct.scatter(spill_buffer, safe_store_h, v_h)
            ct.scatter(spill_buffer, safe_store_e, v_e)
            ct.scatter(spill_buffer, safe_store_f, v_f)
            
            # --- [G] Reduction ---
            safe_v_h = ct.where(active_mask, v_h, c_neg_inf)
            tile_max = ct.max(safe_v_h)
            
            cond = tile_max > best_score
            
            max_local_idx = ct.argmax(safe_v_h)
            cand_q = i_min + diag_offset + max_local_idx
            cand_r = k - cand_q
            
            best_score = ct.where(cond, tile_max, best_score)
            best_q = ct.where(cond, cand_q, best_q)
            best_r = ct.where(cond, cand_r, best_r)

    # [5] Finalize
    res_r = ct.where(best_r > 0, best_r - 1, -1)
    res_q = ct.where(best_q > 0, best_q - 1, -1)
    
    ct.store(out_scores, index=(bid,), tile=best_score)
    ct.store(out_ends_ref, index=(bid,), tile=res_r)
    ct.store(out_ends_query, index=(bid,), tile=res_q)


def run_sw_cutile(
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

    # Padding calculation
    STRIDE = int(max_len) + 256
    if STRIDE % 256 != 0:
        STRIDE += (256 - (STRIDE % 256))
        
    # Allocate with explicit dump zone
    # Valid data ends at: N * 9 * STRIDE
    # Dump zone starts at: N * 9 * STRIDE
    # Size: +4096 elements
    dump_start_offset = N * 9 * STRIDE
    total_spill = dump_start_offset + 4096 
    t_spill = torch.zeros(total_spill, dtype=torch.int16, device=device)
    
    t_out_score = torch.zeros(N, dtype=torch.int16, device=device)
    t_out_r = torch.zeros(N, dtype=torch.int32, device=device)
    t_out_q = torch.zeros(N, dtype=torch.int32, device=device)

    grid = (N, 1, 1)
    stream = torch.cuda.current_stream().cuda_stream
    
    ct.launch(
        stream, grid, sw_affine_cutile_kernel,
        (
            t_ref, t_query, t_ref_off, t_query_off, t_ref_len, t_query_len,
            t_spill,
            t_out_score, t_out_r, t_out_q,
            STRIDE,
            dump_start_offset, # Passing the safe dump offset
            match, mismatch, gap_open, gap_extend
        )
    )

    return t_out_score.cpu().numpy(), t_out_r.cpu().numpy(), t_out_q.cpu().numpy()