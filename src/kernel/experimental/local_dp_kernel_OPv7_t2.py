import triton
import triton.language as tl

"""
2025-08-26

 ksw2_extz2_sse/AGAThA-like Triton-kernel
 OPv7 (Compiler-Friendly):
    - A rollback to a simpler expression of the OPv7 logic.
    - Removes all manual, tile-based scalar pre-calculations (t0, slot_start, thr_e)
      to reduce complex intermediate state.
    - Trusts the compiler to optimize the pure, end-to-end vector expressions.
    - This is an experiment to verify if our manual optimizations were inadvertently
      harming register allocation.
"""

@triton.jit
def sw_kernel(
    # Position arguments
    q_ptrs, r_ptrs,
    m_arr, n_arr,
    outs,
    Hbuf, Fbuf, Ebuf,

    # Scoring parameters
    match_score, mismatch_score,
    gap_open_penalty, gap_extend_penalty,
    drop_threshold,

    # Constexprs
    STRIDE: tl.constexpr,
    BAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # --- KERNEL SETUP ---
    pid = tl.program_id(0)
    m = tl.load(m_arr + pid)
    n = tl.load(n_arr + pid)
    q_ptr = tl.load(q_ptrs + pid).to(tl.pointer_type(tl.int32))
    r_ptr = tl.load(r_ptrs + pid).to(tl.pointer_type(tl.int32))
    
    H32 = Hbuf.to(tl.pointer_type(tl.int32))
    E32 = Ebuf.to(tl.pointer_type(tl.int32))
    F32 = Fbuf.to(tl.pointer_type(tl.int32))
    out = outs + pid * 3

    # --- CONSTANTS ---
    MATCH, MISM = tl.full((), match_score, tl.int32), tl.full((), mismatch_score, tl.int32)
    ALPHA, BETA = tl.full((), gap_open_penalty, tl.int32), tl.full((), gap_extend_penalty, tl.int32)
    GAPOE = ALPHA + BETA
    MINF = tl.full((), -10_000_000, tl.int32)
    NPEN, N_VAL = tl.full((), 1, tl.int32), tl.full((), 14, tl.int32)
    CENTER = STRIDE >> 1

    # --- Z-DROP BOOKKEEPING ---
    best_s = MINF
    best_i, best_j = tl.zeros((), tl.int32), tl.zeros((), tl.int32)

    # --- LOOP SETUP ---
    d = tl.full((), 2, tl.int32)
    prev_lo, prev2_lo = tl.full((), 1, tl.int32), tl.full((), 0, tl.int32)
    stop = tl.zeros((), tl.int32)
    prev_L, prev2_L = tl.full((), 1, tl.int32), tl.full((), 0, tl.int32)
    
    row_off_h32_init = pid * (2 * STRIDE)
    H_slot1_init = H32 + row_off_h32_init + STRIDE
    tl.store(H_slot1_init + CENTER, 0)
    
    R_E, R_F = tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)

    # --- MAIN SWEEP LOOP ---
    while (d <= m + n) & (stop == 0):
        p = d & 1
        
        # --- DIAGONAL-LEVEL SETUP ---
        row_off_h32 = pid * (2 * STRIDE)
        curr_off_h = tl.where(p == 0, 0, STRIDE)
        prev_off_h = tl.where(p == 0, STRIDE, 0)
        Hcurr32, Hprev32, Hprev2_32 = H32 + row_off_h32 + curr_off_h, H32 + row_off_h32 + prev_off_h, H32 + row_off_h32 + curr_off_h
        
        row_off_ef32 = pid * STRIDE
        E_base, F_base = E32 + row_off_ef32, F32 + row_off_ef32
        
        R_E += p
        R_E = tl.where(R_E >= STRIDE, R_E - STRIDE, R_E)
        R_F += p - 1
        R_F = tl.where(R_F < 0, R_F + STRIDE, R_F)

        i_min, i_max = tl.maximum(1, d - n), tl.minimum(m, d - 1)
        half_lo, half_hi = (d - BAND + 2) >> 1, (d + BAND - 1) >> 1
        band_lo, band_hi = tl.maximum(i_min, half_lo), tl.minimum(i_max, half_hi)
        L = band_hi - band_lo + 1
        d_p_half = (d + p) >> 1

        diag_max_s_vec = tl.full((BLOCK,), MINF, tl.int32)
        diag_max_i_vec = tl.zeros((BLOCK,), tl.int32)

        # --- INNER TILE LOOP ---
        off = tl.zeros((), tl.int32)
        while off < L:
            tid = tl.arange(0, BLOCK)
            mask = tid < (L - off)
            
            i_idx = band_lo + off + tid
            j_idx = d - i_idx
            
            # --- PURE VECTOR ADDRESSING ---
            # Calculate all addresses as full vectors, trusting the compiler to optimize.
            t_vec = i_idx - d_p_half
            
            slot_H_curr = CENTER + t_vec
            
            slot_e = CENTER + t_vec + R_E
            slot_e = tl.where(slot_e >= STRIDE, slot_e - STRIDE, slot_e)
            slot_e = tl.where(slot_e < 0, slot_e + STRIDE, slot_e) # Safety wrap

            slot_f = CENTER + t_vec + R_F
            slot_f = tl.where(slot_f >= STRIDE, slot_f - STRIDE, slot_f)
            slot_f = tl.where(slot_f < 0, slot_f + STRIDE, slot_f) # Safety wrap

            # --- DATA LOADING & DP ---
            valid_prev = (i_idx >= prev_lo) & (i_idx <= (prev_lo + prev_L - 1))
            valid_up   = ((i_idx - 1) >= prev_lo) & ((i_idx - 1) <= (prev_lo + prev_L - 1))
            valid_diag = (prev2_L > 0) & ((i_idx - 1) >= prev2_lo) & ((i_idx - 1) <= (prev2_lo + prev2_L - 1))
            
            q_word = tl.load(q_ptr + (i_idx - 1) // 8, mask=mask, other=0)
            r_word = tl.load(r_ptr + (j_idx - 1) // 8, mask=mask, other=0)
            q_code = (q_word >> (((i_idx - 1) % 8) * 4)) & 0xF
            r_code = (r_word >> (((j_idx - 1) % 8) * 4)) & 0xF
            s_val = tl.where((q_code == N_VAL) | (r_code == N_VAL), -NPEN, tl.where(q_code == r_code, MATCH, MISM))

            Hleft = tl.load(Hprev32 + (slot_H_curr + p), mask=mask & valid_prev, other=MINF)
            Hup   = tl.load(Hprev32 + (slot_H_curr + p - 1),   mask=mask & valid_up,   other=MINF)
            Hdiag = tl.load(Hprev2_32 + slot_H_curr,    mask=mask & valid_diag, other=MINF)
            
            Hdiag = tl.where(mask & (i_idx == 1) & (j_idx == 1), 0, Hdiag)
            Hleft = tl.where(mask & (j_idx == 1), GAPOE + BETA * (i_idx - 1), Hleft)
            Hup   = tl.where(mask & (i_idx == 1), GAPOE + BETA * (j_idx - 1), Hup)
            Hdiag = tl.where(mask & (i_idx == 1) & (j_idx > 1), GAPOE + BETA * (j_idx - 2), Hdiag)
            Hdiag = tl.where(mask & (j_idx == 1) & (i_idx > 1), GAPOE + BETA * (i_idx - 2), Hdiag)

            Eleft = tl.load(E_base + slot_e, mask=mask & valid_prev, other=MINF)
            Fup   = tl.load(F_base + slot_f, mask=mask & valid_up,   other=MINF)
            Ecur = tl.maximum(Eleft + BETA, Hleft + GAPOE)
            Fcur = tl.maximum(Fup   + BETA, Hup   + GAPOE)
            Hcur = tl.maximum(Hdiag + s_val, tl.maximum(Ecur, Fcur))

            tl.store(Hcurr32 + slot_H_curr, Hcur, mask=mask)
            tl.store(E_base + slot_e, Ecur, mask=mask)
            tl.store(F_base + slot_f, Fcur, mask=mask)

            is_new_max = Hcur > diag_max_s_vec
            diag_max_s_vec = tl.where(mask & is_new_max, Hcur, diag_max_s_vec)
            diag_max_i_vec = tl.where(mask & is_new_max, i_idx, diag_max_i_vec)
            
            off += BLOCK
        
        # --- END-OF-DIAGONAL REDUCTION ---
        final_mask = tl.arange(0, BLOCK) < L
        masked_s_vec = tl.where(final_mask, diag_max_s_vec, MINF)
        diag_max_s = tl.max(masked_s_vec, axis=0)
        is_max_thread_mask = (masked_s_vec == diag_max_s) & final_mask
        diag_max_i = tl.max(tl.where(is_max_thread_mask, diag_max_i_vec, 0), axis=0)
        diag_max_j = d - diag_max_i
        
        if diag_max_s > best_s:
            best_s, best_i, best_j = diag_max_s, diag_max_i, diag_max_j

        BETA_POS = tl.abs(BETA)
        ZTH = tl.full((), drop_threshold, tl.int32)
        diff_vec = (diag_max_i - diag_max_j) - (best_i - best_j)
        br_mask = (diag_max_i >= best_i) & (diag_max_j >= best_j)
        too_low = (best_s - diag_max_s) > ZTH + BETA_POS * tl.abs(diff_vec)
        if too_low and br_mask: stop = 1
        
        prev2_lo, prev_lo = prev_lo, band_lo
        prev2_L, prev_L = prev_L, L
        d += 1

    # --- FINALIZATION ---
    tl.store(out + 0, best_s)
    tl.store(out + 1, best_i)
    tl.store(out + 2, best_j)