import triton
import triton.language as tl

"""
2025-08-25

ksw2_extz2_sse / AGAThA-like Triton-kernel

OPv7: H uses 2 slots, E/F each use only 1 slot
"""

@triton.jit
def sw_kernel(
    # Position arguments
    q_ptrs, r_ptrs,
    m_arr, n_arr,
    outs,

    # ===== 2025-07-18: a new buffer setup =======
    Hbuf, Fbuf, Ebuf,
    # ============================================

    # Scoring parameters
    match_score, mismatch_score,
    gap_open_penalty, gap_extend_penalty,
    drop_threshold,

    # Constexprs
    SCORING_MODEL: tl.constexpr,
    PRUNING_BAND: tl.constexpr,
    PRUNING_DROP: tl.constexpr,
    IS_EXTENSION: tl.constexpr,
    STRIDE: tl.constexpr,
    BAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # ───── sequence metadata ─────────────────────────────────────
    m = tl.load(m_arr + pid)
    n = tl.load(n_arr + pid)
    q_ptr = tl.load(q_ptrs + pid).to(tl.pointer_type(tl.int32))
    r_ptr = tl.load(r_ptrs + pid).to(tl.pointer_type(tl.int32))

    # Use int32 storage for H/E/F
    H32 = Hbuf.to(tl.pointer_type(tl.int32))
    E32 = Ebuf.to(tl.pointer_type(tl.int32))
    F32 = Fbuf.to(tl.pointer_type(tl.int32))
    out = outs + pid * 3

    # ───── constants in registers ────────────────────────────────
    MATCH = tl.full((), match_score, tl.int32)
    MISM = tl.full((), mismatch_score, tl.int32)
    ALPHA = tl.full((), gap_open_penalty, tl.int32)
    BETA = tl.full((), gap_extend_penalty, tl.int32)
    GAPOE = ALPHA + BETA
    MINF = tl.full((), -10_000_000, tl.int32)
    NPEN = tl.full((), 1, tl.int32)
    N_VAL = tl.full((), 14, tl.int32)

    # --- Pruning Bookkeeping (Correct Initialization) ---
    if PRUNING_DROP != 'NONE':
        init_score = tl.where(IS_EXTENSION, MINF, 0)
        best_s = tl.full((), init_score, tl.int32)
        best_i, best_j = tl.zeros((), tl.int32), tl.zeros((), tl.int32)

    # ───── main loop setup ────────────────────────────────
    d = tl.full((), 2, tl.int32)
    prev_lo = tl.full((), 1, tl.int32)
    prev2_lo = tl.full((), 0, tl.int32)  # d=0 has no elements
    stop = tl.zeros((), tl.int32)
    prev_L = tl.full((), 1, tl.int32)    # d=1 has 1 element
    prev2_L = tl.full((), 0, tl.int32)   # d=0 has 0 elements

    # Initialize d=1: H[1,1] = 0
    row_off_h32_init = pid * (2 * STRIDE)
    H_slot1_init = H32 + row_off_h32_init + STRIDE  # slot 1 for odd d
    init_slot_idx = STRIDE >> 1  # Center position for diagonal offset 0
    tl.store(H_slot1_init + init_slot_idx, 0)

    # --- Unified slot addressing state (E/F rotation accumulators) ---
    R_E = tl.full((), 0, tl.int32)
    R_F = tl.full((), 0, tl.int32)

    # ───────────── main anti-diagonal sweep ──────────────────────
    while (d <= m + n) & (stop == 0):

        # H: 2-SLOT OPTIMIZATION
        row_off_h32 = pid * (2 * STRIDE)

        d_mod2 = d & 1
        if d_mod2 == 0:
            # Even d: curr and prev2 use slot 0, prev uses slot 1
            curr_slot_h32 = 0
            prev_slot_h32 = STRIDE
            prev2_slot_h32 = 0
        else:
            # Odd d: curr and prev2 use slot 1, prev uses slot 0
            curr_slot_h32 = STRIDE
            prev_slot_h32 = 0
            prev2_slot_h32 = STRIDE

        Hcurr32 = H32 + row_off_h32 + curr_slot_h32
        Hprev32 = H32 + row_off_h32 + prev_slot_h32
        Hprev2_32 = H32 + row_off_h32 + prev2_slot_h32

        # Update rotation accumulators once per diagonal
        R_E = R_E + d_mod2
        R_E = tl.where(R_E >= STRIDE, R_E - STRIDE, R_E)
        R_F = R_F + d_mod2 - 1
        R_F = tl.where(R_F < 0, R_F + STRIDE, R_F)

        # E/F: 1-SLOT OPTIMIZATION
        if SCORING_MODEL == 'AFFINE':
            row_off_e32 = pid * STRIDE
            row_off_f32 = pid * STRIDE
            Ecurr32 = E32 + row_off_e32
            Fcurr32 = F32 + row_off_f32
            Eprev32, Fprev32 = Ecurr32, Fcurr32

        i_min = tl.maximum(1, d - n)
        i_max = tl.minimum(m, d - 1)

        # Static centered band
        half_lo = (d - BAND + 2) >> 1
        half_hi = (d + BAND - 1) >> 1
        band_lo = tl.maximum(i_min, half_lo)
        band_hi = tl.minimum(i_max, half_hi)
        L = band_hi - band_lo + 1

        # Track max score per thread for this diagonal
        diag_max_s_vec = tl.full((BLOCK,), MINF, tl.int32)
        diag_max_i_vec = tl.zeros((BLOCK,), tl.int32)

        off = tl.zeros((), tl.int32)

        while off < L:
            tid = tl.arange(0, BLOCK)
            mask = tid < (L - off)
            i_idx = band_lo + off + tid
            j_idx = d - i_idx

            # Load sequence data
            q_word = tl.load(q_ptr + (i_idx - 1) // 8, mask=mask, other=0)
            r_word = tl.load(r_ptr + (j_idx - 1) // 8, mask=mask, other=0)
            q_code = (q_word >> (((i_idx - 1) % 8) * 4)) & 0xF
            r_code = (r_word >> (((j_idx - 1) % 8) * 4)) & 0xF
            s_val = tl.where(
                (q_code == N_VAL) | (r_code == N_VAL), -NPEN,
                tl.where(q_code == r_code, MATCH, MISM)
            )

            # H indexing
            p = d_mod2
            s = i_idx - j_idx
            t_idx = (s - p) >> 1
            center = STRIDE >> 1
            slot_idx = center + t_idx
            slot_prev_left = center + (t_idx + p)
            slot_prev_up = center + (t_idx + p - 1)

            in_cur = (slot_idx >= 0) & (slot_idx < STRIDE)
            in_left = (slot_prev_left >= 0) & (slot_prev_left < STRIDE)
            in_up = (slot_prev_up >= 0) & (slot_prev_up < STRIDE)

            # Validity checks
            valid_prev = (i_idx >= prev_lo) & (i_idx <= prev_lo + prev_L - 1)
            valid_up = ((i_idx - 1) >= prev_lo) & ((i_idx - 1) <= prev_lo + prev_L - 1)
            valid_diag = (prev2_L > 0) & ((i_idx - 1) >= prev2_lo) & ((i_idx - 1) <= prev2_lo + prev2_L - 1)

            Hleft = tl.load(Hprev32 + slot_prev_left, mask=mask & valid_prev & in_left, other=MINF)
            Hup = tl.load(Hprev32 + slot_prev_up, mask=mask & valid_up & in_up, other=MINF)
            Hdiag = tl.load(Hprev2_32 + slot_idx, mask=mask & valid_diag & in_cur, other=MINF)

            Hdiag = tl.where(mask & (i_idx == 1) & (j_idx == 1), 0, Hdiag)

            if SCORING_MODEL == 'AFFINE':
                Hleft = tl.where(mask & (j_idx == 1), GAPOE + BETA * (i_idx - 1), Hleft)
                Hup = tl.where(mask & (i_idx == 1), GAPOE + BETA * (j_idx - 1), Hup)
                Hdiag = tl.where(mask & (i_idx == 1) & (j_idx > 1), GAPOE + BETA * (j_idx - 2), Hdiag)
                Hdiag = tl.where(mask & (j_idx == 1) & (i_idx > 1), GAPOE + BETA * (i_idx - 2), Hdiag)

                # E/F addressing
                slot_e = (center + t_idx + R_E) % STRIDE
                slot_f = (center + t_idx + R_F) % STRIDE
                Eleft = tl.load(Eprev32 + slot_e, mask=mask & valid_prev, other=MINF)
                Fup = tl.load(Fprev32 + slot_f, mask=mask & valid_up, other=MINF)

                Ecur = tl.maximum(Eleft + BETA, Hleft + GAPOE)
                Fcur = tl.maximum(Fup + BETA, Hup + GAPOE)
                Hcur = tl.maximum(Hdiag + s_val, tl.maximum(Ecur, Fcur))

                tl.store(Ecurr32 + slot_e, Ecur, mask=mask)
                tl.store(Fcurr32 + slot_f, Fcur, mask=mask)

            elif SCORING_MODEL == 'LINEAR':
                Hleft = tl.where(mask & (j_idx == 1), BETA * i_idx, Hleft)
                Hup = tl.where(mask & (i_idx == 1), BETA * j_idx, Hup)
                Hdiag = tl.where(mask & (i_idx == 1) & (j_idx > 1), BETA * (j_idx - 1), Hdiag)
                Hdiag = tl.where(mask & (j_idx == 1) & (i_idx > 1), BETA * (i_idx - 1), Hdiag)

                score_left = Hleft + BETA
                score_up = Hup + BETA
                Hcur = tl.maximum(Hdiag + s_val, tl.maximum(score_left, score_up))

            if not IS_EXTENSION:
                Hcur = tl.maximum(0, Hcur)

            tl.store(Hcurr32 + slot_idx, Hcur, mask=mask & in_cur)

            if PRUNING_DROP != 'NONE':
                is_new_max = Hcur > diag_max_s_vec
                diag_max_s_vec = tl.where(mask & is_new_max, Hcur, diag_max_s_vec)
                diag_max_i_vec = tl.where(mask & is_new_max, i_idx, diag_max_i_vec)

            off += BLOCK

        # Block-wide reduction
        diag_max_s = MINF
        diag_max_i = 0
        diag_max_j = 0

        if PRUNING_DROP != 'NONE':
            final_mask = tl.arange(0, BLOCK) < L
            masked_s_vec = tl.where(final_mask, diag_max_s_vec, MINF)
            diag_max_s = tl.max(masked_s_vec, axis=0)
            is_max_thread_mask = (masked_s_vec == diag_max_s) & final_mask
            diag_max_i = tl.max(tl.where(is_max_thread_mask, diag_max_i_vec, 0), axis=0)
            diag_max_j = d - diag_max_i

            if diag_max_s > best_s:
                best_s = diag_max_s
                best_i = diag_max_i
                best_j = diag_max_j

            # Z-drop termination
            BETA_POS = tl.abs(BETA)
            ZTH = tl.full((), drop_threshold, tl.int32)
            diff_vec = (diag_max_i - diag_max_j) - (best_i - best_j)
            br_mask = (diag_max_i >= best_i) & (diag_max_j >= best_j)
            too_low = (best_s - diag_max_s) > ZTH + BETA_POS * tl.abs(diff_vec)

            if too_low and br_mask:
                stop = 1

        # Update tracking
        prev2_lo, prev_lo = prev_lo, band_lo
        prev2_L, prev_L = prev_L, L
        d += 1

    # Store results
    if PRUNING_DROP != 'NONE':
        tl.store(out + 0, best_s)
        tl.store(out + 1, best_i)
        tl.store(out + 2, best_j)