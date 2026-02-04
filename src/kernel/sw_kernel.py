# prototype/core/local_dp_kernel_OPv6.py

import triton
import triton.language as tl

"""
2025-08-21

 ksw2_extz2_sse/AGAThA-like Triton-kernel
 OPv4: Use int16 storage for H/E/F while keeping all math in int32
 OPv5:
    1. Moving the reduction operation outside the while off < L
       In the inner loop, each thread independently tracks the highest score among the cells it has processed. This process requires no inter-thread communication and is very lightweight.
       After the entire backslash (all iterations of off) are calculated, a single block-wide reduction is performed to find the "global highest score" from the "local highest scores" of all threads
    2. Delete 'DYNAMIC' banding option, only keep 'STATIC' banding
       Thus, the H/E/F buffer size can be simplified to `band_width`, not `2*band_width+1` anymore -> important for hacking in smem
 OPv6: int32 storage for H/E/F
"""

@triton.jit
def sw_kernel(
    # Position arguments
    q_ptrs, r_ptrs,
    m_arr, n_arr,
    outs,
    # ===== 2025-07-18: a new buffer setup =======
    Hbuf, Fbuf, Ebuf,
    # ====================

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
    # Use int32 storage for H/E/F (eliminate bank conflicts from 16-bit halves)
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
    # Track best score only if we need drop heuristics
    if PRUNING_DROP != 'NONE':
        init_score = tl.where(IS_EXTENSION, MINF, 0)
        best_s = tl.full((), init_score, tl.int32)
        best_i, best_j = tl.zeros((), tl.int32), tl.zeros((), tl.int32)

    # ───── main loop setup ────────────────────────────────
    d = tl.full((), 2, tl.int32)
    prev_lo = tl.full((), 1, tl.int32)
    prev2_lo = tl.full((), 0, tl.int32)
    stop = tl.zeros((), tl.int32)
    # NOTE: STRIDE is the physical padded stride (>= BAND), rounded to 128B for coalescing; BAND is the logical band width.
    prev_L = tl.full((), STRIDE, tl.int32) 
    prev2_L = tl.full((), STRIDE, tl.int32)

    # ───────────── main anti-diagonal sweep ──────────────────────
    while (d <= m + n) & (stop == 0):

        row_off_h32 = pid * (3 * STRIDE)
        curr_slot_h32  = (d % 3) * STRIDE
        prev_slot_h32  = ((d - 1) % 3) * STRIDE
        prev2_slot_h32 = ((d - 2) % 3) * STRIDE

        Hcurr32 = H32 + row_off_h32 + curr_slot_h32
        Hprev32 = H32 + row_off_h32 + prev_slot_h32
        Hprev2_32 = H32 + row_off_h32 + prev2_slot_h32
        
        if SCORING_MODEL == 'AFFINE':
            row_off_ef32 = pid * (2 * STRIDE)
            curr_slot_ef32 = (d % 2) * STRIDE
            prev_slot_ef32 = ((d - 1) % 2) * STRIDE
            Ecurr32  = E32 + row_off_ef32 + curr_slot_ef32
            Eprev32  = E32 + row_off_ef32 + prev_slot_ef32
            Fcurr32  = F32 + row_off_ef32 + curr_slot_ef32
            Fprev32  = F32 + row_off_ef32 + prev_slot_ef32

        i_min = tl.maximum(1, d - n)
        i_max = tl.minimum(m, d - 1)

        # Static centered band: always center around main diagonal (r_off = 0)
        r_off = 0
        # Use ceil for lower bound and floor for upper bound so L ∈ {B, B-1}
        # half_lo = ceil((d + r_off - BAND + 1)/2) = (d + r_off - BAND + 2) >> 1
        # half_hi = floor((d + r_off + BAND - 1)/2) = (d + r_off + BAND - 1) >> 1
        half_lo = (d + r_off - BAND + 2) >> 1
        half_hi = (d + r_off + BAND - 1) >> 1
        band_lo = tl.maximum(i_min, half_lo)
        band_hi = tl.minimum(i_max, half_hi)

        L = band_hi - band_lo + 1

        # --- MODIFIED ---
        # Instead of scalar variables for the diagonal's max,
        # we now use vectors to track the max score PER THREAD.
        diag_max_s_vec = tl.full((BLOCK,), MINF, tl.int32)
        diag_max_i_vec = tl.zeros((BLOCK,), tl.int32)

        off = tl.zeros((), tl.int32)

        while off < L:
            tid = tl.arange(0, BLOCK)
            mask = tid < (L - off)
            i_idx = band_lo + off + tid
            j_idx = d - i_idx

            q_word = tl.load(q_ptr + (i_idx - 1) // 8, mask=mask, other=0)
            r_word = tl.load(r_ptr + (j_idx - 1) // 8, mask=mask, other=0)
            q_code = (q_word >> (((i_idx - 1) % 8) * 4)) & 0xF
            r_code = (r_word >> (((j_idx - 1) % 8) * 4)) & 0xF
            s_val = tl.where((q_code == N_VAL) | (r_code == N_VAL), -NPEN, tl.where(q_code == r_code, MATCH, MISM))

            idx_prev = i_idx - prev_lo
            valid_prev = (idx_prev >= 0) & (idx_prev < prev_L) 
            idx_prev2 = i_idx - 1 - prev2_lo
            valid_up = (idx_prev > 0) & ((idx_prev - 1) < prev_L)
            valid_diag = (idx_prev2 >= 0) & (idx_prev2 < prev2_L)
            Hleft = tl.load(Hprev32 + idx_prev, mask=mask & valid_prev, other=MINF)
            Hup   = tl.load(Hprev32 + (idx_prev - 1), mask=mask & valid_up, other=MINF)
            Hdiag = tl.load(Hprev2_32 + idx_prev2, mask=mask & valid_diag, other=MINF)
            Hdiag = tl.where(mask & (i_idx == 1) & (j_idx == 1), 0, Hdiag)
            
            # Boundary condition logic
            if SCORING_MODEL == 'AFFINE':
                Hleft = tl.where(mask & (j_idx == 1), GAPOE + BETA * (i_idx - 1), Hleft)
                Hup = tl.where(mask & (i_idx == 1), GAPOE + BETA * (j_idx - 1), Hup)
                Hdiag = tl.where(mask & (i_idx == 1) & (j_idx > 1), GAPOE + BETA * (j_idx - 2), Hdiag)
                Hdiag = tl.where(mask & (j_idx == 1) & (i_idx > 1), GAPOE + BETA * (i_idx - 2), Hdiag)
            elif SCORING_MODEL == 'LINEAR':
                Hleft = tl.where(mask & (j_idx == 1), BETA * i_idx, Hleft)
                Hup = tl.where(mask & (i_idx == 1), BETA * j_idx, Hup)
                Hdiag = tl.where(mask & (i_idx == 1) & (j_idx > 1), BETA * (j_idx - 1), Hdiag)
                Hdiag = tl.where(mask & (j_idx == 1) & (i_idx > 1), BETA * (i_idx - 1), Hdiag)

            if SCORING_MODEL == 'AFFINE':
                Eleft = tl.load(Eprev32 + idx_prev, mask=mask & valid_prev, other=MINF)
                Fup   = tl.load(Fprev32 + (idx_prev - 1), mask=mask & valid_up, other=MINF)
                Ecur = tl.maximum(Eleft + BETA, Hleft + GAPOE)
                Fcur = tl.maximum(Fup + BETA, Hup + GAPOE)
                Hcur = tl.maximum(Hdiag + s_val, tl.maximum(Ecur, Fcur))
                idx_curr = i_idx - band_lo
                tl.store(Ecurr32 + idx_curr, Ecur, mask=mask)
                tl.store(Fcurr32 + idx_curr, Fcur, mask=mask)
            elif SCORING_MODEL == 'LINEAR':
                score_left = Hleft + BETA
                score_up = Hup + BETA
                Hcur = tl.maximum(Hdiag + s_val, tl.maximum(score_left, score_up))

            if not IS_EXTENSION:
                Hcur = tl.maximum(0, Hcur)

            idx_curr = i_idx - band_lo
            tl.store(Hcurr32 + idx_curr, Hcur, mask=mask)

            # --- MODIFIED ---
            # The expensive block-wide reduction is replaced by a lightweight, per-thread update.
            # No communication or synchronization is needed here.
            if PRUNING_DROP != 'NONE':
                is_new_max = Hcur > diag_max_s_vec
                diag_max_s_vec = tl.where(mask & is_new_max, Hcur, diag_max_s_vec)
                diag_max_i_vec = tl.where(mask & is_new_max, i_idx, diag_max_i_vec)
            
            # --- REMOVED ---
            # The old logic with tl.max and tl.argmax inside this loop is removed.
            
            # (Z-drop logic per element was commented out, remains the same)

            off += BLOCK
        
        # --- NEW ---
        # A single block-wide reduction is now performed AFTER the inner loop has finished.
        diag_max_s = MINF
        diag_max_i = 0
        diag_max_j = 0
        if PRUNING_DROP != 'NONE':
            # 1. Mask out threads that were not part of any valid computation.
            #    A thread is valid if its initial tid was less than L.
            final_mask = tl.arange(0, BLOCK) < L
            masked_s_vec = tl.where(final_mask, diag_max_s_vec, MINF)
            
            # 2. Find the maximum score across the block.
            diag_max_s = tl.max(masked_s_vec, axis=0)
            
            # 3. Find the i-coordinate corresponding to that max score.
            #    Create a mask to identify which thread(s) hold the maximum value.
            is_max_thread_mask = (masked_s_vec == diag_max_s) & final_mask
            
            # 4. Extract the i_idx from the thread(s) holding the max.
            #    Using tl.max is a deterministic way to select one 'i' if multiple threads hold the max score.
            diag_max_i = tl.max(tl.where(is_max_thread_mask, diag_max_i_vec, 0), axis=0)
            diag_max_j = d - diag_max_i
            
        # update global max
        if diag_max_s > best_s:
            best_s = diag_max_s
            best_i = diag_max_i
            best_j = diag_max_j

        # doing zdrop terminate after each anti-diag
        if PRUNING_DROP != 'NONE':
            BETA_POS = tl.abs(BETA)
            ZTH = tl.full((), drop_threshold, tl.int32)
            
            diff_vec = (diag_max_i - diag_max_j) - (best_i - best_j)
            br_mask = (diag_max_i >= best_i) & (diag_max_j >= best_j)
            too_low = (best_s - diag_max_s) > ZTH + BETA_POS * tl.abs(diff_vec)
            
            if too_low and br_mask:
                stop = 1
        
        prev2_lo, prev_lo = prev_lo, band_lo
        prev2_L, prev_L = prev_L, L
        d += 1

    # --- FINALIZATION LOGIC (match OPv5 behavior) ---
    if PRUNING_DROP != 'NONE':
        tl.store(out + 0, best_s)
        tl.store(out + 1, best_i)
        tl.store(out + 2, best_j)
    else:
        pass
