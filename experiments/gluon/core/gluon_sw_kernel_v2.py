# filename: local_dp_kernel_gluon_v2_swizzled.py

import triton
# 关键：导入 gluon 和 gluon language
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# -----------------------------------------------------------------------------
# [Dylan's Kernel using Gluon - Attempt 2: Swizzled Layout]
# Goal: Use gl.SwizzledSharedLayout hoping it can handle [3, STRIDE] shape
#       where SharedLinearLayout failed. Rely on descriptor indexing (.index())
#       and load/store (.load([idx])) to handle the abstracted mapping.
# -----------------------------------------------------------------------------

@gluon.jit
def sw_kernel_gluon(
    # Position arguments (Global memory tensors/pointers)
    q_ptrs, r_ptrs, # Base pointers for this PID's sequences
    m_arr, n_arr,   # Lengths for this PID's sequences
    outs,           # Output tensor for this PID

    # Scoring parameters
    match_score: gl.int32, mismatch_score: gl.int32,
    gap_open_penalty: gl.int32, gap_extend_penalty: gl.int32,
    drop_threshold: gl.int32,

    # Constexprs
    SCORING_MODEL: gl.constexpr,
    PRUNING_BAND: gl.constexpr,
    PRUNING_DROP: gl.constexpr,
    IS_EXTENSION: gl.constexpr,
    STRIDE: gl.constexpr, # Physical stride (hopefully usable by Swizzled)
    BAND: gl.constexpr,   # Logical band width
    BLOCK: gl.constexpr,  # Vector processing size (e.g., 256)
):
    pid = gl.program_id(0)

    # ───── sequence metadata ─────────────────────────────────────
    m = gl.load(m_arr + pid)
    n = gl.load(n_arr + pid)
    q_ptr = q_ptrs
    r_ptr = r_ptrs
    out = outs + pid * 3

    # --- Gluon: Shared Memory Allocation using Swizzled Layout ---
    # Define a Swizzled layout. Parameters need tuning for performance,
    # but let's start with something plausible.
    # order=[1, 0] assumes STRIDE (dim 1) is contiguous / faster changing in layout.
    # vec=4 implies 4*int32 = 16 bytes access, common for performance.
    # per_phase/max_phase control the swizzling pattern complexity.
    # We choose simple values first to check if allocation works.
    swizzled_layout = gl.SwizzledSharedLayout(vec=4, per_phase=1, max_phase=4, order=[1, 0])

    # Allocate H, E, F buffers in shared memory using the Swizzled layout.
    # CRITICAL TEST: Does SwizzledLayout allow shape [3, STRIDE]?
    # Note: Swizzled layout might internally require padding or have constraints.
    # We still provide the logical shape we need.

    # NOTE: 问题1，最后两个维度（单个矩阵的行、列），必须是power of 2，shape must have power-of-2 and non-zero dimensions
    H_smem = gl.allocate_shared_memory(gl.int32, [3, STRIDE], layout=swizzled_layout)
    E_smem = gl.allocate_shared_memory(gl.int32, [2, STRIDE], layout=swizzled_layout)
    F_smem = gl.allocate_shared_memory(gl.int32, [2, STRIDE], layout=swizzled_layout)
    # -------------------------------------------------------------

    # ───── constants in registers ────────────────────────────────
    MATCH = match_score
    MISM = mismatch_score
    ALPHA = gap_open_penalty
    BETA = gap_extend_penalty
    GAPOE = ALPHA + BETA
    MINF = -10_000_000
    NPEN = 1
    N_VAL = 14

    # --- Pruning Bookkeeping ---
    if PRUNING_DROP != 'NONE':
        init_score = MINF if IS_EXTENSION else 0
        best_s = init_score
        best_i, best_j = 0, 0

    # ───── main loop setup ────────────────────────────────
    d = 2
    prev_lo = 1
    prev2_lo = 0
    stop = 0
    # prev_L / prev2_L still track logical length based on physical STRIDE
    prev_L = STRIDE
    prev2_L = STRIDE

    # ───────────── main anti-diagonal sweep ──────────────────────
    while (d <= m + n) & (stop == 0):

        # --- Gluon: Get 1D Row Descriptors (Logic remains the same) ---
        Hcurr_desc  = H_smem.index(d % 3)
        Hprev_desc  = H_smem.index((d - 1) % 3)
        Hprev2_desc = H_smem.index((d - 2) % 3)

        if SCORING_MODEL == 'AFFINE':
            Ecurr_desc = E_smem.index(d % 2)
            Eprev_desc = E_smem.index((d - 1) % 2)
            Fcurr_desc = F_smem.index(d % 2)
            Fprev_desc = F_smem.index((d - 1) % 2)
        # -------------------------------------------------------------

        # Calculate band boundaries (Logic remains the same)
        i_min = gl.maximum(1, d - n)
        i_max = gl.minimum(m, d - 1)
        r_off = 0
        half_lo = (d + r_off - BAND + 2) >> 1
        half_hi = (d + r_off + BAND - 1) >> 1
        band_lo = gl.maximum(i_min, half_lo)
        band_hi = gl.minimum(i_max, half_hi)
        L = band_hi - band_lo + 1

        # Per-thread max tracking vectors (Logic remains the same)
        diag_max_s_vec = gl.full((BLOCK,), MINF, gl.int32)
        diag_max_i_vec = gl.zeros((BLOCK,), gl.int32)

        # Inner loop iterates over the band
        off = 0
        while off < L:
            tid = gl.arange(0, BLOCK)
            mask = tid < (L - off)

            # Calculate cell indices (Logic remains the same)
            i_idx = band_lo + off + tid
            j_idx = d - i_idx

            # Load Q and R codes (Global Memory - Logic remains the same)
            q_idx = (i_idx - 1) // 8
            q_shift = ((i_idx - 1) % 8) * 4
            q_mask = mask & (i_idx >= 1)
            q_word = gl.load(q_ptr + q_idx, mask=q_mask, other=0)
            q_code = (q_word >> q_shift) & 0xF

            r_idx = (j_idx - 1) // 8
            r_shift = ((j_idx - 1) % 8) * 4
            r_mask = mask & (j_idx >= 1)
            r_word = gl.load(r_ptr + r_idx, mask=r_mask, other=0)
            r_code = (r_word >> r_shift) & 0xF

            # Calculate score (Logic remains the same)
            s_val = gl.where((q_code == N_VAL) | (r_code == N_VAL), -NPEN, gl.where(q_code == r_code, MATCH, MISM))

            # Calculate indices for accessing previous rows (Logic remains the same)
            idx_prev = i_idx - prev_lo
            idx_prev2 = i_idx - 1 - prev2_lo
            idx_curr = i_idx - band_lo

            # --- Gluon: Load from Shared Memory Descriptors ---
            # Use the 1D row descriptors. The .load() method will handle
            # the address calculation based on the Swizzled layout.
            valid_prev = (idx_prev >= 0) & (idx_prev < prev_L)
            valid_up = (idx_prev > 0) & ((idx_prev - 1) < prev_L)
            valid_diag = (idx_prev2 >= 0) & (idx_prev2 < prev2_L)

            # NOTE: 问题2: shared_memory_descriptor.load() got an unexpected keyword argument 'mask'
            Hleft = Hprev_desc.load([idx_prev], mask=mask & valid_prev, other=MINF)
            Hup   = Hprev_desc.load([idx_prev - 1], mask=mask & valid_up, other=MINF)
            Hdiag = Hprev2_desc.load([idx_prev2], mask=mask & valid_diag, other=MINF)
            Hdiag = gl.where(mask & (i_idx == 1) & (j_idx == 1), 0, Hdiag)
            # ----------------------------------------------------

            # Boundary condition logic (remains the same)
            if SCORING_MODEL == 'AFFINE':
                Hleft = gl.where(mask & (j_idx == 1), GAPOE + BETA * (i_idx - 1), Hleft)
                Hup = gl.where(mask & (i_idx == 1), GAPOE + BETA * (j_idx - 1), Hup)
                Hdiag = gl.where(mask & (i_idx == 1) & (j_idx > 1), GAPOE + BETA * (j_idx - 2), Hdiag)
                Hdiag = gl.where(mask & (j_idx == 1) & (i_idx > 1), GAPOE + BETA * (i_idx - 2), Hdiag)
            elif SCORING_MODEL == 'LINEAR':
                Hleft = gl.where(mask & (j_idx == 1), BETA * i_idx, Hleft)
                Hup = gl.where(mask & (i_idx == 1), BETA * j_idx, Hup)
                Hdiag = gl.where(mask & (i_idx == 1) & (j_idx > 1), BETA * (j_idx - 1), Hdiag)
                Hdiag = gl.where(mask & (j_idx == 1) & (i_idx > 1), BETA * (i_idx - 1), Hdiag)

            # Calculate Hcur, Ecur, Fcur (Logic remains the same)
            if SCORING_MODEL == 'AFFINE':
                Eleft = Eprev_desc.load([idx_prev], mask=mask & valid_prev, other=MINF)
                Fup   = Fprev_desc.load([idx_prev - 1], mask=mask & valid_up, other=MINF)
                Ecur = gl.maximum(Eleft + BETA, Hleft + GAPOE)
                Fcur = gl.maximum(Fup + BETA, Hup + GAPOE)
                Hcur = gl.maximum(Hdiag + s_val, gl.maximum(Ecur, Fcur))

                # --- Gluon: Store E/F to Shared Memory ---
                Ecurr_desc.store([idx_curr], Ecur, mask=mask)
                Fcurr_desc.store([idx_curr], Fcur, mask=mask)
                # ----------------------------------------
            elif SCORING_MODEL == 'LINEAR':
                score_left = Hleft + BETA
                score_up = Hup + BETA
                Hcur = gl.maximum(Hdiag + s_val, gl.maximum(score_left, score_up))

            # Apply floor if not extension (Logic remains the same)
            if not IS_EXTENSION:
                Hcur = gl.maximum(0, Hcur)

            # --- Gluon: Store Hcur to Shared Memory ---
            Hcurr_desc.store([idx_curr], Hcur, mask=mask)
            # -----------------------------------------

            # Update per-thread max (Logic remains the same)
            if PRUNING_DROP != 'NONE':
                is_new_max = Hcur > diag_max_s_vec
                diag_max_s_vec = gl.where(mask & is_new_max, Hcur, diag_max_s_vec)
                diag_max_i_vec = gl.where(mask & is_new_max, i_idx, diag_max_i_vec)

            off += BLOCK

        # Final Reduction (Logic remains the same)
        diag_max_s = MINF
        diag_max_i = 0
        diag_max_j = 0
        if PRUNING_DROP != 'NONE':
            final_mask = gl.arange(0, BLOCK) < L
            masked_s_vec = gl.where(final_mask, diag_max_s_vec, MINF)
            diag_max_s = gl.max(masked_s_vec, axis=0)
            is_max_thread_mask = (masked_s_vec == diag_max_s) & final_mask
            diag_max_i = gl.max(gl.where(is_max_thread_mask, diag_max_i_vec, 0), axis=0)
            diag_max_j = d - diag_max_i

        # update global max (Logic remains the same)
        if PRUNING_DROP != 'NONE':
             if diag_max_s > best_s:
                 best_s = diag_max_s
                 best_i = diag_max_i
                 best_j = diag_max_j

        # Z-drop check (Logic remains the same)
        if PRUNING_DROP != 'NONE':
            BETA_POS = gl.abs(BETA)
            ZTH = drop_threshold
            diff_vec = (diag_max_i - diag_max_j) - (best_i - best_j)
            br_mask = (diag_max_i >= best_i) & (diag_max_j >= best_j)
            too_low = (best_s - diag_max_s) > ZTH + BETA_POS * gl.abs(diff_vec)
            if too_low and br_mask:
                stop = 1

        # Update loop variables (Logic remains the same)
        prev2_lo, prev_lo = prev_lo, band_lo
        prev2_L, prev_L = prev_L, L
        d += 1

    # --- FINALIZATION LOGIC (Store results - Logic remains the same) ---
    if PRUNING_DROP != 'NONE':
        gl.store(out + 0, best_s)
        gl.store(out + 1, best_i)
        gl.store(out + 2, best_j)
    else:
        pass