# filename: local_dp_kernel_gluon_v1.py

import triton
# 关键：导入 gluon 和 gluon language
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# -----------------------------------------------------------------------------
# [Dylan's Kernel using Gluon - Attempt 1]
# Goal: Use gl.allocate_shared_memory with explicit SharedLinearLayout
#       to allocate H/E/F buffers with shapes [3, STRIDE] / [2, STRIDE].
#       Replace pointer arithmetic with shared memory descriptor methods.
# -----------------------------------------------------------------------------

@gluon.jit
def sw_kernel_gluon(
    # Position arguments (Global memory tensors/pointers)
    # NOTE: Gluon automatically converts tensors to pointers based on context.
    # We still need the base pointers for Q and R sequences for this program ID.
    q_ptrs, r_ptrs, # These should now be the base pointers for this PID's sequences
    m_arr, n_arr,   # Lengths for this PID's sequences
    outs,           # Output tensor for this PID

    # Scoring parameters (passed as standard Python types, inferred by JIT)
    match_score: gl.int32, mismatch_score: gl.int32,
    gap_open_penalty: gl.int32, gap_extend_penalty: gl.int32,
    drop_threshold: gl.int32,

    # layout_H_shared: gl.constexpr,
    # layout_EF_shared: gl.constexpr,

    # Constexprs (Compile-time constants)
    SCORING_MODEL: gl.constexpr,
    PRUNING_BAND: gl.constexpr,
    PRUNING_DROP: gl.constexpr,
    IS_EXTENSION: gl.constexpr,
    STRIDE: gl.constexpr, # Physical stride (power-of-2 or aligned)
    BAND: gl.constexpr,   # Logical band width
    BLOCK: gl.constexpr,  # Vector processing size (e.g., 256)
):
    # Get program ID (remains the same)
    pid = gl.program_id(0)

    # ───── sequence metadata (Load sequence lengths) ─────────────────
    m = gl.load(m_arr + pid)
    n = gl.load(n_arr + pid)
    # q_ptrs and r_ptrs are already the correct base pointers for this PID
    # No need to load them again like in the original Triton kernel
    q_ptr = q_ptrs # Direct use
    r_ptr = r_ptrs # Direct use
    out = outs + pid * 3 # Output offset remains the same

    # --- Gluon: Shared Memory Allocation ---
    # Define the explicit C-style row-major layout using SharedLinearLayout.
    # For a shape [R, C] with row-major order [1, 0], the offset bases
    # represent the stride contributions: [Stride_Dim0, Stride_Dim1].
    # Stride_Dim1 (Column) = 1
    # Stride_Dim0 (Row) = C = STRIDE
    # NOTE: We provide the fundamental strides. The actual shape is given
    #       to allocate_shared_memory separately.
    # NOTE: Alignment is crucial for performance, keep it reasonably high (e.g., 16 bytes = 4 * int32)
    # smem_alignment = 16
    # layout_H_shared = gl.SharedLinearLayout(offset_bases=[[STRIDE, 0], [0, 1]], alignment=smem_alignment)
    # layout_EF_shared = gl.SharedLinearLayout(offset_bases=[[STRIDE, 0], [0, 1]], alignment=smem_alignment)

    # Allocate H, E, F buffers in shared memory using the defined layout AND shape.
    # THIS IS THE CRITICAL TEST: Does Gluon allow non-power-of-2 shapes here?
    # H_smem = gl.allocate_shared_memory(gl.int32, [3, STRIDE], layout=layout_H_shared)
    # E_smem = gl.allocate_shared_memory(gl.int32, [2, STRIDE], layout=layout_EF_shared)
    # F_smem = gl.allocate_shared_memory(gl.int32, [2, STRIDE], layout=layout_EF_shared)

    H_smem = gl.allocate_shared_memory(
        gl.int32,
        [3, STRIDE],
        # Directly pass the constructor call as the layout argument
        layout=gl.SharedLinearLayout(
            offset_bases=[[STRIDE, 0], [0, 1]], alignment=16
        )
    )
    E_smem = gl.allocate_shared_memory(
        gl.int32,
        [2, STRIDE],
        layout=gl.SharedLinearLayout(
            offset_bases=[[STRIDE, 0], [0, 1]], alignment=16
        )
    )
    F_smem = gl.allocate_shared_memory(
        gl.int32,
        [2, STRIDE],
        layout=gl.SharedLinearLayout(
            offset_bases=[[STRIDE, 0], [0, 1]], alignment=16
        )
    )
    # ----------------------------------------

    # ───── constants in registers ────────────────────────────────
    MATCH = match_score
    MISM = mismatch_score
    ALPHA = gap_open_penalty
    BETA = gap_extend_penalty
    GAPOE = ALPHA + BETA
    MINF = -10_000_000
    NPEN = 1
    N_VAL = 14 # Assuming int32

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
    prev_L = STRIDE # Use physical stride for tracking previous row length validity
    prev2_L = STRIDE

    # ───────────── main anti-diagonal sweep ──────────────────────
    while (d <= m + n) & (stop == 0):

        # --- Gluon: Get 1D Row Descriptors ---
        # Use .index() on the 2D shared memory descriptor to get a 1D descriptor for the row
        Hcurr_desc  = H_smem.index(d % 3)
        Hprev_desc  = H_smem.index((d - 1) % 3)
        Hprev2_desc = H_smem.index((d - 2) % 3)

        if SCORING_MODEL == 'AFFINE':
            Ecurr_desc = E_smem.index(d % 2)
            Eprev_desc = E_smem.index((d - 1) % 2)
            Fcurr_desc = F_smem.index(d % 2)
            Fprev_desc = F_smem.index((d - 1) % 2)
        # -------------------------------------

        # Calculate band boundaries (logic remains the same)
        i_min = gl.maximum(1, d - n)
        i_max = gl.minimum(m, d - 1)
        r_off = 0
        half_lo = (d + r_off - BAND + 2) >> 1
        half_hi = (d + r_off + BAND - 1) >> 1
        band_lo = gl.maximum(i_min, half_lo)
        band_hi = gl.minimum(i_max, half_hi)
        L = band_hi - band_lo + 1 # Number of valid cells in this diagonal's band

        # Per-thread max tracking vectors (logic remains the same)
        diag_max_s_vec = gl.full((BLOCK,), MINF, gl.int32)
        diag_max_i_vec = gl.zeros((BLOCK,), gl.int32)

        # Inner loop iterates over the band in chunks of BLOCK
        off = 0
        while off < L:
            tid = gl.arange(0, BLOCK) # [0, 1, ..., BLOCK-1]
            mask = tid < (L - off)    # Mask for threads within the valid band length

            # Calculate cell indices (logic remains the same)
            i_idx = band_lo + off + tid # Vector of 'i' coordinates for this chunk
            j_idx = d - i_idx           # Vector of 'j' coordinates for this chunk

            # --- Load Q and R codes (Global Memory) ---
            # Using direct pointer arithmetic as before, assuming q_ptr/r_ptr are correct base pointers
            q_idx = (i_idx - 1) // 8
            q_shift = ((i_idx - 1) % 8) * 4
            q_mask = mask & (i_idx >= 1) # Ensure index is valid
            # NOTE: Global loads in Gluon might require specifying a layout if we want tiling/vectorization,
            # but for this direct translation, we assume scalar loads work like tl.load.
            # Check Gluon docs for explicit pointer types if needed.
            q_word = gl.load(q_ptr + q_idx, mask=q_mask, other=0) # Load 32-bit word
            q_code = (q_word >> q_shift) & 0xF                   # Extract 4-bit code

            r_idx = (j_idx - 1) // 8
            r_shift = ((j_idx - 1) % 8) * 4
            r_mask = mask & (j_idx >= 1)
            r_word = gl.load(r_ptr + r_idx, mask=r_mask, other=0)
            r_code = (r_word >> r_shift) & 0xF
            # ---------------------------------------------

            # Calculate score (logic remains the same)
            s_val = gl.where((q_code == N_VAL) | (r_code == N_VAL), -NPEN, gl.where(q_code == r_code, MATCH, MISM))

            # Calculate indices for accessing previous H/E/F rows relative to their start
            idx_prev = i_idx - prev_lo      # Index within Hprev/Eprev/Fprev rows
            idx_prev2 = i_idx - 1 - prev2_lo # Index within Hprev2 row
            idx_curr = i_idx - band_lo       # Index within Hcurr/Ecurr/Fcurr rows (relative to band_lo)

            # --- Gluon: Load from Shared Memory Descriptors ---
            # Use the 1D row descriptors obtained via .index()
            # Pass the index vector directly.
            valid_prev = (idx_prev >= 0) & (idx_prev < prev_L) # Check against physical stride length
            valid_up = (idx_prev > 0) & ((idx_prev - 1) < prev_L)
            valid_diag = (idx_prev2 >= 0) & (idx_prev2 < prev2_L)

            # Load using the 1D descriptor and the index vector
            Hleft = Hprev_desc.load([idx_prev], mask=mask & valid_prev, other=MINF)
            Hup   = Hprev_desc.load([idx_prev - 1], mask=mask & valid_up, other=MINF)
            Hdiag = Hprev2_desc.load([idx_prev2], mask=mask & valid_diag, other=MINF)
            # Handle the origin cell (remains the same)
            Hdiag = gl.where(mask & (i_idx == 1) & (j_idx == 1), 0, Hdiag)
            # ----------------------------------------------------

            # Boundary condition logic (remains the same, using gl.where)
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

            # Calculate Hcur, Ecur, Fcur (logic remains the same)
            if SCORING_MODEL == 'AFFINE':
                # --- Gluon: Load E/F from Shared Memory ---
                Eleft = Eprev_desc.load([idx_prev], mask=mask & valid_prev, other=MINF)
                Fup   = Fprev_desc.load([idx_prev - 1], mask=mask & valid_up, other=MINF)
                # -----------------------------------------
                Ecur = gl.maximum(Eleft + BETA, Hleft + GAPOE)
                Fcur = gl.maximum(Fup + BETA, Hup + GAPOE)
                Hcur = gl.maximum(Hdiag + s_val, gl.maximum(Ecur, Fcur))

                # --- Gluon: Store E/F to Shared Memory ---
                # Store using the 1D descriptor and the index vector relative to band_lo
                Ecurr_desc.store([idx_curr], Ecur, mask=mask)
                Fcurr_desc.store([idx_curr], Fcur, mask=mask)
                # ----------------------------------------
            elif SCORING_MODEL == 'LINEAR':
                score_left = Hleft + BETA
                score_up = Hup + BETA
                Hcur = gl.maximum(Hdiag + s_val, gl.maximum(score_left, score_up))

            # Apply floor if not extension (logic remains the same)
            if not IS_EXTENSION:
                Hcur = gl.maximum(0, Hcur)

            # --- Gluon: Store Hcur to Shared Memory ---
            Hcurr_desc.store([idx_curr], Hcur, mask=mask)
            # -----------------------------------------

            # Update per-thread max (logic remains the same)
            if PRUNING_DROP != 'NONE':
                is_new_max = Hcur > diag_max_s_vec
                diag_max_s_vec = gl.where(mask & is_new_max, Hcur, diag_max_s_vec)
                diag_max_i_vec = gl.where(mask & is_new_max, i_idx, diag_max_i_vec)

            # Increment offset for the next chunk of the band
            off += BLOCK

        # --- Final Reduction (after inner loop) ---
        # Using Gluon's reduction primitives (assuming gl.max works like tl.max)
        diag_max_s = MINF
        diag_max_i = 0
        diag_max_j = 0
        if PRUNING_DROP != 'NONE':
            final_mask = gl.arange(0, BLOCK) < L
            masked_s_vec = gl.where(final_mask, diag_max_s_vec, MINF)

            # Reduce score vector to scalar max score
            diag_max_s = gl.max(masked_s_vec, axis=0)

            # Find 'i' corresponding to the max score
            is_max_thread_mask = (masked_s_vec == diag_max_s) & final_mask
            diag_max_i = gl.max(gl.where(is_max_thread_mask, diag_max_i_vec, 0), axis=0)
            diag_max_j = d - diag_max_i
        # ------------------------------------------

        # update global max (logic remains the same)
        if PRUNING_DROP != 'NONE':
             if diag_max_s > best_s:
                 best_s = diag_max_s
                 best_i = diag_max_i
                 best_j = diag_max_j

        # Z-drop check (logic remains the same, using gl.abs)
        if PRUNING_DROP != 'NONE':
            BETA_POS = gl.abs(BETA)
            ZTH = drop_threshold
            diff_vec = (diag_max_i - diag_max_j) - (best_i - best_j)
            br_mask = (diag_max_i >= best_i) & (diag_max_j >= best_j)
            too_low = (best_s - diag_max_s) > ZTH + BETA_POS * gl.abs(diff_vec)

            if too_low and br_mask:
                stop = 1

        # Update loop variables (logic remains the same)
        prev2_lo, prev_lo = prev_lo, band_lo
        prev2_L, prev_L = prev_L, L # Update previous band lengths (using physical stride size)
        d += 1 # Increment diagonal counter

    # --- FINALIZATION LOGIC (Store results to global memory) ---
    if PRUNING_DROP != 'NONE':
        gl.store(out + 0, best_s)
        gl.store(out + 1, best_i)
        gl.store(out + 2, best_j)
    else:
        # Handle case where pruning is off if necessary
        pass