# filename: local_dp_kernel_gluon_v3_1x1desc.py

import triton
# 关键：导入 gluon 和 gluon language
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# -----------------------------------------------------------------------------
# [Dylan's Kernel using Gluon - Attempt 3: 1x1 Descriptor Hypothesis]
# Goal: Test if we can obtain a 1x1 descriptor using dynamic indices
#       passed to .index() and .slice(), and then load the single element.
# Assumption Under Test: .index() and .slice() accept dynamic gl.tensor indices.
# Prerequisite: Temporarily change shape to power-of-2 to bypass allocation errors.
#               H: [4, 1024], E/F: [2, 1024], STRIDE_POW2 = 1024
# -----------------------------------------------------------------------------

@gluon.jit
def sw_kernel_gluon_1x1desc(
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
    # --- TEMPORARY MODIFICATION FOR TESTING ---
    STRIDE_POW2: gl.constexpr, # Use power-of-2 stride (e.g., 1024)
    H_SLOTS: gl.constexpr,     # Use power-of-2 slots (e.g., 4)
    EF_SLOTS: gl.constexpr,    # Use power-of-2 slots (e.g., 2)
    # --- END TEMPORARY MODIFICATION ---
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

    # --- Gluon: Shared Memory Allocation (Using Power-of-2 Shape Temporarily) ---
    # Using Swizzled layout as it might be more robust for allocation test
    swizzled_layout_h = gl.SwizzledSharedLayout(vec=4, per_phase=1, max_phase=4, order=[1, 0])
    swizzled_layout_ef = gl.SwizzledSharedLayout(vec=4, per_phase=1, max_phase=4, order=[1, 0])

    # Allocate using temporary power-of-2 shapes
    H_smem = gl.allocate_shared_memory(gl.int32, [H_SLOTS, STRIDE_POW2], layout=swizzled_layout_h)
    E_smem = gl.allocate_shared_memory(gl.int32, [EF_SLOTS, STRIDE_POW2], layout=swizzled_layout_ef)
    F_smem = gl.allocate_shared_memory(gl.int32, [EF_SLOTS, STRIDE_POW2], layout=swizzled_layout_ef)

    # Define a layout for loading/storing a single scalar element
    scalar_layout = gl.BlockedLayout([1], [1], [1], [0])
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
    prev_L = STRIDE_POW2 # Use the allocated stride size
    prev2_L = STRIDE_POW2

    # ───────────── main anti-diagonal sweep ──────────────────────
    while (d <= m + n) & (stop == 0):

        # --- Dynamic Row Indices (as gl.tensor) ---
        # NOTE: Using temporary H_SLOTS and EF_SLOTS for modulo
        curr_row_h_idx_tensor = d % H_SLOTS
        prev_row_h_idx_tensor = (d - 1) % H_SLOTS
        prev2_row_h_idx_tensor = (d - 2) % H_SLOTS
        curr_row_ef_idx_tensor = d % EF_SLOTS
        prev_row_ef_idx_tensor = (d - 1) % EF_SLOTS
        # -------------------------------------------

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
            i_idx = band_lo + off + tid # Dynamic gl.tensor
            j_idx = d - i_idx           # Dynamic gl.tensor

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

            # --- Dynamic Column Indices (as gl.tensor) ---
            idx_prev_tensor = i_idx - prev_lo      # Index within Hprev/Eprev/Fprev rows
            idx_prev_m1_tensor = idx_prev_tensor - 1 # Index for Hup/Fup
            idx_prev2_tensor = i_idx - 1 - prev2_lo # Index within Hprev2 row
            idx_curr_tensor = i_idx - band_lo       # Index within Hcurr/Ecurr/Fcurr rows
            # ------------------------------------------

            # --- Validity Masks (Logic remains the same) ---
            valid_prev = (idx_prev_tensor >= 0) & (idx_prev_tensor < prev_L)
            valid_up = (idx_prev_m1_tensor >= 0) & (idx_prev_m1_tensor < prev_L) # Corrected logic for Hup/Fup index
            valid_diag = (idx_prev2_tensor >= 0) & (idx_prev2_tensor < prev2_L)
            # ---------------------------------------------

            # --- EXPERIMENT: Try to get 1x1 descriptor and load ---
            Hleft = MINF # Default value
            Hup = MINF
            Hdiag = MINF
            Eleft = MINF
            Fup = MINF

            # Need a combined mask for validity AND the main loop mask
            load_mask_left = mask & valid_prev
            load_mask_up = mask & valid_up
            load_mask_diag = mask & valid_diag

            # --- Attempt Hleft = H[prev_row_h_idx][idx_prev] ---
            # EXPECTED FAILURE POINT 1: Passing dynamic tensor to .index()
            # EXPECTED FAILURE POINT 2: Passing dynamic tensor to .slice()
            # try:
            # Get descriptor for the previous H row (shape [STRIDE_POW2])
            Hprev_row_desc = H_smem.index(prev_row_h_idx_tensor)
            # Slice this row to get 1x1 descriptor at dynamic col idx_prev_tensor (shape [1])
            # NOTE: This assumes slice can handle vector `start` index, might need elementwise loop?
            #       Let's assume for now it broadcasts or fails.
            Hleft_elem_desc = Hprev_row_desc.slice(idx_prev_tensor, 1, 0)
            # Load the single element using scalar layout
            _hleft_unmasked = Hleft_elem_desc.load(scalar_layout)
            Hleft = gl.where(load_mask_left, _hleft_unmasked, MINF)
            # except Exception as e_hl: # Catch potential compile error (if possible in JIT)
            #      gl.static_print("Error getting Hleft:", e_hl) # Won't likely run here

            # --- Attempt Hup = H[prev_row_h_idx][idx_prev - 1] ---
            # try:
            Hprev_row_desc_hup = H_smem.index(prev_row_h_idx_tensor) # Re-get row desc (needed?)
            Hup_elem_desc = Hprev_row_desc_hup.slice(idx_prev_m1_tensor, 1, 0)
            _hup_unmasked = Hup_elem_desc.load(scalar_layout)
            Hup = gl.where(load_mask_up, _hup_unmasked, MINF)
            # except Exception as e_hu:
            #      gl.static_print("Error getting Hup:", e_hu)

            # --- Attempt Hdiag = H[prev2_row_h_idx][idx_prev2] ---
            # try:
            Hprev2_row_desc = H_smem.index(prev2_row_h_idx_tensor)
            Hdiag_elem_desc = Hprev2_row_desc.slice(idx_prev2_tensor, 1, 0)
            _hdiag_unmasked = Hdiag_elem_desc.load(scalar_layout)
            Hdiag = gl.where(load_mask_diag, _hdiag_unmasked, MINF)
            # except Exception as e_hd:
            #      gl.static_print("Error getting Hdiag:", e_hd)
            # ----------------------------------------------------

            # Handle origin cell (Logic remains the same)
            Hdiag = gl.where(mask & (i_idx == 1) & (j_idx == 1), 0, Hdiag)

            # Boundary condition logic (remains the same)
            if SCORING_MODEL == 'AFFINE':
                Hleft = gl.where(mask & (j_idx == 1), GAPOE + BETA * (i_idx - 1), Hleft)
                Hup = gl.where(mask & (i_idx == 1), GAPOE + BETA * (j_idx - 1), Hup)
                Hdiag = gl.where(mask & (i_idx == 1) & (j_idx > 1), GAPOE + BETA * (j_idx - 2), Hdiag)
                Hdiag = gl.where(mask & (j_idx == 1) & (i_idx > 1), GAPOE + BETA * (i_idx - 2), Hdiag)
            elif SCORING_MODEL == 'LINEAR':
                 # ... (similar) ...
                 pass

            # Calculate Hcur, Ecur, Fcur (Logic remains the same)
            if SCORING_MODEL == 'AFFINE':
                # --- Attempt Eleft = E[prev_row_ef_idx][idx_prev] ---
                # try:
                Eprev_row_desc = E_smem.index(prev_row_ef_idx_tensor)
                Eleft_elem_desc = Eprev_row_desc.slice(idx_prev_tensor, 1, 0)
                _eleft_unmasked = Eleft_elem_desc.load(scalar_layout)
                Eleft = gl.where(load_mask_left, _eleft_unmasked, MINF)
                # except Exception as e_el:
                #     gl.static_print("Error getting Eleft:", e_el)

                # --- Attempt Fup = F[prev_row_ef_idx][idx_prev - 1] ---
                # try:
                Fprev_row_desc = F_smem.index(prev_row_ef_idx_tensor)
                Fup_elem_desc = Fprev_row_desc.slice(idx_prev_m1_tensor, 1, 0)
                _fup_unmasked = Fup_elem_desc.load(scalar_layout)
                Fup = gl.where(load_mask_up, _fup_unmasked, MINF)
                # except Exception as e_fu:
                #     gl.static_print("Error getting Fup:", e_fu)
                # ----------------------------------------------------

                Ecur = gl.maximum(Eleft + BETA, Hleft + GAPOE)
                Fcur = gl.maximum(Fup + BETA, Hup + GAPOE)
                Hcur = gl.maximum(Hdiag + s_val, gl.maximum(Ecur, Fcur))

                # --- EXPERIMENT: Try to store using 1x1 descriptor ---
                store_mask = mask # Store only for valid elements
                # Get descriptor for E[curr_row_ef_idx][idx_curr]
                # try:
                Ecurr_row_desc = E_smem.index(curr_row_ef_idx_tensor)
                Ecurr_elem_desc = Ecurr_row_desc.slice(idx_curr_tensor, 1, 0)
                # Store scalar value; .store needs a tensor, convert scalar Ecur?
                # Assuming Ecur is already a vector [BLOCK], need to match descriptor somehow.
                # This store logic seems fundamentally incompatible if Ecur is [BLOCK] and desc is [1].
                # Let's assume store works element-wise if target desc matches value shape? Unlikely.
                # --- ABANDONING 1x1 DESCRIPTOR STORE FOR NOW ---
                # Need to store the vector Ecur back. Use the row descriptor?
                Ecurr_row_desc.store([idx_curr_tensor], Ecur, mask=store_mask) # Revert to V1 store logic for now

                # except Exception as e_se:
                #     gl.static_print("Error storing Ecur:", e_se)

                # Similar issue for Fcur store
                # try:
                Fcurr_row_desc = F_smem.index(curr_row_ef_idx_tensor)
                # Fcurr_elem_desc = Fcurr_row_desc.slice(idx_curr_tensor, 1, 0)
                # Fcurr_elem_desc.store(Fcur???) # How to store vector to 1x1 desc?
                Fcurr_row_desc.store([idx_curr_tensor], Fcur, mask=store_mask) # Revert to V1 store logic
                # except Exception as e_sf:
                #     gl.static_print("Error storing Fcur:", e_sf)
                # ----------------------------------------------------

            elif SCORING_MODEL == 'LINEAR':
                 # ... (similar) ...
                 pass

            # Apply floor if not extension (Logic remains the same)
            if not IS_EXTENSION:
                Hcur = gl.maximum(0, Hcur)

            # --- Store Hcur (Reverting to V1 logic for store) ---
            store_mask = mask
            # try:
            Hcurr_row_desc = H_smem.index(curr_row_h_idx_tensor)
            Hcurr_row_desc.store([idx_curr_tensor], Hcur, mask=store_mask)
            # except Exception as e_sh:
            #      gl.static_print("Error storing Hcur:", e_sh)
            # ----------------------------------------------------

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

        prev2_lo, prev_lo = prev_lo, band_lo
        # NOTE: Using STRIDE_POW2 for prev_L tracking
        prev2_L, prev_L = prev_L, L if L > 0 else STRIDE_POW2 # Avoid L=0? Check original logic
        d += 1

    # --- FINALIZATION LOGIC (Store results - Logic remains the same) ---
    if PRUNING_DROP != 'NONE':
        gl.store(out + 0, best_s)
        gl.store(out + 1, best_i)
        gl.store(out + 2, best_j)
    else:
        pass