# src/kernel/experimental/local_dp_kernel_OPv9_smem.py
#
# OPv9 — Generalized shared-memory API variant of OPv6
#
# Key change over OPv6 (sw_kernel.py):
#   - H/E/F ring buffers are NO LONGER passed as kernel arguments.
#   - Instead, each thread block allocates its own private ring buffers in
#     shared memory via tl.allocate_shared, then accesses them through
#     tl.load_shared / tl.store_shared.
#
# This requires the custom triton-custom compiler to be installed so that
# the new tl.allocate_shared / tl.load_shared / tl.store_shared ops are
# available and lower correctly to ttg.local_alloc + ttg.local_*_slice.
#
# Advantages over OPv6:
#   - No host-side pre-allocation of H/E/F device tensors.
#   - No PCIe transfer of H/E/F buffers.
#   - SMEM access (~20x faster than DRAM).
#   - Self-contained kernel: only sequence data and scoring parameters needed.
#
# Scheduling / host API is the same as OPv6 except:
#   - Remove Hbuf / Fbuf / Ebuf from the kernel call-site.
#   - Remove buffer_manager H/E/F allocation.
#
# Memory footprint per block (default STRIDE=768, BLOCK=256):
#   H: 3 * 768 * 4 =  9216 bytes
#   E: 2 * 768 * 4 =  6144 bytes
#   F: 2 * 768 * 4 =  6144 bytes
#   Total per block: 21504 bytes (21 KB), well within the 48-96 KB SMEM limit.

import triton
import triton.language as tl


@triton.jit
def sw_kernel_smem(
    # ── per-query pointer arrays ──────────────────────────────────
    q_ptrs, r_ptrs,
    m_arr, n_arr,
    outs,
    # (H/E/F global-memory buffers removed — now in SMEM below)

    # ── scoring parameters ────────────────────────────────────────
    match_score, mismatch_score,
    gap_open_penalty, gap_extend_penalty,
    drop_threshold,

    # ── compile-time constants ────────────────────────────────────
    SCORING_MODEL: tl.constexpr,
    PRUNING_BAND:  tl.constexpr,
    PRUNING_DROP:  tl.constexpr,
    IS_EXTENSION:  tl.constexpr,
    STRIDE:        tl.constexpr,   # physical ring-buffer stride (>= BAND, %-128B aligned)
    BAND:          tl.constexpr,   # logical band width
    BLOCK:         tl.constexpr,   # threads per block
):
    pid = tl.program_id(0)

    # ── per-block SMEM ring buffers ───────────────────────────────
    # Each block owns its own H/E/F buffers, laid out flat as
    # [num_slots * STRIDE] elements.  Slot k starts at k * STRIDE.
    #
    # H: 3 slots  (rotating mod-3 ring, same as OPv6)
    # E: 2 slots  (rotating mod-2 ring)
    # F: 2 slots  (rotating mod-2 ring)
    #
    # Requires triton-custom with tl.allocate_shared / tl.load_shared /
    # tl.store_shared.
    Hsmem = tl.allocate_shared(3 * STRIDE, tl.int32)
    Esmem = tl.allocate_shared(2 * STRIDE, tl.int32)
    Fsmem = tl.allocate_shared(2 * STRIDE, tl.int32)

    # ── initialize SMEM to MINF sentinel ─────────────────────────
    # Necessary so that masked reads (boundaries / first diagonals) return
    # a safe value even before any write has been issued to that slot.
    _init_tid  = tl.arange(0, BLOCK)
    _MINF_VEC  = tl.full((BLOCK,), -10_000_000, tl.int32)

    # Use strip-mined runtime loops rather than static nested loops so the
    # compiler keeps the init as a compact loop body instead of unrolling many
    # separate shared stores.
    _h_fill_base = tl.zeros((), tl.int32)
    while _h_fill_base < 3 * STRIDE:
        _h_offs = _init_tid + _h_fill_base
        tl.store_shared(Hsmem, _h_offs, _MINF_VEC, mask=_h_offs < 3 * STRIDE)
        _h_fill_base += BLOCK

    _ef_fill_base = tl.zeros((), tl.int32)
    while _ef_fill_base < 2 * STRIDE:
        _ef_offs = _init_tid + _ef_fill_base
        _ef_mask = _ef_offs < 2 * STRIDE
        tl.store_shared(Esmem, _ef_offs, _MINF_VEC, mask=_ef_mask)
        tl.store_shared(Fsmem, _ef_offs, _MINF_VEC, mask=_ef_mask)
        _ef_fill_base += BLOCK

    # Ensure all initialization stores are visible before the DP loop reads.
    tl.debug_barrier()

    # ── sequence metadata ─────────────────────────────────────────
    m     = tl.load(m_arr + pid)
    n     = tl.load(n_arr + pid)
    q_ptr = tl.load(q_ptrs + pid).to(tl.pointer_type(tl.int32))
    r_ptr = tl.load(r_ptrs + pid).to(tl.pointer_type(tl.int32))
    out   = outs + pid * 3

    # ── scoring constants ─────────────────────────────────────────
    MATCH = tl.full((), match_score,         tl.int32)
    MISM  = tl.full((), mismatch_score,      tl.int32)
    ALPHA = tl.full((), gap_open_penalty,    tl.int32)
    BETA  = tl.full((), gap_extend_penalty,  tl.int32)
    GAPOE = ALPHA + BETA
    MINF  = tl.full((), -10_000_000,         tl.int32)
    NPEN  = tl.full((), 1,                   tl.int32)
    N_VAL = tl.full((), 14,                  tl.int32)

    # ── pruning bookkeeping ───────────────────────────────────────
    if PRUNING_DROP != 'NONE':
        init_score = tl.where(IS_EXTENSION, MINF, 0)
        best_s = tl.full((), init_score, tl.int32)
        best_i = tl.zeros((), tl.int32)
        best_j = tl.zeros((), tl.int32)

    # ── main loop state ───────────────────────────────────────────
    d        = tl.full((), 2,      tl.int32)
    prev_lo  = tl.full((), 1,      tl.int32)
    prev2_lo = tl.full((), 0,      tl.int32)
    stop     = tl.zeros((),        tl.int32)
    prev_L   = tl.full((), STRIDE, tl.int32)
    prev2_L  = tl.full((), STRIDE, tl.int32)

    # ── anti-diagonal sweep ───────────────────────────────────────
    while (d <= m + n) & (stop == 0):

        # Flat SMEM slot base offsets for this diagonal.
        # Each slot occupies STRIDE elements; slot k starts at k * STRIDE.
        curr_slot_h  = (d       % 3) * STRIDE
        prev_slot_h  = ((d - 1) % 3) * STRIDE
        prev2_slot_h = ((d - 2) % 3) * STRIDE

        if SCORING_MODEL == 'AFFINE':
            curr_slot_ef = (d       % 2) * STRIDE
            prev_slot_ef = ((d - 1) % 2) * STRIDE

        # Band computation (same as OPv6 STATIC mode).
        i_min   = tl.maximum(1,     d - n)
        i_max   = tl.minimum(m,     d - 1)
        r_off   = 0
        half_lo = (d + r_off - BAND + 2) >> 1
        half_hi = (d + r_off + BAND - 1) >> 1
        band_lo = tl.maximum(i_min, half_lo)
        band_hi = tl.minimum(i_max, half_hi)
        L       = band_hi - band_lo + 1

        diag_max_s_vec = tl.full((BLOCK,), MINF, tl.int32)
        diag_max_i_vec = tl.zeros((BLOCK,),      tl.int32)
        off = tl.zeros((), tl.int32)

        # ── inner loop over BLOCK-sized chunks of the diagonal ────
        while off < L:
            tid   = tl.arange(0, BLOCK)
            mask  = tid < (L - off)
            i_idx = band_lo + off + tid
            j_idx = d - i_idx

            # Sequence character fetch (packed 4-bit encoding).
            q_word = tl.load(q_ptr + (i_idx - 1) // 8, mask=mask, other=0)
            r_word = tl.load(r_ptr + (j_idx - 1) // 8, mask=mask, other=0)
            q_code = (q_word >> (((i_idx - 1) % 8) * 4)) & 0xF
            r_code = (r_word >> (((j_idx - 1) % 8) * 4)) & 0xF
            s_val  = tl.where(
                (q_code == N_VAL) | (r_code == N_VAL),
                -NPEN,
                tl.where(q_code == r_code, MATCH, MISM),
            )

            # Validity masks for dependency access.
            idx_prev   = i_idx - prev_lo
            valid_prev = (idx_prev >= 0) & (idx_prev < prev_L)
            idx_prev2  = i_idx - 1 - prev2_lo
            valid_up   = (idx_prev > 0) & ((idx_prev - 1) < prev_L)
            valid_diag = (idx_prev2 >= 0) & (idx_prev2 < prev2_L)

            # ── H dependency loads from SMEM ──────────────────────
            # Offset = slot_base + relative_index (broadcast: scalar + tensor).
            Hleft = tl.load_shared(
                Hsmem, prev_slot_h + idx_prev,
                mask=mask & valid_prev, other=MINF,
            )
            Hup   = tl.load_shared(
                Hsmem, prev_slot_h + (idx_prev - 1),
                mask=mask & valid_up,   other=MINF,
            )
            Hdiag = tl.load_shared(
                Hsmem, prev2_slot_h + idx_prev2,
                mask=mask & valid_diag, other=MINF,
            )
            # (i=1, j=1) origin cell is always 0 in both SW and alignment.
            Hdiag = tl.where(mask & (i_idx == 1) & (j_idx == 1), 0, Hdiag)

            # Boundary-condition overrides (same as OPv6).
            if SCORING_MODEL == 'AFFINE':
                Hleft = tl.where(mask & (j_idx == 1),
                                 GAPOE + BETA * (i_idx - 1), Hleft)
                Hup   = tl.where(mask & (i_idx == 1),
                                 GAPOE + BETA * (j_idx - 1), Hup)
                Hdiag = tl.where(mask & (i_idx == 1) & (j_idx > 1),
                                 GAPOE + BETA * (j_idx - 2), Hdiag)
                Hdiag = tl.where(mask & (j_idx == 1) & (i_idx > 1),
                                 GAPOE + BETA * (i_idx - 2), Hdiag)
            elif SCORING_MODEL == 'LINEAR':
                Hleft = tl.where(mask & (j_idx == 1),
                                 BETA * i_idx, Hleft)
                Hup   = tl.where(mask & (i_idx == 1),
                                 BETA * j_idx, Hup)
                Hdiag = tl.where(mask & (i_idx == 1) & (j_idx > 1),
                                 BETA * (j_idx - 1), Hdiag)
                Hdiag = tl.where(mask & (j_idx == 1) & (i_idx > 1),
                                 BETA * (i_idx - 1), Hdiag)

            # ── DP recurrence ─────────────────────────────────────
            if SCORING_MODEL == 'AFFINE':
                # E/F loads from SMEM.
                Eleft = tl.load_shared(
                    Esmem, prev_slot_ef + idx_prev,
                    mask=mask & valid_prev, other=MINF,
                )
                Fup   = tl.load_shared(
                    Fsmem, prev_slot_ef + (idx_prev - 1),
                    mask=mask & valid_up,   other=MINF,
                )
                Ecur = tl.maximum(Eleft + BETA, Hleft + GAPOE)
                Fcur = tl.maximum(Fup   + BETA, Hup   + GAPOE)
                Hcur = tl.maximum(Hdiag + s_val, tl.maximum(Ecur, Fcur))

                idx_curr = i_idx - band_lo
                # E/F stores to SMEM.
                tl.store_shared(Esmem, curr_slot_ef + idx_curr, Ecur, mask=mask)
                tl.store_shared(Fsmem, curr_slot_ef + idx_curr, Fcur, mask=mask)

            elif SCORING_MODEL == 'LINEAR':
                Hcur = tl.maximum(
                    Hdiag + s_val,
                    tl.maximum(Hleft + BETA, Hup + BETA),
                )

            if not IS_EXTENSION:
                Hcur = tl.maximum(0, Hcur)

            idx_curr = i_idx - band_lo
            # ── H store to SMEM ───────────────────────────────────
            tl.store_shared(Hsmem, curr_slot_h + idx_curr, Hcur, mask=mask)

            # Per-thread max tracking (no synchronization needed here).
            if PRUNING_DROP != 'NONE':
                is_new_max     = Hcur > diag_max_s_vec
                diag_max_s_vec = tl.where(mask & is_new_max, Hcur,  diag_max_s_vec)
                diag_max_i_vec = tl.where(mask & is_new_max, i_idx, diag_max_i_vec)

            off += BLOCK

        # ── post-diagonal reduction (same as OPv6) ────────────────
        diag_max_s = MINF
        diag_max_i = 0
        diag_max_j = 0
        if PRUNING_DROP != 'NONE':
            final_mask    = tl.arange(0, BLOCK) < L
            masked_s_vec  = tl.where(final_mask, diag_max_s_vec, MINF)
            diag_max_s    = tl.max(masked_s_vec, axis=0)
            is_max_mask   = (masked_s_vec == diag_max_s) & final_mask
            diag_max_i    = tl.max(tl.where(is_max_mask, diag_max_i_vec, 0), axis=0)
            diag_max_j    = d - diag_max_i

        if diag_max_s > best_s:
            best_s = diag_max_s
            best_i = diag_max_i
            best_j = diag_max_j

        # Z-drop termination.
        if PRUNING_DROP != 'NONE':
            BETA_POS = tl.abs(BETA)
            ZTH      = tl.full((), drop_threshold, tl.int32)
            diff_vec = (diag_max_i - diag_max_j) - (best_i - best_j)
            br_mask  = (diag_max_i >= best_i) & (diag_max_j >= best_j)
            too_low  = (best_s - diag_max_s) > ZTH + BETA_POS * tl.abs(diff_vec)
            if too_low and br_mask:
                stop = 1

        prev2_lo, prev_lo = prev_lo, band_lo
        prev2_L,  prev_L  = prev_L,  L
        d += 1

    # ── output ────────────────────────────────────────────────────
    if PRUNING_DROP != 'NONE':
        tl.store(out + 0, best_s)
        tl.store(out + 1, best_i)
        tl.store(out + 2, best_j)
    else:
        pass


# ── Convenience launcher ──────────────────────────────────────────────────────
#
# Minimal call-site example.  Unlike OPv6, no H/E/F device buffers are
# pre-allocated; the kernel manages its own SMEM.
#
# def run_sw_smem(q_ptrs, r_ptrs, m_arr, n_arr, num_pairs,
#                 match=2, mismatch=-4, gap_open=-4, gap_extend=-2,
#                 drop_threshold=0, STRIDE=768, BAND=751, BLOCK=256):
#     import torch
#     outs = torch.zeros(num_pairs * 3, dtype=torch.int32, device='cuda')
#     grid = (num_pairs,)
#     sw_kernel_smem[grid](
#         q_ptrs, r_ptrs, m_arr, n_arr, outs,
#         match, mismatch, gap_open, gap_extend, drop_threshold,
#         SCORING_MODEL='AFFINE',
#         PRUNING_BAND='STATIC',
#         PRUNING_DROP='ZDROP',
#         IS_EXTENSION=True,
#         STRIDE=STRIDE,
#         BAND=BAND,
#         BLOCK=BLOCK,
#     )
#     return outs.view(num_pairs, 3)
