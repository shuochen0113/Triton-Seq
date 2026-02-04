#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton affine‑gap, banded Smith‑Waterman (AGAThA Z‑drop, β = |gap_extend|).
Bit‑exact with the validated CPU implementation.
"""

import torch, triton, triton.language as tl
import numpy as np

# ──────────────── DNA packing ────────────────
DNA_MAP = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 14}

def pack_sequence(seq: str):
    n_words = (len(seq) + 7) // 8
    words   = torch.empty(n_words, dtype=torch.int32)
    acc = off = wid = 0
    for c in seq:
        acc |= DNA_MAP.get(c, 0) << (4 * off)
        off += 1
        if off == 8:
            if acc >= 0x8000_0000:          # cast unsigned→signed
                acc -= 0x1_0000_0000
            words[wid], acc, off, wid = acc, 0, 0, wid + 1
    if off:
        if acc >= 0x8000_0000:
            acc -= 0x1_0000_0000
        words[wid] = acc
    return words.cuda(non_blocking=True)

# ──────────── Triton kernel ────────────
@triton.jit
def sw_kernel_guided_affine(
    q_ptrs, r_ptrs,
    m_arr, n_arr,
    match, mismatch,
    gap_open, gap_ext,
    Zdrop,
    outs,                     # int32[ n_pairs , 3 ]
    prevM, prevD, prevI,      # anti‑diag d‑1
    prev2M, prev2D,           # anti‑diag d‑2   (I not needed)
    currM, currD, currI,      # anti‑diag d
    STRIDE: tl.constexpr,     # == 2*band + 1
    BAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)               # one pair per block

    # sequence meta ----------------------------------------------------------
    m = tl.load(m_arr + pid)
    n = tl.load(n_arr + pid)
    q_ptr = tl.load(q_ptrs + pid).to(tl.pointer_type(tl.int32))
    r_ptr = tl.load(r_ptrs + pid).to(tl.pointer_type(tl.int32))

    # row base pointers ------------------------------------------------------
    row_off = pid * STRIDE
    Mprev  = prevM  + row_off
    Dprev  = prevD  + row_off
    Iprev  = prevI  + row_off
    Mprev2 = prev2M + row_off
    Dprev2 = prev2D + row_off
    Mcurr  = currM  + row_off
    Dcurr  = currD  + row_off
    Icurr  = currI  + row_off
    out    = outs   + pid * 3

    # constants --------------------------------------------------------------
    MATCH = tl.full((), match,     tl.int32)
    MISM  = tl.full((), mismatch,  tl.int32)
    ALPHA = tl.full((), gap_open,  tl.int32)
    BETA  = tl.full((), gap_ext,   tl.int32)
    BETA_POS = tl.abs(BETA)
    ZTH   = tl.full((), Zdrop,     tl.int32)
    MINF  = tl.full((), -1_000_000, tl.int32)
    NPEN  = tl.full((), 1,         tl.int32)  # == N_PENALTY
    N_VAL = tl.full((), 14,        tl.int32)  # == N_CODE & 0xF

    # best‑score bookkeeping -------------------------------------------------
    best_s = tl.zeros((), tl.int32)
    best_i = tl.zeros((), tl.int32)
    best_j = tl.zeros((), tl.int32)
    best_d = tl.zeros((), tl.int32)

    d         = tl.full((), 2, tl.int32)   # first filled anti‑diag
    prev_lo   = tl.full((), 1, tl.int32)   # d‑1 band left‑edge
    prev2_lo  = tl.full((), 1, tl.int32)   # d‑2 band left‑edge
    stop      = tl.zeros((), tl.int32)

    # ──────────── anti‑diagonal sweep ────────────
    while (d <= m + n) & (stop == 0):

        i_min  = tl.maximum(1, d - n)
        i_max  = tl.minimum(m, d - 1)
        center = d // 2
        band_lo = tl.maximum(i_min, center - BAND)
        band_hi = tl.minimum(i_max, center + BAND)
        L       = band_hi - band_lo + 1

        diag_max = tl.zeros((), tl.int32)
        off = tl.zeros((), tl.int32)

        while off < L:
            tid  = tl.arange(0, BLOCK)
            mask = tid < (L - off)

            i_idx = band_lo + off + tid
            j_idx = d - i_idx

            # --- base comparison -------------------------------------------
            q_word = tl.load(q_ptr + (i_idx - 1) // 8, mask=mask)
            r_word = tl.load(r_ptr + (j_idx - 1) // 8, mask=mask)
            q_code = (q_word >> (((i_idx - 1) % 8) * 4)) & 0xF
            r_code = (r_word >> (((j_idx - 1) % 8) * 4)) & 0xF
            s_val = tl.where((q_code == N_VAL) | (r_code == N_VAL), -NPEN,
                             tl.where(q_code == r_code, MATCH, MISM))

            # --- fetch neighbours ------------------------------------------
            idx_prev  = i_idx - prev_lo
            idx_prev2 = i_idx - 1 - prev2_lo

            Mleft = tl.load(Mprev + idx_prev , mask=mask, other=MINF)
            Ileft = tl.load(Iprev + idx_prev , mask=mask, other=MINF)

            valid_up = idx_prev > 0
            Mup = tl.load(Mprev + (idx_prev - 1), mask=mask & valid_up, other=MINF)
            Dup = tl.load(Dprev + (idx_prev - 1), mask=mask & valid_up, other=MINF)

            valid_diag = (idx_prev2 >= 0) & (idx_prev2 < STRIDE)
            Mdiag = tl.load(Mprev2 + idx_prev2, mask=mask & valid_diag, other=0)

            # --- affine‑gap DP ---------------------------------------------
            Icur = tl.maximum(Ileft + BETA, Mleft + ALPHA)
            Dcur = tl.maximum(Dup   + BETA, Mup   + ALPHA)
            Mcur = tl.maximum(0, tl.maximum(Mdiag + s_val, tl.maximum(Icur, Dcur)))

            idx_curr = i_idx - band_lo
            tl.store(Icurr + idx_curr, Icur, mask=mask)
            tl.store(Dcurr + idx_curr, Dcur, mask=mask)
            tl.store(Mcurr + idx_curr, Mcur, mask=mask)

            cur_m   = tl.where(mask, Mcur, MINF)
            row_max = tl.max(cur_m, axis=0)
            diag_max = tl.maximum(diag_max, row_max)

            cidx = tl.argmax(cur_m, axis=0)
            ci   = band_lo + off + cidx
            cj   = d - ci
            upd  = row_max > best_s
            best_s = tl.where(upd, row_max, best_s)
            best_i = tl.where(upd, ci,      best_i)
            best_j = tl.where(upd, cj,      best_j)
            best_d = tl.where(upd, d,       best_d)

            off += BLOCK

        # --- Z‑drop early stop ---------------------------------------------
        delta = d - best_d
        stop  = tl.where(diag_max < best_s - ZTH - BETA_POS * delta,
                         tl.full((), 1, tl.int32), stop)

        # --- rotate buffers: prev2 ← prev ; prev ← curr ; curr ← zero -------
        offset = tl.zeros((), tl.int32)
        while offset < STRIDE:
            idx = tl.arange(0, BLOCK) + offset
            msk = idx < STRIDE

            tl.store(prev2M + row_off + idx, tl.load(Mprev + idx, mask=msk), mask=msk)
            tl.store(prev2D + row_off + idx, tl.load(Dprev + idx, mask=msk), mask=msk)

            tl.store(Mprev + idx, tl.load(Mcurr + idx, mask=msk), mask=msk)
            tl.store(Dprev + idx, tl.load(Dcurr + idx, mask=msk), mask=msk)
            tl.store(Iprev + idx, tl.load(Icurr + idx, mask=msk), mask=msk)

            tl.store(Mcurr + idx, 0,    mask=msk)
            tl.store(Dcurr + idx, MINF, mask=msk)
            tl.store(Icurr + idx, MINF, mask=msk)
            offset += BLOCK

        prev2_lo, prev_lo = prev_lo, band_lo
        d += 1

    tl.store(out + 0, best_s)
    tl.store(out + 1, best_i)
    tl.store(out + 2, best_j)

# ───────────────────────── host wrapper ─────────────────────────
def smith_waterman_gpu_guided_affine(
    q_list, r_list,
    match=1, mismatch=-4,
    gap_open=-6, gap_extend=-2,
    band=400, Z=751,
    BLOCK_SIZE=256,
):
    assert len(q_list) == len(r_list)
    n_pairs = len(q_list)

    q_t = [pack_sequence(s) for s in q_list]
    r_t = [pack_sequence(s) for s in r_list]
    q_ptrs = torch.tensor([t.data_ptr() for t in q_t], dtype=torch.int64, device='cuda')
    r_ptrs = torch.tensor([t.data_ptr() for t in r_t], dtype=torch.int64, device='cuda')
    m_arr  = torch.tensor([len(s) for s in q_list], dtype=torch.int32, device='cuda')
    n_arr  = torch.tensor([len(s) for s in r_list], dtype=torch.int32, device='cuda')

    STRIDE = 2 * band + 1
    def new_buf(init): return torch.full((n_pairs, STRIDE), init,
                                         dtype=torch.int32, device='cuda')
    prevM, prevD, prevI = new_buf(0), new_buf(-10**6), new_buf(-10**6)
    prev2M, prev2D      = new_buf(0), new_buf(-10**6)
    currM, currD, currI = new_buf(0), new_buf(-10**6), new_buf(-10**6)
    outs = torch.zeros((n_pairs, 3), dtype=torch.int32, device='cuda')

    grid = (n_pairs,)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True); t0.record()

    sw_kernel_guided_affine[grid](
        q_ptrs, r_ptrs,
        m_arr, n_arr,
        match, mismatch,
        gap_open, gap_extend,
        Z,
        outs,
        prevM, prevD, prevI,
        prev2M, prev2D,             # <-- new buffers
        currM, currD, currI,
        STRIDE=STRIDE,
        BAND=band,
        BLOCK=BLOCK_SIZE,
    )

    t1.record(); torch.cuda.synchronize()
    print(f"[Triton Affine Guided] Kernel time: {t0.elapsed_time(t1):.3f} ms")

    return [(int(s), (int(i), int(j))) for s, i, j in outs.cpu().tolist()]