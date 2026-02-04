#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton implementation of the **extension + Z-drop** Smith-Waterman
algorithm (ksw_extz-style, β = |gap_extend|).

Bit-exact with the validated CPU routine ``smith_waterman_extz`` in cpu_extz.py.
Each Triton block handles **one (query, ref) pair** and sweeps the DP matrix
anti-diagonal-wise inside a fixed band.

Last Update Date  : 2025-05-26
"""

import torch, triton, triton.language as tl
import numpy as np

# ─────────────────── DNA 4-bit packing helpers ───────────────────
DNA_MAP = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 14}

def pack_sequence(seq: str) -> torch.Tensor:
    """Pack a DNA string into 8*4-bit little-endian int32 words (same as AGAThA)."""
    n_words = (len(seq) + 7) // 8
    words   = torch.empty(n_words, dtype=torch.int32)
    acc = off = wid = 0
    for c in seq:
        acc |= DNA_MAP.get(c.upper(), 0) << (4 * off)
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

# ─────────────────────── Triton kernel ───────────────────────────
@triton.jit
def sw_kernel_extz(
    q_ptrs, r_ptrs,           # int64[ n_pairs ]
    m_arr,  n_arr,            # int32[ n_pairs ]
    match, mismatch,          #  scalar int32
    gap_open, gap_ext,        #  scalar int32  (negative numbers)
    Zdrop,                    #  scalar int32
    outs,                     #  int32[ n_pairs , 3 ]  (score, i, j)

    prevH, prevF, prevE,      #  d-1 anti-diag  (shape: n_pairs × STRIDE)
    prev2H,                   #  d-2 anti-diag
    currH, currF, currE,      #  current anti-diag (d)

    STRIDE: tl.constexpr,     #  2*band + 1
    BAND  : tl.constexpr,
    BLOCK : tl.constexpr,     #  threads per CTA
):
    pid   = tl.program_id(0)               # 1 CTA == 1 alignment pair

    # ───── sequence metadata ─────────────────────────────────────
    m      = tl.load(m_arr + pid)
    n      = tl.load(n_arr + pid)
    q_ptr  = tl.load(q_ptrs + pid).to(tl.pointer_type(tl.int32))
    r_ptr  = tl.load(r_ptrs + pid).to(tl.pointer_type(tl.int32))

    # ───── row base pointers for this pair ───────────────────────
    row_off = pid * STRIDE
    Hprev   = prevH  + row_off
    Fprev   = prevF  + row_off
    Eprev   = prevE  + row_off
    Hprev2  = prev2H + row_off
    Hcurr   = currH  + row_off
    Fcurr   = currF  + row_off
    Ecurr   = currE  + row_off
    out     = outs   + pid * 3

    # ───── constants in registers ────────────────────────────────
    MATCH   = tl.full((), match,    tl.int32)
    MISM    = tl.full((), mismatch, tl.int32)
    ALPHA   = tl.full((), gap_open, tl.int32)   # gap-open  (<= 0)
    BETA    = tl.full((), gap_ext,  tl.int32)   # gap-ext   (<= 0)
    BETA_POS= tl.abs(BETA)
    GAPOE   = ALPHA + BETA 
    ZTH     = tl.full((), Zdrop,    tl.int32)
    MINF    = tl.full((), -10_000_000, tl.int32)
    NPEN    = tl.full((), 1,        tl.int32)   # N penalty
    N_VAL   = tl.full((), 14,       tl.int32)   # code for ‘N’

    # ───── bookkeeping of the global best cell ───────────────────
    best_s = tl.zeros((), tl.int32)
    best_i = tl.zeros((), tl.int32)
    best_j = tl.zeros((), tl.int32)
    best_d = tl.zeros((), tl.int32)   # diagonal of the best cell

    # algorithm starts from anti-diagonal d = 2  (d = 0 / 1 are pre-filled)
    d        = tl.full((), 2, tl.int32)
    prev_lo  = tl.full((), 1, tl.int32)   # band left-edge of d-1
    prev2_lo = tl.full((), 0, tl.int32)   # band left-edge of d-2
    stop     = tl.zeros((), tl.int32)

    # ───────────── main anti-diagonal sweep ──────────────────────
    while (d <= m + n) & (stop == 0):

        i_min   = tl.maximum(1, d - n)
        i_max   = tl.minimum(m, d - 1)
        r_off   = best_i - best_j
        half_lo = (d + r_off - BAND + 1) >> 1
        half_hi = (d + r_off + BAND)     >> 1 
        band_lo = tl.maximum(i_min, half_lo)
        band_hi = tl.minimum(i_max, half_hi)
        L       = band_hi - band_lo + 1

        diag_max = tl.full((), -1_000_000, tl.int32)
        diag_off = tl.zeros((), tl.int32) 
        off      = tl.zeros((), tl.int32)

        # ── iterate over the current anti-diag in BLOCK-sized chunks ──
        while off < L:
            tid   = tl.arange(0, BLOCK)
            mask  = tid < (L - off)

            i_idx = band_lo + off + tid
            j_idx = d - i_idx

            # --- substitution score ------------------------------------
            q_word = tl.load(q_ptr + (i_idx - 1) // 8, mask=mask)
            r_word = tl.load(r_ptr + (j_idx - 1) // 8, mask=mask)
            q_code = (q_word >> (((i_idx - 1) % 8) * 4)) & 0xF
            r_code = (r_word >> (((j_idx - 1) % 8) * 4)) & 0xF
            s_val  = tl.where((q_code == N_VAL) | (r_code == N_VAL), -NPEN, tl.where(q_code == r_code, MATCH, MISM))

            # --- neighbours from previous anti-diagonals ----------------
            idx_prev  = i_idx - prev_lo
            valid_prev = (idx_prev >= 0) & (idx_prev < STRIDE)
            idx_prev2 = i_idx - 1 - prev2_lo

            Hleft = tl.load(Hprev + idx_prev, mask=mask&valid_prev, other=MINF)
            # if j_idx == 1 → the true value is H(i,0) = α + β·i
            Hleft = tl.where(mask & (j_idx == 1), GAPOE + BETA * (i_idx - 1), Hleft)

            Eleft = tl.load(Eprev + idx_prev, mask=mask&valid_prev, other=MINF)

            # valid_up = idx_prev > 0
            valid_up = (idx_prev > 0) & (idx_prev - 1 < STRIDE)
            Hup = tl.load(Hprev + (idx_prev - 1), mask=mask & valid_up, other=MINF)
            Fup = tl.load(Fprev + (idx_prev - 1), mask=mask & valid_up, other=MINF)
            # row-0 fallback  (i_idx == 1 → idx_prev == 0)
            Hup = tl.where(mask & ~valid_up, GAPOE + BETA * (j_idx - 1), Hup)


            valid_diag = (idx_prev2 >= 0) & (idx_prev2 < STRIDE)
            Hdiag = tl.load(Hprev2 + idx_prev2, mask=mask & valid_diag, other=MINF)
            # (0,0) special-case
            Hdiag = tl.where(mask & (i_idx == 1) & (j_idx == 1), 0, Hdiag)
            # row-0 diag fallback: (0, j-1)
            Hdiag = tl.where(mask & (i_idx == 1) & (j_idx > 1), GAPOE + BETA * (j_idx - 1), Hdiag)
            # col-0 diag fallback: (i-1, 0)
            Hdiag = tl.where(mask & (j_idx == 1) & (i_idx > 1), GAPOE + BETA * (i_idx - 1), Hdiag)

            # --- affine-gap DP (no local reset!) ------------------------
            Ecur = tl.maximum(Eleft + BETA, Hleft + GAPOE)
            Fcur = tl.maximum(Fup   + BETA, Hup   + GAPOE)
            Hcur = tl.maximum(Hdiag + s_val, tl.maximum(Ecur, Fcur))

            # --- write current anti-diag --------------------------------
            idx_curr = i_idx - band_lo
            tl.store(Ecurr + idx_curr, Ecur, mask=mask)
            tl.store(Fcurr + idx_curr, Fcur, mask=mask)
            tl.store(Hcurr + idx_curr, Hcur, mask=mask)

            # --- track best & diag max ----------------------------------
            cur_h    = tl.where(mask, Hcur, MINF)
            row_max  = tl.max(cur_h, axis=0)
            # diag_max = tl.maximum(diag_max, row_max)
            cidx  = tl.argmax(cur_h, axis=0)
            ci    = band_lo + off + cidx
            cj    = d - ci

            better   = row_max > diag_max
            diag_off = tl.where(better,  ci - cj, diag_off)
            diag_max = tl.where(better,  row_max, diag_max)

            upd   = row_max > best_s
            best_s = tl.where(upd, row_max, best_s)
            best_i = tl.where(upd, ci,      best_i)
            best_j = tl.where(upd, cj,      best_j)
            best_d = tl.where(upd, d,       best_d)

            # -------- per-cell Z-drop check ----
            diff_vec   = (i_idx - j_idx) - (best_i - best_j)          # delta = (i-j) - (ibest-jbest)
            br_mask    = (i_idx >= best_i) & (j_idx >= best_j)        # only see the bottom-right cells
            too_low    = (best_s - Hcur) > ZTH + BETA_POS * tl.abs(diff_vec)
            hit        = tl.max((too_low & br_mask & mask).to(tl.int32), axis=0)
            stop       = tl.where(hit != 0, tl.full((), 1, tl.int32), stop)

            off += BLOCK

            ### NOTICE: AGAThA's Z-drop is used per sliced-diagonal, not per cell or whole anti-diagonal,
            ### so we can't excatly mimic the AGAThA behavior here. (21 or 28 mismatches in AGAThA's datasets)

        # --- Z-drop early termination -----------------------------------
        # diff_best = best_i - best_j
        # delta = tl.abs(diag_off - diff_best)
        # stop = tl.where(diag_max < best_s - ZTH - BETA_POS * delta, tl.full((), 1, tl.int32), stop)

        # --- rotate buffers (prev2 ← prev, prev ← curr, curr ← 0/-INF) ---
        offset = tl.zeros((), tl.int32)
        while offset < STRIDE:
            idx = tl.arange(0, BLOCK) + offset
            msk = idx < STRIDE

            tl.store(prev2H + row_off + idx, tl.load(Hprev + idx, mask=msk), mask=msk)

            tl.store(Hprev + idx,  tl.load(Hcurr + idx, mask=msk), mask=msk)
            tl.store(Fprev + idx,  tl.load(Fcurr + idx, mask=msk), mask=msk)
            tl.store(Eprev + idx,  tl.load(Ecurr + idx, mask=msk), mask=msk)

            tl.store(Hcurr + idx,  MINF, mask=msk)
            tl.store(Fcurr + idx,  MINF, mask=msk)
            tl.store(Ecurr + idx,  MINF, mask=msk)

            offset += BLOCK

        prev2_lo, prev_lo = prev_lo, band_lo
        d += 1

    tl.store(out + 0, best_s)
    tl.store(out + 1, best_i)
    tl.store(out + 2, best_j)

# ─────────────────── host-side convenience wrapper ──────────────────
def smith_waterman_gpu_extz(
    q_list, r_list,
    match=1, mismatch=-4,
    gap_open=-6, gap_extend=-2,
    band=400, Z=751,
    BLOCK_SIZE=256,
):
    """
    Align *each* (q, r) pair from ``q_list`` & ``r_list`` on the GPU and return
    ``[(score, (i_end, j_end)), …]`` in the **same order**.
    """
    assert len(q_list) == len(r_list)
    n_pairs = len(q_list)

    # --- pack sequences & metadata ----------------------------------------
    q_t = [pack_sequence(s) for s in q_list]
    r_t = [pack_sequence(s) for s in r_list]
    q_ptrs = torch.tensor([t.data_ptr() for t in q_t], dtype=torch.int64, device='cuda')
    r_ptrs = torch.tensor([t.data_ptr() for t in r_t], dtype=torch.int64, device='cuda')
    m_arr  = torch.tensor([len(s) for s in q_list], dtype=torch.int32, device='cuda')
    n_arr  = torch.tensor([len(s) for s in r_list], dtype=torch.int32, device='cuda')

    # --- DP scratch buffers -----------------------------------------------
    STRIDE = 2 * band + 1
    NEG_INF = -10**6
    def new_buf(init):           # helper
        return torch.full((n_pairs, STRIDE), init, dtype=torch.int32, device='cuda')

    prevH, prevF, prevE = new_buf(NEG_INF), new_buf(NEG_INF), new_buf(NEG_INF)
    prev2H              = new_buf(NEG_INF)
    currH, currF, currE = new_buf(NEG_INF), new_buf(NEG_INF), new_buf(NEG_INF)

    # ---- first two anti-diagonals (d = 0, 1) -----------------------------
    # d = 0  ->  only (0,0)  = 0
    prev2H[:, band] = 0         # center of STRIDE is BAND

    # d = 1  ->  cells (1,0)  /  (0,1)
    gapoe = gap_open + gap_extend     # NB: both are negative
    # leftmost position in the band for d = 1 is i = 1
    prevH[:, 0] = gapoe               # H(1,0)
    # E_prev (insert in query) is -INF, F_prev (del in query) identical to H(1,0) – not needed but set
    prevF[:, 0] = gap_open  # F(1,0)

    # everything else stays at NEG_INF

    outs = torch.zeros((n_pairs, 3), dtype=torch.int32, device='cuda')

    # --- launch ------------------------------------------------------------
    grid = (n_pairs,)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True); t0.record()

    sw_kernel_extz[grid](
        q_ptrs, r_ptrs,
        m_arr, n_arr,
        match, mismatch,
        gap_open, gap_extend,
        Z,
        outs,
        prevH, prevF, prevE,
        prev2H,
        currH, currF, currE,
        STRIDE=STRIDE,
        BAND=band,
        BLOCK=BLOCK_SIZE,
    )

    t1.record(); torch.cuda.synchronize()
    print(f"[Triton ExtZ] Kernel time: {t0.elapsed_time(t1):.3f} ms")

    return [(int(s), (int(i), int(j))) for s, i, j in outs.cpu().tolist()]


# ──────────────────────── main function ────────────────────────────────
import time

def read_fasta(path: str) -> list[str]:
    seqs = []
    cur = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append(cur)
                cur = ""
            else:
                cur += line
        if cur:
            seqs.append(cur)
    return seqs

def main():
    q_seqs = read_fasta("datasets/query.fa")
    r_seqs = read_fasta("datasets/ref.fa")
    if len(q_seqs) != len(r_seqs):
        print(f"[Error] query count ({len(q_seqs)}) != ref count ({len(r_seqs)})")
        return

    _ = smith_waterman_gpu_extz(q_seqs[:1], r_seqs[:1])

    start = time.time()
    results = smith_waterman_gpu_extz(q_seqs, r_seqs)
    total = time.time() - start

    for idx, ((score, (i, j))) in enumerate(results, start=1):
        print(f"[Triton GPU ExtZ] Pair {idx}: score={score}, end_i={i}, end_j={j}")

    print(f"[Triton GPU ExtZ] Total time: {total*1000:.2f} ms for {len(results)} pairs")

if __name__ == "__main__":
    main()