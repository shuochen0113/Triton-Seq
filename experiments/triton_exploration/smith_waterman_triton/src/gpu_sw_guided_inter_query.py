import math, torch, triton, triton.language as tl

# ==============================================================================
# Host-side sequence packing utility
# ------------------------------------------------------------------------------
# Converts a DNA string into a list of int32 words where each word encodes
# up to 8 bases using 4 bits per base (A=1, C=2, G=3, T=4).
# ==============================================================================
DNA_MAP = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

def pack_sequence(seq: str):
    n_words = (len(seq) + 7) // 8
    words = torch.empty(n_words, dtype=torch.int32)
    acc = off = wid = 0
    for c in seq:
        acc |= DNA_MAP.get(c, 0) << (4 * off)
        off += 1
        if off == 8:
            words[wid] = acc
            acc = off = 0
            wid += 1
    if off:
        words[wid] = acc
    return words.cuda(non_blocking=True)

# ==============================================================================
# Triton Kernel: Banded Guided Smith-Waterman with Early Termination (X-drop)
# ------------------------------------------------------------------------------
# Processes one sequence pair per block. Uses a diagonal wavefront computation
# bounded within a fixed-width band. Early termination is applied based on the
# X-drop heuristic.
# ==============================================================================
@triton.jit
def sw_kernel_guided_packed(
    q_ptrs, r_ptrs,
    m_arr, n_arr,
    match, mismatch, gap,
    xdrop,
    out_ptrs,
    prev_ptrs, prev2_ptrs, curr_ptrs,
    BAND_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr):

    pid = tl.program_id(0)

    # Load per-pair metadata
    m = tl.load(m_arr + pid)
    n = tl.load(n_arr + pid)
    q_ptr = tl.load(q_ptrs + pid).to(tl.pointer_type(tl.int32))
    r_ptr = tl.load(r_ptrs + pid).to(tl.pointer_type(tl.int32))
    out = tl.load(out_ptrs + pid).to(tl.pointer_type(tl.int32))
    prev = tl.load(prev_ptrs + pid).to(tl.pointer_type(tl.int32))
    prev2 = tl.load(prev2_ptrs + pid).to(tl.pointer_type(tl.int32))
    curr = tl.load(curr_ptrs + pid).to(tl.pointer_type(tl.int32))

    # Scoring constants
    MATCH = tl.full((), match, tl.int32)
    MISM = tl.full((), mismatch, tl.int32)
    GAP = tl.full((), gap, tl.int32)
    XDROP = tl.full((), xdrop, tl.int32)
    MINF = tl.full((), -1_000_000, tl.int32)

    # Track best alignment score and position
    best_score = tl.zeros((), tl.int32)
    best_i = tl.zeros((), tl.int32)
    best_j = tl.zeros((), tl.int32)

    d = tl.full((), 2, tl.int32)
    prev_lo = tl.full((), 1, tl.int32)
    prev2_lo = tl.full((), 1, tl.int32)
    stop = tl.zeros((), tl.int32)

    while (d <= m + n) & (stop == 0):
        # Compute current wavefront band
        i_min = tl.maximum(1, d - n)
        i_max = tl.minimum(m, d - 1)
        center = d // 2
        band_lo = tl.maximum(i_min, center - BAND_WIDTH)
        band_hi = tl.minimum(i_max, center + BAND_WIDTH)
        L = band_hi - band_lo + 1

        # Diagonal wavefront traversal
        off = tl.zeros((), tl.int32)
        while off < L:
            tid = tl.arange(0, BLOCK)
            mask = tid < (L - off)
            i_idx = band_lo + off + tid
            j_idx = d - i_idx

            # Decode bases from packed format
            q_word = tl.load(q_ptr + (i_idx - 1) // 8, mask=mask)
            r_word = tl.load(r_ptr + (j_idx - 1) // 8, mask=mask)
            q_code = (q_word >> (((i_idx - 1) % 8) * 4)) & 0xF
            r_code = (r_word >> (((j_idx - 1) % 8) * 4)) & 0xF
            s_val = tl.where(q_code == r_code, MATCH, MISM)

            # DP recurrence: max of diag, up, left, or zero
            diag = tl.load(prev2 + (i_idx - 1 - prev2_lo), mask=mask, other=0)
            up = tl.load(prev + (i_idx - 1 - prev_lo), mask=mask, other=MINF)
            left = tl.load(prev + (i_idx - prev_lo), mask=mask, other=MINF)
            cur = tl.maximum(0, tl.maximum(diag + s_val, tl.maximum(up + GAP, left + GAP)))

            # Store current score
            tl.store(curr + (i_idx - band_lo), cur, mask=mask)

            # Track best alignment score and coordinates
            cur_m = tl.where(mask, cur, MINF)
            cmax = tl.max(cur_m, axis=0)
            cidx = tl.argmax(cur_m, axis=0)
            ci = band_lo + off + cidx
            cj = d - ci
            upd = cmax > best_score
            best_score = tl.where(upd, cmax, best_score)
            best_i = tl.where(upd, ci, best_i)
            best_j = tl.where(upd, cj, best_j)

            off += BLOCK

        # Early termination using X-drop heuristic
        lane = tl.arange(0, BLOCK)
        lane_m = lane < L
        max_wf = tl.max(tl.load(curr + lane, mask=lane_m, other=MINF), axis=0)
        stop = tl.where(best_score - max_wf > XDROP, stop + 1, tl.zeros((), tl.int32))
        stop = tl.where(stop > 2, tl.full((), 1, tl.int32), stop)

        # Rotate DP buffers
        VALID = 2 * BAND_WIDTH + 1
        offset = tl.zeros((), tl.int32)
        while offset < VALID:
            idx = tl.arange(0, BLOCK) + offset
            m_rot = idx < VALID
            tl.store(prev2 + idx, tl.load(prev + idx, mask=m_rot), mask=m_rot)
            tl.store(prev + idx, tl.load(curr + idx, mask=m_rot), mask=m_rot)
            offset += BLOCK

        # Update wavefront base offset trackers
        prev2_lo = prev_lo
        prev_lo = band_lo
        d += 1

    # Write final result
    tl.store(out + 0, best_score)
    tl.store(out + 1, best_i)
    tl.store(out + 2, best_j)

# ==============================================================================
# Host wrapper for batch guided alignment
# ------------------------------------------------------------------------------
# Runs the banded + X-drop guided alignment kernel on a batch of query/ref pairs.
# Each pair is processed in one GPU block.
# ==============================================================================
def smith_waterman_gpu_guided_packed(
    q_list, r_list,
    match=3, mismatch=-2, gap=-1,
    band=400, xdrop=751, BLOCK_SIZE=256):

    assert len(q_list) == len(r_list)
    n_pairs = len(q_list)

    q_tensors = [pack_sequence(q) for q in q_list]
    r_tensors = [pack_sequence(r) for r in r_list]
    q_ptrs = torch.tensor([t.data_ptr() for t in q_tensors], dtype=torch.int64, device='cuda')
    r_ptrs = torch.tensor([t.data_ptr() for t in r_tensors], dtype=torch.int64, device='cuda')
    m_arr = torch.tensor([len(q) for q in q_list], dtype=torch.int32, device='cuda')
    n_arr = torch.tensor([len(r) for r in r_list], dtype=torch.int32, device='cuda')

    B = 2 * band + 1
    prev = torch.zeros((n_pairs, B), dtype=torch.int32, device='cuda')
    prev2 = torch.zeros_like(prev)
    curr = torch.zeros_like(prev)
    prev_ptrs = torch.tensor([p.data_ptr() for p in prev], dtype=torch.int64, device='cuda')
    prev2_ptrs = torch.tensor([p.data_ptr() for p in prev2], dtype=torch.int64, device='cuda')
    curr_ptrs = torch.tensor([p.data_ptr() for p in curr], dtype=torch.int64, device='cuda')

    outs = torch.zeros((n_pairs, 3), dtype=torch.int32, device='cuda')
    out_ptrs = torch.tensor([o.data_ptr() for o in outs], dtype=torch.int64, device='cuda')

    grid = (n_pairs,)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True); t0.record()

    sw_kernel_guided_packed[grid](
        q_ptrs, r_ptrs,
        m_arr, n_arr,
        match, mismatch, gap,
        xdrop,
        out_ptrs,
        prev_ptrs, prev2_ptrs, curr_ptrs,
        BAND_WIDTH=band,
        BLOCK=BLOCK_SIZE
    )

    t1.record(); torch.cuda.synchronize()
    print(f"[Triton Guided Timing] Kernel time: {t0.elapsed_time(t1):.3f} ms")

    return [(int(s), (int(i), int(j))) for s, i, j in outs.cpu().tolist()]