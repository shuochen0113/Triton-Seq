import triton
import triton.language as tl

@triton.jit
def sw_kernel(
    # ... (all args are identical) ...
    q_ptrs, r_ptrs,
    m_arr, n_arr,
    outs,
    H1buf, H2buf, H3buf, Ebuf, Fbuf,
    match_score, mismatch_score,
    gap_open_penalty, gap_extend_penalty,
    drop_threshold,
    SCORING_MODEL: tl.constexpr,
    PRUNING_DROP: tl.constexpr,
    IS_EXTENSION: tl.constexpr,
    BAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # ... (metadata, chunking ... are identical) ...
    m = tl.load(m_arr + pid)
    n = tl.load(n_arr + pid)
    q_ptr = tl.load(q_ptrs + pid).to(tl.pointer_type(tl.int32))
    r_ptr = tl.load(r_ptrs + pid).to(tl.pointer_type(tl.int32))
    PADDED_BAND = ((BAND + BLOCK - 1) // BLOCK) * BLOCK
    row_offset = pid * PADDED_BAND
    H1_ptr = (H1buf + row_offset).to(tl.pointer_type(tl.int32))
    H2_ptr = (H2buf + row_offset).to(tl.pointer_type(tl.int32))
    H3_ptr = (H3buf + row_offset).to(tl.pointer_type(tl.int32))
    E_ptr  = (Ebuf  + row_offset).to(tl.pointer_type(tl.int32))
    F_ptr  = (Fbuf  + row_offset).to(tl.pointer_type(tl.int32))
    out = outs + pid * 3
    
    # ... (constants ... are identical) ...
    MATCH = tl.full((), match_score, tl.int32)
    MISM  = tl.full((), mismatch_score, tl.int32)
    ALPHA = tl.full((), gap_open_penalty, tl.int32)
    BETA  = tl.full((), gap_extend_penalty, tl.int32)
    GAPOE = ALPHA + BETA
    MINF  = tl.full((), -10_000_000, tl.int32)
    ZERO  = tl.full((), 0, tl.int32)
    N_VAL = tl.full((), 14, tl.int32)
    NPEN  = tl.full((), 1, tl.int32)
    lane   = tl.arange(0, BLOCK)
    minf_v = tl.zeros([BLOCK], tl.int32) + MINF
    zero_v = tl.zeros([BLOCK], tl.int32)

    # ... (Phase 1: initialize buffers ... is identical) ...
    for chunk_id in tl.static_range(0, (BAND + BLOCK - 1)//BLOCK):
        off = chunk_id * BLOCK
        tl.store(H1_ptr + off + lane, minf_v)
        tl.store(H2_ptr + off + lane, minf_v)
        tl.store(H3_ptr + off + lane, minf_v)
        tl.store(E_ptr  + off + lane, minf_v)
        tl.store(F_ptr  + off + lane, minf_v)

    # ... (pruning bookkeeping ... is identical) ...
    if PRUNING_DROP != 'NONE':
        init_score = tl.where(IS_EXTENSION, MINF, ZERO)
        best_s = tl.full((), init_score, tl.int32)
        best_i = tl.zeros((), tl.int32)
        best_j = tl.zeros((), tl.int32)

    # -------- main sweep --------
    d    = tl.full((), 2, tl.int32)
    stop = tl.zeros((), tl.int32)

    # [FIX] 把 pending 寄存器移到 while 循环之外
    pend_up_val   = MINF; pend_up_has   = tl.zeros((), tl.int32)
    pend_diag_val = MINF; pend_diag_has = tl.zeros((), tl.int32)
    pend_F_val    = MINF; pend_F_has    = tl.zeros((), tl.int32)

    while (d <= m + n) & (stop == 0):
        # ... (band_lo / L calculations ... are identical) ...
        i_min_d = tl.maximum(1, d - n)
        i_max_d = tl.minimum(m, d - 1)
        half_lo_d = (d - BAND + 2) >> 1
        half_hi_d = (d + BAND - 1) >> 1
        band_lo_d = tl.maximum(i_min_d, half_lo_d)
        band_hi_d = tl.minimum(i_max_d, half_hi_d)
        L  = tl.maximum(0, band_hi_d - band_lo_d + 1)
        d1 = d + 1
        i_min_d1 = tl.maximum(1, d1 - n)
        i_max_d1 = tl.minimum(m, d1 - 1)
        half_lo_d1 = (d1 - BAND + 2) >> 1
        half_hi_d1 = (d1 + BAND - 1) >> 1
        band_lo_d1 = tl.maximum(i_min_d1, half_lo_d1)
        band_hi_d1 = tl.minimum(i_max_d1, half_hi_d1)
        L1 = tl.maximum(0, band_hi_d1 - band_lo_d1 + 1)
        d2 = d + 2
        i_min_d2 = tl.maximum(1, d2 - n)
        i_max_d2 = tl.minimum(m, d2 - 1)
        half_lo_d2 = (d2 - BAND + 2) >> 1
        half_hi_d2 = (d2 + BAND - 1) >> 1
        band_lo_d2 = tl.maximum(i_min_d2, half_lo_d2)
        band_hi_d2 = tl.minimum(i_max_d2, half_hi_d2)
        L2 = tl.maximum(0, band_hi_d2 - band_lo_d2 + 1)
        delta1 = band_lo_d1 - band_lo_d
        delta2 = band_lo_d2 - band_lo_d
        is_even       = ((d  & 1) == 0)
        even_next     = ((d1 & 1) == 0)
        even_next2    = ((d2 & 1) == 0)
        mask_even_next  = (even_next  == 1)
        mask_odd_next   = (even_next  == 0)
        mask_even_next2 = (even_next2 == 1)
        mask_odd_next2  = (even_next2 == 0)

        # ... (per-diag max ... is identical) ...
        if PRUNING_DROP != 'NONE':
            diag_max_s = MINF
            diag_max_i = tl.zeros((), tl.int32)
        
        # [FIX] 
        # ======== Stage 1: Load ALL Chunks into Registers ========
        # 我们需要一个 list 来存, 但 Triton JIT 不支持
        # 我们必须对 `NUM_CHUNKS` (e.g., =3) 手动 unroll
        
        # 假设 NUM_CHUNKS = 3 (基于 BAND=751, BLOCK=256)
        # 你必须根据你的 (BAND, BLOCK) 调整这里的 unroll 数量
        
        # Chunk 0
        H1_c0 = tl.load(H1_ptr + 0*BLOCK + lane)
        H2_c0 = tl.load(H2_ptr + 0*BLOCK + lane)
        H3_c0 = tl.load(H3_ptr + 0*BLOCK + lane)
        E_c0  = tl.load(E_ptr  + 0*BLOCK + lane)
        F_c0  = tl.load(F_ptr  + 0*BLOCK + lane)
        
        # Chunk 1
        H1_c1 = tl.load(H1_ptr + 1*BLOCK + lane)
        H2_c1 = tl.load(H2_ptr + 1*BLOCK + lane)
        H3_c1 = tl.load(H3_ptr + 1*BLOCK + lane)
        E_c1  = tl.load(E_ptr  + 1*BLOCK + lane)
        F_c1  = tl.load(F_ptr  + 1*BLOCK + lane)

        # Chunk 2
        H1_c2 = tl.load(H1_ptr + 2*BLOCK + lane)
        H2_c2 = tl.load(H2_ptr + 2*BLOCK + lane)
        H3_c2 = tl.load(H3_ptr + 2*BLOCK + lane)
        E_c2  = tl.load(E_ptr  + 2*BLOCK + lane)
        F_c2  = tl.load(F_ptr  + 2*BLOCK + lane)
        
        # (如果 NUM_CHUNKS 更多, 你需要在这里 load 更多 H1_c3, H1_c4...)

        # ======== Stage 2: Compute + Store Loop ========
        # 现在 `load` 和 `store` 被分开了
        # `tl.load` 读的是 (H1_c0...)，`tl.store` 写的是 (H1_ptr)
        # RAW Hazard 消除！
        
        for chunk_id in tl.static_range(0, (BAND + BLOCK - 1)//BLOCK):
            off  = chunk_id * BLOCK
            gidx = off + lane
            maskL = gidx < L

            # ---- flush pendings from previous chunk (to current off) ----
            # (这个逻辑现在是安全的, 因为 store 和 load 分离了)
            valid_up_flush = (pend_up_has == 1) & (off < L1)
            tl.store(H3_ptr + off, pend_up_val, mask = valid_up_flush & mask_even_next)
            tl.store(H2_ptr + off, pend_up_val, mask = valid_up_flush & mask_odd_next)
            pend_up_has = tl.where(valid_up_flush, 0, pend_up_has)

            valid_diag_flush = (pend_diag_has == 1) & (off < L2)
            tl.store(H1_ptr + off, pend_diag_val, mask = valid_diag_flush & mask_even_next2)
            tl.store(H3_ptr + off, pend_diag_val, mask = valid_diag_flush & mask_odd_next2)
            pend_diag_has = tl.where(valid_diag_flush, 0, pend_diag_has)

            valid_F_flush = (pend_F_has == 1) & (off < L1)
            tl.store(F_ptr + off, pend_F_val, mask = valid_F_flush)
            pend_F_has = tl.where(valid_F_flush, 0, pend_F_has)

            # ---- load sources by parity(d) ----
            # [FIX] Load from REGISTERS, not global memory
            
            # 手动选择 chunk
            if chunk_id == 0:
                Hdiag_e = H1_c0; Hdiag_o = H3_c0
                Hleft_e = H2_c0; Hleft_o = H1_c0
                Hup_e   = H3_c0; Hup_o   = H2_c0
                Vec_Eprev = E_c0
                Vec_Fprev = F_c0
            elif chunk_id == 1:
                Hdiag_e = H1_c1; Hdiag_o = H3_c1
                Hleft_e = H2_c1; Hleft_o = H1_c1
                Hup_e   = H3_c1; Hup_o   = H2_c1
                Vec_Eprev = E_c1
                Vec_Fprev = F_c1
            elif chunk_id == 2:
                Hdiag_e = H1_c2; Hdiag_o = H3_c2
                Hleft_e = H2_c2; Hleft_o = H1_c2
                Hup_e   = H3_c2; Hup_o   = H2_c2
                Vec_Eprev = E_c2
                Vec_Fprev = F_c2
            # (如果 NUM_CHUNKS 更多, 你需要在这里加 else if)

            Vec_Hdiag = tl.where(is_even, Hdiag_e, Hdiag_o)
            Vec_Hleft = tl.where(is_even, Hleft_e, Hleft_o)
            Vec_Hup   = tl.where(is_even, Hup_e,   Hup_o)

            # ... (Compute logic ... is identical) ...
            i_idx = band_lo_d + gidx
            j_idx = d - i_idx
            q_word = tl.load(q_ptr + (i_idx - 1) // 8, mask=maskL, other=0)
            r_word = tl.load(r_ptr + (j_idx - 1) // 8, mask=maskL, other=0)
            q_code = (q_word >> (((i_idx - 1) % 8) * 4)) & 0xF
            r_code = (r_word >> (((j_idx - 1) % 8) * 4)) & 0xF
            s_val  = tl.where((q_code == N_VAL) | (r_code == N_VAL), -NPEN,
                                   tl.where(q_code == r_code, MATCH, MISM))
            Hleft = Vec_Hleft
            Hup   = Vec_Hup
            Hdiag = Vec_Hdiag
            if SCORING_MODEL == 'AFFINE':
                Hleft = tl.where(maskL & (j_idx == 1), GAPOE + BETA * (i_idx - 1), Hleft)
                Hup   = tl.where(maskL & (i_idx == 1), GAPOE + BETA * (j_idx - 1), Hup)
                Hdiag = tl.where(maskL & (i_idx == 1) & (j_idx > 1), GAPOE + BETA * (j_idx - 2), Hdiag)
                Hdiag = tl.where(maskL & (j_idx == 1) & (i_idx > 1), GAPOE + BETA * (i_idx - 2), Hdiag)
            else:
                Hleft = tl.where(maskL & (j_idx == 1), BETA * i_idx, Hleft)
                Hup   = tl.where(maskL & (i_idx == 1), BETA * j_idx, Hup)
                Hdiag = tl.where(maskL & (i_idx == 1) & (j_idx > 1), BETA * (j_idx - 1), Hdiag)
                Hdiag = tl.where(maskL & (j_idx == 1) & (i_idx > 1), BETA * (i_idx - 1), Hdiag)
            Hdiag = tl.where(maskL & (i_idx == 1) & (j_idx == 1), ZERO, Hdiag)
            if SCORING_MODEL == 'AFFINE':
                Ecur = tl.maximum(Vec_Eprev + BETA, Hleft + GAPOE)
                Fcur = tl.maximum(Vec_Fprev + BETA, Hup   + GAPOE)
                Hcur = tl.maximum(Hdiag + s_val, tl.maximum(Ecur, Fcur))
            else:
                score_left = Hleft + BETA
                score_up   = Hup   + BETA
                Hcur = tl.maximum(Hdiag + s_val, tl.maximum(score_left, score_up))
                Ecur = minf_v
                Fcur = minf_v
            if not IS_EXTENSION:
                Hcur = tl.maximum(zero_v, Hcur)
            Hval = tl.where(maskL, Hcur, minf_v)
            Eval = tl.where(maskL, Ecur, minf_v)
            Fval = tl.where(maskL, Fcur, minf_v)

            # ... (per-diag max ... is identical) ...
            if PRUNING_DROP != 'NONE':
                masked = tl.where(maskL, Hval, minf_v)
                cmax = tl.max(masked, axis=0)
                better = cmax > diag_max_s
                is_max = (masked == cmax) & maskL
                cand_i = tl.max(tl.where(is_max, i_idx, 0), axis=0)
                diag_max_s = tl.where(better, cmax,   diag_max_s)
                diag_max_i = tl.where(better, cand_i, diag_max_i)

            # ... (shifted stores ... is identical) ...
            # [FIX]
            # `tl.store` 写的都是 H1_ptr, H2_ptr... (全局内存)
            # `tl.load` 读的都是 H1_c0, H2_c1... (寄存器)
            # 读写已分离，这里的 store 不会再污染下一轮 chunk 的 load
            
            dest_left = gidx - delta1
            mask_left = (gidx < L) & (dest_left >= 0) & (dest_left < L1)
            tl.store(H2_ptr + dest_left, Hval, mask = mask_left & mask_even_next)
            tl.store(H1_ptr + dest_left, Hval, mask = mask_left & mask_odd_next)

            dest_up = gidx - (delta1 - 1)
            mask_up_all = (gidx < L) & (dest_up >= 0) & (dest_up < L1)
            cross_up = (delta1 == 0) & (dest_up == (off + BLOCK))
            write_up_now = mask_up_all & (cross_up == 0)
            tl.store(H3_ptr + dest_up, Hval, mask = write_up_now & mask_even_next)
            tl.store(H2_ptr + dest_up, Hval, mask = write_up_now & mask_odd_next)
            up_sel = tl.where(mask_up_all & cross_up, Hval, minf_v)
            new_up = tl.max(up_sel, axis=0)
            pend_up_val = tl.where((delta1 == 0), new_up, pend_up_val)
            pend_up_has = tl.where((delta1 == 0) & (tl.max((mask_up_all & cross_up).to(tl.int32), axis=0) > 0),
                                     1, pend_up_has)

            dest_diag = gidx - (delta2 - 1)
            mask_diag_all = (gidx < L) & (dest_diag >= 0) & (dest_diag < L2)
            cross_diag = (delta2 == 0) & (dest_diag == (off + BLOCK))
            write_diag_now = mask_diag_all & (cross_diag == 0)
            tl.store(H1_ptr + dest_diag, Hval, mask = write_diag_now & mask_even_next2)
            tl.store(H3_ptr + dest_diag, Hval, mask = write_diag_now & mask_odd_next2)
            diag_sel = tl.where(mask_diag_all & cross_diag, Hval, minf_v)
            new_diag = tl.max(diag_sel, axis=0)
            pend_diag_val = tl.where((delta2 == 0), new_diag, pend_diag_val)
            pend_diag_has = tl.where((delta2 == 0) & (tl.max((mask_diag_all & cross_diag).to(tl.int32), axis=0) > 0),
                                       1, pend_diag_has)

            dest_E = gidx - delta1
            mask_E = (gidx < L) & (dest_E >= 0) & (dest_E < L1)
            tl.store(E_ptr + dest_E, Eval, mask = mask_E)

            dest_F = gidx - (delta1 - 1)
            mask_F_all = (gidx < L) & (dest_F >= 0) & (dest_F < L1)
            cross_F = (delta1 == 0) & (dest_F == (off + BLOCK))
            write_F_now = mask_F_all & (cross_F == 0)
            tl.store(F_ptr + dest_F, Fval, mask = write_F_now)
            F_sel = tl.where(mask_F_all & cross_F, Fval, minf_v)
            new_F = tl.max(F_sel, axis=0)
            pend_F_val = tl.where((delta1 == 0), new_F, pend_F_val)
            pend_F_has = tl.where((delta1 == 0) & (tl.max((mask_F_all & cross_F).to(tl.int32), axis=0) > 0),
                                     1, pend_F_has)

        # [FIX] 删除了 L239-L250 的 "flush tail pendings"
        # 已经在 L86 移出了 pending init, 
        # 下一轮 d+1 的 chunk_id=0 会自动 flush
        
        # ... (single-index MINF hole fill ... is identical) ...
        if delta1 == 1:
            t = tl.maximum(0, L1 - 1)
            tl.store(H2_ptr + t, MINF, mask = mask_even_next)
            tl.store(H1_ptr + t, MINF, mask = mask_odd_next)
            tl.store(E_ptr + t, MINF)
        if delta1 == 0:
            tl.store(H3_ptr + 0, MINF, mask = mask_even_next & (L1 > 0))
            tl.store(H2_ptr + 0, MINF, mask = mask_odd_next  & (L1 > 0))
            tl.store(F_ptr + 0, MINF, mask = (L1 > 0))
        if delta2 == 0:
            tl.store(H1_ptr + 0, MINF, mask = mask_even_next2 & (L2 > 0))
            tl.store(H3_ptr + 0, MINF, mask = mask_odd_next2  & (L2 > 0))
        elif (delta2 == 2) | ((delta2 == 1) & (L2 > L)):
            t2 = tl.maximum(0, L2 - 1)
            tl.store(H1_ptr + t2, MINF, mask = mask_even_next2)
            tl.store(H3_ptr + t2, MINF, mask = mask_odd_next2)

        # ... (z-drop ... is identical) ...
        if PRUNING_DROP != 'NONE':
            diag_max_j = d - diag_max_i
            better_g = diag_max_s > best_s
            best_s = tl.where(better_g, diag_max_s, best_s)
            best_i = tl.where(better_g, diag_max_i, best_i)
            best_j = tl.where(better_g, diag_max_j, best_j)

            BETA_POS = tl.abs(BETA)
            ZTH = tl.full((), drop_threshold, tl.int32)
            diff_vec = (diag_max_i - diag_max_j) - (best_i - best_j)
            br_mask  = (diag_max_i >= best_i) & (diag_max_j >= best_j)
            too_low  = (best_s - diag_max_s) > (ZTH + BETA_POS * tl.abs(diff_vec))
            stop = tl.where(too_low & br_mask, tl.full((), 1, tl.int32), stop)

        d = d + 1

    if PRUNING_DROP != 'NONE':
        tl.store(out + 0, best_s)
        tl.store(out + 1, best_i)
        tl.store(out + 2, best_j)