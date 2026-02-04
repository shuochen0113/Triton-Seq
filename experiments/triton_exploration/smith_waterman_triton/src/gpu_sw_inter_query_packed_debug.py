import torch, math
import triton
import triton.language as tl

import torch, math
import triton
import triton.language as tl

# ==============================================================================
# Host 侧 DNA 序列打包工具（4-bit 编码，每32位存8个碱基）
# ==============================================================================
DNA_MAP = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
def pack_sequence(seq: str) -> list:
    """将 DNA 序列打包为 int32 数组，每个 32 位整数存 8 个碱基，不足补0。"""
    packed = []
    L = len(seq)
    num_words = math.ceil(L / 8)
    for i in range(num_words):
        word = 0
        for j in range(8):
            pos = i * 8 + j
            code = DNA_MAP.get(seq[pos], 0) if pos < L else 0
            word |= (code << (4 * j))
        packed.append(word)
    return packed

# ==============================================================================
# Improved Debug Kernel：使用 wavefront 算法计算 Smith–Waterman，同时写入完整 DP 矩阵
# ==============================================================================
@triton.jit
def sw_kernel_inter_query_packed_debug(
    q_ptrs, r_ptrs,
    m_arr, n_arr,
    match, mismatch, gap,
    out_ptrs,
    diag_prev_ptrs,    # scratch 缓冲区：上一波前
    diag_prev2_ptrs,   # scratch 缓冲区：前前波前
    diag_curr_ptrs,    # scratch 缓冲区：当前波前
    debug_dp_ptrs,     # debug 输出：完整 DP 矩阵，大小为 (m+1)*(n+1)
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    m = tl.load(m_arr + pid)          # query 长度
    n = tl.load(n_arr + pid)          # reference 长度
    L_max = tl.minimum(m, n)          # 波前最大长度

    q_ptr   = tl.load(q_ptrs + pid).to(tl.pointer_type(tl.int32))
    r_ptr   = tl.load(r_ptrs + pid).to(tl.pointer_type(tl.int32))
    out_ptr = tl.load(out_ptrs + pid).to(tl.pointer_type(tl.int32))
    
    # 三块 scratch 内存，各自独立（由 host 分配）
    prev_ptr  = tl.load(diag_prev_ptrs + pid).to(tl.pointer_type(tl.int32))
    prev2_ptr = tl.load(diag_prev2_ptrs + pid).to(tl.pointer_type(tl.int32))
    curr_ptr  = tl.load(diag_curr_ptrs + pid).to(tl.pointer_type(tl.int32))
    
    debug_dp_ptr = tl.load(debug_dp_ptrs + pid).to(tl.pointer_type(tl.int32))
    
    MIN_VAL = -1000000000

    global_max = tl.zeros((), tl.int32)
    global_i   = tl.zeros((), tl.int32)
    global_j   = tl.zeros((), tl.int32)

    # -------------------------------
    # 初始化 DP 矩阵边界：第一行和第一列均置零
    # -------------------------------
    j = 0
    while j <= n:
        tl.store(debug_dp_ptr + j, 0)
        j = j + 1
    i = 0
    while i <= m:
        tl.store(debug_dp_ptr + (i*(n+1)), 0)
        i = i + 1

    # -------------------------------
    # 计算第一波前： d = 2，计算 dp[1,1]
    # -------------------------------
    d = 2
    scalar_mask = tl.full((), True, tl.int1)
    q_word = tl.load(q_ptr + 0, mask=scalar_mask)
    r_word = tl.load(r_ptr + 0, mask=scalar_mask)
    q_code = (q_word >> 0) & 0xF
    r_code = (r_word >> 0) & 0xF
    score_val = tl.where(q_code == r_code, match, mismatch)
    val = tl.maximum(0, score_val)
    tl.store(curr_ptr, val, mask=scalar_mask)
    # 将 dp[1,1] 写入 DP 矩阵
    tl.store(debug_dp_ptr + (1*(n+1) + 1), val, mask=scalar_mask)
    global_max = val
    global_i   = tl.full((), 1, tl.int32)
    global_j   = tl.full((), 1, tl.int32)

    # 显式初始化：将第一波前复制到 prev 缓冲区，prev2 置 0
    idx = 0
    while idx < L_max:
        tl.store(prev_ptr + idx, tl.load(curr_ptr + idx))
        tl.store(prev2_ptr + idx, 0)
        idx = idx + 1

    d = d + 1
    d_val = d
    while d_val <= m + n:
        # 当前波前长度：L_curr = min(d_val-1, m, n, m+n-d_val+1)
        L1 = d_val - 1
        L2 = m
        L3 = n
        L4 = m + n - d_val + 1
        L_curr = tl.minimum(tl.minimum(L1, L2), tl.minimum(L3, L4))
        start = tl.maximum(1, d_val - n)
        
        offset = 0
        while offset < L_curr:
            seg_size = tl.minimum(BLOCK_SIZE, L_curr - offset)
            k = tl.arange(0, BLOCK_SIZE)
            mask = k < seg_size
            cur_k = offset + k  # 当前 wavefront 内局部索引
            i_coords = start + cur_k   # 全局 i 坐标
            j_coords = d_val - i_coords  # 全局 j 坐标
            
            # 解码 query[i_coords-1] 与 ref[j_coords-1]
            q_idx = (i_coords - 1) // 8
            q_off = ((i_coords - 1) % 8) * 4
            q_word_seg = tl.load(q_ptr + q_idx, mask=mask)
            q_code_seg = (q_word_seg >> q_off) & 0xF
            
            r_idx = (j_coords - 1) // 8
            r_off = ((j_coords - 1) % 8) * 4
            r_word_seg = tl.load(r_ptr + r_idx, mask=mask)
            r_code_seg = (r_word_seg >> r_off) & 0xF

            s_val = tl.where(q_code_seg == r_code_seg, match, mismatch)

            # 依赖项索引计算——与原逻辑保持一致
            cond = d_val <= (n + 1)
            diag_index_true = tl.where(cur_k > 0, cur_k - 1, 0)
            up_index_true   = tl.where(cur_k > 0, cur_k - 1, 0)
            left_index_true = cur_k

            extra_offset = tl.where(d_val > (n + 2), 1, 0)
            diag_index_false = cur_k + extra_offset
            up_index_false   = cur_k
            left_index_false = cur_k + 1

            diag_term = tl.where(
                cond,
                tl.where(cur_k > 0, tl.load(prev2_ptr + diag_index_true, mask=mask), 0),
                tl.where(cur_k < tl.minimum(L_max, d_val - 2), tl.load(prev2_ptr + diag_index_false, mask=mask), MIN_VAL)
            )
            up_term = tl.where(
                cond,
                tl.where(cur_k > 0, tl.load(prev_ptr + up_index_true, mask=mask), 0),
                tl.load(prev_ptr + up_index_false, mask=mask)
            )
            left_term = tl.where(
                cond,
                tl.load(prev_ptr + left_index_true, mask=mask),
                tl.where((cur_k + 1) < tl.minimum(L_max, d_val - 1), tl.load(prev_ptr + left_index_false, mask=mask), MIN_VAL)
            )
            
            curr_val = tl.maximum(0,
                          tl.maximum(diag_term + s_val,
                          tl.maximum(up_term + gap, left_term + gap)))
                         
            tl.store(curr_ptr + cur_k, curr_val, mask=mask)
            
            # 将当前 wavefront dp 值写入 debug DP 矩阵
            global_index = i_coords * (n+1) + j_coords
            tl.store(debug_dp_ptr + global_index, curr_val, mask=mask)

            curr_val_masked = tl.where(mask, curr_val, MIN_VAL)
            curr_max = tl.max(curr_val_masked, axis=0)
            curr_max_idx = tl.argmax(curr_val_masked, axis=0)
            i_max_local = start + offset + curr_max_idx
            j_max_local = d_val - i_max_local
            update_cond = curr_max > global_max
            global_max = tl.where(update_cond, curr_max, global_max)
            global_i = tl.where(update_cond, i_max_local, global_i)
            global_j = tl.where(update_cond, j_max_local, global_j)

            offset = offset + BLOCK_SIZE
        # end 内层 while
        
        # 明确复制而非指针交换
        # 1. 将上一波前 (prev) 的数据复制到 prev2 缓冲区
        j_copy = 0
        while j_copy < tl.minimum(L_max, d_val - 1):  # d_val-1 为上一波前长度（理论上）
            tl.store(prev2_ptr + j_copy, tl.load(prev_ptr + j_copy))
            j_copy = j_copy + 1

        # 2. 将当前波前 (curr) 的数据复制到 prev 缓冲区
        j_copy = 0
        while j_copy < L_curr:
            tl.store(prev_ptr + j_copy, tl.load(curr_ptr + j_copy))
            j_copy = j_copy + 1

        d_val = d_val + 1
    # end 外层 while

    tl.store(out_ptr + 0, global_max)
    tl.store(out_ptr + 1, global_i)
    tl.store(out_ptr + 2, global_j)

# ==============================================================================
# Host 侧封装 Debug 版本
# ==============================================================================
def smith_waterman_gpu_inter_query_packed_debug(query_list, ref_list, 
                                                match=3, mismatch=-2, gap=-1, 
                                                BLOCK_SIZE=256):
    """
    多序列对比对（Debug 版本）：
      - 序列以 4-bit 压缩格式存储
      - 每对序列由单个 GPU block 使用 wavefront 算法计算，
        仅分配两个前波前缓冲区及一个当前波前缓冲区
      - 同时额外写入一个完整 DP 矩阵，用于调试比较（仅适用于小规模数据）
      - 返回最高得分及其 (i, j) 坐标，同时返回每对序列的 DP 矩阵（CPU端 numpy 数组）
    """
    num_pairs = len(query_list)
    assert num_pairs == len(ref_list)
    
    q_tensors, r_tensors = [], []
    m_list, n_list = [], []
    diag_bufs = []  # 每对用于 wavefront 的 scratch 缓冲，大小 = min(m, n)
    debug_dp_bufs = []  # 每对完整 DP 矩阵，大小 = (m+1)*(n+1)
    for q, r in zip(query_list, ref_list):
        q_tensors.append(torch.tensor(pack_sequence(q), device='cuda', dtype=torch.int32))
        r_tensors.append(torch.tensor(pack_sequence(r), device='cuda', dtype=torch.int32))
        m_list.append(len(q))
        n_list.append(len(r))
        L_max = min(len(q), len(r))
        diag_bufs.append({
            'prev': torch.zeros(L_max, dtype=torch.int32, device='cuda'),
            'prev2': torch.zeros(L_max, dtype=torch.int32, device='cuda'),
            'curr': torch.zeros(L_max, dtype=torch.int32, device='cuda'),
        })
        # 分配完整 DP 矩阵存储缓冲（以一维形式存放，大小为 (m+1)*(n+1)）
        debug_dp_bufs.append(torch.zeros((len(q)+1)*(len(r)+1), dtype=torch.int32, device='cuda'))
    
    q_ptrs = torch.tensor([t.data_ptr() for t in q_tensors], dtype=torch.int64, device='cuda')
    r_ptrs = torch.tensor([t.data_ptr() for t in r_tensors], dtype=torch.int64, device='cuda')
    m_arr  = torch.tensor(m_list, dtype=torch.int32, device='cuda')
    n_arr  = torch.tensor(n_list, dtype=torch.int32, device='cuda')
    
    out_tensors = [torch.zeros(3, dtype=torch.int32, device='cuda') for _ in range(num_pairs)]
    out_ptrs = torch.tensor([t.data_ptr() for t in out_tensors], dtype=torch.int64, device='cuda')

    diag_prev_ptrs  = torch.tensor([buf['prev'].data_ptr() for buf in diag_bufs],
                                     dtype=torch.int64, device='cuda')
    diag_prev2_ptrs = torch.tensor([buf['prev2'].data_ptr() for buf in diag_bufs],
                                     dtype=torch.int64, device='cuda')
    diag_curr_ptrs  = torch.tensor([buf['curr'].data_ptr() for buf in diag_bufs],
                                     dtype=torch.int64, device='cuda')
    
    debug_dp_ptrs = torch.tensor([t.data_ptr() for t in debug_dp_bufs],
                                   dtype=torch.int64, device='cuda')

    grid = (num_pairs,)
    torch.cuda.synchronize()
    s_evt, e_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_evt.record()
    
    sw_kernel_inter_query_packed_debug[grid](
        q_ptrs, r_ptrs,
        m_arr, n_arr,
        match, mismatch, gap,
        out_ptrs,
        diag_prev_ptrs, diag_prev2_ptrs, diag_curr_ptrs,
        debug_dp_ptrs,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    e_evt.record()
    torch.cuda.synchronize()
    ktime = s_evt.elapsed_time(e_evt)
    print(f"[Triton Wavefront Packed Debug Timing] Kernel time: {ktime:.3f} ms")
    
    results = []
    debug_dp_matrices = []
    for idx, out in enumerate(out_tensors):
        s_val, i_val, j_val = out.cpu().tolist()
        results.append((s_val, (i_val, j_val)))
        # 根据对应 m, n 重塑 debug DP 矩阵为二维
        m_val = m_list[idx]
        n_val = n_list[idx]
        debug_dp = debug_dp_bufs[idx].cpu().view(m_val+1, n_val+1)
        debug_dp_matrices.append(debug_dp)
    
    return results, debug_dp_matrices