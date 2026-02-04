import torch, math
import triton
import triton.language as tl

# ==============================================================================
# Host-side DNA sequence packing utility
# ------------------------------------------------------------------------------
# This function converts a DNA sequence (string) into a list of int32 values.
# Each 32-bit integer packs 8 bases (using 4 bits per base). If the number of
# bases is not a multiple of 8, the remaining bits are padded with 0.
# ==============================================================================
DNA_MAP = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
def pack_sequence(seq: str) -> list:
    """Pack a DNA sequence into an int32 array, each 32-bit word packs 8 bases."""
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
# Triton Kernel: Wavefront-based Smith–Waterman algorithm
# ------------------------------------------------------------------------------
# This kernel processes one sequence pair per GPU block (inter-query parallelism).
# The kernel uses a diagonal (wavefront) strategy to compute the DP matrix.
# Instead of storing the entire DP matrix, only three scratch buffers are used:
#   - "prev" for the previous wavefront,
#   - "prev2" for the wavefront before the previous one, and
#   - "curr" for the current wavefront.
#
# To avoid aliasing (stale data when the wavefront becomes shorter), after each
# iteration, we explicitly copy only the valid computed values into "prev" and "prev2"
# and clear the rest of the scratch buffers.
#
# Parameters:
#  - q_ptrs, r_ptrs: Pointers to the packed query and reference sequences (int32).
#  - m_arr, n_arr: Arrays holding the query and reference sequence lengths.
#  - match, mismatch, gap: Scoring parameters.
#  - out_ptrs: Output pointers (each outputs [max_score, best_i, best_j]).
#  - diag_prev_ptrs, diag_prev2_ptrs, diag_curr_ptrs: Scratch buffers for storing the
#       previous wavefront, the one before that, and the current wavefront respectively.
#  - BLOCK_SIZE: Number of threads per segment inside a wavefront (e.g., 256).
# ==============================================================================
@triton.jit
def sw_kernel_inter_query_packed(
    q_ptrs, r_ptrs,
    m_arr, n_arr,
    match, mismatch, gap,
    out_ptrs,
    diag_prev_ptrs,    # Pointer to "prev" wavefront buffer (from the previous iteration)
    diag_prev2_ptrs,   # Pointer to "prev2" wavefront buffer (from two iterations ago)
    diag_curr_ptrs,    # Pointer to "curr" wavefront buffer (stores current wavefront)
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Load sequence lengths (m: query, n: reference)
    m = tl.load(m_arr + pid)
    n = tl.load(n_arr + pid)
    # Maximum possible wavefront length (L_max = min(m, n))
    L_max = tl.minimum(m, n)
    
    # Load pointers to the packed sequences and output
    q_ptr = tl.load(q_ptrs + pid).to(tl.pointer_type(tl.int32))
    r_ptr = tl.load(r_ptrs + pid).to(tl.pointer_type(tl.int32))
    out_ptr = tl.load(out_ptrs + pid).to(tl.pointer_type(tl.int32))
    
    # Load scratch buffers (prev, prev2, and curr)
    prev_ptr  = tl.load(diag_prev_ptrs + pid).to(tl.pointer_type(tl.int32))
    prev2_ptr = tl.load(diag_prev2_ptrs + pid).to(tl.pointer_type(tl.int32))
    curr_ptr  = tl.load(diag_curr_ptrs + pid).to(tl.pointer_type(tl.int32))
    
    MIN_VAL = -1000000000

    # Initialize the global maximum score and coordinate variables.
    global_max = tl.zeros((), tl.int32)
    global_i   = tl.zeros((), tl.int32)
    global_j   = tl.zeros((), tl.int32)
    
    # --------------------------------------------------------------------------
    # Process the first wavefront: d = 2 corresponds to dp[1,1]
    # --------------------------------------------------------------------------
    d = 2
    scalar_mask = tl.full((), True, tl.int1)
    q_word = tl.load(q_ptr + 0, mask=scalar_mask)
    r_word = tl.load(r_ptr + 0, mask=scalar_mask)
    q_code = (q_word >> 0) & 0xF
    r_code = (r_word >> 0) & 0xF
    score_val = tl.where(q_code == r_code, match, mismatch)
    val = tl.maximum(0, score_val)
    tl.store(curr_ptr, val, mask=scalar_mask)
    global_max = val
    global_i = tl.full((), 1, tl.int32)
    global_j = tl.full((), 1, tl.int32)
    
    # Initialize scratch buffers:
    # Copy the first wavefront (only one cell) into "prev" and set "prev2" to 0.
    for idx in range(L_max):
        tl.store(prev_ptr + idx, tl.load(curr_ptr + idx))
        tl.store(prev2_ptr + idx, 0)
    
    d = d + 1
    d_val = d
    
    # --------------------------------------------------------------------------
    # Outer loop: process wavefronts for d = current to m+n
    # --------------------------------------------------------------------------
    while d_val <= m + n:
        # Calculate the valid length of the current wavefront:
        # L_curr = min(d_val-1, m, n, m+n-d_val+1)
        L1 = d_val - 1
        L2 = m
        L3 = n
        L4 = m + n - d_val + 1
        L_curr = tl.minimum(tl.minimum(L1, L2), tl.minimum(L3, L4))
        # The starting row index for the current wavefront:
        start = tl.maximum(1, d_val - n)
        
        # Process the current wavefront in segments of BLOCK_SIZE cells.
        offset = 0
        while offset < L_curr:
            seg_size = tl.minimum(BLOCK_SIZE, L_curr - offset)
            k = tl.arange(0, BLOCK_SIZE)
            mask = k < seg_size
            cur_k = offset + k  # Local index within the wavefront
            # Global coordinates: row = start + cur_k, col = d_val - row
            i_coords = start + cur_k
            j_coords = d_val - i_coords
            
            # Decode the bases from the packed sequences for query and ref.
            q_idx = (i_coords - 1) // 8
            q_off = ((i_coords - 1) % 8) * 4
            q_word_seg = tl.load(q_ptr + q_idx, mask=mask)
            q_code_seg = (q_word_seg >> q_off) & 0xF

            r_idx = (j_coords - 1) // 8
            r_off = ((j_coords - 1) % 8) * 4
            r_word_seg = tl.load(r_ptr + r_idx, mask=mask)
            r_code_seg = (r_word_seg >> r_off) & 0xF
            
            # Compute the score for the current cell.
            s_val = tl.where(q_code_seg == r_code_seg, match, mismatch)

            # ------------------------------------------------------------------
            # Dependency index calculation:
            # If d_val <= n+1, then the starting row is 1 and the index offsets are:
            #   diagonal index = cur_k - 1, up index = cur_k - 1, left index = cur_k.
            # Otherwise (d_val > n+1), an extra offset is needed for the diagonal.
            # ------------------------------------------------------------------
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
            
            # Compute dp[i,j] using the Smith–Waterman recurrence.
            curr_val = tl.maximum(0,
                          tl.maximum(diag_term + s_val,
                          tl.maximum(up_term + gap, left_term + gap)))
                         
            # Store the computed value for the current wavefront.
            tl.store(curr_ptr + cur_k, curr_val, mask=mask)
            
            # Update the global maximum (score and corresponding coordinates) if this cell is higher.
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
        # end inner segmentation loop

        # ----------------------------------------------------------------------
        # Instead of swapping pointers (which may cause aliasing), update the
        # scratch buffers explicitly using copy loops.
        #
        # 1. For the "prev2" buffer, copy the valid region of the previous "prev"
        #    wavefront. The valid length of the previous wavefront is:
        #         L_prev_valid = min(d_val-2, m, n, m+n-d_val+2)
        # 2. For the "prev" buffer, copy the valid region of the current wavefront.
        #    Let L_curr_valid = L_curr.
        # 3. Also, zero out the rest of the buffers up to L_max.
        # ----------------------------------------------------------------------
        L_prev_valid = tl.minimum(tl.minimum(d_val - 2, m), tl.minimum(n, m + n - d_val + 2))
        L_curr_valid = L_curr

        # Copy valid region from prev to prev2.
        j_copy = 0
        while j_copy < L_prev_valid:
            tl.store(prev2_ptr + j_copy, tl.load(prev_ptr + j_copy))
            j_copy = j_copy + 1
        # Zero out the remainder of prev2.
        j_copy = L_prev_valid
        while j_copy < L_max:
            tl.store(prev2_ptr + j_copy, 0)
            j_copy = j_copy + 1
        
        # Copy current wavefront into prev.
        j_copy = 0
        while j_copy < L_curr_valid:
            tl.store(prev_ptr + j_copy, tl.load(curr_ptr + j_copy))
            j_copy = j_copy + 1
        # Zero out the remainder of prev.
        j_copy = L_curr_valid
        while j_copy < L_max:
            tl.store(prev_ptr + j_copy, 0)
            j_copy = j_copy + 1
        
        d_val = d_val + 1
    # end outer wavefront loop
    
    # Store the final maximum score and its position.
    tl.store(out_ptr + 0, global_max)
    tl.store(out_ptr + 1, global_i)
    tl.store(out_ptr + 2, global_j)
    
###############################################################################
# Host-Side Wrapper
###############################################################################
def smith_waterman_gpu_inter_query_packed(query_list, ref_list, match=3, mismatch=-2, gap=-1, BLOCK_SIZE=256):
    """
    Compute Smith–Waterman alignment for multiple sequence pairs on the GPU.
      - Sequences are packed in a 4-bit format.
      - Each sequence pair is processed by a single GPU block using a
        wavefront algorithm with intra-block parallelism.
      - Only two previous wavefront buffers and one current wavefront buffer
        (of size O(min(m,n))) are allocated.
      - The function returns the maximum score and its (i,j) coordinates.
    """
    num_pairs = len(query_list)
    assert num_pairs == len(ref_list)
    
    q_tensors, r_tensors = [], []
    m_list, n_list = [], []
    # Allocate scratch buffers for each sequence pair.
    diag_bufs = []  # For each pair, the buffer size is L_max = min(m, n)
    for q, r in zip(query_list, ref_list):
        q_tensors.append(torch.tensor(pack_sequence(q), device='cuda', dtype=torch.int32))
        r_tensors.append(torch.tensor(pack_sequence(r), device='cuda', dtype=torch.int32))
        m_list.append(len(q))
        n_list.append(len(r))
        # L_max = min(len(q), len(r))
        # diag_bufs.append({
        #     'prev': torch.zeros(L_max, dtype=torch.int32, device='cuda'),
        #     'prev2': torch.zeros(L_max, dtype=torch.int32, device='cuda'),
        #     'curr': torch.zeros(L_max, dtype=torch.int32, device='cuda'),
        # })

        L_max = min(len(q), len(r), len(q) + len(r) - 1)  # wavefront max length is still min(m,n), but be explicit
        L_full = min(len(q), len(r), len(q) + len(r) - 1)
        diag_bufs.append({
            'prev': torch.zeros(L_full, dtype=torch.int32, device='cuda'),
            'prev2': torch.zeros(L_full, dtype=torch.int32, device='cuda'),
            'curr': torch.zeros(L_full, dtype=torch.int32, device='cuda'),
        })
    
    # Build pointer arrays for input sequences and lengths.
    q_ptrs = torch.tensor([t.data_ptr() for t in q_tensors], dtype=torch.int64, device='cuda')
    r_ptrs = torch.tensor([t.data_ptr() for t in r_tensors], dtype=torch.int64, device='cuda')
    m_arr  = torch.tensor(m_list, dtype=torch.int32, device='cuda')
    n_arr  = torch.tensor(n_list, dtype=torch.int32, device='cuda')
    
    out_tensors = [torch.zeros(3, dtype=torch.int32, device='cuda') for _ in range(num_pairs)]
    out_ptrs = torch.tensor([t.data_ptr() for t in out_tensors], dtype=torch.int64, device='cuda')
    
    # Build pointer arrays for the scratch buffers.
    diag_prev_ptrs  = torch.tensor([buf['prev'].data_ptr() for buf in diag_bufs],
                                     dtype=torch.int64, device='cuda')
    diag_prev2_ptrs = torch.tensor([buf['prev2'].data_ptr() for buf in diag_bufs],
                                     dtype=torch.int64, device='cuda')
    diag_curr_ptrs  = torch.tensor([buf['curr'].data_ptr() for buf in diag_bufs],
                                     dtype=torch.int64, device='cuda')
    
    grid = (num_pairs,)
    torch.cuda.synchronize()
    s_evt, e_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_evt.record()
    
    sw_kernel_inter_query_packed[grid](
        q_ptrs, r_ptrs,
        m_arr, n_arr,
        match, mismatch, gap,
        out_ptrs,
        diag_prev_ptrs, diag_prev2_ptrs, diag_curr_ptrs,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    e_evt.record()
    torch.cuda.synchronize()
    ktime = s_evt.elapsed_time(e_evt)
    print(f"[Triton Wavefront Packed Timing] Kernel time: {ktime:.3f} ms")
    
    results = []
    for out in out_tensors:
        s_val, i_val, j_val = out.cpu().tolist()
        results.append((s_val, (i_val, j_val)))
    return results