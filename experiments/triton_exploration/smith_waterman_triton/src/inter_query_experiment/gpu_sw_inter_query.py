import torch
import triton
import triton.language as tl

@triton.jit
def sw_kernel_inter_query(
    query_ptrs, ref_ptrs, dp_ptrs, 
    m_array, n_array, 
    match, mismatch, gap, 
    query_stride, ref_stride, dp_stride_array, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Inter-query Smith-Waterman kernel.
    Each block processes one query-reference pair.
    
    Parameters:
      query_ptrs   : pointer array for query sequences (each query stored as int32 codes).
      ref_ptrs     : pointer array for reference sequences.
      dp_ptrs      : pointer array for the DP matrices for each sequence pair.
      m_array      : array storing length of each query sequence.
      n_array      : array storing length of each reference sequence.
      match, mismatch, gap : scoring parameters.
      query_stride : stride for query sequences (assumed constant = 1).
      ref_stride   : stride for reference sequences (assumed constant = 1).
      dp_stride_array : array storing for each pair the stride (number of columns) of its DP matrix,
                        which equals (n + 1) for that pair.
      BLOCK_SIZE   : number of threads per block.
    """
    # Each block handles one sequence pair.
    pair_idx = tl.program_id(0)
    m = tl.load(m_array + pair_idx)  # length of query
    n = tl.load(n_array + pair_idx)  # length of reference
    q_ptr = tl.load(query_ptrs + pair_idx)  # pointer for query
    r_ptr = tl.load(ref_ptrs + pair_idx)      # pointer for reference
    d_ptr = tl.load(dp_ptrs + pair_idx)        # pointer for DP matrix
    dp_stride = tl.load(dp_stride_array + pair_idx)
    
    max_k = m + n  # total number of diagonals for this pair.
    tid = tl.arange(0, BLOCK_SIZE)

    # Loop over all diagonals.
    for k in range(2, max_k + 1):
        i_min = tl.max(1, k - n)
        i_max = tl.min(m, k - 1)
        num_valid = i_max - i_min + 1

        # Process in chunks of BLOCK_SIZE.
        for chunk in range(0, num_valid, BLOCK_SIZE):
            idx = chunk + tid
            chunk_mask = idx < num_valid

            # Compute the (i, j) coordinates along the current diagonal.
            i = i_min + idx
            j = k - i

            # Load characters from query and reference sequences.
            c1 = tl.load(q_ptr + (i - 1) * query_stride, mask=chunk_mask, other=0)
            c2 = tl.load(r_ptr + (j - 1) * ref_stride, mask=chunk_mask, other=0)

            # Load previously computed DP cells.
            diag = tl.load(d_ptr + (i - 1) * dp_stride + (j - 1), mask=chunk_mask, other=0)
            up   = tl.load(d_ptr + (i - 1) * dp_stride + j, mask=chunk_mask, other=0)
            left = tl.load(d_ptr + i * dp_stride + (j - 1), mask=chunk_mask, other=0)

            current = diag + tl.where(c1 == c2, match, mismatch)
            current = tl.maximum(current, up + gap)
            current = tl.maximum(current, left + gap)
            current = tl.maximum(current, 0)

            tl.store(d_ptr + i * dp_stride + j, current, mask=chunk_mask)
        tl.debug_barrier()


def smith_waterman_gpu_inter_query(query_list, ref_list, match=3, mismatch=-2, gap=-1):
    """
    Compute Smith-Waterman alignment for multiple sequence pairs using inter-query parallelism.
    Each sequence pair is processed concurrently by a separate GPU block.
    
    Parameters:
      query_list : List of query sequences (strings).
      ref_list   : List of reference sequences (strings), must have the same length as query_list.
      match, mismatch, gap : scoring parameters.
    
    Returns:
      results : A list of tuples for each sequence pair:
                (max_score, (max_i, max_j), dp_matrix as a NumPy array).
    """
    num_pairs = len(query_list)
    assert num_pairs == len(ref_list), "Number of queries and references must be equal."

    # For each sequence pair, convert to ASCII tensor (int32) on CUDA.
    query_tensors = []
    ref_tensors = []
    m_list = []
    n_list = []
    dp_tensors = []
    dp_strides = []  # For each pair, dp_stride = (n + 1)

    for q_seq, r_seq in zip(query_list, ref_list):
        q_ascii = [ord(c) for c in q_seq]
        r_ascii = [ord(c) for c in r_seq]
        q_tensor = torch.tensor(q_ascii, device='cuda', dtype=torch.int32)
        r_tensor = torch.tensor(r_ascii, device='cuda', dtype=torch.int32)
        m, n = len(q_seq), len(r_seq)
        dp_tensor = torch.zeros((m + 1, n + 1), dtype=torch.int32, device='cuda')
        query_tensors.append(q_tensor)
        ref_tensors.append(r_tensor)
        m_list.append(m)
        n_list.append(n)
        dp_tensors.append(dp_tensor)
        dp_strides.append(n + 1)  # since dp is shape (m+1, n+1)

    # Create tensor arrays for pointers and parameters.
    # NOTE：Use torch.tensor([x.data_ptr() for x in ...], dtype=torch.int64, device='cuda')
    query_ptrs = torch.tensor([x.data_ptr() for x in query_tensors], dtype=torch.int64, device='cuda')
    ref_ptrs   = torch.tensor([x.data_ptr() for x in ref_tensors], dtype=torch.int64, device='cuda')
    dp_ptrs    = torch.tensor([x.data_ptr() for x in dp_tensors],  dtype=torch.int64, device='cuda')
    m_array    = torch.tensor(m_list, dtype=torch.int32, device='cuda')
    n_array    = torch.tensor(n_list, dtype=torch.int32, device='cuda')
    dp_stride_array = torch.tensor(dp_strides, dtype=torch.int32, device='cuda')
    # For query and ref, stride 1.
    query_stride = torch.tensor(1, dtype=torch.int32, device='cuda')
    ref_stride = torch.tensor(1, dtype=torch.int32, device='cuda')

    BLOCK_SIZE = 256
    # Launch grid: one block per sequence pair.
    grid = (num_pairs,)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    sw_kernel_inter_query[grid](
        query_ptrs, ref_ptrs, dp_ptrs,
        m_array, n_array,
        match, mismatch, gap,
        query_stride, ref_stride, dp_stride_array,
        BLOCK_SIZE=BLOCK_SIZE
    )

    end.record()
    torch.cuda.synchronize()
    kernel_time = start.elapsed_time(end)  # 单位：毫秒
    print(f"[Triton Inter-query Timing] Kernel time: {kernel_time:.3f} ms")

    # Gather results
    results = []
    for dp_tensor, m, n in zip(dp_tensors, m_list, n_list):
        max_score = torch.max(dp_tensor).item()
        max_pos = torch.argmax(dp_tensor.view(-1)).item()
        max_i, max_j = max_pos // (n + 1), max_pos % (n + 1)
        results.append((max_score, (max_i, max_j), dp_tensor.cpu().numpy()))
    return results