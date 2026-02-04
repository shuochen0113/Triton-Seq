import torch
import triton
import triton.language as tl

@triton.jit
def sw_kernel_inter_query_row(
    query_ptrs, ref_ptrs,
    dp_prev_ptrs, dp_curr_ptrs,
    m_array, n_array,
    match, mismatch, gap,
    out_ptrs,
    BLOCK_SIZE: tl.constexpr
):
    # Each block handles one query-reference pair
    pid = tl.program_id(0)

    # Load lengths of the current query and reference sequence
    m = tl.load(m_array + pid)
    n = tl.load(n_array + pid)

    # Load pointers to the current query, reference, and DP buffers
    q_ptr = tl.load(query_ptrs + pid).to(tl.pointer_type(tl.int32))
    r_ptr = tl.load(ref_ptrs + pid).to(tl.pointer_type(tl.int32))
    dp_prev = tl.load(dp_prev_ptrs + pid).to(tl.pointer_type(tl.int32))
    dp_curr = tl.load(dp_curr_ptrs + pid).to(tl.pointer_type(tl.int32))
    out_ptr = tl.load(out_ptrs + pid).to(tl.pointer_type(tl.int32))

    # Initialize dp_prev row to 0: corresponds to DP[i=0][*]
    for j in range(0, n + 1):
        j32 = tl.full((), j, tl.int32)
        tl.store(dp_prev + j32, 0)

    # Track the best alignment score and its coordinates
    max_val = 0
    max_i = 0
    max_j = 0

    # Row-wise DP calculation using two-row rolling buffers
    for i in range(1, m + 1):
        # Set dp_curr[0] = 0 (DP[i][0] = 0)
        tl.store(dp_curr + 0, 0)
        for j in range(1, n + 1):
            # Load neighboring cells
            left = tl.load(dp_curr + (j - 1))      # DP[i][j-1]
            up   = tl.load(dp_prev + j)            # DP[i-1][j]
            diag = tl.load(dp_prev + (j - 1))      # DP[i-1][j-1]

            # Load characters from query and reference
            c1 = tl.load(q_ptr + (i - 1))
            c2 = tl.load(r_ptr + (j - 1))

            # Score match or mismatch
            score = tl.where(c1 == c2, match, mismatch)

            # Compute new DP[i][j] cell
            val = diag + score
            val = tl.maximum(val, up + gap)
            val = tl.maximum(val, left + gap)
            val = tl.maximum(val, 0)

            tl.store(dp_curr + j, val)

            # Update max if needed
            if val > max_val:
                max_val = val
                max_i = i
                max_j = j

        # Copy current row to previous row for next iteration
        for j in range(0, n + 1):
            val = tl.load(dp_curr + j)
            tl.store(dp_prev + j, val)

    # Output: [max_score, max_i, max_j]
    tl.store(out_ptr + 0, max_val)
    tl.store(out_ptr + 1, max_i)
    tl.store(out_ptr + 2, max_j)


def smith_waterman_gpu_inter_query_row(query_list, ref_list, match=3, mismatch=-2, gap=-1):
    num_pairs = len(query_list)
    assert num_pairs == len(ref_list), "Query and reference list lengths must match."

    query_tensors, ref_tensors = [], []
    m_list, n_list = [], []

    # Encode sequences as int32 tensors (ASCII-based)
    for q_seq, r_seq in zip(query_list, ref_list):
        q_ascii = [ord(c) for c in q_seq]
        r_ascii = [ord(c) for c in r_seq]
        query_tensors.append(torch.tensor(q_ascii, device='cuda', dtype=torch.int32))
        ref_tensors.append(torch.tensor(r_ascii, device='cuda', dtype=torch.int32))
        m_list.append(len(q_seq))
        n_list.append(len(r_seq))

    # Allocate rolling DP row buffers (prev & curr) and output buffers
    dp_prev_list = [torch.zeros(n + 1, dtype=torch.int32, device='cuda') for n in n_list]
    dp_curr_list = [torch.zeros(n + 1, dtype=torch.int32, device='cuda') for n in n_list]
    out_list = [torch.zeros(3, dtype=torch.int32, device='cuda') for _ in range(num_pairs)]

    # Pack pointer arrays
    query_ptrs = torch.tensor([t.data_ptr() for t in query_tensors], dtype=torch.int64, device='cuda')
    ref_ptrs   = torch.tensor([t.data_ptr() for t in ref_tensors], dtype=torch.int64, device='cuda')
    dp_prev_ptrs = torch.tensor([x.data_ptr() for x in dp_prev_list], dtype=torch.int64, device='cuda')
    dp_curr_ptrs = torch.tensor([x.data_ptr() for x in dp_curr_list], dtype=torch.int64, device='cuda')
    out_ptrs = torch.tensor([x.data_ptr() for x in out_list], dtype=torch.int64, device='cuda')
    m_arr = torch.tensor(m_list, dtype=torch.int32, device='cuda')
    n_arr = torch.tensor(n_list, dtype=torch.int32, device='cuda')

    # Launch grid: one block per sequence pair
    grid = (num_pairs,)

    # Run the kernel
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    sw_kernel_inter_query_row[grid](
        query_ptrs, ref_ptrs,
        dp_prev_ptrs, dp_curr_ptrs,
        m_arr, n_arr,
        match, mismatch, gap,
        out_ptrs,
        BLOCK_SIZE=256
    )
    end.record()
    torch.cuda.synchronize()
    ktime = start.elapsed_time(end)
    print(f"[Triton Inter-query Row Timing] Kernel time: {ktime:.3f} ms")

    # Collect results: [(score, (i, j)), ...]
    results = []
    for out in out_list:
        s, i, j = out.cpu().tolist()
        results.append((s, (i, j)))
    return results