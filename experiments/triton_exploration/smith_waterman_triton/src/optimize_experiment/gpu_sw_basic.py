import torch
import triton
import triton.language as tl


@triton.jit
def sw_kernel_basic(
    seq1_ptr, seq2_ptr, dp_ptr,
    k, m, n,
    match, mismatch, gap,
    seq1_stride, seq2_stride, dp_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    Basic Smith-Waterman kernel that computes one diagonal in parallel.
    (Pointer arithmetic is done in 64-bit to avoid overflow on large matrices.)

    Parameters:
      seq1_ptr (pointer): Pointer to first sequence in GPU memory.
      seq2_ptr (pointer): Pointer to second sequence in GPU memory.
      dp_ptr (pointer): Pointer to the DP matrix in GPU memory.
      k (int): Current diagonal index.
      m (int): Length of sequence 1.
      n (int): Length of sequence 2.
      match (int): Score for a match.
      mismatch (int): Score for a mismatch.
      gap (int): Gap penalty.
      seq1_stride (int): Stride for accessing sequence 1.
      seq2_stride (int): Stride for accessing sequence 2.
      dp_stride (int): Stride for the DP matrix.
      BLOCK_SIZE (int): Number of threads per block.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Determine the valid range for the current diagonal.
    i_min = max(1, k - n)
    i_max = min(m, k - 1)
    num_valid = i_max - i_min + 1

    # Create a mask for valid indices.
    mask = idx < num_valid
    i = i_min + idx
    j = k - i
    mask &= (i >= 1) & (i <= m) & (j >= 1) & (j <= n)

    # Load the sequence elements using 64-bit arithmetic for offsets.
    c1 = tl.load(seq1_ptr + tl.cast(i - 1, tl.int64) * tl.cast(seq1_stride, tl.int64),
                 mask=mask, other=0)
    c2 = tl.load(seq2_ptr + tl.cast(j - 1, tl.int64) * tl.cast(seq2_stride, tl.int64),
                 mask=mask, other=0)

    # Load DP matrix dependencies with 64-bit offsets.
    diag = tl.load(dp_ptr + tl.cast(i - 1, tl.int64) * tl.cast(dp_stride, tl.int64) +
                   tl.cast(j - 1, tl.int64), mask=mask, other=0)
    up   = tl.load(dp_ptr + tl.cast(i - 1, tl.int64) * tl.cast(dp_stride, tl.int64) +
                   tl.cast(j, tl.int64), mask=mask, other=0)
    left = tl.load(dp_ptr + tl.cast(i, tl.int64) * tl.cast(dp_stride, tl.int64) +
                   tl.cast(j - 1, tl.int64), mask=mask, other=0)

    # Compute the current DP cell.
    current = diag + tl.where(c1 == c2, match, mismatch)
    current = tl.maximum(current, up + gap)
    current = tl.maximum(current, left + gap)
    current = tl.maximum(current, 0)

    # Store the computed value back.
    tl.store(dp_ptr + tl.cast(i, tl.int64) * tl.cast(dp_stride, tl.int64) +
             tl.cast(j, tl.int64), current, mask=mask)


def smith_waterman_gpu_basic(seq1, seq2, match=, mismatch=-1, gap=-1):
    """
    Compute Smith-Waterman alignment using the basic GPU kernel.

    Parameters:
      seq1 (str): First sequence.
      seq2 (str): Second sequence.
      match (int): Score for a match.
      mismatch (int): Score for a mismatch.
      gap (int): Gap penalty.

    Returns:
      tuple: (max_score, (max_i, max_j), dp_matrix as a NumPy array)
    """
    # Convert sequences to ASCII codes.
    seq1_ascii = [ord(c) for c in seq1]
    seq2_ascii = [ord(c) for c in seq2]

    # Transfer sequences to GPU.
    seq1_tensor = torch.tensor(seq1_ascii, device='cuda', dtype=torch.int32)
    seq2_tensor = torch.tensor(seq2_ascii, device='cuda', dtype=torch.int32)
    m, n = len(seq1), len(seq2)

    # Initialize the DP matrix.
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32, device='cuda')

    # Process each diagonal (k=0 and k=1 are already zero-initialized).
    max_k = m + n
    BLOCK_SIZE = 256

    for k in range(2, max_k + 1):
        i_min = max(1, k - n)
        i_max = min(m, k - 1)
        if i_min > i_max:
            continue

        num_i = i_max - i_min + 1
        grid = lambda meta: (triton.cdiv(num_i, meta['BLOCK_SIZE']),)
        sw_kernel_basic[grid](
            seq1_tensor, seq2_tensor, dp, k,
            m, n, match, mismatch, gap,
            seq1_tensor.stride(0), seq2_tensor.stride(0), dp.stride(0),
            BLOCK_SIZE=BLOCK_SIZE
        )

    # Extract the maximum score and its position.
    max_score = torch.max(dp).item()
    max_pos = torch.argmax(dp.view(-1)).item()
    max_i, max_j = max_pos // (n + 1), max_pos % (n + 1)

    return max_score, (max_i, max_j), dp.cpu().numpy()