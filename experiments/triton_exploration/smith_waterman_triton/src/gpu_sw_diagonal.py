import torch
import triton
import triton.language as tl


@triton.jit
def sw_kernel_diagonal(
    seq1_ptr, seq2_ptr, dp_ptr,
    k, m, n,
    match, mismatch, gap,
    seq1_stride, seq2_stride, dp_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    Smith-Waterman kernel that computes one diagonal in parallel.

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
    # Get the thread index range within the block
    tid = tl.arange(0, BLOCK_SIZE)

    # Determine the valid range for the current diagonal.
    i_min = max(1, k - n)
    i_max = min(m, k - 1)
    num_valid = i_max - i_min + 1

    offset = 0
    while offset < num_valid:
        # Compute current chunk indices.
        cur_idx = offset + tid
        mask = cur_idx < num_valid

        # Compute actual indices.
        i = i_min + cur_idx
        j = k - i
        mask &= (i >= 1) & (i <= m) & (j >= 1) & (j <= n)

        # Load sequence elements.
        c1 = tl.load(seq1_ptr + (i - 1) * seq1_stride, mask=mask, other=0)
        c2 = tl.load(seq2_ptr + (j - 1) * seq2_stride, mask=mask, other=0)

        # Load DP matrix dependencies.
        diag = tl.load(dp_ptr + (i - 1) * dp_stride + (j - 1), mask=mask, other=0)
        up = tl.load(dp_ptr + (i - 1) * dp_stride + j, mask=mask, other=0)
        left = tl.load(dp_ptr + i * dp_stride + (j - 1), mask=mask, other=0)

        # Compute the current DP cell using Smith-Waterman recurrence.
        current = diag + tl.where(c1 == c2, match, mismatch)
        current = tl.maximum(current, up + gap)
        current = tl.maximum(current, left + gap)
        current = tl.maximum(current, 0)

        # Store the computed value back into the DP matrix.
        tl.store(dp_ptr + i * dp_stride + j, current, mask=mask)

        offset += BLOCK_SIZE


def smith_waterman_gpu_diagonal(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Compute Smith-Waterman alignment using the diagonal-based GPU kernel.

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

    max_k = m + n
    BLOCK_SIZE = 256

    # Process each diagonal.
    for k in range(2, max_k + 1):
        i_min = max(1, k - n)
        i_max = min(m, k - 1)
        if i_min > i_max:
            continue

        grid = (1,)  # Single block handles the entire diagonal.
        sw_kernel_diagonal[grid](
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