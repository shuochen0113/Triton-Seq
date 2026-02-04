import torch
import triton
import triton.language as tl


@triton.jit
def sw_kernel_single_block(
    seq1_ptr, seq2_ptr, dp_ptr,
    m, n, match, mismatch, gap,
    seq1_stride, seq2_stride, dp_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    Smith-Waterman kernel using a single launch with an inner loop over all diagonals.

    Parameters:
      seq1_ptr (pointer): Pointer to first sequence in GPU memory.
      seq2_ptr (pointer): Pointer to second sequence in GPU memory.
      dp_ptr (pointer): Pointer to the DP matrix in GPU memory.
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
    max_k = m + n # total number of diagonals
    tid = tl.arange(0, BLOCK_SIZE)

    for k in range(2, max_k + 1):
        i_min = max(1, k - n)
        i_max = min(m, k - 1)
        num_valid = i_max - i_min + 1

        for chunk in range(0, num_valid, BLOCK_SIZE):
            idx = chunk + tid
            chunk_mask = idx < num_valid

            i = i_min + idx
            j = k - i

            c1 = tl.load(seq1_ptr + tl.cast(i - 1, tl.int64) * tl.cast(seq1_stride, tl.int64),
                         mask=chunk_mask, other=0)
            c2 = tl.load(seq2_ptr + tl.cast(j - 1, tl.int64) * tl.cast(seq2_stride, tl.int64),
                         mask=chunk_mask, other=0)

            diag = tl.load(dp_ptr + tl.cast(i - 1, tl.int64) * tl.cast(dp_stride, tl.int64) +
                           tl.cast(j - 1, tl.int64), mask=chunk_mask, other=0)
            up   = tl.load(dp_ptr + tl.cast(i - 1, tl.int64) * tl.cast(dp_stride, tl.int64) +
                           tl.cast(j, tl.int64), mask=chunk_mask, other=0)
            left = tl.load(dp_ptr + tl.cast(i, tl.int64) * tl.cast(dp_stride, tl.int64) +
                           tl.cast(j - 1, tl.int64), mask=chunk_mask, other=0)

            current = diag + tl.where(c1 == c2, match, mismatch)
            current = tl.maximum(current, up + gap)
            current = tl.maximum(current, left + gap)
            current = tl.maximum(current, 0)

            tl.store(dp_ptr + tl.cast(i, tl.int64) * tl.cast(dp_stride, tl.int64) +
                     tl.cast(j, tl.int64), current, mask=chunk_mask)
        
        tl.debug_barrier()


def smith_waterman_gpu_single_block(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Compute Smith-Waterman alignment using a single-launch GPU kernel.

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

    # Initialize the DP matrix.
    m, n = len(seq1), len(seq2)
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32, device='cuda')

    # Define block size.
    BLOCK_SIZE = 256

    # Launch kernel with a single grid block.
    grid = (1,)
    sw_kernel_single_block[grid](
        seq1_tensor, seq2_tensor, dp,
        m, n, match, mismatch, gap,
        seq1_tensor.stride(0), seq2_tensor.stride(0), dp.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Extract the maximum score and its position.
    max_score = torch.max(dp).item()
    max_pos = torch.argmax(dp.view(-1)).item()
    max_i, max_j = max_pos // (n + 1), max_pos % (n + 1)

    return max_score, (max_i, max_j), dp.cpu().numpy()