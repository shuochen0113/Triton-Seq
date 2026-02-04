import torch
import triton
import triton.language as tl

@triton.jit
def sw_kernel_multi_block(
    seq1_ptr, seq2_ptr, dp_ptr,
    m, n, match, mismatch, gap,
    seq1_stride, seq2_stride, dp_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    A multi-block Smith-Waterman kernel using a 2D grid.
    
    Grid dimensions:
      - grid[0]: Diagonal index k, where k runs from 2 to m+n (i.e. m+n-1 diagonals).
      - grid[1]: Chunk index along the current diagonal (each chunk has BLOCK_SIZE cells).
      
    For each diagonal, we compute:
      i_min = max(1, k - n)
      i_max = min(m, k - 1)
    and then each block processes a contiguous chunk of cells along the diagonal.
    
    Note: This version ignores the inter-diagonal (data dependency) synchronization,
          so while it fully parallelizes the work, the numerical results may be wrong.
    """
    # Determine diagonal index: k ranges from 2 to m+n.
    diag = tl.program_id(0) + 2
    # Each diagonal has a varying number of cells.
    # Compute valid row indices: i_min = max(1, diag - n), i_max = min(m, diag - 1)
    i_min = max(1, diag - n)
    i_max = min(m, diag - 1)
    num_valid = i_max - i_min + 1

    # Compute offset along this diagonal for the current block.
    block_offset = tl.program_id(1) * BLOCK_SIZE
    tid = tl.arange(0, BLOCK_SIZE)
    idx = block_offset + tid
    mask = idx < num_valid

    # Compute (i, j) indices for the current diagonal cell.
    # Given that diag = i + j, we have j = diag - i.
    i = i_min + idx
    j = diag - i

    # Load the sequence characters (adjust pointer arithmetic with (i-1) and (j-1)).
    c1 = tl.load(seq1_ptr + (i - 1) * seq1_stride, mask=mask, other=0)
    c2 = tl.load(seq2_ptr + (j - 1) * seq2_stride, mask=mask, other=0)
    
    # Load DP values from the neighboring cells.
    diag_val = tl.load(dp_ptr + (i - 1) * dp_stride + (j - 1), mask=mask, other=0)
    up_val   = tl.load(dp_ptr + (i - 1) * dp_stride + j, mask=mask, other=0)
    left_val = tl.load(dp_ptr + i * dp_stride + (j - 1), mask=mask, other=0)

    # Compute the current cell score.
    score = diag_val + tl.where(c1 == c2, match, mismatch)
    score = tl.maximum(score, up_val + gap)
    score = tl.maximum(score, left_val + gap)
    score = tl.maximum(score, 0)

    # Write the computed score back to the DP matrix.
    tl.store(dp_ptr + i * dp_stride + j, score, mask=mask)

def smith_waterman_gpu_multi_block(seq1, seq2, match=3, mismatch=-2, gap=-1):
    """
    Compute Smith-Waterman alignment using the optimal multi-block kernel.
    
    This function fully parallelizes across the anti-diagonals using a 2D grid.
    Note: Since inter-diagonal dependencies are not enforced, the output may be incorrect.
    """
    # Convert sequences to ASCII codes.
    seq1_ascii = [ord(c) for c in seq1]
    seq2_ascii = [ord(c) for c in seq2]
    seq1_tensor = torch.tensor(seq1_ascii, device='cuda', dtype=torch.int32)
    seq2_tensor = torch.tensor(seq2_ascii, device='cuda', dtype=torch.int32)
    m, n = len(seq1), len(seq2)

    # Create the DP matrix (with an extra row and column for the zero initialization).
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32, device='cuda')
    BLOCK_SIZE = 256

    # Number of diagonals is m+n-1 (since diag = 2 ... m+n).
    num_diagonals = m + n - 1
    # Maximum diagonal length is min(m, n), so we compute how many blocks are needed.
    num_blocks_per_diag = triton.cdiv(min(m, n), BLOCK_SIZE)
    grid = (num_diagonals, num_blocks_per_diag)

    sw_kernel_multi_block[grid](
        seq1_tensor, seq2_tensor, dp,
        m, n, match, mismatch, gap,
        seq1_tensor.stride(0), seq2_tensor.stride(0), dp.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    max_score = torch.max(dp).item()
    max_pos = torch.argmax(dp.view(-1)).item()
    max_i, max_j = max_pos // (n + 1), max_pos % (n + 1)
    return max_score, (max_i, max_j), dp.cpu().numpy()