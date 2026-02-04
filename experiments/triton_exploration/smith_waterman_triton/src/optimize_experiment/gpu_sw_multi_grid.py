import torch
import triton
import triton.language as tl

@triton.jit
def sw_kernel_multi_grid(
    seq1_ptr, seq2_ptr, dp_ptr,
    m, n, match, mismatch, gap,
    seq1_stride, seq2_stride, dp_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    Experimental multi-grid Smith-Waterman kernel.
    Each grid block is assigned a contiguous block of diagonals.
    Pointer arithmetic is done in 64-bit to avoid overflow.
    WARNING: This version does NOT enforce inter-grid synchronization.
    """
    max_k = m + n  # Total number of diagonals.
    total_diags = max_k - 1  # (k from 2 to m+n)
    grid_num = tl.num_programs(0)
    diags_per_block = tl.cdiv(total_diags, grid_num)
    grid_id = tl.program_id(0)
    start_k = 2 + grid_id * diags_per_block
    end_k = min(max_k + 1, start_k + diags_per_block)
    
    tid = tl.arange(0, BLOCK_SIZE)

    for k in range(start_k, end_k):
        i_min = max(1, k - n)
        i_max = min(m, k - 1)
        num_valid = i_max - i_min + 1

        for chunk in range(0, num_valid, BLOCK_SIZE):
            idx = chunk + tid
            chunk_mask = idx < num_valid

            i = i_min + idx
            j = k - i

            # Load characters using 64-bit arithmetic.
            c1 = tl.load(seq1_ptr + tl.cast(i - 1, tl.int64) * tl.cast(seq1_stride, tl.int64),
                         mask=chunk_mask, other=0)
            c2 = tl.load(seq2_ptr + tl.cast(j - 1, tl.int64) * tl.cast(seq2_stride, tl.int64),
                         mask=chunk_mask, other=0)

            # Load DP dependencies.
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

def smith_waterman_gpu_multi_grid(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Compute Smith-Waterman alignment using an experimental multi-grid GPU kernel.
    (Pointer arithmetic is done in 64-bit.)
    WARNING: Results may be incorrect due to data dependencies and lack of global synchronization.
    """
    seq1_ascii = [ord(c) for c in seq1]
    seq2_ascii = [ord(c) for c in seq2]
    seq1_tensor = torch.tensor(seq1_ascii, device='cuda', dtype=torch.int32)
    seq2_tensor = torch.tensor(seq2_ascii, device='cuda', dtype=torch.int32)

    m, n = len(seq1), len(seq2)
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32, device='cuda')
    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv((m+n-1), meta['BLOCK_SIZE']),)
    sw_kernel_multi_grid[grid](
        seq1_tensor, seq2_tensor, dp,
        m, n, match, mismatch, gap,
        seq1_tensor.stride(0), seq2_tensor.stride(0), dp.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    max_score = torch.max(dp).item()
    max_pos = torch.argmax(dp.view(-1)).item()
    max_i, max_j = max_pos // (n + 1), max_pos % (n + 1)
    return max_score, (max_i, max_j), dp.cpu().numpy()