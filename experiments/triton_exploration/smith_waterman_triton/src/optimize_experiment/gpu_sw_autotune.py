import torch
import triton
import triton.language as tl

# Auto-tune the BLOCK_SIZE
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['BLOCK_SIZE']
)
@triton.jit
def sw_kernel_single_block(
    seq1_ptr, seq2_ptr, dp_ptr,
    m, n, match, mismatch, gap,
    seq1_stride, seq2_stride, dp_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    Single-launch Smith-Waterman kernel with auto-tuning over BLOCK_SIZE.
    """
    max_k = m + n
    tid = tl.arange(0, BLOCK_SIZE)
    print("Current BLOCK_SIZE: ", BLOCK_SIZE)

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

def smith_waterman_gpu_single_block(seq1, seq2, match=3, mismatch=-2, gap=-1):
    """
    Compute Smith-Waterman alignment using an auto-tuned GPU kernel.
    """
    seq1_ascii = [ord(c) for c in seq1]
    seq2_ascii = [ord(c) for c in seq2]
    seq1_tensor = torch.tensor(seq1_ascii, device='cuda', dtype=torch.int32)
    seq2_tensor = torch.tensor(seq2_ascii, device='cuda', dtype=torch.int32)
    m, n = len(seq1), len(seq2)

    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32, device='cuda')

    grid = (1,)
    # Run the kernel with auto-tuning enabled
    kernel = sw_kernel_single_block[grid]
    kernel(
        seq1_tensor, seq2_tensor, dp,
        m, n, match, mismatch, gap,
        seq1_tensor.stride(0), seq2_tensor.stride(0), dp.stride(0)
    )

    # Extract the best-tuned configuration
    # best_config = kernel.config

    max_score = torch.max(dp).item()
    max_pos = torch.argmax(dp.view(-1)).item()
    max_i, max_j = max_pos // (n + 1), max_pos % (n + 1)

    return max_score, (max_i, max_j), dp.cpu().numpy()