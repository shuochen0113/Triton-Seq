import torch
import triton
import triton.language as tl

def convert_fasta_to_ascii(sequence):
    """Convert a DNA sequence into ASCII integer representation."""
    return [ord(c) if c != 'N' else -1 for c in sequence]  # 'N' as -1 for fuzzy matching

@triton.jit
def sw_kernel_single_block(
    seq1_ptr, seq2_ptr, dp_ptr,
    m, n, match, mismatch, gap, n_score,
    seq1_stride, seq2_stride, dp_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    Smith-Waterman kernel using a single launch with an inner loop over all diagonals.
    Supports 'N' as a fuzzy matching character.
    """
    max_k = m + n
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

            c1 = tl.load(seq1_ptr + (i - 1) * seq1_stride, mask=chunk_mask, other=0)
            c2 = tl.load(seq2_ptr + (j - 1) * seq2_stride, mask=chunk_mask, other=0)

            diag = tl.load(dp_ptr + (i - 1) * dp_stride + (j - 1), mask=chunk_mask, other=0)
            up = tl.load(dp_ptr + (i - 1) * dp_stride + j, mask=chunk_mask, other=0)
            left = tl.load(dp_ptr + i * dp_stride + (j - 1), mask=chunk_mask, other=0)

            match_penalty = tl.where((c1 == -1) | (c2 == -1), n_score, 
                                     tl.where(c1 == c2, match, mismatch))
            
            current = diag + match_penalty
            current = tl.maximum(current, up + gap)
            current = tl.maximum(current, left + gap)
            current = tl.maximum(current, 0)

            tl.store(dp_ptr + i * dp_stride + j, current, mask=chunk_mask)
        
        tl.debug_barrier()

def smith_waterman_gpu_single_block(seq1, seq2, match=2, mismatch=-1, gap=-1, n_score=0):
    """
    Compute Smith-Waterman alignment using a single-launch GPU kernel.
    Supports 'N' as a fuzzy matching character.
    """
    seq1_ascii = convert_fasta_to_ascii(seq1)
    seq2_ascii = convert_fasta_to_ascii(seq2)

    seq1_tensor = torch.tensor(seq1_ascii, device='cuda', dtype=torch.int32)
    seq2_tensor = torch.tensor(seq2_ascii, device='cuda', dtype=torch.int32)

    m, n = len(seq1), len(seq2)
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32, device='cuda')

    BLOCK_SIZE = 256
    grid = (1,)
    
    sw_kernel_single_block[grid](
        seq1_tensor, seq2_tensor, dp,
        m, n, match, mismatch, gap, n_score,
        seq1_tensor.stride(0), seq2_tensor.stride(0), dp.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    max_score = torch.max(dp).item()
    max_pos = torch.argmax(dp.view(-1)).item()
    max_i, max_j = max_pos // (n + 1), max_pos % (n + 1)

    return max_score, (max_i, max_j), dp.cpu().numpy()
