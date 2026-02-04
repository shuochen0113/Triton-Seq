import torch
import triton
import triton.language as tl

@triton.jit
def sw_kernel_single_block_packed(
    seq1_ptr, seq2_ptr, dp_ptr,
    m, n, match, mismatch, gap,
    seq1_stride, seq2_stride, dp_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    Single-launch Smith-Waterman kernel with packed input.
    The sequences are packed: 8 bases per 32-bit word, with each base stored in 4 bits.
    """
    max_k = m + n  # Total number of diagonals
    tid = tl.arange(0, BLOCK_SIZE)

    for k in range(2, max_k + 1):
        i_min = tl.maximum(1, k - n)
        i_max = tl.minimum(m, k - 1)
        num_valid = i_max - i_min + 1

        for chunk in range(0, num_valid, BLOCK_SIZE):
            idx = chunk + tid
            chunk_mask = idx < num_valid

            i = i_min + idx
            j = k - i

            # Decode base from packed sequence for seq1 (index i-1)
            i_minus = i - 1
            # Compute floor division using tl.floor; cast back to int32
            word_idx1 = tl.cast(tl.floor(i_minus / 8), tl.int32)
            offset1 = i_minus - word_idx1 * 8
            packed_word1 = tl.load(seq1_ptr + word_idx1 * seq1_stride, mask=chunk_mask, other=0)
            c1 = (packed_word1 >> (offset1 * 4)) & 15

            # Decode base for seq2 (index j-1)
            j_minus = j - 1
            word_idx2 = tl.cast(tl.floor(j_minus / 8), tl.int32)
            offset2 = j_minus - word_idx2 * 8
            packed_word2 = tl.load(seq2_ptr + word_idx2 * seq2_stride, mask=chunk_mask, other=0)
            c2 = (packed_word2 >> (offset2 * 4)) & 15

            diag = tl.load(dp_ptr + (i - 1) * dp_stride + (j - 1), mask=chunk_mask, other=0)
            up   = tl.load(dp_ptr + (i - 1) * dp_stride + j, mask=chunk_mask, other=0)
            left = tl.load(dp_ptr + i * dp_stride + (j - 1), mask=chunk_mask, other=0)

            current = diag + tl.where(c1 == c2, match, mismatch)
            current = tl.maximum(current, up + gap)
            current = tl.maximum(current, left + gap)
            current = tl.maximum(current, 0)
            tl.store(dp_ptr + i * dp_stride + j, current, mask=chunk_mask)
        
        tl.debug_barrier()

def smith_waterman_gpu_single_block_packed(seq1, seq2, match=3, mismatch=-2, gap=-1):
    """
    Compute Smith-Waterman alignment using a single-launch GPU kernel
    with packed input sequences.
    """
    def pack_sequence(seq):
        # Mapping: A->0, C->1, G->2, T->3. (Adjust if necessary.)
        mapping = {ord('A'): 0, ord('C'): 1, ord('G'): 2, ord('T'): 3}
        packed = 0
        result = []
        for i, c in enumerate(seq):
            base = mapping.get(ord(c), 0)
            # Shift and pack base into the current 32-bit word.
            packed |= base << (4 * (i % 8))
            if (i + 1) % 8 == 0:
                result.append(packed)
                packed = 0
        if len(seq) % 8 != 0:
            result.append(packed)
        return result

    packed_seq1 = pack_sequence(seq1)
    packed_seq2 = pack_sequence(seq2)
    seq1_tensor = torch.tensor(packed_seq1, device='cuda', dtype=torch.int32)
    seq2_tensor = torch.tensor(packed_seq2, device='cuda', dtype=torch.int32)
    
    m, n = len(seq1), len(seq2)
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32, device='cuda')
    BLOCK_SIZE = 256

    grid = (1,)
    sw_kernel_single_block_packed[grid](
        seq1_tensor, seq2_tensor, dp,
        m, n, match, mismatch, gap,
        seq1_tensor.stride(0), seq2_tensor.stride(0), dp.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    max_score = torch.max(dp).item()
    max_pos = torch.argmax(dp.view(-1)).item()
    max_i, max_j = max_pos // (n + 1), max_pos % (n + 1)
    return max_score, (max_i, max_j), dp.cpu().numpy()