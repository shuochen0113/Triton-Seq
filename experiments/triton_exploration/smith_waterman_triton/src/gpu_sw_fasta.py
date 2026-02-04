import torch
import triton
import triton.language as tl

@triton.jit
def sw_kernel(
    seq1_ptr, seq2_ptr, dp_ptr,
    k, m, n,
    match, mismatch, gap, n_score,
    seq1_stride, seq2_stride, dp_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    i_min = max(1, k - n)
    i_max = min(m, k - 1)
    num_valid = i_max - i_min + 1

    mask = idx < num_valid
    i = i_min + idx
    j = k - i

    valid = (i >= 1) & (i <= m) & (j >= 1) & (j <= n)
    mask &= valid

    c1 = tl.load(seq1_ptr + (i-1) * seq1_stride, mask=mask, other=0)
    c2 = tl.load(seq2_ptr + (j-1) * seq2_stride, mask=mask, other=0)

    diag = tl.load(dp_ptr + (i-1) * dp_stride + (j-1), mask=mask, other=0)
    up = tl.load(dp_ptr + (i-1) * dp_stride + j, mask=mask, other=0)
    left = tl.load(dp_ptr + i * dp_stride + (j-1), mask=mask, other=0)

    # Transfer 'N' into ASCII code
    match_penalty = tl.where((c1 == 78) | (c2 == 78), n_score, tl.where(c1 == c2, match, mismatch))

    current = diag + match_penalty
    current = max(current, up + gap)
    current = max(current, left + gap)
    current = max(current, 0)

    tl.store(dp_ptr + i * dp_stride + j, current, mask=mask)

def smith_waterman_gpu(seq1, seq2, match=2, mismatch=-1, gap=-1, n_score=0):
    seq1_ascii = [ord(c) for c in seq1]
    seq2_ascii = [ord(c) for c in seq2]

    seq1_tensor = torch.tensor(seq1_ascii, device='cuda', dtype=torch.int32)
    seq2_tensor = torch.tensor(seq2_ascii, device='cuda', dtype=torch.int32)
    m, n = len(seq1), len(seq2)
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32, device='cuda')

    max_k = m + n
    for k in range(2, max_k + 1):
        i_min = max(1, k - n)
        i_max = min(m, k - 1)
        if i_min > i_max:
            continue

        num_i = i_max - i_min + 1

        grid = lambda meta: (triton.cdiv(num_i, meta['BLOCK_SIZE']),)
        sw_kernel[grid](
            seq1_tensor, seq2_tensor, dp, k,
            m, n, match, mismatch, gap, n_score,
            seq1_tensor.stride(0), seq2_tensor.stride(0), dp.stride(0),
            BLOCK_SIZE=256
        )

    max_score = torch.max(dp).item()
    max_pos = torch.argmax(dp.view(-1)).item()
    max_i, max_j = max_pos // (n+1), max_pos % (n+1)

    return max_score, (max_i, max_j), dp.cpu().numpy()