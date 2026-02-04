"""
extz_cpu.codon

CPU implementation of extension + Z-drop (global alignment) algorithm.
Equivalent to Triton `gpu_extz.py` functionality:
- affine gap penalties (open + extend)
- banded alignment
- Z-drop early termination

Last Update Date  : 2025-05-26
"""

# ───────────────── Parameters ─────────────────
match: int = 1           # match score
mismatch: int = -4       # mismatch penalty
gapo: int = -6       # gap open penalty (negative)
gape: int = -2     # gap extension penalty (negative)
Z: int = 751         # Z-drop threshold
band: int = 400          # half-bandwidth for banded alignment

# Number of threads for parallel execution (adjust based on your CPU)
num_threads: int = 8

# ───────────────────── Utilities ─────────────────────
def dna_to_code(seq: str) -> list[int]:
    """
    Convert DNA string to integer codes: A=1, C=2, G=3, T=4, N=14, others=0.
    """
    code_map = {'A':1, 'C':2, 'G':3, 'T':4, 'N':14}
    return [code_map.get(base.upper(), 0) for base in seq]


def read_fasta(path: str) -> list[str]:
    """
    Read sequences from a FASTA file (ignores header lines).
    Returns a list of sequence strings.
    """
    sequences: list[str] = []
    buffer = ""
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if buffer:
                sequences.append(buffer)
                buffer = ""
        else:
            buffer += line
    if buffer:
        sequences.append(buffer)
    return sequences

# ───────────────── Alignment ─────────────────
def extz_align_cpu(query: list[int], ref: list[int]) -> tuple[int,int,int]:
    """
    Perform anti-diagonal global alignment with:
    - affine-gap penalties
    - fixed band constraint
    - Z-drop early termination based on best score trajectory

    Returns (best_score, end_i, end_j).
    """
    m, n = len(query), len(ref)
    NEG_INF = -10**6
    stride = 2 * band + 1

    # Initialize banded DP buffers:
    prev2H = [NEG_INF] * stride  # stores H(d-2)
    prevH  = [NEG_INF] * stride  # stores H(d-1)
    prevE  = [NEG_INF] * stride  # stores E(d-1)
    prevF  = [NEG_INF] * stride  # stores F(d-1)
    currH  = [NEG_INF] * stride  # stores H(d)
    currE  = [NEG_INF] * stride  # stores E(d)
    currF  = [NEG_INF] * stride  # stores F(d)

    # === Initialization for d=0 and d=1 ===
    prev2H[band] = 0                           # H(0,0)
    prevH[0]     = gapo + gape      # H(1,0)
    prevF[0]     = gapo                   # F(1,0)
    # prevE remains NEG_INF for d=1

    best_score = 0
    best_i = 0
    best_j = 0
    prev_lo  = 1
    prev2_lo = 0

    # Loop over anti-diagonals d = 2 .. m+n
    for d in range(2, m + n + 1):
        # valid i range for diagonal
        i_min = max(1, d - n)
        i_max = min(m, d - 1)

        # slide band center according to best_i - best_j
        center_offset = best_i - best_j
        lo = (d + center_offset - band + 1) // 2
        hi = (d + center_offset + band) // 2
        lo = max(lo, i_min)
        hi = min(hi, i_max)

        # scan cells on this diagonal
        for i in range(lo, hi + 1):
            j = d - i
            q_code = query[i-1]
            r_code = ref[j-1]
            # substitution score (N treated as mismatch)
            if q_code == 14 or r_code == 14:
                sub_score = -1
            else:
                sub_score = match if q_code == r_code else mismatch

            # indices into band buffers
            idx_prev = i - prev_lo
            idx_up   = idx_prev - 1
            idx_diag = i - 1 - prev2_lo

            # H from left (i, j-1)
            if j == 1:
                Hleft = gapo + gape * (i - 1)
            elif 0 <= idx_prev < stride:
                Hleft = prevH[idx_prev]
            else:
                Hleft = NEG_INF
            Eleft = prevE[idx_prev] if 0 <= idx_prev < stride else NEG_INF

            # H from up (i-1, j)
            if i == 1:
                Hup = gapo + gape * (j - 1)
            elif 0 <= idx_up < stride:
                Hup = prevH[idx_up]
            else:
                Hup = NEG_INF
            Fup = prevF[idx_up] if 0 <= idx_up < stride else NEG_INF

            # H from diagonal (i-1, j-1)
            if i == 1 and j == 1:
                Hdiag = 0
            elif i == 1:
                Hdiag = gapo + gape * (j - 1)
            elif j == 1:
                Hdiag = gapo + gape * (i - 1)
            elif 0 <= idx_diag < stride:
                Hdiag = prev2H[idx_diag]
            else:
                Hdiag = NEG_INF

            # affine-gap calculation
            open_ext = gapo + gape
            Eval = max(Eleft + gape, Hleft + open_ext)
            Fval = max(Fup    + gape, Hup    + open_ext)

            # final H
            Hval = max(Hdiag + sub_score, Eval, Fval)

            # store into curr buffers
            buf_idx = i - lo
            currH[buf_idx] = Hval
            currE[buf_idx] = Eval
            currF[buf_idx] = Fval

            # update best
            if Hval > best_score:
                best_score = Hval
                best_i = i
                best_j = j

            # Z-drop early exit
            delta = (i - j) - (best_i - best_j)
            if i >= best_i and j >= best_j:
                penalty = Z + abs(gape) * abs(delta)
                if best_score - Hval > penalty:
                    return best_score, best_i, best_j

        # rotate buffers for next diagonal
        prev2H, prevH, currH = prevH, currH, [NEG_INF] * stride
        prevF, currF = currF, [NEG_INF] * stride
        prevE, currE = currE, [NEG_INF] * stride
        prev2_lo, prev_lo = prev_lo, lo

    return best_score, best_i, best_j

# ───────────────── Main & Parallel Loop ─────────────────
queries = read_fasta("datasets/query_odd.fa")
refs    = read_fasta("datasets/ref_odd.fa")

if len(queries) != len(refs):
    print(f"[Error] query count ({len(queries)}) != ref count ({len(refs)})")
else:
    # parallel over all pairs
    # @par(schedule="dynamic", chunk_size=1, num_threads=num_threads)
    for idx in range(len(queries)):
        q_codes = dna_to_code(queries[idx])
        r_codes = dna_to_code(refs[idx])
        score, end_i, end_j = extz_align_cpu(q_codes, r_codes)
        print(f"[Codon CPU ExtZ] Pair {idx+1}: score={score}, end_i={end_i}, end_j={end_j}")
