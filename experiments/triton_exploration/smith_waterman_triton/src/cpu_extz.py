import numpy as np
from Bio import SeqIO

# parameters
MATCH     = 1
MISMATCH  = -4
GAP_OPEN  = -6
GAP_EXT   = -2
ZDROP     = 751
BAND      = 400
N_PENALTY = -1
DNA_MAP = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 14}

def encode(seq):
    return [DNA_MAP.get(c.upper(), 0) for c in seq]

def smith_waterman_extz(query: str, target: str):
    q = encode(query)
    t = encode(target)
    m, n = len(q), len(t)

    gapo = -GAP_OPEN
    gape = -GAP_EXT
    gapoe = gapo + gape
    NEG_INF = -10**9

    H = [[NEG_INF]*(m+1) for _ in range(n+1)]
    E = [[NEG_INF]*(m+1) for _ in range(n+1)]
    F = [[NEG_INF]*(m+1) for _ in range(n+1)]

    H[0][0] = 0
    for j in range(1, m+1):
        if j <= BAND:
            H[0][j] = -(gapoe + gape*(j-1))
            E[0][j] = -(gapoe + gapoe + gape*j)

    for i in range(1, n+1):
        if i <= BAND:
            H[i][0] = -(gapoe + gape*(i-1))
            F[i][0] = -(gapoe + gapoe + gape*i)

    global_max = NEG_INF
    end_i = end_j = 0

    # not wavefront, using the row-major order
    for i in range(1, n+1):
        row_max = NEG_INF
        row_max_j = 0
        st = max(1, i - BAND)
        en = min(m, i + BAND)

        for j in range(st, en+1):
            E[i][j] = max(E[i][j-1] - gape, H[i][j-1] - gapoe)
            F[i][j] = max(F[i-1][j] - gape, H[i-1][j] - gapoe)

            if q[j-1] == 14 or t[i-1] == 14:
                sc = N_PENALTY
            else:
                sc = MATCH if q[j-1] == t[i-1] else MISMATCH

            h_diag = H[i-1][j-1] + sc
            H[i][j] = h = max(h_diag, E[i][j], F[i][j])

            if h > row_max:
                row_max = h
                row_max_j = j

            if h > global_max:
                global_max = h
                end_i, end_j = i, j

        delta = abs((i - end_i) - (row_max_j - end_j))
        if global_max - row_max > ZDROP + gape * delta:
            break

    return global_max, end_j, end_i

if __name__ == "__main__":
    query_seqs = list(SeqIO.parse("query1.fasta", "fasta"))
    ref_seqs = list(SeqIO.parse("ref1.fasta", "fasta"))

    if len(query_seqs) != len(ref_seqs):
        raise ValueError("query.fasta and ref.fasta must have the same number of sequences.")

    with open("alignment.log", "w") as f:
        for idx, (q_record, r_record) in enumerate(zip(query_seqs, ref_seqs), 1):
            q_seq = str(q_record.seq)
            r_seq = str(r_record.seq)

            score, q_end, r_end = smith_waterman_extz(q_seq, r_seq)
            f.write(f"score = {score}\ttarget_end = {r_end}\tquery_end = {q_end}\n")