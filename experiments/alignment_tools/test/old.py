import numpy as np

# parameters
MATCH = 1
MISMATCH = -4
GAP_OPEN = -6
GAP_EXT = -2
Z = 751
BAND = 400
N_PENALTY = -1  # penalty for 'N', same as AGAThA
DNA_MAP = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 14}

query = "CCTATACGCTAATCATCAAAATCAGCCCATACACATTAACACACCGGCCGGACAGGGGCAACCTTGCTCCAAAGGCAGCAGCCACCAAAGGAGCTGGGGCGGAGAATCTAGCCAGCGAGAGGGCGGGGACTGGCATTCCCAG"
ref   = "CCTATACGCTAAATCATCTAAAACAGCCCATACACATTAACACACCGGCGGCAGCGGCAACCTTGCTCCAGAAGGGCAGCGAGCCCACCAAGGGAGCTGGGGCGGAGAATGATAGCCAGCGAAGAGGGCGGGACTGGCGATTCCCAGCAAGCAGATAAAAGATAGGGGAATTAAAATGGAAAAGACTCCATAAGACACGCATCTCGGCGA"

def encode(seq):
    return [DNA_MAP.get(c, 0) for c in seq]

q = encode(query)
r = encode(ref)

def smith_waterman_guided_cpu(q, r):
    m, n = len(q), len(r)
    # score matrices
    H = np.zeros((m + 1, n + 1), dtype=np.int32)
    I = np.full((m + 1, n + 1), -10**6, dtype=np.int32)
    D = np.full((m + 1, n + 1), -10**6, dtype=np.int32)

    max_score = 0
    end_i = end_j = 0
    best_i = best_j = 0

    # fill DP with banded, guided termination
    for d in range(2, m + n + 1):
        i_min = max(1, d - n)
        i_max = min(m, d - 1)
        center = d // 2
        band_lo = max(i_min, center - BAND)
        band_hi = min(i_max, center + BAND)

        # track diagonal-wise max position for Z-drop
        diag_max = 0
        diag_i = diag_j = 0

        for i in range(band_lo, band_hi + 1):
            j = d - i
            if j <= 0 or j > n:
                continue

            qi, rj = q[i - 1], r[j - 1]
            # match/mismatch or N penalty
            if qi == 14 or rj == 14:
                sc = N_PENALTY
            else:
                sc = MATCH if qi == rj else MISMATCH

            I[i][j] = max(I[i][j - 1] + GAP_EXT, H[i][j - 1] + GAP_OPEN)
            D[i][j] = max(D[i - 1][j] + GAP_EXT, H[i - 1][j] + GAP_OPEN)
            H[i][j] = max(0, H[i - 1][j - 1] + sc, I[i][j], D[i][j])

            if H[i][j] > max_score:
                max_score = H[i][j]
                end_i, end_j = i, j
                best_i, best_j = i, j
            if H[i][j] > diag_max:
                diag_max = H[i][j]
                diag_i, diag_j = i, j

        # guided Z-drop termination using diagonal difference
        delta = abs((best_i - best_j) - (diag_i - diag_j))
        if max_score - diag_max > Z + abs(GAP_EXT) * delta:
            break

    return max_score, end_i, end_j

# run
score, end_i, end_j = smith_waterman_guided_cpu(q, r)
print(f"Score: {score}")
print(f"End position: ({end_i}, {end_j})")