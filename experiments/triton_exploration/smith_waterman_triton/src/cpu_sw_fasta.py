import time

def smith_waterman_cpu(seq1, seq2, match=3, mismatch=-2, gap=-1, n_score=0):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_score = 0
    max_pos = (0, 0)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == 'N' or seq2[j - 1] == 'N':
                match_penalty = n_score  # 'N' is fuzzy matching
            else:
                match_penalty = match if seq1[i - 1] == seq2[j - 1] else mismatch

            match_score = dp[i - 1][j - 1] + match_penalty
            delete = dp[i - 1][j] + gap
            insert = dp[i][j - 1] + gap
            dp[i][j] = max(match_score, delete, insert, 0)

            if dp[i][j] > max_score:
                max_score = dp[i][j]
                max_pos = (i, j)

    return max_score, max_pos, dp