import time

def smith_waterman_cpu(seq1, seq2, match=3, mismatch=-2, gap=-1):
    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    max_score = 0
    max_pos = (0, 0)
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            match_score = dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            delete = dp[i-1][j] + gap
            insert = dp[i][j-1] + gap
            dp[i][j] = max(match_score, delete, insert, 0)
            
            if dp[i][j] > max_score:
                max_score = dp[i][j]
                max_pos = (i, j)
    
    return max_score, max_pos, dp