#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Affine‑gap, banded Smith‑Waterman with AGAThA Z‑drop termination (CPU).

Parameters default to AGAThA presets:
    match=1, mismatch=-4, gap_open=-6, gap_extend=-2,
    band=400, Z=751
Returns (best_score, (i,j)) in 1‑based coordinates.
"""

from typing import List, Tuple

N_CODE = 14     # Consistent with AGAThA: 'N' is encoded as 14
N_PENALTY = 1   # Matches AGAThA's -N_PENALTY treatment


def smith_waterman_cpu_guided(
    query: str,
    ref: str,
    match: int = 1,
    mismatch: int = -4,
    gap_open: int = -6,
    gap_extend: int = -2,
    band: int = 400,
    Z: int = 751,
) -> Tuple[int, Tuple[int, int]]:

    def encode_base(c: str) -> int:
        return {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': N_CODE}.get(c, 0)

    q_encoded = list(map(encode_base, query))
    r_encoded = list(map(encode_base, ref))

    m, n = len(q_encoded), len(r_encoded)
    if m == 0 or n == 0:
        return 0, (0, 0)

    beta = abs(gap_extend)          # β must be positive in Z‑drop

    # previous-row buffers
    prev_M: list[int] = []
    prev_D: list[int] = []
    jlo_prev = jhi_prev = -1

    best_score = 0
    best_i = best_j = 0
    best_diag = 0                   # anti-diagonal where best_score first seen

    for i in range(1, m + 1):
        jlo = max(1, i - band)
        jhi = min(n, i + band)
        width = jhi - jlo + 1

        curr_M = [0] * width
        curr_D = [0] * width
        curr_I = [0] * width

        diag_max = 0

        for k, j in enumerate(range(jlo, jhi + 1)):
            # helpers to safely fetch prev-row cells inside band
            def fetch(vec, col):
                return vec[col - jlo_prev] if jlo_prev <= col <= jhi_prev else 0

            M_diag = fetch(prev_M, j - 1)
            M_up   = fetch(prev_M, j)
            D_up   = fetch(prev_D, j)

            # I (insertions) : from left
            if k > 0:
                curr_I[k] = max(curr_I[k-1] + gap_extend,
                                curr_M[k-1] + gap_open)
            else:
                curr_I[k] = float('-inf')   # outside band on the left

            # D (deletions) : from above
            curr_D[k] = max(D_up + gap_extend,
                            M_up + gap_open)

            q_base = q_encoded[i-1]
            r_base = r_encoded[j-1]
            if q_base == N_CODE or r_base == N_CODE:
                s = -N_PENALTY
            else:
                s = match if q_base == r_base else mismatch

            curr_M[k] = max(0, M_diag + s, curr_I[k], curr_D[k])

            # update running maxima
            if curr_M[k] > diag_max:
                diag_max = curr_M[k]

            if curr_M[k] > best_score:
                best_score = curr_M[k]
                best_i, best_j = i, j
                best_diag = i + j

        # Z‑drop early termination (AGAThA)
        curr_diag_idx = i + jhi          # anti‑diag number of last processed cell
        delta = curr_diag_idx - best_diag
        if diag_max < best_score - Z - beta * delta:
            break

        # roll buffers
        prev_M, prev_D, jlo_prev, jhi_prev = curr_M, curr_D, jlo, jhi

    return best_score, (best_i, best_j)


def batch_sw_cpu_guided(queries: List[str],
                        refs: List[str],
                        **kw) -> List[Tuple[int, Tuple[int, int]]]:
    return [smith_waterman_cpu_guided(q, r, **kw)
            for q, r in zip(queries, refs)]