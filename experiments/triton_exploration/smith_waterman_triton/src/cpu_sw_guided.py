#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Banded Smith‑Waterman (CPU reference) — matches AGAThA / Triton.

• fixed band，|i‑j| ≤ band
• gap = open = extend = `gap`
• X‑drop：continuous two rows row_max < best_score - xdrop, then terminate
• O(band) memory per row
"""

from typing import List, Tuple

def smith_waterman_cpu_guided(
    query: str,
    ref: str,
    match: int = 3,
    mismatch: int = -2,
    gap: int = -1,
    band: int = 400,
    xdrop: int = 751,
) -> Tuple[int, Tuple[int, int]]:
    m, n = len(query), len(ref)
    if m == 0 or n == 0:
        return 0, (0, 0)

    best_score = 0
    best_i = best_j = 0
    below_cnt = 0

    # previous row buffer & window
    prev: list[int] = []
    j_lo_prev = j_hi_prev = -1        # invalid on first row

    for i in range(1, m + 1):
        j_lo = max(1, i - band)
        j_hi = min(n, i + band)
        width = j_hi - j_lo + 1
        curr = [0] * width
        row_max = 0

        for k, j in enumerate(range(j_lo, j_hi + 1)):  # k = j - j_lo
            # safe fetch from previous row
            if j_lo_prev <= j - 1 <= j_hi_prev:
                diag = prev[j - 1 - j_lo_prev]
            else:
                diag = 0
            if j_lo_prev <= j <= j_hi_prev:
                up = prev[j - j_lo_prev]
            else:
                up = 0
            left = curr[k - 1] if k > 0 else 0

            s = match if query[i - 1] == ref[j - 1] else mismatch
            score = max(0, diag + s, up + gap, left + gap)
            curr[k] = score

            if score > row_max:
                row_max = score
            if score > best_score:
                best_score, best_i, best_j = score, i, j

        # X‑drop early‑stop
        if best_score - row_max > xdrop:
            below_cnt += 1
            if below_cnt >= 2:
                break
        else:
            below_cnt = 0

        # rotate
        prev, j_lo_prev, j_hi_prev = curr, j_lo, j_hi

    return best_score, (best_i, best_j)


def batch_sw_cpu_guided(queries: List[str],
                        refs: List[str],
                        **kw) -> List[Tuple[int, Tuple[int, int]]]:
    return [smith_waterman_cpu_guided(q, r, **kw) for q, r in zip(queries, refs)]