# ksw2_extz Algorithm

## Overview

This algorithm is an optimized variant of the Smith-Waterman local alignment algorithm (banded, affine-gap, Z-drop). It is the backbone of the alignment engine in tools such as AGAThA.

---

## 1. Problem Definition

Given two sequences:

- Query: `Q[0 ... m-1]`
- Target: `T[0 ... n-1]`

The goal is to compute a local alignment between a substring of `Q` and a substring of `T` that maximizes the alignment score, considering:

- A substitution matrix `mat[a][b]` for base pair scores.
- Affine gap penalties:  
  - Gap open penalty: `gapo`
  - Gap extension penalty: `gape`
  - Gap open + gap extension penalty: `gapoe`

The algorithm seeks to compute:

- The best local alignment score.
- The alignment ending positions on `Q` and `T`.

---

## 2. Data Structures and Scoring

EXTZ maintains three primary dynamic programming variables per alignment column:

- `H[i][j]`: Optimal score ending at `Q[j]` and `T[i]`.
- `E[i][j]`: Best score ending at `(i,j)` with a gap in **target** (i.e., gap extends in `T`).
- `F[i][j]`: Best score ending at `(i,j)` with a gap in **query** (i.e., gap extends in `Q`).

The affine gap penalty is defined as:

```
E[i][j] = max(E[i][j-1] - gape, H[i][j-1] - gapoe)
F[i][j] = max(F[i-1][j] - gape, H[i-1][j] - gapoe)
H[i][j] = max(H[i-1][j-1] + mat[Q[j]][T[i]], E[i][j], F[i][j])
```

> *Note: There is no operation to normalize to 0 when it is less than 0, and when open a new gap, a **gapoe** penalty will be added, not traditional **gapo**.*

---

## 3. Initialization

EXTZ is a **semi-global** alignment algorithm in practice, allowing gaps at the start of the query or target with affine penalties. Initialization is:

- `H[0][0] = 0`
- `E[0][j], H[0][j]`: Initialized based on `gapo` and `gape`, for `j ≤ band width`
- `F[i][0], H[i][0]`: Likewise for `i ≤ band width`
- Other values initialized to `-∞` (practically, a large negative constant).

---

## 4. Banding Strategy

To reduce computational complexity from `O(mn)` to `O(nw)` (where `w` is band width), only evaluates DP cells satisfying:

```
|i - j| ≤ w
```
For each row `i` (i.e., position in `T`), computes only those `j` in `[i - w, i + w]`, clamped to `[0, m-1]`.

---

## 5. Main Computation Loop

For each target position `i`:

1. Determine valid query range `[st, en]` based on banding.
2. For each position `j ∈ [st, en]`:
   - Compute `E[i][j]` from `E[i][j-1]` and `H[i][j-1]`
   - Compute `F[i][j]` from `F[i-1][j]` and `H[i-1][j]`
   - Compute `H[i][j]` from diagonal match/mismatch, and max of `E`, `F`
   - Track current row maximum `(row_max, row_max_j)`
   - Track global maximum `(global_max, end_i, end_j)`

> *Note: Only one row/column of DP matrices is required at any time. Rolling buffers or linear arrays are used in optimized implementations.*

---

## 6. Z-drop Early Termination

To avoid unnecessary computation when alignment quality drops too far from the best observed path, it uses a **Z-drop heuristic**:

Let:

- `(i, j)` = current local maximum on diagonal `i + j = c`
- `(i', j')` = global maximum at earlier diagonal `i′ + j′ < c`
- `H(i, j)` = current local max score
- `H(i′, j′)` = global max score so far

Compute:
```
delta = |(i - i′) - (j - j′)|
if H(i′, j′) - H(i, j) > Z + β * delta:
terminate early
```
Where:

- `Z` is a predefined Z-drop threshold
- `β` is typically equal to `gape`

---

## 7. Output

- The best alignment score `global_max`
- Alignment endpoints in query and target: `(end_j, end_i)`

---

## 8. Core Differences Between EXTZ and Naive Smith-Waterman

| Aspect              | Naive Smith-Waterman                  | EXTZ                                        |
|---------------------|----------------------------------------|---------------------------------------------|
| **Initialization**  | All `H[0][*]` and `H[*][0]` set to 0   | Affine-penalized semi-global initialization |
| **Score Normalization** | `H[i][j] = max(0, …)` (no negative scores) | No normalization; scores can drop below 0   |
| **Gap Penalty**     | Affine: `gapo + (l - 1) * gape`, or only `gap`              | `gapoe = gapo + gape` for first gap unit    |
| **Alignment Model** | Strict local alignment                | Semi-global extension with banding          |
| **Termination**     | Stops when score drops to 0           | Z-drop heuristic based on score decline     |

EXTZ replaces score normalization with an early-termination mechanism and uses banded, affine-gap, semi-global alignment to efficiently extend seeds in practical long-read alignment scenarios.