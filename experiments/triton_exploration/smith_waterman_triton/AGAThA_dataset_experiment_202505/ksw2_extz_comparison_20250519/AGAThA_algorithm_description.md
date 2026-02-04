# Triton EXTZ Sequence Alignment (AGAThA-style)

## Overview

This document describes the full technical structure and implementation details of a GPU-based **Smith-Waterman alignment algorithm with affine gap penalties and Z-drop early termination**, inspired by the AGAThA system. The kernel is implemented in Triton and is **bit-exact** with a validated CPU reference implementation. It operates in a **wavefront (anti-diagonal)** order and supports **dynamic banded pruning**.

---

## 1. Problem Definition

Given two DNA sequences:

* Query: `Q[0 ... m-1]`
* Target: `T[0 ... n-1]`

We aim to compute the **best local alignment score** and **alignment endpoints** using:

* Substitution matrix with match/mismatch scores.
* Affine gap penalties:

  * `gap_open = α` (penalty for opening a gap)
  * `gap_extend = β` (penalty for extending a gap)
* Z-drop heuristic to terminate unpromising paths early.

---

## 2. Core Scoring Logic

### Variables:

* `H(i,j)`: Highest score ending at (i,j) via match/mismatch.
* `E(i,j)`: Best score ending with an insertion in the target (gap in query).
* `F(i,j)`: Best score ending with a deletion in the target (gap in reference).

### Affine Dynamic Programming Recurrence:

```
E(i,j) = max(E(i,j-1) + β, H(i,j-1) + α + β)
F(i,j) = max(F(i-1,j) + β, H(i-1,j) + α + β)
H(i,j) = max(H(i-1,j-1) + s(q_j, t_i), E(i,j), F(i,j))
```

> No reset-to-zero as in classic SW. The algorithm retains negative values and behaves as a semi-global local alignment.

### Substitution Score `s(q_j, t_i)`:

* `+MATCH` if bases match
* `-MISMATCH` if mismatch
* `-NPENALTY` for ambiguous base `'N'`

---

## 3. Initialization Strategy

The kernel uses **three anti-diagonal buffers**: `prev2`, `prev`, `curr` to roll across the matrix.

* `H(0,0) = 0`
* `H(i,0) = GAPOE + β * (i-1)`
* `H(0,j) = GAPOE + β * (j-1)`
* `E(i,j), F(i,j) = -INF` unless explicitly initialized.

Initial anti-diagonals:

* `d = 0` → `prev2H[BAND] = 0`
* `d = 1` → `prevH[0] = GAPOE`

---

## 4. Wavefront Anti-Diagonal Sweep

* Iterate anti-diagonals `d = 2 ... m+n`
* Each point (i,j) satisfies `i + j = d`
* Compute values on each anti-diagonal in **parallel**

---

## 5. Dynamic Banded Strategy

To restrict computation to a narrow band around the main alignment path:

```
offset = best_i - best_j
half_lo = (d + offset - BAND + 1) // 2
half_hi = (d + offset + BAND) // 2
band_lo = max(i_min, half_lo)
band_hi = min(i_max, half_hi)
```

* Dynamically centers the band around the best known (i,j) diagonal.
* Ensures compute only focuses on likely alignment paths.

---

## 6. Z-drop Early Termination

Used to terminate alignment when score drops far below the current best:

```
delta = |(i - j) - (i' - j')|
if H(i', j') - H(i, j) > Z + β * delta:
    terminate
```

* `(i, j)` is current anti-diagonal max
* `(i', j')` is global best so far
* Avoids continuing computation down low-quality regions
