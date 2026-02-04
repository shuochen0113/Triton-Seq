## Understanding `ksw_extz` and the Differences Between `old.py` and `new.py`

### 1. Nature of the `extz` (Extension + Z-drop) Algorithm

#### 1.1 Not Standard Smith–Waterman

* Smith–Waterman is a classic *local* alignment:

  * Each DP cell is calculated with `H(i,j) = max(0, ...)`.
  * Negative scores are clipped to zero (local-reset).
  * Final alignment is the substring with the highest local score.
* `ksw_extz` is an *extension* alignment with Z-drop:

  * No local reset — negative scores are kept.
  * Scoring starts from `(0,0)` and accumulates regardless of sign.
  * A Z-drop heuristic terminates alignment when deviating too far from the diagonal.

#### 1.2 DP Structure: H, E, F Matrices

* **H(i,j)**: best score reaching cell (i,j).
* **E(i,j)**: score ending with an insertion (gap in ref).
* **F(i,j)**: score ending with a deletion (gap in query).

**Update formulas (per `ksw_extz.c`):**

```text
E(i,j) = max{ E(i,j-1) - gape, H(i,j-1) - (gapo + gape) }
F(i,j) = max{ F(i-1,j) - gape, H(i-1,j) - (gapo + gape) }
h_n = H(i-1,j-1) + match/mismatch/N
H(i,j) = max{ h_n, E(i,j), F(i,j) }
```

* `gapo` = gap open penalty
* `gape` = gap extend penalty

#### 1.3 Banding

* Only fill cells where `|i-j| <= BAND`.
* Others treated as `-inf`.

#### 1.4 Z-drop Termination

After each row `i`, compute:

```text
row_max = max_j H(i,j)
global_max = max score so far

delta = | (i - best_i) - (row_max_j - best_j) |
if (global_max - row_max) > ZDROP + gape * delta:
    break
```

#### Why does `extz` give score 31, while original SW gave 48?

* Smith–Waterman resets negatives to 0, capturing highest local subsequence alignment (score = 48).
* `ksw_extz` accumulates from the start, and terminates early with Z-drop (score = 31 at approx (70,69)).

### 2. Comparison: guided_sw vs. ksw2_extz

| Aspect             | guided SW              | ksw\_extz-aligned                 |     |           |
| ------------------ | --------------------------------- | --------------------------------------------- | --- | --------- |
| **DP Type**        | Local SW: `H(i,j)=max(0,...)`     | Extension: no reset                           |     |           |
| **Init Rows/Cols** | All `H[0][*]` and `H[*][0] = 0`   | Match `ksw_extz.c` exactly: special penalties |     |           |
| **Local Clipping** | `max(0,...)` applied              | No local reset, can go negative               |     |           |
| **Gap Matrices**   | I/D style SW reset                | Full E/F per `ksw_extz.c`, no reset           |     |           |
| **Banding**        | Diagonal scan, manual band        | Per row/col with `abs(i-j) <= BAND` |
| **Z-drop**         | Manual diag vs. global comparison | Full Z-drop heuristic, exit early             |     |           |
| **Termination**    | Optional early break              | Z-drop forced break                           |     |           |
| **Return**         | `(score, end\_i, end\_j)`, score=48 | `(score, end\_query, end\_target)`, score=31    |     |           |
| **Behavior**       | Can reset and start over          | One pass from (0,0), no restart               |     |           |

### Summary

* **Core difference:**

  * `old.py`: Band + guided Z-drop **local** Smith–Waterman
  * `new.py`: Band + Z-drop **extension** alignment (no reset)

* **Score difference reason:**

  * SW: finds highest-scoring local segment (score=48)
  * extz: accumulates from start, terminates early (score=31)

* **Code-level implications:**

  * To replicate AGAThA/GASAL/Minimap2-style alignment, use extz-style logic: no reset, proper init, Z-drop.
  * To use classic SW, reintroduce local reset and reset-style gap logic.

### Reproduce:
  - ksw: `./ksw2-test -w 751 -z 400 -A 1 -B 4 -O 6,6 -E 2,2 -t extz2_sse qry.fa ref.fa`