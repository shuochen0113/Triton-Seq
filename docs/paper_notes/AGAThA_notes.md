# AGAThA Detailed Summary

## PRELIMINARY

### Sequence Alignment Algorithm of AGAThA
- Initial scores are not all zero. Banded alignment with a termination strategy.
- Score computation follows standard Smith-Waterman with affine gap penalties (gap open + gap extend).
- Guiding strategy = k-banding + termination condition.
  - Banded alignment only calculates k-band and disregards the rest.
  - Termination: stop if current score is far below the global maximum (threshold-based).
  - Problem: huge overhead on GPU for this logic.

### 2.2. GPU Acceleration
- **Input Packing**: Uses GASAL2; genome has only 5 bases, so 4 bits suffice (why not 3 bits? 32-bit aligns better with 4-bit encoding).
- **Intra-query Parallelism**: 4 threads compute four 8x8 blocks concurrently. Threads proceed horizontally across chunks, then move down.
- **Subwarp Strategy**: Threads compute the same anti-diagonal in sync. High-numbered threads start late and low-numbered ones finish early.
  - Warp is split into subwarps to reduce external fragmentation (with some warp divergence cost).

**Figure 2: GPU Processing Model**
- (a) Sequences are 4-bit encoded; 8 bases fit into 32-bit word.
- (b) Subwarps are assigned one alignment task and process chunks of banded DP matrix. Each thread computes blocks left-to-right and proceeds to next chunk.
- (c) Warp splitting into subwarps improves utilization.

## MOTIVATION

### Problems in GPU Implementation
- Termination latency + intra-warp imbalance + inter-warp imbalance.
- Chunks can't finish full diagonals, which Z-drop requires.
- Leads to run-ahead processing → over-computation.
- Heavy intra-warp imbalance when subwarp workloads differ.
- No inter-warp balancing → some warps starve.

### Motivational Experiment
- Compare: CPU alignment, baseline GPU acceleration, guiding techniques.
- Measure overhead from:
  - Tracking local maxima.
  - Load imbalance.

### Some Thoughts: How to Use This Section in Your Work
1. Design experiments:
   - CPU baseline
   - Naive Triton kernel
   - Single-block Triton kernel (`gpu_sw_single_block`)
   - Add banding + Z-drop progressively
2. Measure:
   - Kernel time
   - Time per alignment
   - Occupancy
   - Warp/thread utilization
3. Analyze bottlenecks:
   - Memory latency?
   - Warp divergence?
   - Launch overhead?
   - Use AGAThA's techniques (e.g., sliced diagonals, rolling window)

## AGAThA DESIGN

### 4.1. Tracking Local Maximums with Rolling Window
- Purpose: Efficient Z-drop termination by tracking max of each anti-diagonal.
- Shared memory buffer (LMB) size: 3 * block_size × num_threads.
- First 7 rows (2*block_size - 1) store values for a 4x4 block.

**Steps:**
- Each thread processes top-left cell → writes to first row of LMB.
- Threads compute vertically → fill next rows.
- As columns proceed, same anti-diagonal slots may be reused → compare & replace values.
- When block anti-diagonal (7 total) is filled → rows 0–6 filled.
- Once diagonals 0–3 are complete → warp-wide reduction (`__reduce_max_sync`).
- Max is stored in global memory (GMB), rows cleared.
- Proceed horizontally to next block → LMB fills rows 4–10, so on and so forth, and rolls over to beginning rows as needed.
- If LMB large enough → no need to spill.

**Further Thoughts:**
- Why 3 * block_size rows? Technically (2 * block_size - 1) might be enough.
- Compare & replace values within LMB.
- Max-reduction is cooperative (not one thread per row).

### 4.2. Sliced Diagonal Strategy
- Problem: Horizontal progression leads to excessive run-ahead and shared memory usage.
- Solution: Partition DP matrix into anti-diagonal **slices** (width = `s`).
- Within each slice:
  - Horizontally chunked by subwarp size (rows = #threads in subwarp).
  - Threads process top-down, row-by-row, then check termination.

**Benefits:**
- Run-ahead is bounded by `s × band_width`.
- Shared memory usage reduced.

**Trade-off:**
- Start/end of a slice must read/write horizontal intermediate values (e.g., `H[i][j-1]`).
- As `s` decreases → more slices → more memory access.
- Empirically tuned → `s = 3` works best.

### 4.3. Reducing Warp Divergence with Subwarp Rejoining
- Purpose: Reduce idle time caused by imbalanced subwarps.
- Strategy: When one subwarp finishes, it can **rejoin** others.

**Steps:**
1. Subwarp sets `AS` flag (active status) in shared memory.
2. If finished → `AS = 0`
3. Idle subwarp searches `AS` table → joins active task from another subwarp (`TA`)
4. Copies metadata and updates local thread IDs using `__match_any_sync`
5. Compute remaining work cooperatively.
6. After task → reset subwarp.

- Sync happens only at **slice boundaries**.

### 4.4. Workload Balancing with Uneven Bucketing
- Purpose: Avoid inter-warp imbalance from sequential assignment.
- Strategy:
  - Sort all alignment tasks by sequence length.
  - Pick top `1/N` longest tasks (where N = subwarps per warp).
  - Redistribute: each warp gets at most one long task.

- Benefit: Subwarp rejoining now handles long tasks dynamically.
- This is where the **sort kernel** is used.

### 4.5. Performance Modeling
- Total cells = Anti-diagonals × Band_width + Runahead
- RW reduces `AR_anti`; SD reduces `AR_anti` and `AR_term`.
- SR improves `AVG_subwarps`, UB improves `AVG_warps`

## EVALUATION

### Experiment Setup
- Data: GRCh38, Genome in a Bottle (9 datasets)
- Minimap2's preset parameters

### Baselines
- **Manymap**: Based on Minimap2, fixed to support CUDA streams & multiple reads
- **GASAL2**: Input packing + inter-query parallelism
- **SALoBa**: Intra-query + banding heuristic
- **LOGAN**: Own guiding algo, adaptive band width per anti-diagonal

→ Measured under:
- **Diff-Target** (original baseline version)
- **MM2-Target** (modified to mimic Minimap2)

### Performance Comparison
- SALoBa is fastest in baselines, but AGAThA is much better

### Ablation Study
- **RW**: Reduces global mem access, fast anti-diag max via reduction
- **SD**: Limits run-ahead
- **SR**: Reduces divergence
- **UB**: Redistributes extreme workloads

### Sensitivity Study on Slice Width
- Tested s = 1 to 128
- Performance improves (fewer mem accesses) till s = 3–4, then flattens, increases again (more run-ahead)

→ **s = 3 is optimal**

### SR and UB
- With RW + SD + SR + UB → best load balance
- Tested on generated datasets with % of long reads varied
- UB is best when few long reads cause imbalance
- Reminds us: identify bottlenecks based on **scenario**

### Subwarp Size Sensitivity
- Tested subwarp = 8, 16, 32
- Smaller subwarps → more divergence
- With only RW + SD, full warp is better
- With SR + UB → subwarp (e.g., 8) is better
- Idea: Can Triton auto-tune subwarp size?

### Hardware Flexibility
- GPU > CPU by large margin
- Even on RTX 2080 (no warp reduce) → still better
- Multi-GPU scalability → linear when tasks are distributed equally (manual splitting)

### Applying AGAThA to BWA-MEM
- Applied AGAThA kernel to BWA-MEM’s backend
- Gains smaller than Minimap2 (due to differences in termination logic & band shape)

## DISCUSSION

- Potential of **applying DPX** to AGAThA
- Exploring **different bucketing parameters**
  - E.g., could you terminate alignment **before kernel launch** using estimation?

---

## Technical Deep Dive of `agatha_kernel`
> **Note**: All the code snippets below are written in simplified pseudocode for easier understanding. They correspond to actual logic in `agatha_kernel.h`, though variable names and expressions may slightly differ in implementation.

### Input Packing & Reverse Complement
- **Packing**: Each 8 DNA bases (A/T/C/G/N) are encoded into one `uint32_t` using 4-bit encoding per base.
- **Reverse Complement**: Uses a bitwise reverse and complement trick **without decoding** back to ASCII.
- Implemented in `gasal_pack_kernel` (see `pack_rc_seqs.h`).

```cpp
packed_reg |= ((reg >> 8) & 15) << 24; // packs 4-bit base into 32-bit word
```

---

### Pre-Alignment Scheduling
- **`agatha_sort` kernel** implements **uneven bucketing** to balance workload:
  - Long jobs are distributed across different subwarps
  - This improves inter-warp workload balance
- After sorting, jobs are passed to `agatha_kernel`

---

### Key Structures in `agatha_kernel`

- Each **job** = one sequence alignment (query vs. reference)
- **Subwarp** (8 threads) processes one job at a time

#### Intermediate Buffers (Not full DP matrix!)
- `global_buffer_top` → row-wise (H and E)
- `global_buffer_left` → column-wise (H and F)
- `global_buffer_topleft` → diagonal (P)

#### On-the-fly Bit Decoding
```cpp
uint32_t qbase = (packed_query_literal >> k) & 15;
uint32_t rbase = (packed_ref_literal >> l) & 15;
```
- Decodes 4-bit base **just in time**, saves memory and latency

---

### Core Smith-Waterman Logic (Affine Gap)
- 3 paths: **Diagonal (P), Left (F), Up (E)**
- Scoring via macro: `DEV_GET_SUB_SCORE_GLOBAL`

#### Example:
```cpp
f[m] = max(h[m] - gap_open_extend, f[m] - gap_extend);
h[m] = max(p[m] + match_score, f[m], e);
```

#### Band Restriction:
```cpp
if (query_idx ± band_width > ref_idx + offset) skip;
```

---

### Rolling Window Anti-Diagonal Max Tracker
- `antidiag_max[]` in shared memory buffers scores for **Z-drop termination**
- Each score packed as:
```cpp
(score << 16) | ref_idx
```
- Used with `__reduce_max_sync()` for warp-wide maximum

---

### Z-drop Termination
```cpp
if (max_score - current_score > Z + len_diff * gap_extend) terminate;
```
- Stops alignment early when score drops significantly from best-so-far

---

### Subwarp Rejoining (Dynamic Load Balancing)
- Idle subwarps **rejoin** ongoing jobs of slower subwarps
- Synchronization via shared memory + `__match_any_sync`
- Avoids underutilization due to imbalance in sequence lengths

---

### Final Result
- Returns only:
  - Max alignment score
  - End position in query & reference
- No traceback
- No full matrix
- Optimized for Minimap2-like workflows

---

## Thoughts of My Triton Kernel: Current Limitations & Next Steps

### Current Limitations:
- Uses **single block** for all jobs — lacks inter-query parallelism
- Sequence pairs processed **sequentially**, not in parallel
- Naive input packing: decode happens **inside kernel** inefficiently

### Next Steps for Improvement:
1. **Enable Inter-Query Parallelism**
   - Try to use Triton’s `tl.program_id(axis)` to assign each block one sequence pair
   - Launch many blocks for many jobs (sequence pair)
2. **Optimize Packing**
   - Apply AGAThA-style bit-packing with minimal decoding
   - Decode bases on-the-fly with bitmasks
3. **Use Buffers Instead of Full Matrices**
   - Store H/E/F in registers or strip buffers
   - Avoid full DP table
> **Note**: Triton doesn't support explicitly control shared memory and subwarp, so other optimizations regarding these might be complex.
