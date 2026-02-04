# DP-HLS — A High-Level Synthesis Framework for Efficient Bioinformatics DP Kernel Acceleration

> Source: Yingqi Cao et al., UCSD.  
> Paper: *"DP-HLS: A High-Level Synthesis Framework for Accelerating Dynamic Programming Algorithms in Bioinformatics"*

---

## One-Sentence Summary

**DP-HLS** proposes a high-level synthesis framework for efficiently deploying a variety of 2D dynamic programming (DP) algorithms on FPGAs. By separating the algorithmic front-end from the optimized hardware back-end, the framework significantly simplifies kernel development while maintaining competitive performance and resource efficiency.

---

## 1. Motivation and Contributions

### Background

- Dynamic programming algorithms such as Smith-Waterman (SW), Needleman-Wunsch (NW), DTW, etc., are widely used in bioinformatics.
- The main performance bottleneck lies in computing and tracing back the DP matrix.
- Although GPUs, FPGAs, and ASICs have been used to accelerate DP, most designs rely on low-level RTL, which is hard to generalize or modify.

### Objectives of DP-HLS

1. **Support a wide family of DP kernels**, including local/global, affine gap, two-piece affine, DTW, profile alignment, etc.
2. **Automatically generate efficient hardware** through high-level synthesis, without requiring HDL expertise.
3. **Decouple the algorithmic specification (front-end)** from **low-level compiler/hardware optimization (back-end)**.

### Key Contributions

- Defines computation (PE_func) and traceback logic using C++ and HLS pragmas.
- Automatically schedules parallelism using the `(NPE, NB, NK)` hierarchy.
- Implements 15 bioinformatics kernels, several of which had no prior FPGA implementation.
- Achieves comparable or better throughput than hand-optimized RTL and GPU implementations on AWS EC2 F1.
- Releases source code and documentation ([GitHub: TurakhiaLab/DP-HLS](https://github.com/TurakhiaLab/DP-HLS)).

---

## 2. Technical Architecture

### PE_func (Processing Element Logic)

- Developers specify only the recurrence relation for a single DP cell `(i, j)`:
  - `wt_scr = max(...)`
  - `wt_tbp = TB_LEFT/UP/DIAG/...`
- The rest (dataflow, buffering, synchronization) is handled by the back-end.

### Traceback FSM

- Users describe a finite state machine (FSM) defining traceback pointer transitions.
- Supports transitions across multiple scoring matrices (e.g., affine gap, 2-piece gap).
- Uses FSM logic to compress traceback logic into compact hardware.

### Parallelism Configuration: NPE / NB / NK

| Parameter | Meaning |
|----------|---------|
| **NPE** | Number of PEs per block (intra-query parallelism) |
| **NB**  | Number of blocks per kernel (inter-query parallelism) |
| **NK**  | Number of kernel instances (multi-threaded host parallelism) |

> Equivalent to using one GPU block per query-reference pair (inter-query parallelism) in Triton.

---

## 3. Experimental Results (Section 6–7)

### Performance Comparison

- Deployed all 15 kernels on AWS EC2 F1 instances.
- Compared to baselines:
  - **CPU (SeqAn3)**: DP-HLS achieves 1.5× to 32× speedup.
  - **GPU (GASAL2, CUDASW++)**: DP-HLS achieves 1.4× to 17× speedup.
  - **Handwritten RTL (GACT, BSW)**: DP-HLS is within 7.7%–16.8% of peak throughput, with significantly better programmability.

### Resource Scaling

- Results show:
  - LUT/FF usage scales linearly with NPE.
  - Resource usage scales proportionally with NB due to identical blocks.
  - DSP/BRAM usage depends on scoring complexity (e.g., affine vs linear).

---

## 4. Observations Related to Current Triton-Based Work

### Implemented Components (e.g., `gpu_extz.py`)

- Affine gap penalty support.
- Z-drop early termination heuristic.
- Banded wavefront DP matrix filling.
- Each Triton GPU block handles one (query, reference) pair — inter-query parallelism.

✅ This closely aligns with DP-HLS Kernel #12 (Banded Local Affine Alignment + Z-drop heuristic).

Current results include:
- 99.5% output match with AGAThA / KSW2_EXTZ baseline.
- Full support for banded and early-drop heuristics.
- Structural equivalence with established GPU/CUDA pipelines.

---

## 5. Future Research Directions (Inspired by DP-HLS)

### Core Research Vision

> "The goal is not just to write alignment kernels in Triton, but to **extend Triton's compiler** to natively support and optimize for alignment-style dynamic programming kernels."

The aim is to repurpose Triton beyond matrix-centric deep learning workloads and make it a **first-class compiler for generalized sequence alignment and 2-D DP problems**.

---

### Roadmap Sketch: Toward Triton-DP Compiler Extension

| Area | Goal |
|------|------|
| 🧠 IR Extensions | Introduce `wavefront_for`, `dp_tile`, `dp_load`, and diagonal loop patterns |
| 🧵 New Primitives | Define domain-specific ops like `tl.dp_max`, `tl.dp_traceback` |
| 🧰 Shared Memory Hack | Enable `dp_shared_memory(region)` to reuse local wavefront buffers |
| 🔄 Inter-block Communication | Support tile-chaining and long-sequence stitching across blocks |
| 🏗 Scheduling Framework | Implement `Scheduler(algo="affine", zdrop=True, tile=256)` to generate appropriate kernel |
| 📚 Generalized API Layer | Abstract `smith_waterman(config)` interface to support SW/NW/DTW/Profile variants |
| ⚡ Performance Baselines | Benchmark against AGAThA, GASAL2, CUDA, and others under iso-band, iso-gap settings |
| 👤 Developer Productivity | Ensure faster, safer, and more portable kernel writing than CUDA or hand-optimized libraries |

---

## 6. Conclusion

DP-HLS provides a highly efficient and generalizable FPGA-based acceleration framework for bioinformatics dynamic programming. More importantly, it offers a system design inspiration:

> Can a Triton-based infrastructure be extended to become a **domain-specific compiler stack** for DP algorithms, just as DP-HLS did for FPGAs?

Triton kernels such as `gpu_extz.py` have already demonstrated structural equivalence and near-match accuracy with leading CUDA implementations. The next step lies in **transforming Triton itself** — not just using it — into a natively optimized compiler path for sequence alignment.

---

## Related Work Summary Table

| Component | Status |
|-----------|--------|
| `gpu_extz.py` (affine + Z-drop + banded) | ✅ Fully implemented and tested |
| AGAThA/KSW2 EXTZ output match | ✅ Achieved >99.5% accuracy |
| DP-HLS paper & baseline analysis | ✅ Completed via this session |
| Kernel-level adaptivity design | 🚧 Defined (e.g., adaptive block size, scoring model switch) |
| Triton IR / primitive hack plan | 🚧 In progress — compiler IR pass and primitive design pending |
| Generalized Sequence Alignment Framework | 🚧 In progress — goal to evolve from kernel-set to compiler-stack |
