### UPDATE: 20250519:
- AGAThA (& Minimap2) don’t use “Smith-Waterman” Algorithm!!!
- It’s alignment algorithm is called: ksw2_extz, see: https://github.com/lh3/ksw2/

#### Current Implementations:
- `/alignment_related_20250519/test/new.py`: the CPU implementation (same as ksw2-extz/AGAThA's algorithm).
- `/alignment_related_20250519/summerization.md`: a detailed comparison between the "guided_alignemt" and "ksw2"
- `/triton_experiment/smith_waterman_triton/src/gpu_extz.py`: A triton implementation of extz Correctness has been verified. (Same as CPU implementation)
- To get the output of this triton kernel, use `/triton_experiment/smith_waterman_triton/tests/sw_benchmark_ksw2.py`
- Compare with AGAThA's output (see `/triton_experiment/smith_waterman_triton/AGAThA_dataset_experiment)202505`): 
  - Total pairs: 16384
  - ✔️  Exact match (score & position): 16363
  - ⚠️  Score match, but position mismatch: 0
  - ⚠️  Position match, but score mismatch: 0
  - ❌  Score & position both mismatch: 21 （0.13%）

#### Comparison: Triton vs. AGAThA

---

### 20250509, new kernels (/src)
**Completely Same** algorithm with AGAThA:
  - ` cpu_sw_affine.py `: guided alignment CPU baseline w/ affine gap penalty;
  - ` gpu_sw_guided_affine.py `: guided alignment Triton kernel.
  - Parameters: match +1, mismatch -4, gap open -6, gap extend -2, band-width: 400, Z-drop parameter: 751
  - Algorithm details (Z-drop): TODO

**Other** kernels:
  - ` cpu_sw_guided.py `: band-limited alignment (same algorithm w/ following triton kernel).
  - ` gpu_sw_guided_inter_query.py `: band-limited alignment triton kernel.
  - parameters: match +3, mismatch -2, gap -1.

### benchmark scripts (/tests):
  - ` sw_benchmark_guided.py `: self-generate datasets to compare **triton kernel** and **AGAThA**. (for both `affine` and `guided_inter_query` kernel)
  - ` sw_benchmark_guided_cpu.py `: verify correctness of cpu & gpu guided.
  - ` sw_benchmark_guided_affine_cpu.py`: verify correctness of cpu & gpu guided alignment w/ affine gap penalty.


### outputs (/outputs):
  - `guided_alignments_cpu/gpu_{ts}.json`: output of ` sw_benchmark_guided_cpu.py `.
  - `guided_affine_alignments_cpu/gpu_{ts}.json`: output of ` sw_benchmark_guided_affine_cpu.py`.
  - `/triton_(affine_)guided_alignment_results/***.json` and `sw_guided_vs_agatha_{ts}.json`: output of ` sw_benchmark_guided.py `.

### evaluations:
  - see this directory.

### the logic `AGAThA` deals with `N`
In the current build of AGAThA, N_PENALTY is enabled with a value of 1. This means that any occurrence of an ambiguous base ‘N’—encoded as 0x4E and internally represented as 14—is penalized with a fixed score of -1 during alignment, regardless of the base it aligns with (even another ‘N’). This behavior is controlled via the macro DEV_GET_SUB_SCORE_LOCAL in gasal_kernels.h, and is fully active due to the -DN_PENALTY=1 definition passed at compile time in the Makefile.


