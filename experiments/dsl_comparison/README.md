# GPU DSLs for Sequence Alignment: A Comparative Benchmark

A high-performance benchmark suite evaluating various DSLs against native CUDA/CPU implementations for the **Smith-Waterman** sequence alignment algorithm (Affine Gap Penalty).

**Supported Backends:**
* **Baselines:** CPU (OpenMP + AVX-512), Native CUDA.
* **DSLs:** Triton, TileLang (PKU/Microsoft), ThunderKittens (Stanford), CuTile (NVIDIA). (TODO: Codon, etc.)

---

## Algorithm & Implementation 

### Smith-Waterman (Affine Gap)
We implement the standard **Smith-Waterman** local sequence alignment algorithm with an **Affine Gap Penalty** model.

* **Recurrence Formulas:**
    * $E_{i,j} = \max(H_{i,j-1} + GapOpen, E_{i,j-1} + GapExtend)$  *(Horizontal Extension)*
    * $F_{i,j} = \max(H_{i-1,j} + GapOpen, F_{i-1,j} + GapExtend)$  *(Vertical Extension)*
    * $H_{i,j} = \max(0, H_{i-1,j-1} + Score(q_i, r_j), E_{i,j}, F_{i,j})$ *(Main Score)*

* **Parallelism Strategy: Wavefront (Anti-Diagonal)**
    * Since cell $(i,j)$ depends on $(i-1,j)$, $(i,j-1)$, and $(i-1,j-1)$, cells along the diagonal $d = i+j$ are independent.
    * **GPU Strategy**: 
        * We compute one anti-diagonal at a time. All threads in a block collaborate to compute cells where $i+j = d$. 
        * We use **one** block to compute **one** pair of sequences. 
        * We **don't** use shared memory to store DP dependency buffer (H/E/F).

---

## Project Structure

* **Kernels (`*_sw.*`)**: Pure compute logic. No I/O, no timing, just math.
* **Harness (`bench.py`)**: Unified control plane for data loading, verification, and timing.

```text
.
├── bench.py               # [Entry Point] Unified Benchmark Engine
├── cpu_sw.cpp             # CPU Kernel
├── cuda_sw.cu             # CUDA Kernel
├── tk_sw.cu               # ThunderKittens Kernel
├── triton_sw.py           # Triton Kernel
├── tilelang_sw.py         # TileLang Kernel
├── cutile_sw.py           # CuTile Kernel
├── codon_sw.codon         # Codon Kernel
├── ThunderKittens/        # [Submodule] Library Headers
├── datasets/              # Input FASTA files (Required)
│   ├── query.fa
│   └── ref.fa
└── results/               # Benchmark Logs & Summaries

```

---

## Quick Start

### 1\. Prerequisites

  * **OS:** Linux x86\_64
  * **Hardware:**
      * **GPU:** NVIDIA Ampere (A6000), or Blackwell (RTX 5090).
      * **CPU:** Intel CPU with AVX-512 support.
  * **Software:**
      * Conda (Python 3.8+)
      * CUDA Toolkit 12.0+ (13.0+ required for CuTile)
      * GCC with OpenMP support

### 2\. Environment Setup

```bash
conda activate /root/emhua/scchen/conda_envs/triton_seq
# Ensure submodules (ThunderKittens) are pulled
git submodule update --init dsl_comparison_202512/ThunderKittens
```

### 3\. Compilation

We must Compile the C++ based kernels into Shared Objects (`.so`) before running Python.

> **⚠️ Architecture Note:** Adjust `-arch` flags based on the GPU:
>
>   * **Ampere (A6000):** `sm_86`
>   * **Blackwell (RTX 5090):** `sm_120`

```bash
# 1. CPU Baseline (AVX-512)
g++ -shared -fPIC -O3 -march=native -fopenmp -std=c++17 cpu_sw.cpp -o cpu_sw.so

# 2. CUDA Native
nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_89 -std=c++20 --extended-lambda --expt-relaxed-constexpr cuda_sw.cu -o cuda_sw.so

# 3. ThunderKittens (Requires C++20)
sed -i 's/static __device__ inline constexpr bf16_2/static __device__ inline bf16_2/g' ThunderKittens/include/common/base_types.cuh
sed -i 's/static __device__ inline constexpr half_2/static __device__ inline half_2/g' ThunderKittens/include/common/base_types.cuh
sed -i 's/cuCtxCreate_v4(&contexts\[i\], 0, devices\[i\])/0/g' ThunderKittens/include/types/device/ipc.cuh
nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_86 -std=c++20 --extended-lambda --expt-relaxed-constexpr -I./ThunderKittens/include tk_sw.cu -o tk_sw.so
```

---

## Running Benchmarks

### Option A: One-Click Benchmark

Automatically runs all available backends, removes outliers (warmup/jitter), calculates average GCUPS, and saves a summary report.

```bash
python bench.py --backend all
# Results saved to: results/bench_history/summary_YYYYMMDD_HHMMSS.txt
```

### Option B: Individual Development Run

Useful for debugging specific kernels.

1.  **Generate Gold Standard (CPU):**

    ```bash
    python bench.py --backend cpu --out results/cpu_result.txt
    ```

2.  **Run & Verify DSL (e.g., TileLang):**

    ```bash
    python bench.py --backend tilelang --out results/tilelang_result.txt --verify results/cpu_result.txt
    ```

---

## Configuration & Customization

### Runtime Parameters (CLI)

You can modify scoring parameters directly via the command line:

```bash
python bench.py --backend cuda \
    --match 2 --mismatch -3 --gap_open -5 --gap_extend -1
```

### Compile-Time Constants (Hardcoded)

To change fundamental architectural parameters, edit the respective kernel files:

  * **Block Size:** Default is `256`.
      * *CUDA/TK:* Edit `#define BLOCK_SIZE 256` in `.cu` files.
      * *Triton/TileLang:* Edit `BLOCK_SIZE = 256` variable in `.py` files.
  * **Data Packing:** The `bench.py` automatically packs variable-length sequences into a CSR-like format (Flat Array + Offsets). This is consistent across all backends.

---

## Extensibility Guide

We designed `bench.py` to be easily extensible for new DSLs or algorithms.

### How to add a new Backend?

1.  **Implement the Kernel:**
    Create a new file (e.g., `new_dsl_sw.py` or `.cpp`). It must accept raw pointers (flat arrays) and standard C types (`int`, `short`).

2.  **Create a Runner Class:**
    In `bench.py`, inherit from `BackendRunner` (or `CTypesBackend` if loading a `.so`).

    ```python
    class NewDSLRunner(BackendRunner):
        def __init__(self):
            # Load library or import module
            pass
        
        def run(self, dataset, threads, params):
            # 1. Marshal data to device
            # 2. Call kernel
            # 3. Return list of AlignResult objects
            pass
    ```

3.  **Register:**
    Add your runner to the `candidates` list in `benchmark_all` and the `main` dispatch logic.

---

## Troubleshooting

  * **`FileNotFoundError: ... .so`**: Forgot to compile the C++ kernels. See Section 3.
  * **`Score Mismatch`**: The kernel logic is incorrect. Use `--verify` to debug.
  * **`Tie-break diff` (Warning)**: This is normal. Wavefront (GPU) and Row-Major (CPU) processing orders may select different optimal coordinates for the same score.
  * **ThunderKittens Compile Error**: Ensure using `nvcc` with `-std=c++20`.