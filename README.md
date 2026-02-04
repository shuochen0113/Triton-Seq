# Triton-Seq: A Compiler-Extended Triton Framework for GPU Sequence Alignment

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A high-performance GPU-accelerated framework for Smith-Waterman sequence alignment, featuring optimized Triton kernels and custom compiler extensions for automatic shared memory optimization.

## Highlights

- **Optimized Triton Kernel**: Production-ready Smith-Waterman implementation with async pipeline
- **Custom Compiler Extension**: Novel MLIR passes for automatic shared memory promotion (~35% speedup)
- **Comprehensive DSL Evaluation**: Systematic comparison of 6 GPU DSLs for bioinformatics
- **Full Validation**: Bit-exact correctness verification against AGAThA, and ksw2

## Performance

| GPU | Framework | GCUPS | Speedup vs CPU |
|-----|-----------|-------|----------------|
| RTX 4090 | Triton-Seq (OPv6) | ~135* | ~80x |
| RTX 4090 | + Compiler Hack | ~182* | ~108x |
| A6000 | Triton-Seq (OPv6) | ~118* | ~70x |
| A6000 | + Compiler Hack | ~159* | ~95x |

*Approximate values based on 16,384 sequence pairs (AGAThA dataset)

### DSL Comparison (A6000, 16K pairs)

| Backend | GCUPS | Performance vs CUDA |
|---------|-------|---------------------|
| Native CUDA | 61.37 | 100% (baseline) |
| TileLang | 45.12 | 73.5% |
| ThunderKittens | 41.19 | 67.1% |
| Triton (vanilla) | 38.32 | 62.4% |
| CPU (OpenMP+AVX512) | 7.10 | 11.6% |

See [experiments/dsl_comparison](experiments/dsl_comparison/) for detailed analysis.

## Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:shuochen0113/Triton-Seq.git
cd Triton-Seq

# Setup environment
bash scripts/setup_env.sh
conda activate triton-seq

# Initialize submodules
git submodule update --init --recursive

# (Optional) Build custom Triton compiler for additional speedup
bash scripts/build_triton.sh
```

### Basic Usage

```python
from src.api import align_sequences
from src.utils.io import load_fasta

# Load sequences
queries = load_fasta("datasets/small/query_small.fa")
references = load_fasta("datasets/small/ref_small.fa")

# Run alignment
results = align_sequences(queries, references)

# Access results
for result in results:
    print(f"Score: {result.score}, Position: ({result.query_end}, {result.ref_end})")
```

### Run Benchmarks

```bash
# Run baseline OPv6 benchmark
python benchmarks/scripts/run_baseline.py

# Analyze results
python benchmarks/scripts/analyze_results.py
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed instructions.

## Architecture

Triton-Seq employs a highly optimized architecture:

1. **Kernel Level**: Triton-based Smith-Waterman with:
   - Wavefront (anti-diagonal) parallelism
   - 3+2 ring buffer design (3 slots for H, 2 for E/F)
   - Optimized reduction placement (outer loop vs inner loop)
   - 4-bit DNA sequence encoding

2. **Host Pipeline**: Asynchronous double-buffered execution:
   - Overlapping CPU preparation and GPU execution
   - Two CUDA streams for concurrent operations
   - Stages: H2D → Packing → DP Init → Alignment → D2H

3. **Compiler Extension** (optional): Custom MLIR passes for:
   - Linear shared memory layout (vs. swizzled)
   - Automatic promotion of DP buffers to shared memory
   - ~35% kernel speedup through reduced global memory traffic

![Architecture Overview](figures/Fig3_host%20pipeline%20%26%20packing.png)

## Project Structure

```
Triton-Seq/
├── src/                    # Core framework (OPv6 stable)
├── compiler/               # Custom Triton compiler extension
├── benchmarks/             # Performance testing suite
├── experiments/            # Research experiments
│   ├── ptx_modification/  # Manual PTX proof-of-concept (35% speedup)
│   ├── dsl_comparison/    # 6 DSL evaluation study
│   ├── alignment_tools/   # Correctness validation
│   ├── triton_exploration/# Early Triton experiments
│   ├── gluon/             # Triton 3.5 Gluon experiments
│   └── codon/             # Codon DSL exploration
├── datasets/               # Test datasets
├── docs/                   # Documentation & research logs
└── tests/                  # Unit and integration tests
```

## Research Background

This work was developed during an internship at Cornell University under the guidance of **Prof. Zhiru Zhang** and **Jiajie Li**.

### Key Contributions

1. **Kernel Optimization Journey**: Iterative optimization from OPv1 to OPv6
   - OPv5: Key breakthrough - moving reduction outside inner loop (30% speedup)
   - OPv6: Stable production version with int32 buffers and outer-loop reduction

2. **PTX Validation**: Manual PTX modification proved ~35% speedup via shared memory
   - Hand-edited PTX to place H/E/F buffers in shared memory
   - Validated hypothesis that bandwidth > occupancy as bottleneck

3. **Compiler Hacking**: Automated the PTX optimization via custom MLIR passes
   - Created `LinearSharedEncodingAttr` for deterministic memory layout
   - Implemented `PromoteSeqAlignToShared` and `MaterializeSWSmem` passes
   - Achieved similar performance to manual PTX editing

4. **DSL Comparative Study**: First systematic evaluation of GPU DSLs for sequence alignment
   - Evaluated: Triton, TileLang, ThunderKittens, CuTile, Codon, Native CUDA
   - Identified strengths/weaknesses of each approach
   - TileLang performed best among DSLs (73.5% of CUDA)

## Experiments

### PTX Modification

Manual PTX editing experiment validating shared memory optimization:
- **Location**: [experiments/ptx_modification](experiments/ptx_modification/)
- **Result**: ~35% kernel speedup (88ms → 67ms on RTX 4090)
- **Key Insight**: Shared memory beneficial despite reduced occupancy (12 → 4 blocks/SM)

### DSL Comparison

Comprehensive evaluation of 6 GPU programming frameworks:
- **Location**: [experiments/dsl_comparison](experiments/dsl_comparison/)
- **Implementations**: CUDA, Triton, TileLang, ThunderKittens, CuTile, Codon
- **Benchmark**: Smith-Waterman with affine gap penalty
- **Dataset**: 16,384 sequence pairs (AGAThA dataset)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Shuochen Chen
- **Email**: shuochen0113@gmail.com
- **GitHub**: [@shuochen0113](https://github.com/shuochen0113)
