# Triton-Seq Quickstart Guide

**Last Updated:** February 4, 2026

This guide will help you get started with Triton-Seq, from installation to running your first alignment.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Running Benchmarks](#running-benchmarks)
4. [Understanding the Framework](#understanding-the-framework)
5. [Advanced: Custom Compiler](#advanced-custom-compiler)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- **CUDA**: Version 12.1 or later
- **Python**: 3.10 or later
- **conda**: Miniconda or Anaconda

### Quick Setup

```bash
# 1. Clone the repository
git clone git@github.com:shuochen0113/Triton-Seq.git
cd Triton-Seq

# 2. Run setup script
bash scripts/setup_env.sh

# 3. Activate environment
conda activate triton-seq

# 4. Initialize submodules
git submodule update --init --recursive
```

### Verify Installation

```bash
# Test Python import
python -c "import torch; import triton; print('✓ PyTorch:', torch.__version__); print('✓ Triton:', triton.__version__)"

# Test CUDA
python -c "import torch; print('✓ CUDA available:', torch.cuda.is_available()); print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Basic Usage

### Example 1: Simple Alignment

```python
from src.api import align_sequences
from src.utils.io import load_fasta

# Load small test dataset
queries = load_fasta("datasets/small/query_small.fa")
references = load_fasta("datasets/small/ref_small.fa")

# Run alignment
results = align_sequences(queries, references)

# Print results
for i, result in enumerate(results):
    print(f"Pair {i}: Score={result.score}, Position=({result.query_end}, {result.ref_end})")
```

### Example 2: Custom Scoring Matrix

```python
from src.api import TritonSW

# Initialize with custom scoring
aligner = TritonSW(
    match_score=2,
    mismatch_penalty=-3,
    gap_open=-5,
    gap_extend=-1,
    device='cuda:0',
    batch_size=2048
)

# Align sequences
results = aligner.align(queries, references)
```

See [examples/](../examples/) directory for more usage examples.

---

## Running Benchmarks

### Benchmark the Baseline (OPv6)

```bash
# From project root
python benchmarks/scripts/run_baseline.py
```

**Expected output:**
```
Loading dataset...
Sequences loaded: 16,384 pairs
Running alignment...
Kernel time: ~88 ms (RTX 4090) or ~105 ms (A6000)
Total time: ~120 ms
GCUPS: ~135 (RTX 4090)
```

### Analyze Results

```bash
python benchmarks/scripts/analyze_results.py
```

This compares outputs with pre-computed reference results from AGAThA and ksw2.

### Performance Profiling

```bash
# Profile with Nsight Systems (requires nsys)
nsys profile -o triton_seq_profile python benchmarks/scripts/run_baseline.py

# View in Nsight Systems GUI
nsys-ui triton_seq_profile.nsys-rep
```

---

## Understanding the Framework

### Directory Structure

```
Triton-Seq/
├── src/                    # Core framework
│   ├── api.py             # Main user API
│   ├── kernel/            # Triton kernels
│   │   └── sw_kernel.py  # OPv6 stable kernel
│   ├── host/              # Host-side pipeline
│   │   ├── scheduler.py   # Async scheduler
│   │   └── buffer_manager.py
│   └── utils/             # Utilities
│       ├── packing.py     # 4-bit encoding
│       └── io.py          # FASTA I/O
├── benchmarks/            # Performance testing
├── experiments/           # Research experiments
│   ├── ptx_modification/  # Manual PTX hack
│   ├── dsl_comparison/    # DSL evaluation
│   └── alignment_tools/   # Correctness validation
└── datasets/              # Test data
```

### Key Components

**1. Kernel (src/kernel/sw_kernel.py)**
- Implements Smith-Waterman with affine gap penalty
- Wavefront (anti-diagonal) parallelism
- 3+2 ring buffer design
- Optimized reduction placement

**2. Host Pipeline (src/host/scheduler.py)**
- Asynchronous double-buffered execution
- Overlaps CPU preparation and GPU execution
- Stages: H2D → Packing → DP Init → Alignment → D2H

**3. Buffer Manager (src/host/buffer_manager.py)**
- Pre-allocates all device and pinned host memory
- Manages CUDA streams
- Handles data transfers

**4. API (src/api.py)**
- User-friendly interface
- Handles batching automatically
- Returns structured results

---

## Advanced: Custom Compiler

The custom Triton compiler provides an additional ~35% speedup through automatic shared memory optimization.

### Why Use the Custom Compiler?

- **Performance**: ~35% faster kernel execution
- **Automatic**: No code changes needed
- **Research**: Demonstrates compiler-level optimization potential

### Installation

```bash
# Activate environment first
conda activate triton-seq

# Build custom Triton
bash scripts/build_triton.sh
```

This will:
1. Initialize the `compiler/triton` submodule
2. Build Triton from source (~10-20 minutes)
3. Install in development mode

### Verify Custom Compiler

```bash
# Check Triton version
python -c "import triton; print(triton.__version__)"

# Run benchmark (should see ~35% speedup)
python benchmarks/scripts/run_baseline.py
```

**Expected performance with custom compiler:**
- RTX 4090: ~67 ms (vs 88 ms baseline)
- A6000: ~75 ms (vs 105 ms baseline)

### Understanding the Compiler Extension

The custom compiler adds:
1. **LinearSharedEncodingAttr**: New memory layout attribute
2. **SeqAlignDetect Pass**: Detects SW kernel pattern
3. **PromoteSeqAlignToShared Pass**: Promotes DP buffers to shared memory
4. **MaterializeSWSmem Pass**: Generates optimized shared memory code

See [compiler/README.md](../compiler/README.md) for technical details.

---

## Datasets

### Included Datasets

**Small Dataset** (`datasets/small/`)
- 10 sequence pairs
- For quick testing (~1 second)

**Standard Dataset** (`datasets/standard/`)
- 16,384 sequence pairs (AGAThA dataset)
- Standard benchmark dataset (~120 ms on RTX 4090)

### Generate Larger Datasets

```bash
# Generate 10x scaled dataset
python datasets/generate_scaled.py --scale 10 --output datasets/large/

# Supported scales: 5x, 10x, 50x, 100x
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size
```python
aligner = TritonSW(batch_size=1024)  # Default is 2048
```

### Issue: Import Error

```bash
# Ensure you're in the right environment
conda activate triton-seq

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Slow Performance

**Check:**
1. GPU is being used: `torch.cuda.is_available()`
2. Running on correct GPU: `CUDA_VISIBLE_DEVICES=0 python ...`
3. No other processes using GPU: `nvidia-smi`

### Issue: Custom Compiler Build Fails

**Common causes:**
1. Missing build tools: `sudo apt-get install cmake ninja-build`
2. Insufficient disk space (needs ~5GB)
3. Wrong branch: Ensure submodule is on `hack/sw_kernel-v1`

```bash
# Reset submodule
cd compiler/triton
git checkout hack/sw_kernel-v1
git pull
cd ../..
bash scripts/build_triton.sh
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/shuochen0113/Triton-Seq/issues)
- **Email**: shuochen0113@gmail.com
- **Documentation**: [docs/](.)

---

## Environment Details

Development and benchmarking environment:

- **CPU**: Intel Xeon Platinum 8481C (25 vCPU)
- **GPUs**: NVIDIA RTX 4090 (24GB), NVIDIA A6000 (48GB)
- **CUDA**: 12.1
- **PyTorch**: 2.1.0
- **Python**: 3.10
- **OS**: Ubuntu 22.04

Your results may vary depending on hardware.
