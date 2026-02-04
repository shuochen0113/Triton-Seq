# Triton Compiler Extension for Sequence Alignment

This directory contains a custom fork of the Triton compiler with specialized MLIR passes for optimizing sequence alignment kernels.

## Overview

The standard Triton compiler uses swizzled shared memory layouts to avoid bank conflicts in matrix operations. However, for sequence alignment algorithms like Smith-Waterman, we need **linear addressing** for ring buffer operations. This compiler extension adds support for linear shared memory layouts and automatic promotion of DP buffers.

## Performance Impact

- **Baseline (vanilla Triton)**: ~88 ms kernel time (RTX 4090, 16K pairs)
- **Custom compiler**: ~67 ms kernel time (**~35% speedup**)
- **Mechanism**: Promotes H/E/F DP buffers from global to shared memory

## Installation

```bash
# From Triton-Seq root directory
bash scripts/build_triton.sh
```

This will:
1. Initialize the `triton/` submodule (branch: `hack/sw_kernel-v1`)
2. Build Triton from source (~10-20 minutes)
3. Install in development mode

## What's Modified?

The extension consists of:

### 1. New Memory Layout Attribute

**File**: `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`

```cpp
def TTG_LinearSharedEncodingAttr : TTG_Attr<"LinearSharedEncoding"> {
  // Deterministic linear layout for sequence alignment ring buffers
  // Unlike swizzled layouts, maintains simple pointer arithmetic
}
```

### 2. New MLIR Operations

**File**: `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`

- `ttg.local_load_slice`: Load slice from shared memory with linear addressing
- `ttg.local_store_slice`: Store slice to shared memory with linear addressing

### 3. Custom Compiler Passes

**Files**: `lib/Dialect/TritonGPU/Transforms/*.cpp`

#### Pass 1: SeqAlignDetect
- **Purpose**: Identify Smith-Waterman kernel patterns
- **Pattern**: H/E/F tensor operations in anti-diagonal wavefront
- **Output**: Marks eligible tensors for promotion

#### Pass 2: PromoteSeqAlignToShared
- **Purpose**: Promote DP buffers to shared memory
- **Transformation**: `blocked` → `linear_shared` encoding
- **Criteria**: Ring buffer access pattern, size fits in shared memory

#### Pass 3: MaterializeSWSmem
- **Purpose**: Generate optimized shared memory code
- **Lowering**: `linear_shared` encoding → LLVM/PTX
- **Optimization**: Coalesced access, no bank conflicts

## Compiler Pipeline

```
Triton AST
    ↓
TTIR (Triton IR)
    ↓
TTGIR (Triton GPU IR)
    ↓
[SeqAlignDetect] ← Identify SW patterns
    ↓
[PromoteSeqAlignToShared] ← Change encoding
    ↓
[MaterializeSWSmem] ← Generate shared memory code
    ↓
LLVM IR
    ↓
PTX
```

## Verification

The compiler extension produces the same numerical results as the baseline (bit-exact), with only performance differences.

### Correctness Test

```bash
# Run with custom compiler
python benchmarks/scripts/run_baseline.py > results_custom.txt

# Switch to vanilla Triton
pip uninstall triton
pip install triton==2.1.0

# Run with vanilla Triton
python benchmarks/scripts/run_baseline.py > results_baseline.txt

# Compare outputs (should be identical)
diff results_custom.txt results_baseline.txt
```

### Performance Test

```bash
# Benchmark custom compiler
python benchmarks/scripts/run_baseline.py --profile
```

Expected results:
- Kernel time: ~67 ms (RTX 4090)
- Speedup: ~35% over baseline

## Technical Details

### Why Linear Layout?

Smith-Waterman uses a **ring buffer** for H/E/F matrices:
```python
H[t % 3, i] = max(H[(t-1) % 3, i-1] + match, ...)
```

Swizzled layouts break the modulo arithmetic:
```
Swizzled: address = base ⊕ offset  # XOR transform
Linear:   address = base + offset  # Simple addition
```

We need linear layout for correct ring buffer semantics.

### Shared Memory Benefits

**Global memory version:**
- Latency: ~400 cycles
- Bandwidth: ~900 GB/s (RTX 4090)
- Bottleneck: Memory bandwidth

**Shared memory version:**
- Latency: ~20 cycles
- Bandwidth: ~15 TB/s
- Benefit: 20x faster access

Even though shared memory reduces occupancy (12 → 4 blocks/SM), the bandwidth improvement dominates.

## Repository

**GitHub**: https://github.com/shuochen0113/triton-sw-hack
**Branch**: `hack/sw_kernel-v1`

## Future Work

- Generalize to other DP algorithms (Needleman-Wunsch, etc.)
- Support variable shared memory allocation
- Tile-level wavefront scheduling
- Integration with Triton 3.5+ (Gluon framework)

## Contact

For technical questions about the compiler extension:
- **Email**: shuochen0113@gmail.com

## References

- **PTX Proof-of-Concept**: [experiments/ptx_modification/](../experiments/ptx_modification/)
- **Technical Documentation**: See experiments directory for detailed implementation notes
