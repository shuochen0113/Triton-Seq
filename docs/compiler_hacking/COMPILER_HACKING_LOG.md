# Compiler Hacking Log — Triton-Seq Custom SMEM API

**Author:** shuochen0113
**Last Updated:** 2026-03-16

This document is a running log of the custom Triton compiler work for Triton-Seq.
It records what has been accomplished, the current state of the compiler fork,
known limitations, and what still needs to be done.

---

## Update — March 16, 2026 (A6000 investigation)

This was the first full Triton-Seq-side compiler comparison pass on the new
benchmark harness with saved TTIR/TTGIR/LLVM/PTX artifacts.

### Measured artifacts

Relevant runs in `benchmarks/results/compiler_compare/`:

- `20260316T063711Z_upstream_opv6`
- `20260316T063553Z_hack-v2_opv9`

### Measured A6000 results

| Run | Time | Throughput | PTX stats |
|-----|------|------------|-----------|
| Upstream OPv6 | `303.13 ms` | `404.80 GCUPS` | `ld_shared=4`, `st_shared=4`, `ld_global=18`, `st_global=9`, `bar_sync=6` |
| Hack-v2 OPv9 | `147.49 ms` | `832.74 GCUPS` | `ld_shared=14`, `st_shared=52`, `ld_global=8`, `st_global=3`, `bar_sync=7` |

### Manual PTX reference

The manual hacked PTX in `experiments/ptx_modification/ptx/hacked_HEF.ptx` still
defines the target code shape:

- `ld_shared=14`
- `st_shared=14`
- `ld_global=8`
- `st_global=3`
- `bar_sync=7`

### What was learned

1. The V2 generalized SMEM API is clearly working.
   - The A6000 speedup versus the newest upstream OPv6 baseline is already large.

2. The masked-store lowering bug was real and is now fixed.
   - Before the fix, masked shared stores lowered as:
     `ld.shared -> selp -> bar.sync -> st.shared`
   - After the fix, `ttg.local_store_slice` carries an optional mask and lowers
     to predicated `st.shared`.

3. The extra barrier problem was real and is now fixed.
   - `lib/Analysis/Membar.cpp` was conservatively inserting barriers between
     explicit `ttg.local_load_slice` / `ttg.local_store_slice` accesses because
     it only knew whole-buffer intervals.
   - Those ops now bypass the generic membar path.

4. The remaining `st.shared` inflation is mostly initialization.
   - In the measured OPv9 PTX, the dominant extra stores come from static H/E/F
     initialization, not from the main recurrence.
   - The DP loop itself is already close to the manual PTX structure.

### Code changes landed on March 16, 2026

#### Compiler-side

- `compiler/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`
  - `ttg.local_store_slice` now has an optional mask operand.
- `compiler/triton/lib/Dialect/TritonGPU/IR/Ops.cpp`
  - verifier updated for optional mask.
- `compiler/triton/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp`
  - `StoreSharedPattern` now preserves masks instead of synthesizing RMW.
- `compiler/triton/lib/Dialect/TritonGPU/Transforms/MaterializeSWSmem.cpp`
  - masked stores preserve masks
  - H/E/F initialization rewritten into strip-mined runtime fill loops
- `compiler/triton/lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp`
  - `LocalStoreSliceOpConversion` now lowers to predicated `targetInfo.storeShared(...)`
- `compiler/triton/lib/Analysis/Membar.cpp`
  - explicit local slice ops bypass generic membar insertion

#### Triton-Seq kernel-side

- `src/kernel/experimental/local_dp_kernel_OPv9_smem.py`
  - shared-buffer initialization rewritten from static nested loops to
    strip-mined runtime fill loops

### Important status note

The newest init-compaction patches were added **after** the latest A6000 run.
So the currently saved `st_shared=52` result is still the **pre-rerun** number.

The immediate next check after rebuilding the compiler is:

- does `ld_shared` stay near `14`?
- does `bar_sync` stay near `7`?
- does `st_shared` fall substantially from `52`?

If `st_shared` is still too high after the rerun, the next likely step is a
dedicated `tl.fill_shared` / TTIR op instead of relying only on strip-mined
source/compiler loops.

---

## Background

The core performance bottleneck of the Smith-Waterman kernel (OPv6) is H/E/F
ring-buffer bandwidth.  Each GPU thread block reads and writes three per-block
ring buffers (H: 3 slots × STRIDE int32 elements, E/F: 2 slots × STRIDE each).
At BAND=751, STRIDE=768, BLOCK=256 these are 768×3 + 768×2×2 = 5376 int32s =
21 KB per block.

Hypothesis (validated via manual PTX editing in `experiments/ptx_modification/`):
placing these buffers in **shared memory** instead of global memory provides
a ~35% kernel speedup by eliminating L2 pressure from ring-buffer traffic.

The compiler hacking effort automates this promotion — first via automatic MLIR
passes (V1), then via a generalized Python-level API (V2).

---

## V1 Hack — Automatic MLIR Promotion (`hack/sw_kernel-v1`)

**Goal:** Automatically detect H/E/F ring-buffer accesses in the original `sw_kernel`
and promote them to shared memory, with zero changes to the Python kernel code.

**How it works:**
1. `SeqAlignDetect` — detects that we are inside `@sw_kernel`
2. `PromoteSeqAlignToShared` — inserts `ttg.convert_layout` anchors tagged
   `smem.anchor = "H"|"E"|"F"` on H/E/F pointer loads
3. `MaterializeSWSmem` — allocates flat 1D `ttg.local_alloc` MemDescs for each
   buffer, derives ring slot indices from pointer arithmetic, rewrites loads/stores
   to `ttg.local_load_slice` / `ttg.local_store_slice`

**Status:** Working but hardcoded:
- STRIDE=768 and BLOCK=256 are compiled into the C++ pass
- Pattern matching is fragile: requires exact `addptr(splat(base), offset_tensor)` IR shape
- Only processes `@sw_kernel` by name
- `SeqAlignDetect.cpp` and `PromoteSeqAlignToShared.cpp` were removed from upstream
  Triton main during the rebase (March 2026); only `MaterializeSWSmem.cpp` is kept
  as a reference

**Where the code lives:**
- `compiler/triton/lib/Dialect/TritonGPU/Transforms/MaterializeSWSmem.cpp`
- Branch: `hack/smem-api-v2-rebased` (kept for reference; SeqAlign passes removed)

---

## V2 — Generalized SMEM API (`hack/smem-api-v2` / `hack/smem-api-v2-rebased`)

**Goal:** A first-class Python API for explicit per-block SMEM scratch buffers,
supporting arbitrary BAND/STRIDE parameters, with correctness verified end-to-end.

### New Language Primitives

```python
buf  = tl.allocate_shared(size: constexpr, dtype: constexpr) -> tl.shared_buf
data = tl.load_shared(buf, offsets, mask=None, other=None)   -> tensor
       tl.store_shared(buf, offsets, value, mask=None)
```

### Compiler Pipeline (full-stack)

```
Python @triton.jit
  tl.allocate_shared(N, tl.int32)
  tl.load_shared(buf, offsets, mask, other)
  tl.store_shared(buf, offsets, value, mask)
        ↓  Python JIT (core.py / semantic.py)
TTIR:   tt.alloc_shared : !tt.shared_buf<N, i32>
        tt.load_shared %buf[%off], mask=%m, other=%x
        tt.store_shared %buf[%off], %val, mask=%m
        ↓  TritonToTritonGPUPass.cpp
TTGIR:  ttg.local_alloc (1D flat LinearSharedEncodingAttr)
        ttg.local_load_slice %memdesc, %flat_offsets
        ttg.local_store_slice %val, %memdesc, %flat_offsets
        ↓  MemoryOpToLLVM.cpp
PTX:    ld.shared.b32 / st.shared.b32
```

### OPv9 Kernel (`src/kernel/experimental/local_dp_kernel_OPv9_smem.py`)

Same logic as OPv6, but H/E/F ring buffers are allocated with `tl.allocate_shared`
inside the kernel body.  No `Hbuf`/`Fbuf`/`Ebuf` global tensor arguments.  Each
GPU thread block's ring buffers live entirely in shared memory.

```python
Hsmem = tl.allocate_shared(3 * STRIDE, tl.int32)
Esmem = tl.allocate_shared(2 * STRIDE, tl.int32)
Fsmem = tl.allocate_shared(2 * STRIDE, tl.int32)

Hleft = tl.load_shared(Hsmem, slot_base + lane_off, mask=lane_mask & valid, other=MINF)
tl.store_shared(Hsmem, curr_slot + lane_off, H_new, mask=lane_mask)
```

### Verification Results (H100 80GB, CUDA 13.1)

| Config | OPv9 SMEM | OPv6 Global | Match |
|--------|-----------|-------------|-------|
| BAND=127, STRIDE=128 | ✓ | ✓ | ✓ |
| BAND=251, STRIDE=256 | ✓ | ✓ | ✓ |
| BAND=501, STRIDE=512 | ✓ | ✓ | ✓ |
| BAND=751, STRIDE=768 | ✓ | ✓ | ✓ |
| BAND=1023, STRIDE=1024 | ✓ | ✓ | ✓ |
| BAND=51, STRIDE=64 | Rejected by verifier (STRIDE < BLOCK) | — | Expected |

**Correctness:** 16,384 real sequence pairs — all `(score, best_i, best_j)` match
OPv6 exactly.

**Performance (BAND=751, 122.6 GCells):**

| Kernel | Time | Throughput | vs OPv6 |
|--------|------|------------|---------|
| OPv6 (global mem) | 97.4 ms | 1260 GCUPS | baseline |
| OPv9 (SMEM) | 99.6 ms | 1232 GCUPS | 0.98× |

Performance parity is expected: at this batch size, OPv6's 21 KB/block ring buffers
fit in H100 L2 cache, providing similar bandwidth to SMEM.  The main benefit of OPv9
is **eliminating ~344 MB of pre-allocated global ring-buffer memory**.

**PTX confirmed real SMEM accesses:**
```
ld.shared.b32   %r133, [%r132];       // H ring read
st.shared.b32   [%r266], %r272;       // H ring write
ld.shared.b32   %r214, [%r213+9216];  // E ring (base = 2304*4 B)
st.shared.b32   [%r251+15360], %r261; // F ring (base = 3840*4 B)
```

### Key Design Decisions (Bugs Fixed During Development)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `shared_buf_type has no attr is_block` | `semantic.py` returned `tl.tensor(handle, shared_buf_type(...))` but `tl.tensor.__init__` calls `type.is_block()` | Return `tl.shared_buf(handle, shared_buf_type(...))` instead |
| `'ir.value' has no attr 'handle'` | `shared_buf_type._unflatten_ir` returned raw `ir.value` instead of `shared_buf` — breaks for-loop scope cloning | `return shared_buf(handles[cursor], self), cursor + 1` |
| `arith.select type mismatch` | `LoadSharedPattern` passed scalar `other` (MINF from `tl.full((),...)`) to `arith.select` expecting tensor | Splat scalar `other` to result tensor type before select |
| CSE merges Esmem and Fsmem | `TT_AllocSharedOp` was `[Pure]` — CSE merges two same-size allocs | Remove `[Pure]`; use `DeclareOpInterfaceMethods<MemoryEffectsOpInterface>` with `Allocate+Write` |
| No `BlockedEncodingAttr` on load result | `LoadSharedPattern` used original TTIR type for `LocalLoadSliceOp` result | Use `getTypeConverter()->convertType(origResultTy)` |
| `MemDescType::verify` power-of-2 failure | 2D MemDesc `[slots, STRIDE=768]` fails because 768 is not power-of-2 | Use 1D flat `[N]` shape — `drop_front(1)` is vacuously empty |

---

## Repository Structure

```
compiler/triton/                  # git submodule → triton-sw-hack
  branch: hack/smem-api-v2
  remote: git@github.com:shuochen0113/triton-sw-hack.git

  Key modified files:
    include/triton/Dialect/Triton/IR/
      TritonTypes.td              # TT_SharedBufType
      TritonOps.td                # TT_AllocSharedOp/LoadSharedOp/StoreSharedOp
      Dialect.h                   # SharedMemory resource (mlir::triton namespace)
    include/triton/Dialect/TritonGPU/IR/
      TritonGPUOps.td             # TTG_LocalLoadSliceOp / TTG_LocalStoreSliceOp
    lib/Dialect/Triton/IR/Ops.cpp
    lib/Dialect/TritonGPU/IR/Ops.cpp
    lib/Conversion/TritonToTritonGPU/
      TritonToTritonGPUPass.cpp   # AllocSharedPattern / LoadSharedPattern / StoreSharedPattern
      TritonGPUConversion.cpp     # SharedBufType → MemDescType type converter
    lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp
    lib/Dialect/TritonGPU/Transforms/MaterializeSWSmem.cpp  # V1 legacy
    python/src/ir.cc
    python/triton/language/{core,semantic,__init__}.py

  Full detail: compiler/triton/docs/smem-api/SMEM_GENERALIZED_API.md
```

---

## What Still Needs Development

### Short-Term (next session)

1. **Upstream rebase (deferred)**
   A `hack/smem-api-v2-rebased` branch was attempted (cherry-pick onto upstream Triton
   main, March 2026) but the build failed due to upstream API breakage
   (`llvm/Plugins/PassPlugin.h` missing + other changes).  Branch was deleted.
   The working, verified version is `hack/smem-api-v2`.  Rebasing onto a newer upstream
   is left for a future session when Triton API differences can be addressed properly.

2. **MaterializeSWSmem V1 re-integration (optional)**
   `SeqAlignDetect.cpp` and `PromoteSeqAlignToShared.cpp` are not in the current fork.
   If V1 (automatic promotion without code changes to `sw_kernel`) is still desired,
   these passes need to be re-added and updated to current Triton APIs.

3. **H100 performance with larger batches**
   At BAND=751, BLOCK=256 with 16,384 pairs, ring buffers fit in L2 → OPv9 ≈ OPv6.
   Test with much larger batches (e.g., 128K pairs) or bigger BAND widths where L2
   eviction would penalize OPv6, to see real SMEM advantage.

### Medium-Term

4. **Vectorized SMEM accesses (`ld.shared.v4.b32`)**
   Currently generates scalar `ld.shared.b32` per thread.  If offsets are 128-bit
   aligned, vectorizing to 4-element loads would 4× the effective SMEM bandwidth per
   instruction and potentially unlock more MIO throughput.

5. **STRIDE < BLOCK support**
   Currently rejected by the `LocalLoadSliceOp` verifier (`STRIDE ≥ BLOCK` required).
   Could be supported by tiling the offset tensor into multiple sub-BLOCK iterations
   inside the lowering — useful for very narrow band widths.

6. **Warp-level `ld.shared` pipeline (pipelining)**
   Insert `cp.async.wait` / `ldmatrix` style double-buffering for the ring buffer
   reads if the DP loop latency becomes register-bound at high occupancy.

7. **Integration into the main `src/` API**
   `sw_kernel_smem` (OPv9) is currently in `src/kernel/experimental/`.  Once the
   rebase is verified and a performance benefit is demonstrated at scale, promote it
   to `src/kernel/sw_kernel_smem.py` and wire it into `src/api.py` as the default
   when the custom compiler is installed.

### Long-Term / Research

8. **General-purpose SMEM API for other kernels**
   The `tl.allocate_shared` API is domain-agnostic.  Other Triton kernels that do
   repeated reads from small working sets (e.g., attention QK tiles, scan prefix
   trees) could benefit from the same approach.

9. **Upstream contribution**
   The `TTG_LocalLoadSliceOp` / `TTG_LocalStoreSliceOp` ops and the `tl.allocate_shared`
   Python API could potentially be contributed back to upstream Triton after cleanup and
   a test suite is added.

---

## Branches at a Glance

| Branch | Base | Description |
|--------|------|-------------|
| `hack/sw_kernel-v1` | old upstream merge | Original V1 automatic MLIR passes; OPv6 + SeqAlign passes |
| `hack/smem-api-v2` | `hack/sw_kernel-v1` | **Current** — V2 generalized SMEM API, verified on H100 |

---

## Build Instructions

```bash
cd compiler/triton
git checkout hack/smem-api-v2

# Build (do NOT run pip install -e . unless ready)
pip install -r python/requirements.txt
TRITON_BUILD_TESTING=OFF pip install -e .

# Verify SMEM API is available
python -c "import triton.language as tl; print(tl.allocate_shared)"

# Test OPv9 correctness
python benchmarks/scripts/test_opv9_correctness.py

# Performance benchmark
python benchmarks/scripts/benchmark_extz_OPv9.py
```
