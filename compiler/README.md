# Triton Compiler Work for Triton-Seq

This directory tracks the custom Triton compiler work used by Triton-Seq.
The compiler fork itself lives in `compiler/triton/` and is used to support
shared-memory H/E/F ring buffers for the Smith-Waterman kernels.

## Current State

There are two distinct compiler paths in this project:

| Path | Status | Purpose |
|------|--------|---------|
| **V1 automatic promotion** | Legacy, still useful | Auto-promote OPv6 H/E/F traffic into SMEM via custom MLIR passes |
| **V2 generalized SMEM API** | Active path | Expose `tl.allocate_shared` / `tl.load_shared` / `tl.store_shared` for explicit SMEM scheduling |

As of **March 16, 2026**, the V2 path is functionally working and benchmarked on
an **RTX A6000**:

- **Upstream OPv6**: `303.13 ms`
- **Hack-v2 OPv9**: `147.49 ms`
- **PTX stats (hack-v2 OPv9)**: `ld_shared=14`, `st_shared=52`, `ld_global=8`, `st_global=3`, `bar_sync=7`

The important interpretation is:

- `ld_shared` and `bar_sync` are already close to the manual-PTX target
- the remaining `st_shared` inflation is mostly **initialization code shape**
- new patches were added on **March 16, 2026** to compact initialization, but
  those patches still need a fresh rebuild and rerun

## What The Compiler Fork Adds

### V2 generalized shared-memory API

```python
buf  = tl.allocate_shared(size: constexpr, dtype: constexpr)
data = tl.load_shared(buf, offsets, mask=None, other=None)
       tl.store_shared(buf, offsets, value, mask=None)
```

This lowers through:

```text
Python tl.* builtins
  -> TTIR: tt.alloc_shared / tt.load_shared / tt.store_shared
  -> TTGIR: ttg.local_alloc / ttg.local_load_slice / ttg.local_store_slice
  -> LLVM/PTX: ld.shared / st.shared
```

### V1 automatic promotion

The older V1 route keeps the original OPv6 kernel source unchanged and rewrites
selected H/E/F traffic into shared memory through custom passes, centered on
`MaterializeSWSmem.cpp`.

## What Was Fixed Recently

The March 16, 2026 hacking round addressed three concrete codegen issues:

1. **Masked shared stores were lowered as read-modify-write**
   - fixed by letting `ttg.local_store_slice` carry an optional mask
   - lowered to predicated shared stores instead of `ld.shared + selp + st.shared`

2. **Generic membar analysis inserted too many barriers**
   - fixed by treating explicit `ttg.local_load_slice` / `ttg.local_store_slice`
     as user-managed shared-memory primitives

3. **Shared-buffer initialization still produced too many static stores**
   - OPv9 kernel init in `src/kernel/experimental/local_dp_kernel_OPv9_smem.py`
     was rewritten into strip-mined runtime fill loops
   - V1 `MaterializeSWSmem.cpp` init path was rewritten to the same runtime-fill shape
   - these init-shape changes are implemented but still need a rebuild/rerun

## Manual PTX Target

The reference target is still the manual PTX hack in
`experiments/ptx_modification/ptx/hacked_HEF.ptx`.

Its useful static PTX profile is:

- `ld_shared=14`
- `st_shared=14`
- `ld_global=8`
- `st_global=3`
- `bar_sync=7`

That manual version is the template for what “good” codegen should look like:
keep the core upstream structure, but move H/E/F traffic from global to shared
memory without introducing extra masked-store RMW sequences or bloated init code.

## Most Important Files

### In `compiler/triton/`

- `include/triton/Dialect/Triton/IR/TritonOps.td`
- `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`
- `lib/Dialect/Triton/IR/Ops.cpp`
- `lib/Dialect/TritonGPU/IR/Ops.cpp`
- `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp`
- `lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp`
- `lib/Analysis/Membar.cpp`
- `lib/Dialect/TritonGPU/Transforms/MaterializeSWSmem.cpp`
- `python/src/ir.cc`
- `python/triton/language/core.py`
- `python/triton/language/semantic.py`

### In Triton-Seq

- `src/kernel/sw_kernel.py` — OPv6 stable baseline
- `src/kernel/experimental/local_dp_kernel_OPv9_smem.py` — explicit SMEM kernel
- `benchmarks/results/compiler_compare/` — saved TTIR/TTGIR/LLVM/PTX/timing artifacts
- `docs/compiler_hacking/COMPILER_HACKING_LOG.md` — running log of the hacking work

## Build / Rerun

From `compiler/triton/`:

```bash
TRITON_BUILD_TESTING=OFF pip install -e . --no-build-isolation
```

From Triton-Seq root after rebuild:

```bash
python benchmarks/scripts/experiment_triton_compiler.py \
  --compiler-label hack-v2 \
  --kernel opv9
```

Then inspect the newest `benchmarks/results/compiler_compare/*/summary.json` and
`kernel.ptx`. The immediate expectation for the next rerun is a **lower
`st_shared` count** while keeping `ld_shared` and `bar_sync` near their current values.

## Related Docs

- `compiler/triton/README.md`
- `compiler/triton/CLAUDE.md`
- `compiler/triton_build_notes.md`
- `docs/compiler_hacking/COMPILER_HACKING_LOG.md`
- `experiments/ptx_modification/README.md`
