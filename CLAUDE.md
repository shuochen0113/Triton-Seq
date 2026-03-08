# CLAUDE.md — Triton-Seq

Guidance for Claude Code working in this repository.

## What This Project Is

GPU-accelerated Smith-Waterman (KSW2-ESTZ) sequence alignment using Triton.
The core kernel is `src/kernel/sw_kernel.py` (OPv6).  A custom Triton compiler
fork (`compiler/triton`) adds a generalized shared-memory API for the OPv9 kernel.

## Repository Layout

```
Triton-Seq/
├── src/kernel/
│   ├── sw_kernel.py                          # OPv6 — stable, global-mem ring buffers
│   └── experimental/
│       └── local_dp_kernel_OPv9_smem.py      # OPv9 — SMEM ring buffers (tl.allocate_shared)
├── compiler/triton/                          # git submodule → triton-sw-hack
├── benchmarks/scripts/
│   ├── test_opv9_correctness.py              # OPv9 vs OPv6 correctness check
│   └── benchmark_extz_OPv9.py               # Head-to-head perf benchmark
├── docs/compiler_hacking/COMPILER_HACKING_LOG.md  # Dev log for compiler work
└── datasets/small/                           # Small test dataset (10 pairs)
```

## Two-Repo Structure

This repo (`Triton-Seq`) contains the application code.
The compiler fork lives in the **submodule** `compiler/triton`:
- Remote: `git@github.com:shuochen0113/triton-sw-hack.git`
- Active branch: `hack/smem-api-v2-rebased`
- GIT_DIR for submodule: `/workspace/Triton-Seq/.git/modules/compiler/triton`

To run git commands against the submodule use:
```bash
GIT_DIR=/workspace/Triton-Seq/.git/modules/compiler/triton \
GIT_WORK_TREE=/workspace/triton-custom \
git <command>
```

## Critical Rules

- **NEVER run `pip install -e .` in the compiler** unless the user explicitly asks.
  The compiler build takes 10-20 minutes and must be run manually by the user.
- **Do not modify `src/kernel/sw_kernel.py` (OPv6)** — it is the stable reference baseline.
- **Do not commit to `main` of Triton-Seq without explicit user instruction.**
- Kernel version naming convention: OPv1 → OPv9.  Next kernel should be OPv10.

## Kernel Versions (History)

| Version | Description |
|---------|-------------|
| OPv1–v4 | Early iterations, int16 buffers, various optimizations |
| OPv5 | Key breakthrough: reduction moved outside inner loop |
| OPv6 | Stable: int32 ring buffers, outer-loop reduction, global mem |
| OPv7–v8 | Experimental variants (affine/linear scoring, H2E2F2 config) |
| OPv9 | SMEM ring buffers via `tl.allocate_shared` (requires custom compiler) |

## Compiler State

The custom Triton fork adds three new `tl.*` builtins:
```python
tl.allocate_shared(size, dtype)  → tl.shared_buf
tl.load_shared(buf, offsets, mask, other)  → tensor
tl.store_shared(buf, offsets, value, mask)
```

Full pipeline: `tl.*` → `tt.alloc/load/store_shared` → `ttg.local_alloc` + `ttg.local_load/store_slice` → `ld/st.shared.b32` PTX.

Key constraints:
- `STRIDE ≥ BLOCK` (enforced by `LocalLoadSliceOp` verifier)
- `TT_AllocSharedOp` must NOT be `[Pure]` (CSE aliasing bug)
- `_unflatten_ir` must return `shared_buf`, not raw `ir.value`

See `docs/compiler_hacking/COMPILER_HACKING_LOG.md` for the full dev log.
See `compiler/triton/docs/smem-api/SMEM_GENERALIZED_API.md` for compiler internals.

## Test Commands

```bash
# OPv9 correctness (requires custom compiler installed)
python benchmarks/scripts/test_opv9_correctness.py

# OPv6 baseline benchmark
python benchmarks/scripts/run_baseline.py

# OPv9 vs OPv6 performance
python benchmarks/scripts/benchmark_extz_OPv9.py
```

## Environment

- GPU: H100 80GB (primary target), also tested on A6000/RTX4090
- CUDA: 12.1+
- Python: 3.10+
- PyTorch: 2.1.0+
- Conda env: `triton-seq`
