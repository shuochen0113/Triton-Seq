"""
benchmarks/scripts/test_opv9_correctness.py

Correctness test: OPv9 (SMEM, tl.allocate_shared) vs OPv6 (global memory).
Both kernels should produce identical (score, best_i, best_j) for every pair.

Usage:
    python benchmarks/scripts/test_opv9_correctness.py

Requires:
    - triton-custom installed (for tl.allocate_shared / tl.load_shared / tl.store_shared)
    - torch with CUDA
"""

import sys, time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
from utils.packing import pack_batch_into_buffers
from utils.io import read_fasta_as_bytes
from kernel.sw_kernel import sw_kernel            # OPv6 (global memory)
from kernel.experimental.local_dp_kernel_OPv9_smem import sw_kernel_smem  # OPv9 (SMEM)

# ── shared config ────────────────────────────────────────────────────────────
MATCH         =  1
MISMATCH      = -4
GAP_OPEN      = -6
GAP_EXTEND    = -2
BAND          = 751
Z_DROP        = 400
BLOCK         = 256
STRIDE        = ((BAND + 31) // 32) * 32  # 768

KERNEL_KWARGS = dict(
    match_score       = MATCH,
    mismatch_score    = MISMATCH,
    gap_open_penalty  = GAP_OPEN,
    gap_extend_penalty= GAP_EXTEND,
    drop_threshold    = Z_DROP,
    SCORING_MODEL     = 'AFFINE',
    PRUNING_BAND      = 'STATIC',
    PRUNING_DROP      = 'ZDROP',
    IS_EXTENSION      = True,
    STRIDE            = STRIDE,
    BAND              = BAND,
    BLOCK             = BLOCK,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def pack_seqs(seqs, device='cuda'):
    """Pack a list of byte-sequences into GPU buffers; return (q_ptrs, lengths)."""
    max_len  = max(len(s) for s in seqs)
    n        = len(seqs)
    max_raw  = n * ((max_len + 7) & ~7)

    d_raw    = torch.empty(max_raw,       dtype=torch.uint8,  device=device)
    d_packed = torch.empty(max_raw // 8,  dtype=torch.int32,  device=device)
    d_lens   = torch.empty(n,             dtype=torch.int32,  device=device)
    d_offs   = torch.empty(n,             dtype=torch.int32,  device=device)

    bufs = {'raw_u8': d_raw, 'packed_u32': d_packed, 'lengths': d_lens, 'offsets': d_offs}
    pack_batch_into_buffers(seqs, bufs, {})
    torch.cuda.synchronize()

    # Build pointer array: int64 absolute addresses, one per sequence
    ptrs = (d_packed.data_ptr() + d_offs[:n].to(torch.int64) * 4)
    return ptrs, d_lens[:n]


def run_opv6(q_ptrs, r_ptrs, m_arr, n_arr, n_pairs):
    """Run OPv6 (global memory ring buffers) and return outs tensor."""
    neg_inf = -10_000_000
    Hbuf = torch.full((n_pairs, 3 * STRIDE), neg_inf, dtype=torch.int32, device='cuda')
    Ebuf = torch.full((n_pairs, 2 * STRIDE), neg_inf, dtype=torch.int32, device='cuda')
    Fbuf = torch.full((n_pairs, 2 * STRIDE), neg_inf, dtype=torch.int32, device='cuda')
    outs = torch.zeros((n_pairs, 3), dtype=torch.int32, device='cuda')

    sw_kernel[n_pairs,](
        q_ptrs, r_ptrs, m_arr, n_arr, outs,
        Hbuf, Fbuf, Ebuf,
        **KERNEL_KWARGS,
    )
    torch.cuda.synchronize()
    return outs


def run_opv9(q_ptrs, r_ptrs, m_arr, n_arr, n_pairs):
    """Run OPv9 (per-block SMEM ring buffers) and return outs tensor."""
    outs = torch.zeros((n_pairs, 3), dtype=torch.int32, device='cuda')

    sw_kernel_smem[n_pairs,](
        q_ptrs, r_ptrs, m_arr, n_arr, outs,
        **KERNEL_KWARGS,
    )
    torch.cuda.synchronize()
    return outs


# ── test driver ──────────────────────────────────────────────────────────────

def test_dataset(q_seqs, r_seqs, label):
    n = len(q_seqs)
    print(f"\n[{label}] {n} pairs")

    q_ptrs, m_arr = pack_seqs(q_seqs)
    r_ptrs, n_arr = pack_seqs(r_seqs)

    # Warm-up (JIT compile both kernels)
    _ = run_opv6(q_ptrs, r_ptrs, m_arr, n_arr, n)
    _ = run_opv9(q_ptrs, r_ptrs, m_arr, n_arr, n)

    # Timed runs
    t0 = time.perf_counter()
    out_v6 = run_opv6(q_ptrs, r_ptrs, m_arr, n_arr, n)
    t1 = time.perf_counter()
    out_v9 = run_opv9(q_ptrs, r_ptrs, m_arr, n_arr, n)
    t2 = time.perf_counter()

    print(f"  OPv6 kernel : {(t1-t0)*1000:.2f} ms")
    print(f"  OPv9 kernel : {(t2-t1)*1000:.2f} ms")

    v6 = out_v6.cpu().tolist()
    v9 = out_v9.cpu().tolist()

    mismatches = 0
    for i, (a, b) in enumerate(zip(v6, v9)):
        if a != b:
            print(f"  MISMATCH pair {i}: OPv6={a}  OPv9={b}")
            mismatches += 1
        else:
            s, bi, bj = a
            print(f"  pair {i}: score={s}  best=({bi},{bj})  ✓")

    if mismatches == 0:
        print(f"  ✓  All {n} pairs match!")
    else:
        print(f"  ✗  {mismatches}/{n} pairs MISMATCH")
    return mismatches == 0


def main():
    print("=" * 60)
    print("OPv9 Correctness Test (SMEM vs Global Memory)")
    print("=" * 60)

    # ── 1. Synthetic micro-test ───────────────────────────────────
    q_micro = [
        b"AAAAA",
        b"ACGT" * 50,       # 200 bp
        b"ACGTACGT" * 20,   # 160 bp
    ]
    r_micro = [
        b"AAAAA",
        b"ACGT" * 52,       # 208 bp – slightly longer
        b"TGCATGCA" * 20,   # 160 bp – complement-ish
    ]
    ok1 = test_dataset(q_micro, r_micro, "Synthetic micro-test")

    # ── 2. Small FASTA dataset ────────────────────────────────────
    small_q = project_root / "datasets" / "small" / "query_small.fa"
    small_r = project_root / "datasets" / "small" / "ref_small.fa"
    if small_q.exists():
        q_seqs = read_fasta_as_bytes(small_q)
        r_seqs = read_fasta_as_bytes(small_r)
        ok2 = test_dataset(q_seqs, r_seqs, "datasets/small")
    else:
        print("\n[datasets/small] NOT FOUND – skipping")
        ok2 = True

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if ok1 and ok2:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
