"""
benchmarks/scripts/benchmark_extz_OPv9.py

Head-to-head benchmark: OPv9 (SMEM) vs OPv6 (global memory).

Key difference:
  OPv6: allocates H/E/F ring buffers in global memory (host pre-allocates,
        must re-initialize with fill_ before each batch).
  OPv9: each thread block self-allocates ring buffers in shared memory;
        no host-side DP buffer allocation or init required.

Usage:
    python benchmarks/scripts/benchmark_extz_OPv9.py

Requires:
    - triton-custom installed (for tl.allocate_shared / tl.load_shared /
      tl.store_shared used by OPv9)
    - datasets/standard/query.fa + ref.fa
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
from kernel.sw_kernel import sw_kernel                                       # OPv6
from kernel.experimental.local_dp_kernel_OPv9_smem import sw_kernel_smem   # OPv9

# ── config ───────────────────────────────────────────────────────────────────
MATCH         =  1
MISMATCH      = -4
GAP_OPEN      = -6
GAP_EXTEND    = -2
BAND          = 751
Z_DROP        = 400
BLOCK         = 256
STRIDE        = ((BAND + 31) // 32) * 32   # 768

BATCH_SIZE    = 16384   # pairs per launch; tune to GPU VRAM
MAX_SEQ_LEN   = 10_000

KERNEL_KWARGS = dict(
    match_score        = MATCH,
    mismatch_score     = MISMATCH,
    gap_open_penalty   = GAP_OPEN,
    gap_extend_penalty = GAP_EXTEND,
    drop_threshold     = Z_DROP,
    SCORING_MODEL      = 'AFFINE',
    PRUNING_BAND       = 'STATIC',
    PRUNING_DROP       = 'ZDROP',
    IS_EXTENSION       = True,
    STRIDE             = STRIDE,
    BAND               = BAND,
    BLOCK              = BLOCK,
)

# ── packing helpers ──────────────────────────────────────────────────────────

def alloc_seq_bufs(n, device='cuda'):
    max_raw = n * ((MAX_SEQ_LEN + 7) & ~7)
    return {
        'raw_u8':     torch.empty(max_raw,      dtype=torch.uint8,  device=device),
        'packed_u32': torch.empty(max_raw // 8, dtype=torch.int32,  device=device),
        'lengths':    torch.empty(n,             dtype=torch.int32,  device=device),
        'offsets':    torch.empty(n,             dtype=torch.int32,  device=device),
    }

def pack_and_ptrs(seqs, bufs, stream=None):
    ctx = torch.cuda.stream(stream) if stream else torch.no_grad()
    with ctx:
        pack_batch_into_buffers(seqs, bufs, {})
    n    = len(seqs)
    ptrs = bufs['packed_u32'].data_ptr() + bufs['offsets'][:n].to(torch.int64) * 4
    lens = bufs['lengths'][:n]
    return ptrs, lens

# ── benchmark core ───────────────────────────────────────────────────────────

def benchmark(q_seqs, r_seqs, n_warmup_batches=1):
    n_total = len(q_seqs)
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  {n_total} pairs, {n_batches} batch(es) of up to {BATCH_SIZE}")

    # Pre-allocate sequence buffers (shared between OPv6 and OPv9)
    q_bufs = alloc_seq_bufs(BATCH_SIZE)
    r_bufs = alloc_seq_bufs(BATCH_SIZE)

    # Pre-allocate OPv6 DP buffers (not needed for OPv9)
    neg_inf = -10_000_000
    Hbuf = torch.full((BATCH_SIZE, 3 * STRIDE), neg_inf, dtype=torch.int32, device='cuda')
    Ebuf = torch.full((BATCH_SIZE, 2 * STRIDE), neg_inf, dtype=torch.int32, device='cuda')
    Fbuf = torch.full((BATCH_SIZE, 2 * STRIDE), neg_inf, dtype=torch.int32, device='cuda')

    outs_v6 = torch.zeros((BATCH_SIZE, 3), dtype=torch.int32, device='cuda')
    outs_v9 = torch.zeros((BATCH_SIZE, 3), dtype=torch.int32, device='cuda')

    # ── warm-up: JIT-compile both kernels ────────────────────────
    print("  Warming up (JIT compile)...")
    q_b = q_seqs[:min(BATCH_SIZE, n_total)]
    r_b = r_seqs[:min(BATCH_SIZE, n_total)]
    nb  = len(q_b)
    q_ptrs, m_arr = pack_and_ptrs(q_b, q_bufs)
    r_ptrs, n_arr = pack_and_ptrs(r_b, r_bufs)

    sw_kernel[nb,](q_ptrs, r_ptrs, m_arr, n_arr, outs_v6[:nb],
                   Hbuf[:nb], Fbuf[:nb], Ebuf[:nb], **KERNEL_KWARGS)
    sw_kernel_smem[nb,](q_ptrs, r_ptrs, m_arr, n_arr, outs_v9[:nb], **KERNEL_KWARGS)
    torch.cuda.synchronize()
    print("  JIT done.")

    # ── timed runs ────────────────────────────────────────────────
    results = {'v6': {'kernel_ms': 0.0, 'init_ms': 0.0},
               'v9': {'kernel_ms': 0.0}}

    for run_label, run_fn in [
        ('OPv6 (global mem)', 'v6'),
        ('OPv9 (SMEM)',       'v9'),
    ]:
        print(f"\n  Running {run_label}...")
        total_kernel_ms = 0.0
        total_init_ms   = 0.0

        for i in range(n_batches):
            s = i * BATCH_SIZE
            e = min(s + BATCH_SIZE, n_total)
            nb = e - s
            q_b = q_seqs[s:e]
            r_b = r_seqs[s:e]

            q_ptrs, m_arr = pack_and_ptrs(q_b, q_bufs)
            r_ptrs, n_arr = pack_and_ptrs(r_b, r_bufs)
            torch.cuda.synchronize()

            if run_fn == 'v6':
                # Re-init DP buffers (necessary for OPv6, counted as overhead)
                init_s = torch.cuda.Event(enable_timing=True)
                init_e = torch.cuda.Event(enable_timing=True)
                init_s.record()
                Hbuf[:nb].fill_(neg_inf)
                Ebuf[:nb].fill_(neg_inf)
                Fbuf[:nb].fill_(neg_inf)
                init_e.record()

                k_s = torch.cuda.Event(enable_timing=True)
                k_e = torch.cuda.Event(enable_timing=True)
                k_s.record()
                sw_kernel[nb,](q_ptrs, r_ptrs, m_arr, n_arr, outs_v6[:nb],
                               Hbuf[:nb], Fbuf[:nb], Ebuf[:nb], **KERNEL_KWARGS)
                k_e.record()
                torch.cuda.synchronize()
                total_init_ms   += init_s.elapsed_time(init_e)
                total_kernel_ms += k_s.elapsed_time(k_e)

            else:  # OPv9
                k_s = torch.cuda.Event(enable_timing=True)
                k_e = torch.cuda.Event(enable_timing=True)
                k_s.record()
                sw_kernel_smem[nb,](q_ptrs, r_ptrs, m_arr, n_arr, outs_v9[:nb],
                                    **KERNEL_KWARGS)
                k_e.record()
                torch.cuda.synchronize()
                total_kernel_ms += k_s.elapsed_time(k_e)

        results[run_fn]['kernel_ms'] = total_kernel_ms
        if run_fn == 'v6':
            results[run_fn]['init_ms'] = total_init_ms

    return results

# ── correctness spot-check ────────────────────────────────────────────────────

def spot_check(q_seqs, r_seqs, n=8):
    """Quick sanity-check that OPv9 scores match OPv6 on the first n pairs."""
    n = min(n, len(q_seqs))
    q_b, r_b = q_seqs[:n], r_seqs[:n]
    q_bufs = alloc_seq_bufs(n)
    r_bufs = alloc_seq_bufs(n)
    q_ptrs, m_arr = pack_and_ptrs(q_b, q_bufs)
    r_ptrs, n_arr = pack_and_ptrs(r_b, r_bufs)

    neg_inf = -10_000_000
    Hbuf = torch.full((n, 3 * STRIDE), neg_inf, dtype=torch.int32, device='cuda')
    Ebuf = torch.full((n, 2 * STRIDE), neg_inf, dtype=torch.int32, device='cuda')
    Fbuf = torch.full((n, 2 * STRIDE), neg_inf, dtype=torch.int32, device='cuda')
    out6 = torch.zeros((n, 3), dtype=torch.int32, device='cuda')
    out9 = torch.zeros((n, 3), dtype=torch.int32, device='cuda')

    sw_kernel[n,](q_ptrs, r_ptrs, m_arr, n_arr, out6, Hbuf, Fbuf, Ebuf, **KERNEL_KWARGS)
    sw_kernel_smem[n,](q_ptrs, r_ptrs, m_arr, n_arr, out9, **KERNEL_KWARGS)
    torch.cuda.synchronize()

    v6 = out6.cpu().tolist()
    v9 = out9.cpu().tolist()
    ok = True
    for i, (a, b) in enumerate(zip(v6, v9)):
        match = "✓" if a == b else "✗ MISMATCH"
        print(f"    pair {i:3d}: OPv6={a}  OPv9={b}  {match}")
        if a != b:
            ok = False
    return ok


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Benchmark: OPv9 (SMEM) vs OPv6 (Global Memory)")
    print("=" * 65)

    dataset_path = project_root / "datasets" / "standard"
    q_path = dataset_path / "query.fa"
    r_path = dataset_path / "ref.fa"

    if not q_path.exists():
        print(f"Dataset not found: {q_path}")
        print("Falling back to small dataset for demo.")
        q_path = project_root / "datasets" / "small" / "query_small.fa"
        r_path = project_root / "datasets" / "small" / "ref_small.fa"

    print(f"Loading {q_path.name} / {r_path.name} ...")
    t0 = time.perf_counter()
    q_seqs = read_fasta_as_bytes(q_path)
    r_seqs = read_fasta_as_bytes(r_path)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  {len(q_seqs)} pairs loaded in {load_ms:.1f} ms")

    # ── Spot-check correctness on first 8 pairs ──────────────────
    print("\n--- Spot-check correctness (first 8 pairs) ---")
    ok = spot_check(q_seqs, r_seqs, n=8)
    if not ok:
        print("\n!!! CORRECTNESS FAILURE — aborting benchmark !!!")
        sys.exit(1)
    print("  Spot-check PASSED ✓\n")

    # ── Full benchmark ───────────────────────────────────────────
    print("--- Full benchmark ---")
    results = benchmark(q_seqs, r_seqs)

    v6_k  = results['v6']['kernel_ms']
    v6_i  = results['v6']['init_ms']
    v9_k  = results['v9']['kernel_ms']

    n_total = len(q_seqs)
    total_cells = sum(len(q) * len(r) for q, r in zip(q_seqs, r_seqs))

    def gcups(ms):
        return (total_cells / (ms / 1000)) / 1e9 if ms > 0 else float('nan')

    print("\n" + "=" * 65)
    print("Results")
    print("=" * 65)
    print(f"  Total pairs : {n_total:,}")
    print(f"  Total cells : {total_cells / 1e9:.3f} GCells")
    print()
    print(f"  OPv6 (global mem)")
    print(f"    DP buf init time : {v6_i:.2f} ms")
    print(f"    Kernel time      : {v6_k:.2f} ms")
    print(f"    Total (init+ker) : {v6_i + v6_k:.2f} ms")
    print(f"    Throughput       : {gcups(v6_k):.3f} GCUPS (kernel only)")
    print(f"    Throughput       : {gcups(v6_i + v6_k):.3f} GCUPS (kernel+init)")
    print()
    print(f"  OPv9 (SMEM - tl.allocate_shared)")
    print(f"    Kernel time      : {v9_k:.2f} ms")
    print(f"    Throughput       : {gcups(v9_k):.3f} GCUPS")
    print()
    speedup_vs_kernel = v6_k / v9_k if v9_k > 0 else float('nan')
    speedup_vs_total  = (v6_i + v6_k) / v9_k if v9_k > 0 else float('nan')
    print(f"  OPv9 speedup vs OPv6 kernel      : {speedup_vs_kernel:.2f}x")
    print(f"  OPv9 speedup vs OPv6 kernel+init : {speedup_vs_total:.2f}x")
    print("=" * 65)


if __name__ == "__main__":
    main()
