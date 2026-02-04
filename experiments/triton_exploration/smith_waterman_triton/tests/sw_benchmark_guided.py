#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import json, time, random, re, sys, subprocess
from io import StringIO
from datetime import datetime
from pathlib import Path
import torch
# from src import gpu_sw_guided_inter_query as k_guided
from src import gpu_sw_guided_affine as k_guided

# ──────────────────────────────────────────────────────────────────────────────
# Helpers to extract kernel-time printed by Triton
# ──────────────────────────────────────────────────────────────────────────────
def _extract_timing(stdout: str, label: str) -> float:
    pat = rf"\[{label}\].*?Kernel time:\s+([0-9.]+)\s+ms"
    m = re.search(pat, stdout)
    if not m:
        raise RuntimeError(f"Cannot find timing for {label}")
    return float(m.group(1))

def _run_with_capture(fn, label, *args, **kw):
    bak, sys.stdout = sys.stdout, StringIO()
    try:
        fn(*args, **kw)
        return _extract_timing(sys.stdout.getvalue(), label)
    finally:
        sys.stdout = bak

def _wall_time(fn, *a, **kw):
    t0 = time.time()
    fn(*a, **kw)
    torch.cuda.synchronize()
    return (time.time() - t0) * 1e3

# ──────────────────────────────────────────────────────────────────────────────
# Sequence generation
# ──────────────────────────────────────────────────────────────────────────────
DNA = "ATCG"
def _mutate(seq, rate=0.05):
    n = max(1, int(len(seq) * rate))
    s = list(seq)
    for _ in range(n):
        idx = random.randrange(len(s))
        s[idx] = random.choice(DNA)
    return "".join(s)

def gen_pair(L: int):
    ref = "".join(random.choice(DNA) for _ in range(L))
    qry = _mutate(ref)
    return qry, ref

def generate_pairs(N: int):
    qs, rs = [], []
    for _ in range(N):
        p = random.random()
        if p < .30:   L = random.randint(50, 200)
        elif p < .60: L = random.randint(500, 2000)
        elif p < .90: L = random.randint(5000, 10000)
        else:         L = random.randint(15000, 20000)
        q, r = gen_pair(L)
        qs.append(q); rs.append(r)
    return qs, rs

# ──────────────────────────────────────────────────────────────────────────────
# Run AGAThA and parse stderr and output/time.json
# ──────────────────────────────────────────────────────────────────────────────
AGATHA_ROOT = Path("../AGAThA")
DATASET_DIR = AGATHA_ROOT / "dataset"

def _write_fasta(seq_list, path: Path):
    with open(path, "w") as f:
        for i, s in enumerate(seq_list, 1):
            f.write(f">{i}\n{s}\n")

def _parse_agatha_times(stderr: str):
    values = {}
    for name in ["load", "distribution", "process", "local kernel", "total"]:
        m = re.search(rf"{name} time .*?: ([0-9.]+)", stderr)
        values[name.replace(" ", "_") + "_ms"] = float(m.group(1)) if m else -1.0
    return values

def _parse_agatha_wall_time_json():
    time_json_path = AGATHA_ROOT / "output" / "time.json"
    try:
        with open(time_json_path) as f:
            data = json.load(f)
            return data.get("AGAThA", {}).get("test", -1.0)
    except Exception:
        return -1.0

def run_agatha(qs, rs):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    _write_fasta(qs, DATASET_DIR / "generated_query.fasta")
    _write_fasta(rs, DATASET_DIR / "generated_ref.fasta")

    result = subprocess.run(["bash", str(AGATHA_ROOT / "AGAThA.sh")], capture_output=True, text=True)
    print("=== AGAThA STDERR ===")
    print(result.stderr)

    times = _parse_agatha_times(result.stderr)
    times["agatha_wall_time_json_ms"] = _parse_agatha_wall_time_json()
    return times

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark core
# ──────────────────────────────────────────────────────────────────────────────
def run_batch_kernel(qs, rs, label, kernel_fn):
    kernel_fn(qs[:1], rs[:1])  # warm up
    kt = _run_with_capture(kernel_fn, label, qs, rs)
    wt = _wall_time(kernel_fn, qs, rs)
    return kt, wt

TEST_BATCHES = [1, 10, 50, 100, 500, 1000, 2000, 5000, 10000]

def run_suite():
    print("Warm-up kernels …")
    wq, wr = generate_pairs(1)
    # k_guided.smith_waterman_gpu_guided_packed(wq, wr, band=400, xdrop=751)
    k_guided.smith_waterman_gpu_guided_affine(wq, wr, band=400, Z=751)

    run_agatha(wq, wr)

    all_res = []
    for n in TEST_BATCHES:
        print(f"\n=== {n} pair(s) ===")
        qs, rs = generate_pairs(n)

        # triton_local, triton_total = run_batch_kernel(
        #     qs, rs, "Triton Guided Timing",
        #     lambda q, r: k_guided.smith_waterman_gpu_guided_packed(q, r, band=400, xdrop=751)
        # )

        # triton_results = k_guided.smith_waterman_gpu_guided_packed(qs, rs, band=400, xdrop=751)
        triton_results = k_guided.smith_waterman_gpu_guided_affine(qs, rs, band=400, Z=751)

        triton_local = _run_with_capture(
            # lambda q, r: k_guided.smith_waterman_gpu_guided_packed(q, r, band=400, xdrop=751),
            lambda q, r: k_guided.smith_waterman_gpu_guided_affine(q, r, band=400, Z=751),
            # "Triton Guided Timing", qs, rs
            "Triton Affine Guided", qs, rs
        )
        triton_total = _wall_time(
            # lambda q, r: k_guided.smith_waterman_gpu_guided_packed(q, r, band=400, xdrop=751),
            lambda q, r: k_guided.smith_waterman_gpu_guided_affine(q, r, band=400, Z=751),
            qs, rs
        )

        alignment_output_dir = Path("outputs/triton_affine_guided_alignment_results"); alignment_output_dir.mkdir(exist_ok=True)
        alignment_file = alignment_output_dir / f"triton_alignments_{n}_pairs.json"
        with open(alignment_file, "w") as f:
            json.dump([
                {
                    "pair_id": i,
                    "query": qs[i],
                    "reference": rs[i],
                    "score": s,
                    "position": [i_pos, j_pos]
                }
                for i, (s, (i_pos, j_pos)) in enumerate(triton_results)
            ], f, indent=2)

        agatha_times = run_agatha(qs, rs)

        all_res.append({
            "num_pairs": n,
            "seq_len_min": min(map(len, qs)),
            "seq_len_max": max(map(len, qs)),
            "triton_local_kernel_time_ms": triton_local,
            "triton_total_time_ms": triton_total,
            **agatha_times
        })

    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"sw_guided_vs_agatha_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\n✓  Benchmark saved to {out_file.resolve()}")

if __name__ == "__main__":
    torch.manual_seed(0); random.seed(0)
    run_suite()
