#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Triton‑GPU and CPU guided Smith‑Waterman and launch AGAThA.

Produces three files in ./outputs/
  • guided_alignments_cpu_<ts>.json
  • guided_alignments_gpu_<ts>.json
  • guided_compare_cpu_gpu_<ts>.json
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"          # pick your GPU

import json, random, subprocess, gc
from pathlib import Path
from datetime import datetime
import torch

# ── implementations -----------------------------------------------------------
from src import gpu_sw_guided_inter_query as k_guided
from src import cpu_sw_guided                    # has batch_sw_cpu_guided

# ── sequence helpers ----------------------------------------------------------
DNA = "ATCG"
def _mutate(seq, rate=0.05):
    s = list(seq)
    for _ in range(max(1, int(len(seq) * rate))):
        s[random.randrange(len(seq))] = random.choice(DNA)
    return "".join(s)

def gen_pair(L: int):
    ref = "".join(random.choice(DNA) for _ in range(L))
    return _mutate(ref), ref

def generate_pairs(N: int):
    qs, rs = [], []
    for _ in range(N):
        p = random.random()
        if p < .20:   L = random.randint(50,   200)
        elif p < .40: L = random.randint(500,  2000)
        elif p < .70: L = random.randint(5000, 8000)
        else:         L = random.randint(8000, 10000)
        q, r = gen_pair(L); qs.append(q); rs.append(r)
    return qs, rs

# ── AGAThA launcher (kept minimal) -------------------------------------------
AGATHA_ROOT = Path("../AGAThA")
DATASET_DIR  = AGATHA_ROOT / "dataset"

def _write_fasta(seqs, path):
    with open(path, "w") as f:
        for i, s in enumerate(seqs, 1):
            f.write(f">{i}\n{s}\n")

def run_agatha(qs, rs):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    _write_fasta(qs, DATASET_DIR / "generated_query.fasta")
    _write_fasta(rs, DATASET_DIR / "generated_ref.fasta")
    subprocess.run(["bash", str(AGATHA_ROOT / "AGAThA.sh")])

# ── main benchmark ------------------------------------------------------------
def run_suite(n_pairs: int = 10, band: int = 400, xdrop: int = 751):
    qs, rs = generate_pairs(n_pairs)

    # warm‑ups (small input to compile kernels / JIT once)
    k_guided.smith_waterman_gpu_guided_packed(qs[:1], rs[:1], band=band, xdrop=xdrop)
    cpu_sw_guided.batch_sw_cpu_guided(qs[:1], rs[:1], band=band, xdrop=xdrop)

    # ---------- CPU (reference) ----------------------------------------------
    cpu_res = cpu_sw_guided.batch_sw_cpu_guided(qs, rs, band=band, xdrop=xdrop)

    # ---------- GPU -----------------------------------------------------------
    gpu_res = k_guided.smith_waterman_gpu_guided_packed(qs, rs, band=band, xdrop=xdrop)
    # release temporary GPU buffers ASAP
    torch.cuda.empty_cache(); gc.collect()

    # ---------- compare -------------------------------------------------------
    mismatches = []
    for i, ((gs, gp), (cs, cp)) in enumerate(zip(gpu_res, cpu_res)):
        if (gs, gp) != (cs, cp):
            mismatches.append({
                "pair_id": i,
                "query": qs[i],
                "reference": rs[i],
                "gpu_score": gs, "gpu_pos": list(gp),
                "cpu_score": cs, "cpu_pos": list(cp)
            })

    # ---------- write JSON outputs -------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)

    with open(out_dir / f"guided_alignments_cpu_{ts}.json", "w") as f:
        json.dump([{"pair_id": i, "score": s, "position": list(pos)}
                   for i, (s, pos) in enumerate(cpu_res)], f, indent=2)

    with open(out_dir / f"guided_alignments_gpu_{ts}.json", "w") as f:
        json.dump([{"pair_id": i, "score": s, "position": list(pos)}
                   for i, (s, pos) in enumerate(gpu_res)], f, indent=2)

    # with open(out_dir / f"guided_compare_cpu_gpu_{ts}.json", "w") as f:
    #     if mismatches:
    #         json.dump({"mismatches": len(mismatches), "details": mismatches}, f, indent=2)
    #         print(f"⚠  {len(mismatches)} mismatch(es) – see guided_compare_cpu_gpu_{ts}.json")
    #     else:
    #         json.dump({"mismatches": 0}, f, indent=2)
    #         print("✓ CPU and GPU results are identical.")

    # ---------- launch AGAThA -------------------------------------------------
    run_agatha(qs, rs)
    print("✓ AGAThA run completed.")

# ----------------------------------------------------------------------------- 
if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    run_suite()