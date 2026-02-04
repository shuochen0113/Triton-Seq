import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import json, subprocess, gc
from pathlib import Path
from datetime import datetime
import torch

# ── implementations ──────────────────────────────
# from src import gpu_sw_guided_affine as k_guided        # GPU
from src import gpu_extz as k_guided                 # GPU
from src import cpu_sw_affine as cpu_sw                 # new CPU

# ── FASTA parser ────────────────────────────────────
from Bio import SeqIO

def read_fasta_seqs(file: Path) -> list[str]:
    return [str(record.seq).upper() for record in SeqIO.parse(file, "fasta")]

# ── AGAThA launcher (unchanged) ─────────────────
AGATHA_ROOT = Path("../AGAThA")
DATASET_DIR  = AGATHA_ROOT / "dataset"
def _write_fasta(seqs, path):
    with open(path,"w") as f:
        for i,s in enumerate(seqs,1):
            f.write(f">{i}\n{s}\n")

def run_agatha(qs, rs):
    # DATASET_DIR.mkdir(parents=True, exist_ok=True)
    # _write_fasta(qs, DATASET_DIR/"generated_query.fasta")
    # _write_fasta(rs, DATASET_DIR/"generated_ref.fasta")
    subprocess.run(["bash", str(AGATHA_ROOT/"AGAThA.sh")])

# ── benchmark ────────────────────────────────────

def run_suite(band=400, match=1, mismatch=-4, gap_open=-6, gap_extend=-2, Z=751):
    dataset_path = Path("./dataset")
    qs = read_fasta_seqs(dataset_path / "query.fasta")
    rs = read_fasta_seqs(dataset_path / "ref.fasta")
    assert len(qs) == len(rs), "Query and reference lengths do not match."

    # warm‑ups
    k_guided.smith_waterman_gpu_extz(qs[:1], rs[:1],
                                              band=band, match=match, mismatch=mismatch, gap_open=gap_open,
                                              gap_extend=gap_extend, Z=Z)
    # cpu_sw.batch_sw_cpu_guided(qs[:1], rs[:1],
    #                            band=band, gap_open=gap_open,
    #                            gap_extend=gap_extend, Z=Z)

    # # CPU gold standard
    # cpu_res = cpu_sw.batch_sw_cpu_guided(
    #     qs, rs, band=band,
    #     gap_open=gap_open, gap_extend=gap_extend, Z=Z)

    # GPU run
    gpu_res = k_guided.smith_waterman_gpu_extz(
        qs, rs, band=band, match=match, mismatch=mismatch, gap_open=gap_open, gap_extend=gap_extend, Z=Z)
    torch.cuda.empty_cache(); gc.collect()

    # write JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)

    # with open(out_dir/f"guided_affine_alignments_cpu_{ts}.json","w") as f:
    #     json.dump([{"pair_id":i,"score":s,"position":list(pos)}
    #                for i,(s,pos) in enumerate(cpu_res)], f, indent=2)

    with open(out_dir/f"guided_ksw2_gpu_{ts}.json","w") as f:
        json.dump([{"pair_id":i,"score":s,"position":list(pos)}
                   for i,(s,pos) in enumerate(gpu_res)], f, indent=2)

    # run AGAThA
    run_agatha(qs, rs)
    print("✓ AGAThA run completed.")

# ----------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    run_suite()