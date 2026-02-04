#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ptx_bench.py
Read the sweep CSV and produce a set of figures comparing:
- kernel time vs #pairs
- GCUPS vs #pairs
- active blocks/SM vs #pairs
- GCUPS per active block (per-block efficiency)
- speedup(shared vs triton-native) vs #pairs
Also emits a text report that detects crossover point(s).

CSV columns expected:
  n_pairs,label,kernel_ms,GCUPS,active_blocks_sm,dyn_smem_used_B,
  regs,shared_static_B,local_bytes,max_threads_per_block,reqntid,ptx
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LABELS_ORDER = [
    "[PTX Shared ]",
    "[PTX Triton nat]",
    "[PTX Triton eq ]",
]
COLORS = {
    "[PTX Shared ]": None,
    "[PTX Triton nat]": None,
    "[PTX Triton eq ]": None,
}

def _ensure_outdir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    # normalize label spacing (some prints may differ by trailing spaces)
    def norm(x: str) -> str:
        x = x.strip()
        # unify known variants
        if "PTX Shared" in x: return "[PTX Shared ]"
        if "Triton nat" in x: return "[PTX Triton nat]"
        if "Triton eq"  in x: return "[PTX Triton eq ]"
        return x
    df["label"] = df["label"].map(norm)
    df = df[df["label"].isin(LABELS_ORDER)].copy()
    df = df.sort_values(["n_pairs", "label"])
    return df

def _fmt_absm(vals) -> str:
    # 用中位数做代表，整数就不带小数，否则保留两位
    import numpy as np
    v = float(np.median(vals))
    return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.2f}"

def _plot_xy(df: pd.DataFrame, ycol: str, ylabel: str, path: Path, logy=False):
    plt.figure(figsize=(8,5))
    for lab in LABELS_ORDER:
        sub = df[df["label"] == lab]
        if sub.empty: 
            continue
        x = sub["n_pairs"].values
        y = sub[ycol].values

        legend_lab = lab
        if "active_blocks_sm" in sub.columns and not sub["active_blocks_sm"].empty:
            absm_repr = _fmt_absm(sub["active_blocks_sm"].values)
            legend_lab = f"{lab} (actBlk/SM={absm_repr})"

        plt.plot(x, y, marker="o", label=legend_lab)

    plt.xscale("log", base=10)
    if logy:
        plt.yscale("log", base=10)
    plt.xlabel("#pairs (log10)")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _compute_per_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Avoid div by zero
    df["GCUPS_per_active_block"] = df["GCUPS"] / df["active_blocks_sm"].clip(lower=1)
    return df

def _plot_speedup_shared_vs_tn(df: pd.DataFrame, path: Path):
    # merge by n_pairs
    sh = df[df["label"] == "[PTX Shared ]"][["n_pairs","kernel_ms"]].rename(columns={"kernel_ms":"ms_shared"})
    tn = df[df["label"] == "[PTX Triton nat]"][["n_pairs","kernel_ms"]].rename(columns={"kernel_ms":"ms_tn"})
    m = pd.merge(sh, tn, on="n_pairs", how="inner")
    if m.empty:
        return None
    # speedup = (tn - sh)/tn
    m["speedup"] = (m["ms_tn"] - m["ms_shared"]) / m["ms_tn"]
    plt.figure(figsize=(8,5))
    plt.plot(m["n_pairs"], 100*m["speedup"], marker="o")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xscale("log", base=10)
    plt.xlabel("#pairs (log10)")
    plt.ylabel("Speedup of Shared over Triton-native (%)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return m

def _write_crossover_report(speedup_df: pd.DataFrame, out_txt: Path):
    if speedup_df is None or speedup_df.empty:
        out_txt.write_text("No shared/native pair to compute crossover.\n")
        return
    # Find first n_pairs where speedup < 0 (shared slower) and last where > 0
    m = speedup_df.sort_values("n_pairs")
    neg = m[m["speedup"] < 0]
    pos = m[m["speedup"] > 0]
    lines = []
    if not neg.empty and not pos.empty:
        # pick nearest neighbors around zero crossing if exists
        # simple heuristic: last positive and first negative
        last_pos = pos.iloc[-1]
        first_neg = neg.iloc[0]
        lines.append("Crossover detected:")
        lines.append(f"  last positive @ n_pairs={int(last_pos['n_pairs'])}, speedup={last_pos['speedup']*100:.2f}%")
        lines.append(f"  first negative @ n_pairs={int(first_neg['n_pairs'])}, speedup={first_neg['speedup']*100:.2f}%")
    else:
        lines.append("No sign change in speedup (shared always better or always worse on tested sizes).")
    # also dump full table
    lines.append("\nSpeedup table (n_pairs, %):")
    for _, row in m.iterrows():
        lines.append(f"{int(row['n_pairs'])}, {row['speedup']*100:.2f}")
    out_txt.write_text("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to sweep CSV.")
    ap.add_argument("--outdir", required=False, default=None, help="Output directory for figures.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent / "figs"
    _ensure_outdir(outdir)

    df = pd.read_csv(csv_path)
    df = _prep(df)

    # 1) kernel_ms vs n_pairs
    _plot_xy(df, "kernel_ms", "Kernel time (ms)", outdir / "kernel_ms_vs_pairs.png", logy=False)

    # 2) GCUPS vs n_pairs
    _plot_xy(df, "GCUPS", "GCUPS (Gcells/s)", outdir / "gcups_vs_pairs.png", logy=False)

    # 3) active_blocks/SM vs n_pairs
    _plot_xy(df, "active_blocks_sm", "Active blocks per SM", outdir / "active_blocks_vs_pairs.png", logy=False)

    # 4) GCUPS per active block (per-block efficiency)
    df_pb = _compute_per_block(df)
    _plot_xy(df_pb, "GCUPS_per_active_block", "GCUPS per active block", outdir / "gcups_per_active_block.png", logy=False)

    # 5) Speedup(shared vs triton-native)
    sp = _plot_speedup_shared_vs_tn(df, outdir / "speedup_shared_vs_triton_native.png")
    _write_crossover_report(sp, outdir / "crossover_report.txt")

    print(f"Figures saved to: {outdir}")

if __name__ == "__main__":
    main()