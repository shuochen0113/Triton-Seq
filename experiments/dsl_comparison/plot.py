"""
======================================================================================
Project: DSL for Sequence Alignment
File: plot.py
Date: Dec 11, 2025 (Fix: Better Aesthetics & Line Charts)
======================================================================================
"""

import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ====================================================================================
# Configuration & Style
# ====================================================================================

# Set a professional academic theme
sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

# Font adjustments for publication quality
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "Bitstream Vera Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})

# 1. DSL Color Palette (Refined for aesthetics)
DSL_PALETTE = {
    "CPU":  "#7f7f7f",  # Gray
    "CUDA":  "#2ca02c",  # Green
    "Triton":         "#ff7f0e",  # Orange
    "TileLang":       "#1f77b4",  # Blue
    "ThunderKittens": "#9467bd",  # Purple
    "CuTile":         "#17becf",  # Teal
    "Codon":          "#d62728"   # Red
}

# 2. Hardware Color Palette (Distinct colors for lines)
GPU_PALETTE = {
    "RTX 4090": "#1f77b4", # Blue
    "RTX 5090": "#d62728", # Red
    "RTX A6000": "#7f7f7f" # Grey
}

# User-defined ordering for Abstraction Level (Low/Hard -> High/Easy)
ABSTRACTION_ORDER = [
    "CUDA",
    "ThunderKittens",
    "TileLang",
    "Triton",
    "CuTile"
]

# ====================================================================================
# Data Loading
# ====================================================================================

def parse_summary_file(filepath):
    with open(filepath, 'r') as f:
        content = f.readlines()

    data = {"cpu": "Unknown", "gpu": "Unknown", "results": []}
    
    for line in content:
        line = line.strip()
        if line.startswith("Hardware:"):
            parts = line.replace("Hardware:", "").split(",")
            for p in parts:
                if "GPU=" in p: data["gpu"] = p.split("=")[1].strip()
        
        if "|" in line and "Backend" not in line and "---" not in line and "=" not in line:
            parts = [x.strip() for x in line.split("|")]
            if len(parts) == 3:
                try:
                    data["results"].append({"Backend": parts[0], "GCUPS": float(parts[2])})
                except ValueError: pass
    return data

def load_all_data(input_dir):
    files = glob.glob(os.path.join(input_dir, "summary_*.txt"))
    if not files: return pd.DataFrame()

    all_records = []
    print(f"[Info] Found {len(files)} log files.")
    
    for f in files:
        parsed = parse_summary_file(f)
        gpu_raw = parsed["gpu"]
        
        # Normalize GPU Names
        if "RTX 4090" in gpu_raw: gpu_label = "RTX 4090"
        elif "RTX 5090" in gpu_raw: gpu_label = "RTX 5090"
        elif "A6000" in gpu_raw: gpu_label = "RTX A6000"
        else: gpu_label = gpu_raw
        
        for res in parsed["results"]:
            res["GPU"] = gpu_label
            res["File"] = os.path.basename(f)
            all_records.append(res)
            
    return pd.DataFrame(all_records)

# ====================================================================================
# Plotters
# ====================================================================================

def plot_shootout(df, output_dir):
    """
    Chart 1: Absolute Performance (GCUPS) - Bar Chart
    """
    plt.figure(figsize=(14, 8))
    
    # Filter order
    order = ["CPU"] + [x for x in ABSTRACTION_ORDER if x in df["Backend"].unique()]
    
    # Grouped Bar Plot
    ax = sns.barplot(
        data=df, 
        x="GPU",       
        y="GCUPS", 
        hue="Backend", 
        hue_order=order,
        palette=DSL_PALETTE, 
        edgecolor="white",
        linewidth=1.5,
        errorbar=None
    )
    
    # Aesthetics
    sns.despine(left=True)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    
    # Add number labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=5, fontsize=11, fontweight='bold')

    plt.title("Sequence Alignment Throughput\n(Smith-Waterman, Affine Gap Penalty, 16384 Dataset)", fontsize=20, fontweight='bold', pad=20)
    plt.ylabel("Throughput (GCUPS)", fontsize=16, labelpad=10)
    plt.xlabel("", fontsize=16)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14)
    
    # Legend
    plt.legend(title="", bbox_to_anchor=(1.0, 1.0), loc='upper left', frameon=False, fontsize=13)
    
    plt.tight_layout()
    
    out = os.path.join(output_dir, "plot_gcups_shootout.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"[Output] Saved {out}")
    plt.close()

def plot_speedup(df, output_dir):
    """
    Chart 2: Speedup vs Abstraction Level - Line Chart
    Trends across abstraction levels for different GPUs.
    """
    # 1. Calculate Speedup
    df_speedup = df.copy()
    df_speedup = df_speedup[df_speedup["Backend"] != "CPU"]
    
    # Filter only backends present in ABSTRACTION_ORDER
    df_speedup = df_speedup[df_speedup["Backend"].isin(ABSTRACTION_ORDER)]

    def calc_rel(x):
        cuda_row = x[x["Backend"] == "CUDA"]
        if cuda_row.empty: 
            x["Speedup"] = 0.0
            return x
        baseline = cuda_row["GCUPS"].mean()
        x["Speedup"] = x["GCUPS"] / baseline
        return x

    df_speedup = df_speedup.groupby("GPU", group_keys=False).apply(calc_rel)
    
    # 2. Plot Setup
    plt.figure(figsize=(14, 8))
    
    # Ensure categorical order for X-axis
    df_speedup["Backend"] = pd.Categorical(df_speedup["Backend"], categories=ABSTRACTION_ORDER, ordered=True)
    df_speedup.sort_values("Backend", inplace=True)

    # Line Plot
    # markers=True adds dots, dashes=False makes lines solid
    ax = sns.lineplot(
        data=df_speedup, 
        x="Backend", 
        y="Speedup", 
        hue="GPU", 
        style="GPU",
        palette=GPU_PALETTE,
        markers=True, 
        dashes=False,
        linewidth=3, 
        markersize=12
    )

    # Add Baseline Line (y=1.0)
    plt.axhline(1.0, color='gray', linestyle=':', linewidth=2, alpha=0.8, zorder=0)
    plt.text(-0.4, 0.95, "Baseline (Native CUDA)", color='gray', fontsize=12, style='italic')

    # Aesthetics
    sns.despine()
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    
    # Add value labels next to points
    # We iterate over the data to place text manually for clarity
    for gpu_name in df_speedup["GPU"].unique():
        subset = df_speedup[df_speedup["GPU"] == gpu_name]
        color = GPU_PALETTE.get(gpu_name, "black")
        for _, row in subset.iterrows():
            if row["Backend"] == "CUDA": continue # Skip baseline label to avoid clutter
            ax.text(
                x=row["Backend"], 
                y=row["Speedup"] + 0.02, 
                s=f"{row['Speedup']:.2f}x", 
                color=color, 
                ha="center", 
                va="bottom", 
                fontsize=11, 
                fontweight='bold'
            )

    plt.title("Performance Scaling vs. Abstraction Level\n(Smith-Waterman, Affine Gap Penalty, 16384 Dataset)", fontsize=20, fontweight='bold', pad=20)
    plt.ylabel("Speedup relative to Native CUDA", fontsize=16, labelpad=10)
    plt.xlabel("Abstraction Level (Lower → Higher)", fontsize=16, labelpad=10)
    
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=14)
    
    # Legend
    plt.legend(title="", bbox_to_anchor=(0.02, 0.98), loc='upper left', frameon=True, fontsize=13)
    
    plt.tight_layout()

    out = os.path.join(output_dir, "plot_speedup_abstraction.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"[Output] Saved {out}")
    plt.close()

# ====================================================================================
# Main
# ====================================================================================

if __name__ == "__main__":
    input_dir = "results/bench_history"
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Plot] Loading data from {input_dir}...")
    df = load_all_data(input_dir)
    
    if not df.empty:
        print(f"[Info] Found {len(df)} data points.")
        print("[Plot] Generating GCUPS Shootout...")
        plot_shootout(df, output_dir)
        
        print("[Plot] Generating Speedup vs Abstraction...")
        plot_speedup(df, output_dir)
    else:
        print("[Error] No data found.")