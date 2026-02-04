# /benchmarking/scripts/benchmark_logan.py

import sys
import json
import time
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Framework and configuration imports
from framework_api import run_alignment_task
from configs.task_config import (
    AlignmentTask,
    SolutionConfiguration,
    create_linear_scoring,
    create_logan_x_drop_pruning
)
from utils.io import read_fasta

# --- Configuration for LOGAN ---
# Using the parameters from our successful validation runs.
CONFIG = {
    "match": 1,
    "mismatch": -1,
    "gap_penalty": -1, # Linear gap model
    "x_drop": 21
}

def calculate_total_cells(q_seqs, r_seqs):
    """Calculates the total number of cells in the DP matrices (m * n) for all pairs."""
    total_cells = 0
    for q, r in zip(q_seqs, r_seqs):
        total_cells += len(q) * len(r)
    return total_cells

def main():
    print("--- Starting Performance Benchmark for Triton LOGAN Kernel ---")
    
    # --- Load Dataset ---
    dataset_path = project_root / "datasets"
    try:
        q_seqs = read_fasta(dataset_path / "query.fa")
        r_seqs = read_fasta(dataset_path / "ref.fa")
        print(f"Loaded {len(q_seqs)} sequence pairs from dataset.")
    except FileNotFoundError:
        print(f"Error: Dataset not found in {dataset_path}. Halting benchmark.", file=sys.stderr)
        return

    total_cells = calculate_total_cells(q_seqs, r_seqs)
    print(f"Total cells to be computed (theoretical max): {total_cells / 1e9:.3f} GCells")

    # --- Configure the LOGAN Alignment Task ---
    print("\n[Configuring Alignment Task for LOGAN...]")
    logan_task = AlignmentTask(
        # LOGAN performs extension alignment from a seed.
        align_problem='Extension',
        configuration=SolutionConfiguration(
            scoring=create_linear_scoring(
                match=CONFIG['match'],
                mismatch=CONFIG['mismatch'],
                gap_penalty=CONFIG['gap_penalty']
            ),
            pruning=create_logan_x_drop_pruning(
                threshold=CONFIG['x_drop']
            )
        )
    )
    
    # --- Run Benchmark ---
    print("\n[Benchmarking Triton Framework...]")
    
    # 1. Perform a warm-up run to avoid initialization overhead in measurements
    print("Performing warm-up run with one sequence pair...")
    try:
        # Assumes your API returns 5 values on success
        _ = run_alignment_task(q_seqs[:1], r_seqs[:1], logan_task)
    except Exception as e:
        print(f"Warm-up run failed. Error: {e}")
        return

    # 2. Perform the main timed execution
    print(f"Performing timed execution run on {len(q_seqs)} sequence pairs...")
    
    # Measure total wall-clock time for the synchronous API call
    start_time = time.perf_counter()
    
    # Call your framework's API
    # This assumes `run_alignment_task` can be configured to return detailed timings.
    try:
        _, kernel_ms, malloc_h2d_ms, packing_ms, d2h_ms = run_alignment_task(q_seqs, r_seqs, logan_task)
    except ValueError:
        print("\n[Warning] `run_alignment_task` did not return detailed timings.")
        print("Falling back to measuring kernel time only.")
        try:
            _, kernel_ms = run_alignment_task(q_seqs, r_seqs, logan_task)
            malloc_h2d_ms = packing_ms = d2h_ms = 0.0 # Set other times to zero
        except Exception as e:
            print(f"FATAL: Timed run failed. Error: {e}")
            return

    end_time = time.perf_counter()
    
    total_ms = (end_time - start_time) * 1000
    
    # --- Results & Time Breakdown ---
    print("\n--- Performance Breakdown ---")
    other_ms = total_ms - (kernel_ms + malloc_h2d_ms + packing_ms + d2h_ms)

    print(f"Total End-to-End Runtime   : {total_ms:.3f} ms")
    print("-" * 35)
    print(f"  - Malloc, H2D, Packing     : {malloc_h2d_ms:.3f} ms")
    print(f"  - GPU Kernel Execution       : {kernel_ms:.3f} ms")
    print(f"  - D2H (Results Transfer)   : {d2h_ms:.3f} ms")
    print(f"  - Python Post-processing   : {other_ms:.3f} ms")
    
    # --- Final Summary Table ---
    print("\n--- Benchmark Summary ---")
    print(f"{'Tool':<18} | {'Platform':<8} | {'Kernel Time (ms)':<18} | {'Total Runtime (ms)':<20} | {'Throughput (GCUPS)':<20}")
    print("-" * 95)
    
    platform = "GPU (Triton)"
    kernel_t_str = f"{kernel_ms:.3f}"
    total_t_str = f"{total_ms:.3f}"
    gcups = f"{(total_cells / (total_ms / 1000)) / 1e9:.3f}" if total_ms > 0 else "N/A"
        
    print(f"{'Our LOGAN Kernel':<18} | {platform:<8} | {kernel_t_str:<18} | {total_t_str:<20} | {gcups:<20}")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()