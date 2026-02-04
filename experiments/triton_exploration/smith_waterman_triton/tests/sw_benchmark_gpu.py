import os
import json
import time
import random
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
sys.path.append("../src")

import gpu_sw_basic
from optimize_experiment import gpu_sw_multi_grid, gpu_sw_single_block, gpu_sw_basic

def generate_test_case(base_len=5000, var_len=1000):
    """
    Generate a synthetic DNA sequence pair with mutations and insertions.
    """
    base = [random.choice("ATCG") for _ in range(base_len)]
    test1 = base + [random.choice("ATCG") for _ in range(var_len)]
    test2 = base.copy()

    # Introduce mutations in the last 100 bases
    test2[-100:] = [random.choice("ATCG") for _ in range(100)]
    # Insert a gap of 50 characters at position 300
    test2.insert(300, "-" * 50)
    # Introduce 100 random mutations
    for _ in range(100):
        idx = random.randint(0, len(test2) - 1)
        test2[idx] = random.choice("ATCG")
    
    return "".join(test1).replace("-", ""), "".join(test2).replace("-", "")

def run_experiment(test_case):
    """
    Run Smith-Waterman alignment using three GPU kernels.
    """
    seq_a, seq_b = test_case
    results = {}
    print(f"\nStarting experiment for sequences of length {len(seq_a)} and {len(seq_b)}")
    
    # GPU Execution: Basic Kernel
    # print("Running Smith-Waterman on GPU (Basic Kernel)...")
    # torch.cuda.synchronize()
    # gpu_basic_start = time.time()
    # gpu_score_basic, gpu_pos_basic, _ = gpu_sw_basic.smith_waterman_gpu_basic(seq_a, seq_b)
    # torch.cuda.synchronize()
    # gpu_basic_time = time.time() - gpu_basic_start
    # print(f"Basic GPU execution completed in {gpu_basic_time:.4f} seconds")

    # GPU Execution: Single-Launch Kernel
    print("Running Smith-Waterman on GPU (Single-Launch Kernel)...")
    torch.cuda.synchronize()
    gpu_single_start = time.time()
    gpu_score_single, gpu_pos_single, _ = gpu_sw_single_block.smith_waterman_gpu_single_block(seq_a, seq_b)
    torch.cuda.synchronize()
    gpu_single_time = time.time() - gpu_single_start
    print(f"Single-Launch GPU execution completed in {gpu_single_time:.4f} seconds")

    # GPU Execution: Multi-Grid Kernel
    print("Running Smith-Waterman on GPU (Multi-Grid Kernel)...")
    torch.cuda.synchronize()
    gpu_multi_start = time.time()
    gpu_score_multi, gpu_pos_multi, _ = gpu_sw_multi_grid.smith_waterman_gpu_multi_grid(seq_a, seq_b)
    torch.cuda.synchronize()
    gpu_multi_time = time.time() - gpu_multi_start
    print(f"Multi-Grid GPU execution completed in {gpu_multi_time:.4f} seconds")

    # Check for consistency
    outputs_match = (
        # gpu_score_basic == gpu_score_single == gpu_score_multi and
        # gpu_pos_basic == gpu_pos_single == gpu_pos_multi

        gpu_score_single == gpu_score_multi and
        gpu_pos_single == gpu_pos_multi
    )

    if outputs_match:
        print("All GPU implementations produced matching results ✅")
    else:
        print("Warning: GPU implementations do not match ❌")
        # print(f"Basic GPU Score: {gpu_score_basic}, Position: {gpu_pos_basic}")
        print(f"Single-Launch GPU Score: {gpu_score_single}, Position: {gpu_pos_single}")
        print(f"Multi-Grid GPU Score: {gpu_score_multi}, Position: {gpu_pos_multi}")

    # Store performance results
    results["base_length"] = len(seq_a)
    results["variant_length"] = len(seq_b)
    results["outputs_match"] = outputs_match
    # results["gpu_time_basic"] = gpu_basic_time
    results["gpu_time_single"] = gpu_single_time
    results["gpu_time_multi"] = gpu_multi_time
    # results["speedup_single_vs_basic"] = gpu_basic_time / gpu_single_time if gpu_single_time > 0 else 0
    # results["speedup_multi_vs_basic"] = gpu_basic_time / gpu_multi_time if gpu_multi_time > 0 else 0
    results["speedup_multi_vs_single"] = gpu_single_time / gpu_multi_time if gpu_multi_time > 0 else 0

    return results

def main():
    print(f"Total GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    """
    Run experiments on multiple test cases and save the benchmark results.
    """
    test_cases = [
        (100, 50),
        (500, 200),
        (2000, 1000),
        (5000, 3000),
        (10000, 5000),
        # (30000, 20000),
        # (50000, 50000)
    ]
    
    all_results = []
    print("Starting Smith-Waterman Benchmarking...\n")

    for base_len, var_len in test_cases:
        print(f"\nGenerating test case with base_len={base_len}, var_len={var_len}")
        test_case = generate_test_case(base_len, var_len)
        result = run_experiment(test_case)
        all_results.append(result)
    
    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"sw_gpu_optimizing_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nBenchmark results saved to {output_path}")

if __name__ == "__main__":
    main()