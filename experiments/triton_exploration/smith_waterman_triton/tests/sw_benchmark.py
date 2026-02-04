"""
sw_benchmark.py

Benchmark the performance of the CPU Smith-Waterman algorithm and three GPU Smith-Waterman
kernels:
  - Basic GPU kernel (file: gpu_sw_basic.py, function: smith_waterman_gpu_basic)
  - Block-per-diagonal GPU kernel (file: gpu_sw_diagonal.py, function: smith_waterman_gpu_diagonal)
  - Single-launch GPU kernel (file: gpu_sw_single_block.py, function: smith_waterman_gpu_single_block)

The benchmark generates synthetic test cases and then measures execution time, correctness,
and speedup for each implementation.
"""

import os
import json
import time
import random
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Import the CPU and GPU implementations.
# Adjust the import paths if needed.
from src import cpu_sw
from src import gpu_sw_basic
from src import gpu_sw_diagonal
from src import gpu_sw_single_block


def generate_test_case(base_len=5000, var_len=1000):
    """
    Generate a synthetic DNA sequence pair with mutations and insertions.
    
    Parameters:
        base_len (int): Base length of the common sequence.
        var_len (int): Additional length for the first sequence.
    
    Returns:
        tuple: (sequence_a, sequence_b) as strings.
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

    # Remove gap characters and return as strings.
    return (
        "".join(test1).replace("-", ""),
        "".join(test2).replace("-", "")
    )


def run_experiment(test_case):
    """
    Run Smith-Waterman alignment using the CPU and three GPU kernels.
    
    Parameters:
        test_case (tuple): (sequence_a, sequence_b)
    
    Returns:
        dict: A dictionary containing performance results and outputs.
    """
    seq_a, seq_b = test_case
    results = {}

    print(f"\nStarting experiment for sequences of length {len(seq_a)} and {len(seq_b)}")

    # Warm-up GPU (optional, helps stabilize timings)
    _ = gpu_sw_basic.smith_waterman_gpu_basic(seq_a, seq_b)
    torch.cuda.synchronize()

    # CPU Execution
    print("Running Smith-Waterman on CPU...")
    cpu_start = time.time()
    cpu_score, cpu_pos, _ = cpu_sw.smith_waterman_cpu(seq_a, seq_b)
    cpu_time = time.time() - cpu_start
    print(f"CPU execution completed in {cpu_time:.4f} seconds")

    # GPU Execution: Basic Kernel
    print("Running Smith-Waterman on GPU (Basic Kernel)...")
    torch.cuda.synchronize()
    gpu_basic_start = time.time()
    gpu_score_basic, gpu_pos_basic, _ = gpu_sw_basic.smith_waterman_gpu_basic(seq_a, seq_b)
    torch.cuda.synchronize()
    gpu_basic_time = time.time() - gpu_basic_start
    print(f"Basic GPU execution completed in {gpu_basic_time:.4f} seconds")

    # GPU Execution: Block-per-Diagonal Kernel
    print("Running Smith-Waterman on GPU (Block Kernel)...")
    torch.cuda.synchronize()
    gpu_diagonal_start = time.time()
    gpu_score_diagonal, gpu_pos_diagonal, _ = gpu_sw_diagonal.smith_waterman_gpu_diagonal(seq_a, seq_b)
    torch.cuda.synchronize()
    gpu_diagonal_time = time.time() - gpu_diagonal_start
    print(f"Block GPU execution completed in {gpu_diagonal_time:.4f} seconds")

    # GPU Execution: Single-Launch Kernel
    print("Running Smith-Waterman on GPU (Single-Launch Kernel)...")
    torch.cuda.synchronize()
    gpu_single_start = time.time()
    gpu_score_single, gpu_pos_single, _ = gpu_sw_single_block.smith_waterman_gpu_single_block(seq_a, seq_b)
    torch.cuda.synchronize()
    gpu_single_time = time.time() - gpu_single_start
    print(f"Single-Launch GPU execution completed in {gpu_single_time:.4f} seconds")

    # Check for consistency in outputs
    outputs_match = (
        cpu_score == gpu_score_basic == gpu_score_diagonal == gpu_score_single and
        cpu_pos == gpu_pos_basic == gpu_pos_diagonal == gpu_pos_single
    )

    if outputs_match:
        print("All implementations produced matching results ✅")
    else:
        print("Warning: Outputs do not match ❌")
        print(f"CPU Score: {cpu_score}, Position: {cpu_pos}")
        print(f"Basic GPU Score: {gpu_score_basic}, Position: {gpu_pos_basic}")
        print(f"Block GPU Score: {gpu_score_diagonal}, Position: {gpu_pos_diagonal}")
        print(f"Single-Launch GPU Score: {gpu_score_single}, Position: {gpu_pos_single}")

    # Store performance and output data.
    results["base_length"] = len(seq_a)
    results["variant_length"] = len(seq_b)
    results["outputs_match"] = outputs_match
    results["cpu_time"] = cpu_time
    results["gpu_time_basic"] = gpu_basic_time
    results["gpu_time_diagonal"] = gpu_diagonal_time
    results["gpu_time_single"] = gpu_single_time
    results["speedup_basic_vs_cpu"] = cpu_time / gpu_basic_time if gpu_basic_time > 0 else 0
    results["speedup_diagonal_vs_cpu"] = cpu_time / gpu_diagonal_time if gpu_diagonal_time > 0 else 0
    results["speedup_single_vs_cpu"] = cpu_time / gpu_single_time if gpu_single_time > 0 else 0
    results["speedup_single_vs_basic"] = gpu_basic_time / gpu_single_time if gpu_single_time > 0 else 0

    # Store scores.
    results["cpu_score"] = cpu_score
    results["gpu_score_basic"] = gpu_score_basic
    results["gpu_score_diagonal"] = gpu_score_diagonal
    results["gpu_score_single"] = gpu_score_single

    return results


def main():
    """
    Run experiments on multiple test cases and save the benchmark results.
    """
    # Define test cases as tuples: (base_length, variant_length)
    test_cases = [
        (100, 20),    # Small scale
        (500, 100),   # Medium-small scale
        (2000, 500),  # Medium scale
        (5000, 1000), # Large scale
        (10000, 2000) # Very large scale
    ]

    all_results = []
    print("Starting Smith-Waterman Benchmarking...\n")

    for base_len, var_len in test_cases:
        print(f"\nGenerating test case with base_len={base_len}, var_len={var_len}")
        test_case = generate_test_case(base_len, var_len)
        result = run_experiment(test_case)
        all_results.append(result)

    # Save the results to a JSON file.
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"sw_benchmark_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBenchmark results saved to {output_path}")


if __name__ == "__main__":
    main()