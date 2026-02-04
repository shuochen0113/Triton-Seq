import os
import json
import time
import random
import torch
import subprocess
import sys
import re
from io import StringIO
from datetime import datetime
from pathlib import Path

# Import Triton GPU kernels
from src import gpu_sw_single_block
from src import gpu_sw_single_block_packed

def extract_timing_from_output(output, label):
    """
    Extract the kernel time (in ms) from the output that contains a line like:
    "[{label} Timing] Kernel time: xxx ms"
    """
    pattern = rf"\[{label} Timing\]\s+Kernel time:\s+([0-9.]+)\s+ms"
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not extract '{label}' timing from output.")

def capture_function_output(func, *args, label=None, **kwargs):
    """
    Run a function and capture its stdout output (used to extract kernel timing).
    """
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    output = mystdout.getvalue()
    return extract_timing_from_output(output, label)

def measure_wall_time(func, *args, **kwargs):
    """
    Measure wall-clock time (in ms) for a function call, including GPU synchronization.
    """
    start_time = time.time()
    func(*args, **kwargs)
    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) * 1000

def generate_test_case(length):
    """
    Generate two equal-length sequences:
      - Reference: A random DNA sequence of the given length.
      - Query: A copy of the reference with several random small block mutations.
    This ensures the two sequences are of equal length and have realistic variations.
    """
    ref = "".join(random.choice("ATCG") for _ in range(length))
    query = list(ref)
    n_modifications = max(1, length // 20)
    for _ in range(n_modifications):
        idx = random.randint(0, length - 1)
        block_size = random.randint(1, min(5, length - idx))
        for i in range(block_size):
            orig = query[idx + i]
            choices = [n for n in "ATCG" if n != orig]
            query[idx + i] = random.choice(choices)
    return "".join(query), ref

def save_fasta_pair(seq_a, seq_b, query_path, ref_path):
    """
    Save the query and reference sequences to FASTA files.
    The header is fixed as ">1" to ensure compatibility with Agatha.
    """
    with open(query_path, 'w') as qf:
        qf.write(">1\n" + seq_a + "\n")
    with open(ref_path, 'w') as rf:
        rf.write(">1\n" + seq_b + "\n")

def run_experiment(test_case):
    """
    For a given test case (two equal-length sequences), run the following:
      - Triton Unpacked Kernel (both printed kernel time and wall-clock time)
      - Triton Packed Kernel (both printed kernel time and wall-clock time)
      - AGAThA kernel via the bash script, where:
          * The internal kernel time is read from the first line of score.log.
          * The wall-clock time is read from raw.log (which contains a single floating-point number).
    All timing measurements are printed to the terminal and saved.
    """
    seq_a, seq_b = test_case
    results = {}
    seq_len = len(seq_a)
    print(f"\nStarting experiment for sequence length = {seq_len}")

    # Save FASTA files for Agatha (adjust paths as necessary)
    agatha_dataset_dir = "../AGAThA/dataset/"
    os.makedirs(agatha_dataset_dir, exist_ok=True)
    query_path = os.path.join(agatha_dataset_dir, "generated_query.fasta")
    ref_path = os.path.join(agatha_dataset_dir, "generated_ref.fasta")
    save_fasta_pair(seq_a, seq_b, query_path, ref_path)

    # ------------------ Triton Unpacked Kernel ------------------
    print("Warming up Triton Unpacked Kernel...")
    gpu_sw_single_block.smith_waterman_gpu_single_block(seq_a, seq_b)
    torch.cuda.synchronize()

    print("Running Triton Unpacked Kernel (printed kernel time)...")
    try:
        triton_unpacked_kernel_time = capture_function_output(
            gpu_sw_single_block.smith_waterman_gpu_single_block, seq_a, seq_b, label="Triton Unpacked"
        )
    except Exception as e:
        print("Error capturing Triton Unpacked timing:", e)
        triton_unpacked_kernel_time = -1
    print(f"[Triton Unpacked Timing] Kernel time: {triton_unpacked_kernel_time:.3f} ms")

    print("Running Triton Unpacked Kernel (wall-clock measurement)...")
    try:
        triton_unpacked_wall_time = measure_wall_time(
            gpu_sw_single_block.smith_waterman_gpu_single_block, seq_a, seq_b
        )
    except Exception as e:
        print("Error measuring Triton Unpacked wall-clock time:", e)
        triton_unpacked_wall_time = -1
    print(f"[Triton Unpacked Wall Time] Elapsed time: {triton_unpacked_wall_time:.3f} ms")

    # ------------------ Triton Packed Kernel ------------------
    print("Warming up Triton Packed Kernel...")
    gpu_sw_single_block_packed.smith_waterman_gpu_single_block_packed(seq_a, seq_b)
    torch.cuda.synchronize()

    print("Running Triton Packed Kernel (printed kernel time)...")
    try:
        triton_packed_kernel_time = capture_function_output(
            gpu_sw_single_block_packed.smith_waterman_gpu_single_block_packed, seq_a, seq_b, label="Triton Packed"
        )
    except Exception as e:
        print("Error capturing Triton Packed timing:", e)
        triton_packed_kernel_time = -1
    print(f"[Triton Packed Timing] Kernel time: {triton_packed_kernel_time:.3f} ms")

    print("Running Triton Packed Kernel (wall-clock measurement)...")
    try:
        triton_packed_wall_time = measure_wall_time(
            gpu_sw_single_block_packed.smith_waterman_gpu_single_block_packed, seq_a, seq_b
        )
    except Exception as e:
        print("Error measuring Triton Packed wall-clock time:", e)
        triton_packed_wall_time = -1
    print(f"[Triton Packed Wall Time] Elapsed time: {triton_packed_wall_time:.3f} ms")

    # ------------------ AGAThA Kernel ------------------
    print("Running AGAThA kernel...")
    try:
        result = subprocess.run(["bash", "../AGAThA/AGAThA.sh"], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print("Error running AGAThA:", e)

    # Extract Agatha kernel time from score.log (internal kernel time)
    agatha_score_path = "../AGAThA/output/score.log"
    agatha_kernel_time = -1
    if os.path.exists(agatha_score_path):
        with open(agatha_score_path, 'r') as f:
            first_line = f.readline().strip()
            if "AGATHA Timing" in first_line:
                try:
                    agatha_kernel_time = float(first_line.split(":")[-1].strip().split()[0])
                except Exception as e:
                    print("Error parsing AGATHA timing from score.log:", e)
    else:
        print(f"AGATHA score file not found at {agatha_score_path}")

    # Read Agatha wall time from raw.log (this file contains a single floating-point number)
    agatha_raw_path = "../AGAThA/output/raw.log"
    agatha_wall_time = -1
    if os.path.exists(agatha_raw_path):
        with open(agatha_raw_path, 'r') as f:
            raw_content = f.read().strip()
            try:
                agatha_wall_time = float(raw_content)
            except Exception as e:
                print("Error parsing AGATHA wall time from raw.log:", e)
    else:
        print(f"AGATHA raw file not found at {agatha_raw_path}")

    # Print Agatha timing information to the terminal
    if agatha_kernel_time >= 0:
        print(f"[AGATHA Timing] Kernel time (from score.log): {agatha_kernel_time:.3f} ms")
    else:
        print("[AGATHA Timing] Kernel time (from score.log): Not available")
    print(f"[AGATHA Wall Time] Elapsed time (from raw.log): {agatha_wall_time:.3f} ms")

    # Save results
    results["sequence_length"] = seq_len
    results["triton_unpacked_kernel_time_ms"] = triton_unpacked_kernel_time
    results["triton_unpacked_wall_time_ms"] = triton_unpacked_wall_time
    results["triton_packed_kernel_time_ms"] = triton_packed_kernel_time
    results["triton_packed_wall_time_ms"] = triton_packed_wall_time
    results["agatha_kernel_time_ms"] = agatha_kernel_time
    results["agatha_wall_time_ms"] = agatha_wall_time

    return results

def main():
    """
    Run experiments for multiple test cases, including sequences with lengths:
    100, 500, 2000, 5000, 10000, and 30000.
    Save the benchmark results to a JSON file.
    """
    test_lengths = [100, 500, 2000, 5000, 10000, 15000, 20000, 30000, 40000]

    all_results = []
    print("Starting Smith-Waterman Benchmarking with AGAThA...\n")
    for length in test_lengths:
        print(f"\nGenerating test case with sequence length = {length}")
        test_case = generate_test_case(length)
        result = run_experiment(test_case)
        all_results.append(result)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"sw_benchmark_agatha_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBenchmark results saved to {output_path}")

if __name__ == "__main__":
    main()