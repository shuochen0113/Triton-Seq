#!/usr/bin/env python3

# ==============================================================================
# This benchmark script compares three Smith–Waterman implementations:
#   1. Agatha (CUDA, inter‑query)
#   2. gpu_sw_single_block_packed (sequential, packed format)
#   3. gpu_sw_inter_query_packed (inter‑query, packed format)

# For each test set, we randomly generate N sequence pairs (N chosen from 1 to 15,000)
# with various length distributions. The sequences are written to FASTA files (for Agatha),
# and each kernel's performance is measured in two ways:
#   - Kernel time (as reported by the kernel's printed output)
#   - Total wall time (measured on the host, including data transfer and kernel launch)

# The results of all tests (along with dataset and warmup information) are then saved
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Set GPU device (adjust as needed)

import json, time, random, re, sys, subprocess
from io import StringIO
from datetime import datetime
from pathlib import Path
import torch
from src import gpu_sw_single_block_packed
from src import gpu_sw_inter_query_packed

# ==============================================================================
# Helper Functions to Extract and Measure Kernel Timing
# ==============================================================================
def _extract_timing(stdout: str, label: str) -> float:
    """
    Parse a line of text and return the first floating point number found after
    "[label] Kernel time:".
    """
    pat = rf"\[{label}\].*?Kernel time:\s+([0-9.]+)\s+ms"
    m = re.search(pat, stdout)
    if not m:
        raise RuntimeError(f"Cannot find timing for {label}")
    return float(m.group(1))

def _run_with_capture(fn, label, *args, **kwargs):
    """
    Run a kernel function, capturing its standard output. Then extract and return
    the kernel time (in ms) from that output.
    """
    old = sys.stdout
    sys.stdout = mystd = StringIO()
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return _extract_timing(mystd.getvalue(), label)

def _wall_time(fn, *args, **kwargs):
    """
    Measure the total wall-time (in ms) for the function call. This includes
    all overhead of transferring data and launching the kernel.
    """
    start = time.time()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.time() - start) * 1e3  # Convert seconds to ms

# ==============================================================================
# Sequence Generation Utilities
# ==============================================================================
DNA = "ATCG"

def _mutate(seq: list[int], rate: float = 0.05) -> list[int]:
    """
    Mutate a sequence by replacing ~5% of the bases at random positions.
    """
    n = max(1, int(len(seq) * rate))
    for _ in range(n):
        idx = random.randint(0, len(seq) - 1)
        seq[idx] = random.choice(DNA)
    return seq

def gen_pair(length: int) -> tuple[str, str]:
    """
    Generate one sequence pair (query and reference) of the specified length.
    The reference is randomly generated and the query is a mutated copy.
    """
    ref = [random.choice(DNA) for _ in range(length)]
    qry = _mutate(ref.copy(), rate=0.05)
    return "".join(qry), "".join(ref)

def generate_pairs(num_pairs: int) -> tuple[list[str], list[str]]:
    """
    Generate two lists (query_list and ref_list) of sequence pairs.
    The lengths are chosen from mixed distributions.
    """
    qs, rs = [], []
    for _ in range(num_pairs):
        p = random.random()
        if p < 0.30:     L = random.randint(50, 200)
        elif p < 0.60:   L = random.randint(500, 2000)
        elif p < 0.90:   L = random.randint(5000, 10000)
        else:            L = random.randint(15000, 20000)
        q, r = gen_pair(L)
        qs.append(q)
        rs.append(r)
    return qs, rs

# ==============================================================================
# FASTA File Helpers
# ==============================================================================
def write_fasta(seq_list: list[str], path: Path):
    """
    Write a list of sequences to a FASTA file. Each sequence is prefixed
    with a header ">{index}".
    """
    with open(path, "w") as f:
        for i, s in enumerate(seq_list, 1):
            f.write(f">{i}\n{s}\n")

# ==============================================================================
# Sequential Kernel Runner (for gpu_sw_single_block_packed)
# ==============================================================================
def run_sequential_total(kernel_fn, label: str, q_list: list[str], r_list: list[str]) -> tuple[float, float]:
    """
    Run the provided sequential kernel function for all sequence pairs in q_list and r_list.
    For each pair, the kernel time (extracted from printed output) is accumulated.
    Also measure the total wall time for processing all sequence pairs.
    
    Returns a tuple: (total_kernel_time in ms, total_wall_time in ms)
    """
    total_kernel_time = 0.0
    start_time = time.time()
    for q, r in zip(q_list, r_list):
        # Warm up per pair (optional)
        kernel_fn(q, r)
        ktime = _run_with_capture(kernel_fn, label, q, r)
        total_kernel_time += ktime
    torch.cuda.synchronize()
    total_wall_time = (time.time() - start_time) * 1e3  # milliseconds
    return total_kernel_time, total_wall_time

# ==============================================================================
# Inter-query Kernel Runner (for gpu_sw_inter_query_packed)
# ==============================================================================
def _run_inter_query(q_list: list[str], r_list: list[str], label: str, kernel_fn):
    """
    Run the provided inter-query kernel function for the entire list of sequence pairs.
    This kernel is expected to process all pairs at once.
    
    Returns a tuple: (kernel_time in ms, wall_time in ms)
    """
    # Warm up with one pair (optional)
    kernel_fn(q_list[:1], r_list[:1])
    kt = _run_with_capture(kernel_fn, label, q_list, r_list)
    wt = _wall_time(kernel_fn, q_list, r_list)
    return kt, wt

# ==============================================================================
# Agatha Helpers
# ==============================================================================
AGATHA_ROOT = Path("../AGAThA")
DATASET_DIR = AGATHA_ROOT / "dataset"

def run_agatha(q_list: list[str], r_list: list[str]) -> tuple[float, float]:
    """
    Run the Agatha kernel benchmark.
    This function writes the generated sequence pairs to FASTA files and
    invokes Agatha through a shell script. It then reads the reported kernel
    time (from score.log) and wall time (from a JSON file).
    
    Returns a tuple: (kernel_time in ms, wall_time in ms)
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    write_fasta(q_list, DATASET_DIR / "generated_query.fasta")
    write_fasta(r_list, DATASET_DIR / "generated_ref.fasta")

    subprocess.run(["bash", str(AGATHA_ROOT / "AGAThA.sh")], capture_output=True)
    # Extract kernel time from Agatha's score.log
    ktime = -1.0
    score_log = AGATHA_ROOT / "output/score.log"
    if score_log.exists():
        with open(score_log) as f:
            line = f.readline()
            m = re.search(r"Kernel time:\s+([0-9.]+)\s+ms", line)
            if m:
                ktime = float(m.group(1))
    # Extract wall time from Agatha's time.json
    wtime = -1.0
    time_json_path = AGATHA_ROOT / "output/time.json"
    if time_json_path.exists():
        try:
            with open(time_json_path) as f:
                wtime = json.load(f).get("AGAThA", {}).get("test", -1.0)
        except Exception as e:
            print(f"[!] Failed to read AGATHA time.json: {e}")

    return ktime, wtime

# Benchmark Test Parameters: List of numbers of sequence pairs to test.
TEST_PAIRS = [1, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 15000]

# ==============================================================================
# Main Benchmark Loop: run_suite()
# ==============================================================================
def run_suite():
    """
    Main loop that runs benchmarks for each test batch. For each number of sequence
    pairs specified in TEST_PAIRS, this function does the following:
      1. Generate a batch of random sequence pairs.
      2. Run the sequential kernel (gpu_sw_single_block_packed) on each pair individually,
         accumulating the kernel and wall times.
      3. Run the inter-query kernel (gpu_sw_inter_query_packed) once for all pairs.
      4. Run Agatha and record its kernel and wall times.
    The results for all batches are stored in a JSON file.
    """
    print("\nWarming up all kernels to eliminate cold-start overhead...")

    # Generate a small warm-up batch (one sequence pair)
    warm_q, warm_r = generate_pairs(1)

    # Warm-up for sequential (packed) kernel
    gpu_sw_single_block_packed.smith_waterman_gpu_single_block_packed(warm_q[0], warm_r[0])
    
    # Warm-up for inter-query kernel (packed)
    gpu_sw_inter_query_packed.smith_waterman_gpu_inter_query_packed(warm_q, warm_r)

    # Warm-up for Agatha
    run_agatha(warm_q, warm_r)

    all_res = []
    for n_pairs in TEST_PAIRS:
        print(f"\n=== Running batch with {n_pairs} sequence pairs ===")
        q_list, r_list = generate_pairs(n_pairs)

        # Run the sequential kernel: Process each sequence pair one-by-one and sum their timings.
        kt_seq, wt_seq = run_sequential_total(
            gpu_sw_single_block_packed.smith_waterman_gpu_single_block_packed,
            "Triton Packed Timing",  # Label used in kernel printout
            q_list,
            r_list
        )

        # Run the inter-query kernel: This kernel processes all sequence pairs in one call.
        kt_inter, wt_inter = _run_inter_query(
            q_list, r_list,
            "Triton Wavefront Packed Timing",  # Label as printed in the kernel
            gpu_sw_inter_query_packed.smith_waterman_gpu_inter_query_packed
        )

        # Run Agatha
        kt_ag, wt_ag = run_agatha(q_list, r_list)

        # Record additional details such as minimum and maximum sequence length.
        all_res.append({
            "num_pairs": n_pairs,
            "seq_len_min": min(len(s) for s in q_list),
            "seq_len_max": max(len(s) for s in q_list),
            "triton_sequential_packed_kernel_ms": kt_seq,
            "triton_sequential_packed_wall_ms":   wt_seq,
            "triton_interquery_packed_kernel_ms": kt_inter,
            "triton_interquery_packed_wall_ms":   wt_inter,
            "agatha_kernel_ms": kt_ag,
            "agatha_wall_ms":   wt_ag
        })

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sw_interquery_suite_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nBenchmark saved to {out_path}")

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    run_suite()