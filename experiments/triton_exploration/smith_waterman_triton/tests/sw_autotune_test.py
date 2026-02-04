import os
import json
import time
import torch
import sys
sys.stdout.flush()

from datetime import datetime
from pathlib import Path

# Import your Smith-Waterman GPU Triton implementation
smith_waterman_triton = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(smith_waterman_triton, "src/optimize_experiment"))

# Import the auto-tuned single-launch kernel
import gpu_sw_autotune  # This should contain the Triton auto-tuned kernel

# Read .fasta file, return {sequence_id: sequence}
def read_fasta(file_path):
    sequences = {}
    current_seq_id = None
    current_seq = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq_id is not None:
                    sequences[current_seq_id] = "".join(current_seq)
                current_seq_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_seq_id is not None:
            sequences[current_seq_id] = "".join(current_seq)

    return sequences

# Traceback function to reconstruct the optimal alignment
def traceback(dp, seq1, seq2, pos, match=2, mismatch=-1, gap=-1, n_score=0):
    i, j = pos
    alignment1, alignment2 = [], []

    while i > 0 and j > 0 and dp[i][j] > 0:
        current = dp[i][j]
        diag = dp[i - 1][j - 1]
        up = dp[i - 1][j]
        left = dp[i][j - 1]

        if seq1[i - 1] == 'N' or seq2[j - 1] == 'N':
            match_penalty = n_score  # 'N' is fuzzy matching
        else:
            match_penalty = match if seq1[i - 1] == seq2[j - 1] else mismatch

        if current == diag + match_penalty:
            alignment1.append(seq1[i - 1])
            alignment2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif current == up + gap:
            alignment1.append(seq1[i - 1])
            alignment2.append('-')
            i -= 1
        else:
            alignment1.append('-')
            alignment2.append(seq2[j - 1])
            j -= 1

    return ''.join(reversed(alignment1)), ''.join(reversed(alignment2))

# Run Smith-Waterman using auto-tuned GPU kernel
def run_experiment(seq1, seq2):
    torch.cuda.synchronize()
    gpu_start = time.time()

    # Run the auto-tuned kernel and capture best configuration
    gpu_score, gpu_pos, gpu_dp = gpu_sw_autotune.smith_waterman_gpu_single_block(seq1, seq2)

    torch.cuda.synchronize()
    gpu_time = time.time() - gpu_start

    alignment_a, alignment_b = traceback(gpu_dp, seq1, seq2, gpu_pos)

    return {
        "lengths": (len(seq1), len(seq2)),
        "gpu_time": gpu_time,
        "gpu_score": gpu_score,
        "position": gpu_pos,
        "alignment": (alignment_a, alignment_b)
        # "best_config": best_config  # Save chosen BLOCK_SIZE
    }

# Benchmarking multiple configurations
def benchmark():
    reference_path = "/root/shuochen/Cornell_intern/triton_experiment/smith_waterman_triton/dataset/ref_profile.fasta"
    query_path = "/root/shuochen/Cornell_intern/triton_experiment/smith_waterman_triton/dataset/query_profile.fasta"

    # Ensure .fasta files exist
    if not os.path.exists(reference_path) or not os.path.exists(query_path):
        print(f"Error: Missing {reference_path} or {query_path}.")
        return

    reference_sequences = read_fasta(reference_path)
    query_sequences = read_fasta(query_path)

    total_ref = len(reference_sequences)
    total_query = len(query_sequences)

    results = []
    total_gpu_time = 0
    count = 0
    
    # Run each query against each reference sequence
    for (ref_id, ref_seq), (query_id, query_seq) in zip(reference_sequences.items(), query_sequences.items()):
        print(f"Running Smith-Waterman Auto-Tuned Kernel on:\nReference ({ref_id}): {len(ref_seq)} bases\nQuery ({query_id}): {len(query_seq)} bases", flush=True)

        # Run the experiment
        result = run_experiment(ref_seq, query_seq)
        result["reference_id"] = ref_id
        result["query_id"] = query_id
        results.append(result)

        # Update total execution time
        total_gpu_time += result["gpu_time"]
        count += 1

        print(f"Completed Smith-Waterman on reference {ref_id} and query {query_id}.", flush=True)

    # Compute average time
    avg_gpu_time = total_gpu_time / count if count > 0 else 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / f"auto_tune_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    summary = {
        "total_reference_sequences": total_ref,
        "total_query_sequences": total_query,
        "average_gpu_time": avg_gpu_time
    }

    summary_path = output_dir / f"auto_tune_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    benchmark()