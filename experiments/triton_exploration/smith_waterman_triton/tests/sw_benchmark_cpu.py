import os
import json
import time
import sys
from datetime import datetime
from pathlib import Path
from src import cpu_sw_fasta

sys.stdout.flush()

# read .fasta file, return {sequence_id: sequence}
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

# traceback
def traceback(dp, seq1, seq2, pos, match=3, mismatch=-2, gap=-1, n_score=0):
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

def run_experiment(seq1, seq2):
    # CPU execution only
    cpu_start = time.time()
    cpu_score, cpu_pos, cpu_dp = cpu_sw_fasta.smith_waterman_cpu(seq1, seq2, n_score=0)
    cpu_time = time.time() - cpu_start

    alignment_a, alignment_b = traceback(cpu_dp, seq1, seq2, cpu_pos)

    return {
        "lengths": (len(seq1), len(seq2)),
        "cpu_time": cpu_time,
        "cpu_score": cpu_score,
        "position": cpu_pos,
        "alignment": (alignment_a, alignment_b)
    }

def main():
    reference_path = "dataset/ref_profile.fasta"
    query_path = "dataset/query_profile.fasta"

    if not os.path.exists(reference_path) or not os.path.exists(query_path):
        print(f"Error: Missing {reference_path} or {query_path}.")
        return

    reference_sequences = read_fasta(reference_path)
    query_sequences = read_fasta(query_path)

    total_ref = len(reference_sequences)
    total_query = len(query_sequences)

    results = []
    total_cpu_time = 0
    count = 0
    
    for (ref_id, ref_seq), (query_id, query_seq) in zip(reference_sequences.items(), query_sequences.items()):
        print(f"Running Smith-Waterman on:\nReference ({ref_id}): {len(ref_seq)} bases\nQuery ({query_id}): {len(query_seq)} bases", flush=True)
        result = run_experiment(ref_seq, query_seq)
        result["reference_id"] = ref_id
        result["query_id"] = query_id
        results.append(result)

        total_cpu_time += result["cpu_time"]
        count += 1

        print(f"Finish running Smith-Waterman on reference {ref_id} and query {query_id}.", flush=True)

    avg_cpu_time = total_cpu_time / count if count > 0 else 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / f"fasta_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    summary = {
        "total_reference_sequences": total_ref,
        "total_query_sequences": total_query,
        "average_cpu_time": avg_cpu_time
    }
    summary_path = output_dir / f"fasta_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
