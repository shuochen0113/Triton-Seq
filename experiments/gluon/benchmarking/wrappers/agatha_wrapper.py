# benchmarking/wrappers/agatha_wrapper.py
import subprocess
import re
import json
from pathlib import Path
from typing import List, Tuple
import subprocess
import time


# IMPORTANT: Update this path to the correct location of your AGAThA directory
AGATHA_ROOT = Path("/root/autodl-tmp/Cornell_intern/triton_experiment/AGAThA/")

def _write_fasta_for_agatha(seqs, path):
    """Writes a list of sequences to a FASTA file for AGAThA."""
    with open(path, "w") as f:
        for i, s in enumerate(seqs, 1):
            f.write(f">seq{i}\n{s}\n")

def run_agatha(q_seqs: List[str], r_seqs: List[str]) -> Tuple[List[Tuple[int, Tuple[int, int]]], float]:
    """
    Wrapper to run AGAThA and parse its output files.
    Returns a tuple: (alignment_results, kernel_time_ms)
    """
    if not AGATHA_ROOT.exists() or not (AGATHA_ROOT / "AGAThA.sh").exists():
        raise FileNotFoundError(f"AGAThA not found at the specified path: {AGATHA_ROOT.resolve()}")

    # agatha_dataset_dir = AGATHA_ROOT / "dataset"
    # agatha_dataset_dir.mkdir(exist_ok=True)
    # _write_fasta_for_agatha(q_seqs, agatha_dataset_dir / "query.fa")
    # _write_fasta_for_agatha(r_seqs, agatha_dataset_dir / "ref.fa")
    
    agatha_output_dir = AGATHA_ROOT / "output"
    score_log_path = agatha_output_dir / "score.log"
    time_json_path = agatha_output_dir / "time.json"
    
    # Clean up previous results to ensure a fresh run
    # if score_log_path.exists():
    #     score_log_path.unlink()
    # if time_json_path.exists():
    #     time_json_path.unlink()

    print("Running AGAThA...")
    total_start_time = time.perf_counter()

    try:
        # Execute the AGAThA script from within its own directory
        subprocess.run(
            ["bash", "AGAThA.sh"], 
            check=True, capture_output=True, text=True, cwd=AGATHA_ROOT
        )
        print("AGAThA run complete. Parsing results...")
    except subprocess.CalledProcessError as e:
        print("--- AGAThA Execution Failed ---")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e

    total_end_time = time.perf_counter()
    agatha_total_time_ms = (total_end_time - total_start_time) * 1000


    # Parse alignment results from score.log
    if not score_log_path.exists():
        raise FileNotFoundError(f"AGAThA did not produce the expected output file: {score_log_path}")
    
    results = []
    pattern = re.compile(r"(\d+)\s+query_batch_end=(\d+)\s+target_batch_end=(\d+)")
    with open(score_log_path, 'r') as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                score, end_t_raw, end_q_raw = map(int, match.groups())
                # Adjust coordinates to match our framework's convention (1-based, query-first)
                end_q = end_q_raw + 1
                end_t = end_t_raw + 1
                results.append((score, (end_q, end_t)))
                
    # Parse kernel time from time.json
    kernel_time_ms = -1.0
    if time_json_path.exists():
        with open(time_json_path, 'r') as f:
            time_data = json.load(f)
            kernel_time_ms = time_data.get("AGAThA", {}).get("test", -1.0)
    else:
        print(f"Warning: AGAThA did not produce the time.json file.")

    return results, kernel_time_ms, agatha_total_time_ms