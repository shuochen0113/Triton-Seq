# benchmarking/wrappers/logan_wrapper.py
import subprocess
import tempfile
import re
from pathlib import Path
from typing import List, Tuple

# IMPORTANT: Update this path to your LOGAN directory
LOGAN_ROOT = Path("/root/shuochen/Cornell_intern/alignment_related_202505/LOGAN")
# Define a smaller batch size for the wrapper to process at a time
WRAPPER_BATCH_SIZE = 3000

def _write_logan_input(q_seqs_batch: List[str], r_seqs_batch: List[str]) -> Path:
    fp = tempfile.NamedTemporaryFile(mode='w+', prefix="logan_input_", suffix='.txt', delete=False)
    for q, r in zip(q_seqs_batch, r_seqs_batch):
        fp.write(f"{q}\t0\t{r}\t0\tn\n")
    fp.close()
    return Path(fp.name)

def run_logan_cli(q_seqs: List[str], r_seqs: List[str], x_drop: int) -> Tuple[List[int], float, float]:
    """
    Runs the MODIFIED LOGAN demo in batches to handle large datasets
    and parses its stdout for scores and timing.
    Returns: (list_of_all_scores, total_gpu_time_ms, total_exec_time_ms)
    """
    logan_executable = LOGAN_ROOT / "demo"
    if not logan_executable.exists():
        raise FileNotFoundError(f"LOGAN executable not found at {logan_executable}. Did you run 'make demo'?")

    all_scores = []
    total_gpu_time = 0.0
    total_exec_time = 0.0
    num_total_seqs = len(q_seqs)

    # Loop through the data in smaller batches
    for i in range(0, num_total_seqs, WRAPPER_BATCH_SIZE):
        batch_end = min(i + WRAPPER_BATCH_SIZE, num_total_seqs)
        q_batch = q_seqs[i:batch_end]
        r_batch = r_seqs[i:batch_end]
        
        print(f"  -> Running LOGAN batch: pairs {i} to {batch_end-1}...")
        
        input_file_path = _write_logan_input(q_batch, r_batch)
        cmd = [str(logan_executable), str(input_file_path), "17", str(x_drop), "1"]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=LOGAN_ROOT)
            
            # Use regex to find and parse values
            time_match = re.search(r"GPU only time:\s+([\d\.]+)", result.stdout)
            total_time_match = re.search(r"Total Execution time:\s+([\d\.]+)", result.stdout)
            
            if time_match: total_gpu_time += float(time_match.group(1))
            if total_time_match: total_exec_time += float(total_time_match.group(1))

            # Find all score results
            # RESULT\t<pair_id_local>\t<score>
            score_matches = re.findall(r"RESULT\t(\d+)\t(-?\d+)", result.stdout)
            
            # Create a temporary list to hold batch results in order
            batch_scores = [None] * len(q_batch)
            for pair_id_local_str, score_str in score_matches:
                pair_id_local = int(pair_id_local_str)
                score = int(score_str)
                if pair_id_local < len(batch_scores):
                    batch_scores[pair_id_local] = score
            
            all_scores.extend(batch_scores)

        except subprocess.CalledProcessError as e:
            print(f"--- LOGAN Execution Failed on batch {i} ---")
            print("STDERR:", e.stderr)
            raise e
        finally:
            input_file_path.unlink()
            
    # Convert total seconds to milliseconds
    return all_scores, total_gpu_time * 1000, total_exec_time * 1000