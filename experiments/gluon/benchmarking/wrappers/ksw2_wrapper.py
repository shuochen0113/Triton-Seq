# benchmarking/wrappers/ksw2_wrapper.py

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

# IMPORTANT: Update this path to point to your ksw2 directory
# This directory should contain the compiled 'ksw2-test' executable.
KSW2_ROOT = Path("/root/autodl-tmp/Cornell_intern/alignment_related_202505/ksw2") # Assuming it's in the project root

def _write_fasta_temp(seqs: List[str], prefix: str) -> Path:
    """Writes a list of sequences to a temporary FASTA file."""
    # We use tempfile to avoid cluttering the directory
    fp = tempfile.NamedTemporaryFile(mode='w+', prefix=prefix, suffix='.fa', delete=False)
    for i, s in enumerate(seqs, 1):
        fp.write(f">seq{i}\n{s}\n")
    fp.close()
    return Path(fp.name)

def run_ksw2_cli(q_seqs: List[str], r_seqs: List[str], config: Dict[str, Any]) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Runs the ksw2-test command-line executable and parses its stdout.
    This is a robust, simple, and reliable way to get the baseline results.
    """
    ksw2_executable = KSW2_ROOT / "ksw2-test"
    if not ksw2_executable.exists():
        raise FileNotFoundError(
            f"ksw2-test executable not found at {ksw2_executable}.\n"
            "Please run 'make' inside the ksw2 directory first."
        )

    # Create temporary FASTA files for query and reference
    query_file_path = _write_fasta_temp(q_seqs, "temp_qry_")
    ref_file_path = _write_fasta_temp(r_seqs, "temp_ref_")

    # Construct the command from the config dictionary
    # Note: ksw2-test CLI expects positive penalties
    cmd = [
        str(ksw2_executable),
        "-w", str(config['band_width']),
        "-z", str(config['z_drop']),
        "-A", str(config['match']),
        "-B", str(abs(config['mismatch'])), # Mismatch penalty as a positive number
        "-O", str(abs(config['gap_open'])),
        "-E", str(abs(config['gap_extend'])),
        "-t", "extz2_sse",
        # "-t", "extz",
        str(query_file_path),
        str(ref_file_path),
    ]
    
    print(f"Running ksw2 command: {' '.join(cmd)}")
    
    try:
        # Execute the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # --- Parse the stdout from ksw2-test ---
        output_lines = result.stdout.strip().split('\n')
        parsed_results = []
        for line in output_lines:
            parts = line.split()
            if len(parts) < 7: continue # Skip any non-data lines
            
            # Per your analysis: 4th value is score, 5th/6th are end points
            score = int(parts[3])
            end_q = int(parts[4]) + 1
            end_t = int(parts[5]) + 1
            parsed_results.append((score, (end_q, end_t)))
            
        return parsed_results

    except subprocess.CalledProcessError as e:
        print("--- ksw2-test Execution Failed ---")
        print("COMMAND:", ' '.join(cmd))
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e
    finally:
        # Clean up the temporary files
        query_file_path.unlink()
        ref_file_path.unlink()