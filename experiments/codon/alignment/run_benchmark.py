#!/usr/bin/env python3
"""
run_benchmarks.py

Run and time multiple ExtZ alignment implementations on the same dataset,
logging both total runtime and per-tool outputs to separate log files.

Usage:
    python3 run_benchmarks.py
"""

import subprocess
import time
import sys
from pathlib import Path

# Configuration
DATA_QUERY = "datasets/query.fa"
DATA_REF   = "datasets/ref.fa"
LOG_DIR    = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Define each benchmark: name, command list, log filename
BENCHMARKS = [
    {
        "name": "triton_extz",
        "cmd": [sys.executable, "gpu_extz.py"],
        "log": LOG_DIR / "triton_extz.log"
    },
    {
        "name": "codon_cpu_extz",
        "cmd": ["codon", "run", "extz_cpu.codon"],
        "log": LOG_DIR / "codon_cpu_extz.log"
    },
    {
        "name": "seq_plugin",
        "cmd": ["codon", "run", "-plugin", "seq", "seq_test.codon"],
        "log": LOG_DIR / "seq_plugin.log"
    },
    {
        "name": "ksw2_extz",
        "cmd": [
            "ksw2/ksw2-test",
            "-w", "400", "-z", "751",
            "-A", "1", "-B", "4",
            "-O", "6,6", "-E", "2,2",
            "-t", "extz",
            DATA_QUERY, DATA_REF
        ],
        "log": LOG_DIR / "ksw2_extz.log"
    },
]

def run_benchmark(name, cmd, log_path):
    print(f"=== Running {name} ===")
    with open(log_path, "w") as log_f:
        log_f.write(f"# {name} @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_f.write("## Command\n```\n" + " ".join(cmd) + "\n```\n\n")
        log_f.write("## Output\n```\n")
        log_f.flush()

        start = time.time()
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
        proc.wait()
        elapsed = time.time() - start

        log_f.write("\n```\n")
        log_f.write(f"\n**Total elapsed time:** {elapsed*1000:.2f} ms\n")
        log_f.flush()
    print(f"{name} done, logged to {log_path}\n")
    
def main():
    for bench in BENCHMARKS:
        run_benchmark(bench["name"], bench["cmd"], bench["log"])
    print("All benchmarks complete. Logs are in", LOG_DIR)

if __name__ == "__main__":
    main()