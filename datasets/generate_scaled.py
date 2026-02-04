#!/usr/bin/env python3

"""
2025-07-23:
scale fasta files by simple repetition.

Usage:
    python scale_fasta.py query.fa          # directly scale
    python scale_fasta.py ref.fa --unique_headers  # scale with unique headers
"""

import argparse
from pathlib import Path
import itertools
import re

FACTORS = [5, 10, 20, 30, 40, 50, 75, 100]
# FACTORS = [4]

def read_fasta_lines(path: Path):
    return path.read_text().rstrip("\n").splitlines(keepends=True)

def write_scaled_fasta(src_lines, out_path: Path, k: int, unique_headers: bool):
    if not unique_headers:
        out_path.write_text("".join(src_lines * k))
        return

    header_pat = re.compile(r"^>(\S+)(.*)")       # >id [rest]
    with out_path.open("w") as fout:
        for idx in range(k):
            for line in src_lines:
                if line.startswith(">"):
                    m = header_pat.match(line)
                    if m:
                        new_id = f"{m.group(1)}/{idx}"
                        fout.write(f">{new_id}{m.group(2)}\n")
                    else:  # if header is malformed, write as is
                        fout.write(line)
                else:
                    fout.write(line)

def generate_scaled_files(src_path: Path, unique_headers=False):
    src_lines = read_fasta_lines(src_path)
    for k in FACTORS:
        out_path = src_path.with_name(f"{src_path.stem}_x{k}{src_path.suffix}")
        write_scaled_fasta(src_lines, out_path, k, unique_headers)
        print(f"✅ wrote {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scale FASTA files by simple repetition.")
    parser.add_argument("fasta", type=Path,
                        help="Path to query.fa or ref.fa")
    parser.add_argument("--unique_headers", action="store_true",
                        help="Append /idx to each header to guarantee uniqueness")
    args = parser.parse_args()

    generate_scaled_files(args.fasta, unique_headers=args.unique_headers)