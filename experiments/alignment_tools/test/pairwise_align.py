#!/usr/bin/env python3
"""
Batch local Smith–Waterman alignments using Biopython.
Reads paired sequences from two FASTA files and writes JSON results.

Usage:
  ./align_batch.py <query.fasta> <ref.fasta> [output.json]

Each output entry contains:
  - pair_id: integer index of sequence pair (0-based)
  - score: alignment score
  - position: [query_end, ref_end] (1-based end positions)
"""
import sys
import json
from Bio import pairwise2
from Bio.SeqIO import parse

# Alignment parameters
MATCH_SCORE = 1
MISMATCH_PENALTY = -4
GAP_OPEN = -6
GAP_EXTEND = -2


def align_pair(query_seq: str, ref_seq: str):
    """
    Perform local alignment on a sequence pair and return score and end positions.
    Returns:
      score (int), query_end (1-based), ref_end (1-based)
    """
    # Run Smith–Waterman local alignment
    alignments = pairwise2.align.localms(
        query_seq, ref_seq,
        MATCH_SCORE, MISMATCH_PENALTY,
        GAP_OPEN, GAP_EXTEND
    )
    # Take the best alignment
    aligned_q, aligned_r, score, begin, end = alignments[0]

    # Compute original sequence end positions (0-based)
    # Count non-gap characters in aligned_q up to 'end'
    orig_q_end = sum(1 for c in aligned_q[:end] if c != '-') - 1
    orig_r_end = sum(1 for c in aligned_r[:end] if c != '-') - 1

    # Convert to 1-based coordinates
    return score, orig_q_end + 1, orig_r_end + 1


def main():
    # Parse command-line arguments
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <query.fasta> <ref.fasta> [output.json]")
        sys.exit(1)

    query_file = sys.argv[1]
    ref_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "alignment_results.json"

    # Read sequences from FASTA files
    query_seqs = [str(rec.seq) for rec in parse(query_file, "fasta")]
    ref_seqs = [str(rec.seq) for rec in parse(ref_file, "fasta")]

    if len(query_seqs) != len(ref_seqs):
        print("Error: query and reference files contain different number of sequences.")
        sys.exit(1)

    # Perform alignment for each pair and collect results
    results = []
    for idx, (q, r) in enumerate(zip(query_seqs, ref_seqs)):
        score, q_end, r_end = align_pair(q, r)
        results.append({
            "pair_id": idx,
            "score": score,
            "position": [q_end, r_end]
        })

    # Write JSON output
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} alignments to {output_file}")


if __name__ == "__main__":
    main()
