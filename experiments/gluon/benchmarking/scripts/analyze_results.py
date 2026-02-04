# benchmarking/scripts/analyze_results.py
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from utils.io import read_fasta # To load sequences for mismatch reports

def compare_results(data_a, name_a, data_b, name_b, sequences, report_file):
    """A helper function to compare two sets of alignment results."""
    mismatches = []
    total_pairs = len(data_a)
    
    for i in range(total_pairs):
        score_a, (i_a, j_a) = data_a[i]
        score_b, (i_b, j_b) = data_b[i]
        
        if score_a != score_b or i_a != i_b or j_a != j_b:
            mismatches.append({
                "pair_id": i,
                name_a: {"score": score_a, "end": (i_a, j_a)},
                name_b: {"score": score_b, "end": (i_b, j_b)},
                "query_len": len(sequences['q_seqs'][i]),
                "ref_len": len(sequences['r_seqs'][i]),
            })
            
    mismatch_rate = len(mismatches) / total_pairs
    
    print(f"\n--- Comparison: {name_a} vs. {name_b} ---")
    print(f"Found {len(mismatches)} mismatches out of {total_pairs} pairs ({mismatch_rate:.4%}).")
    
    # Write detailed mismatches to the report file
    report_file.write(f"\n===== Mismatch Report: {name_a} vs. {name_b} =====\n")
    if not mismatches:
        report_file.write("All results match perfectly.\n")
    else:
        # Log first 50 mismatches to avoid overly large files
        for item in mismatches[:50]:
            report_file.write(f"Pair {item['pair_id']}:\n")
            report_file.write(f"  - {name_a}: {item[name_a]}\n")
            report_file.write(f"  - {name_b}: {item[name_b]}\n")
            report_file.write(f"  - Seq Lens: (q={item['query_len']}, r={item['ref_len']})\n\n")
        if len(mismatches) > 50:
            report_file.write(f"... and {len(mismatches) - 50} more mismatches.\n")

def main():
    print("--- Running Comprehensive Analysis of Benchmark Results ---")
    results_dir = project_root / "benchmarking" / "results"
    
    # Load all result files
    try:
        with open(results_dir / "our_framework_extz.json", 'r') as f: our_data = json.load(f)
        with open(results_dir / "our_framework_extz_test.json", 'r') as f: our_test_data = json.load(f)
        with open(results_dir / "ksw2_extz.json", 'r') as f: ksw2_data = json.load(f)
        with open(results_dir / "agatha_extz.json", 'r') as f: agatha_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: A result file is missing. Please run benchmark_extz.py first. Details: {e}")
        return
        
    # Load sequences to report on mismatches
    dataset_path = project_root / "datasets"
    q_seqs = read_fasta(dataset_path / "query.fa")
    r_seqs = read_fasta(dataset_path / "ref.fa")
    sequences = {'q_seqs': q_seqs, 'r_seqs': r_seqs}

    with open(results_dir / "mismatch_report.txt", 'w') as report_file:
        print(f"Detailed mismatch report will be saved to: {results_dir / 'mismatch_report.txt'}")
        
        # compare_results(our_data, "OurFramework", ksw2_data, "ksw2", sequences, report_file)
        compare_results(agatha_data, "AGAThA", ksw2_data, "ksw2", sequences, report_file)
        # compare_results(our_data, "OurFramework", agatha_data, "AGAThA", sequences, report_file)
        # compare_results(our_data, "OurFramework", our_test_data, "OurOpFramework", sequences, report_file)
        compare_results(our_test_data, "OurOpFramework", agatha_data, "AGAThA", sequences, report_file)
        compare_results(our_test_data, "OurOpFramework", ksw2_data, "ksw2", sequences, report_file)

 
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()