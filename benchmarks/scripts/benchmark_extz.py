# benchmarking/scripts/benchmark_extz.py

import sys
import json
import time
import torch
from pathlib import Path

# =======2025-07-16: verfity JIT cost==========
# import triton
# import triton.language as tl

# @triton.jit
# def dummy_kernel():
#     pass
#==========================================

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import the restored, single API function
from framework_api_OPv1 import run_alignment_task
from configs.task_config import AlignmentTask, SolutionConfiguration, create_affine_scoring, create_banded_z_drop_pruning
from utils.io import read_fasta
from benchmarking.wrappers.ksw2_wrapper import run_ksw2_cli
from benchmarking.wrappers.agatha_wrapper import run_agatha

CONFIG = { "match": 1, "mismatch": -4, "gap_open": -6, "gap_extend": -2, "band_width": 751, "z_drop": 400 }

def calculate_total_cells(q_seqs, r_seqs):
    total_cells = 0
    for q, r in zip(q_seqs, r_seqs):
        total_cells += len(q) * len(r)
    return total_cells

def main():
    print("--- Starting Systematic Benchmark for ksw_extz (Final Timing Architecture) ---")
    
    # Loading data and defining directories remains the same
    dataset_path = project_root / "datasets"
    try:
        q_seqs = read_fasta(dataset_path / "query_x20.fa")
        r_seqs = read_fasta(dataset_path / "ref_x20.fa")
        print(f"Loaded {len(q_seqs)} sequence pairs from dataset.")
    except FileNotFoundError:
        print(f"Error: Dataset not found in {dataset_path}.")
        return

    total_cells = calculate_total_cells(q_seqs, r_seqs)
    print(f"Total cells to be computed: {total_cells / 1e9:.3f} GCells")

    results_dir = project_root / "benchmarking" / "results"
    results_dir.mkdir(exist_ok=True)
    
    benchmark_data = {}

# # =======2025-07-16: verfity JIT cost==========
#     print("\n[Running Dummy Kernel to absorb JIT startup cost...]")
#     dummy_start_time = time.perf_counter()
#     dummy_kernel[(1,)]() 
#     torch.cuda.synchronize()
#     dummy_end_time = time.perf_counter()
#     dummy_total_time_ms = (dummy_end_time - dummy_start_time) * 1000
#     print(f"Dummy kernel startup tax absorption time: {dummy_total_time_ms:.3f} ms")
# #============================================

    # --- Run Our Framework ---
    print("\n[Benchmarking Our Framework...]")
    our_task = AlignmentTask(
        align_problem='Extension',
        configuration=SolutionConfiguration(
            scoring=create_affine_scoring(CONFIG['match'], CONFIG['mismatch'], CONFIG['gap_open'], CONFIG['gap_extend']),
            pruning=create_banded_z_drop_pruning(CONFIG['band_width'], CONFIG['z_drop'])
        )
    )
    
    print("Performing warm-up run...")
    try:
        _, _, _, _, _ = run_alignment_task(q_seqs[:1], r_seqs[:1], our_task)
    except ValueError:
        _, _ = run_alignment_task(q_seqs[:1], r_seqs[:1], our_task)
    
    print("Performing timed execution run...")
    # Measure total wall-clock time for the synchronous API call
    start_time = time.perf_counter()
    try:
        # The function now returns both results and the internally measured kernel time and other times
        our_results, our_kernel_time_ms, malloc_H2D_packing_time_ms, packing_time_ms, d2h_time_ms = run_alignment_task(q_seqs, r_seqs, our_task)
    except ValueError:
        # If framework doesn't return the additional times, set them to default values
        our_results, our_kernel_time_ms = run_alignment_task(q_seqs, r_seqs, our_task)
        malloc_H2D_packing_time_ms = packing_time_ms = d2h_time_ms = 0.0
    
    end_time = time.perf_counter()
    
    our_total_time_ms = (end_time - start_time) * 1000
    
    # Print detailed time breakdown for the framework
    print(f"Our Framework GPU Kernel Time: {our_kernel_time_ms:.3f} ms")
    print(f"Our Framework malloc+H2D+Packing Time: {malloc_H2D_packing_time_ms:.3f} ms")
    print(f"Our Framework Packing Kernel Time: {packing_time_ms:.3f} ms")
    print(f"Our Framework D2H Time: {d2h_time_ms:.3f} ms")
    print(f"Our Framework Post-processing Time: {our_total_time_ms - (malloc_H2D_packing_time_ms + d2h_time_ms + our_kernel_time_ms):.3f} ms")
    print(f"Our Framework Total Runtime: {our_total_time_ms:.3f} ms")
    
    benchmark_data['Our Framework'] = {'kernel_ms': our_kernel_time_ms, 'total_ms': our_total_time_ms}

    our_output_path = results_dir / "our_framework_extz_test.json"
    with open(our_output_path, 'w') as f:
        json.dump(our_results, f)
    print(f"Results saved to {our_output_path}")

    # # --- Benchmark ksw2 (CPU Baseline) ---
    # print("\n[Benchmarking ksw2 (CPU)...]")
    # start_time = time.perf_counter()
    # ksw2_results = run_ksw2_cli(q_seqs, r_seqs, CONFIG)
    # end_time = time.perf_counter()
    
    # ksw2_total_time_ms = (end_time - start_time) * 1000
    # print(f"ksw2 CPU Total Time (including CLI overhead): {ksw2_total_time_ms:.3f} ms")
    # benchmark_data['ksw2'] = {'kernel_ms': None, 'total_ms': ksw2_total_time_ms}

    # ksw2_output_path = results_dir / "ksw2_extz.json"
    # with open(ksw2_output_path, 'w') as f:
    #     json.dump(ksw2_results, f)
    # print(f"Results saved to {ksw2_output_path}")

    # # --- Benchmark AGAThA (GPU Baseline) ---
    # print("\n[Benchmarking AGAThA (GPU)...]")
    # try:
    #     # total_start_time = time.perf_counter()
    #     agatha_results, agatha_kernel_time_ms, agatha_total_time_ms = run_agatha(q_seqs, r_seqs)
    #     # total_end_time = time.perf_counter()    
    #     # agatha_total_time_ms = (total_end_time - total_start_time) * 1000

    #     print(f"AGAThA Kernel Time: {agatha_kernel_time_ms:.3f} ms")
    #     print(f"AGAThA Total Runtime: {agatha_total_time_ms:.3f} ms")
    #     benchmark_data['AGAThA'] = {'kernel_ms': agatha_kernel_time_ms, 'total_ms': agatha_total_time_ms}

    #     agatha_output_path = results_dir / "agatha_extz.json"
    #     with open(agatha_output_path, 'w') as f:
    #         json.dump(agatha_results, f)
    #     print(f"AGAThA results parsed and saved to {agatha_output_path}")
    # except (FileNotFoundError, subprocess.CalledProcessError) as e:
    #     print(f"Could not run or parse AGAThA. Skipping. Error: {e}")
    #     benchmark_data['AGAThA'] = {'kernel_ms': -1, 'total_ms': -1}

    # # --- Print Final Summary Table ---
    # print("\n--- Benchmark Summary ---")
    # print(f"{'Tool':<18} | {'Platform':<8} | {'Kernel Time (ms)':<18} | {'Total Runtime (ms)':<20} | {'Throughput (GCUPS)':<20}")
    # print("-" * 90)
    
    # for tool, data in benchmark_data.items():
    #     platform = "GPU" if tool != "ksw2" else "CPU"
    #     kernel_t = f"{data['kernel_ms']:.3f}" if data['kernel_ms'] is not None and data['kernel_ms'] > 0 else "N/A"
    #     total_t = f"{data['total_ms']:.3f}"
    #     gcups = f"{(total_cells / (data['total_ms'] / 1000)) / 1e9:.3f}" if data['total_ms'] > 0 else "N/A"
        
    #     print(f"{tool:<18} | {platform:<8} | {kernel_t:<18} | {total_t:<20} | {gcups:<20}")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
