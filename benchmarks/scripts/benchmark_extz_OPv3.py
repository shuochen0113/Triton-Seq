# benchmarking/scripts/benchmark_extz.py

"""
A benchmarking script for evaluating the performance of Triton/AGAThA/extz2_sse
Usage:
    python benchmark_extz_OPv3.py

2025-07-24
"""

import sys
import json
import time
import torch
import os
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
from framework_api_OPv3 import run_alignment_task
from configs.task_config import AlignmentTask, SolutionConfiguration, create_affine_scoring, create_banded_z_drop_pruning
# from utils.io import read_fasta
from utils.io_OPv3 import read_fasta_as_bytes
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
    load_start_time = time.perf_counter()

    dataset_path = project_root / "datasets"
    try:
        q_seqs = read_fasta_as_bytes(dataset_path / "query.fa")
        r_seqs = read_fasta_as_bytes(dataset_path / "ref.fa")
        print(f"Loaded {len(q_seqs)} sequence pairs from dataset.")
    except FileNotFoundError:
        print(f"Error: Dataset not found in {dataset_path}.")
        return

    load_end_time = time.perf_counter()
    load_time_ms = (load_end_time - load_start_time) * 1000

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
    
    # warm-up run to ensure JIT compilation
    print("Performing warm-up run with full data to ensure JIT compilation...")
    # This untimed run ensures all kernels are compiled with the final, full-sized tensors.
    # The results are discarded.
    _ = run_alignment_task(q_seqs, r_seqs, our_task)
    torch.cuda.synchronize() # Ensure warm-up is completely finished.

    print("Performing timed execution run...")
    # Measure total wall-clock time for the synchronous API call
    start_time = time.perf_counter()
    # =========== 2025-07-23: batch control ===========
    all_results, total_kernel_time_ms, malloc_H2D_packing_time_ms, packing_kernel_time_ms, total_d2h_time_ms = run_alignment_task(q_seqs, r_seqs, our_task)
    end_time = time.perf_counter()
    our_process_time_ms = (end_time - start_time) * 1000

    # ======================== 2025-07-24: PROFILING CODE ========================
    # print("\n--- Starting Phase 0: Profiling Run on 'Ours (batch vers)' ---")

    # # define a directory to save profiler traces
    # trace_dir = "profiler_traces"
    # os.makedirs(trace_dir, exist_ok=True)

    # # use torch.profiler.profile context manager
    # # with_stack=True is crucial for generating flame graphs, it records the Python call stack
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     # on_trace_ready could automatically export results in TensorBoard/Perfetto readable format
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir)
    # ) as prof:
    #     # run the alignment once to profile
    #     run_alignment_task(q_seqs, r_seqs, our_task)

    # print("--- Profiling Complete ---")

    # # print the profiling results
    # print("\n--- Top 15 CPU Self-Time Events (Where CPU spends most of its time) ---")
    # print(prof.key_averages(group_by_stack_n=8).table(sort_by="self_cpu_time_total", row_limit=15))

    # print("\n--- Top 15 CUDA Time Events (GPU Kernel Runtimes) ---")
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    # print(f"\nTrace file saved in '{trace_dir}' directory.")
    # print("You can now analyze the generated .pt.trace.json file.")
    # ======================== END OF PROFILING ========================

    # --- Print detailed time breakdown for the PIPELINED framework ---
    # Rename for clarity: the 3rd return value is our new accumulated setup time
    total_setup_ms = malloc_H2D_packing_time_ms 
    
    # Calculate the total work scheduled on the GPU
    total_gpu_work_time = total_setup_ms + total_kernel_time_ms + total_d2h_time_ms
    
    # Calculate the time saved by overlapping CPU/GPU operations
    pipeline_overlap_ms = total_gpu_work_time - our_process_time_ms
    # If the total time is slightly longer, it's due to tiny CPU overheads, clamp to zero.
    pipeline_overlap_ms = max(0, pipeline_overlap_ms)

    print("\n--- Pipelined Framework Performance Breakdown ---")
    print(f"Total Wall-Clock Time: {our_process_time_ms + load_time_ms:.3f} ms")
    print(f"Load Time: {load_time_ms:.3f} ms")
    print(f"Process Time: {our_process_time_ms:.3f} ms")
    print("-" * 50)
    print("Accumulated GPU Task Times (Sum of all scheduled work):")
    print(f"  - Alignment Kernel Time        : {total_kernel_time_ms:.3f} ms")
    print(f"  - Setup Time (H2D, Pack, Init) : {total_setup_ms:.3f} ms")
    print(f"      (-> Packing Kernel Time)   : {packing_kernel_time_ms:.3f} ms")
    print(f"  - D2H Transfer Time            : {total_d2h_time_ms:.3f} ms")
    print(f"  -------------------------------------------------")
    print(f"  = Total GPU Work Scheduled     : {total_gpu_work_time:.3f} ms")
    print("-" * 50)
    print("Pipeline Efficiency:")
    if our_process_time_ms > 0:
      print(f"  => Time Saved by Overlap        : {pipeline_overlap_ms:.3f} ms ({pipeline_overlap_ms / total_gpu_work_time * 100:.2f}% of total work)")
    print("-" * 50)    
    benchmark_data['Our Framework'] = {'kernel_ms': total_kernel_time_ms, 'total_ms': our_process_time_ms}

    # Save all aggregated results
    with open(results_dir / "our_framework_extz_test.json", 'w') as f:
        json.dump(all_results, f)
    
    print(f"Results for {len(all_results)} pairs saved.")
    # ================================================
    

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
