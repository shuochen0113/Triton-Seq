import torch
import numpy as np
import random
from src import gpu_sw_single_block_packed
from src import gpu_sw_inter_query_packed_debug
from src import cpu_sw

def run_debug_case():
    # Create short test pair
    query = "AACTTAACTG"
    ref = "AAGTTTACTT"
    print("\nGenerated Sequences")
    print(f"Query:  {query}")
    print(f"Ref:    {ref}")

    print("\nCPU Reference DP Matrix")
    score_cpu, pos_cpu, dp_cpu = cpu_sw.smith_waterman_cpu(query, ref)
    print(dp_cpu)
    print(f"Max Score = {score_cpu}, Position = {pos_cpu}")

    print("\nRunning Triton Single-Block Packed")
    score_sb, pos_sb, dp_sb = gpu_sw_single_block_packed.smith_waterman_gpu_single_block_packed(query, ref)
    print(f"Score = {score_sb}, Position = {pos_sb}")

    print("\nRunning Triton Inter-Query Packed Debug")
    results, dp_debug_list = gpu_sw_inter_query_packed_debug.smith_waterman_gpu_inter_query_packed_debug(
        [query], [ref]
    )
    score_debug, pos_debug = results[0]
    dp_debug = dp_debug_list[0]
    print(dp_debug)
    print(f"Score = {score_debug}, Position = {pos_debug}")

    print("\nComparison Summary")
    print(f"CPU vs SingleBlock: Score = {score_cpu} vs {score_sb}, Pos = {pos_cpu} vs {pos_sb}")
    print(f"CPU vs InterQuery : Score = {score_cpu} vs {score_debug}, Pos = {pos_cpu} vs {pos_debug}")
    print(f"SingleBlock vs InterQuery : Score = {score_sb} vs {score_debug}, Pos = {pos_sb} vs {pos_debug}")

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    run_debug_case()