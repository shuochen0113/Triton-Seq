import torch
import random
import traceback

from src import gpu_sw_single_block_packed
from src import gpu_sw_guided_inter_query

def generate_mutated_pair(length):
    bases = "ATCG"
    ref = [random.choice(bases) for _ in range(length)]
    query = ref.copy()
    for _ in range(length // 2):  # 50% mutation
        idx = random.randint(0, length - 1)
        query[idx] = random.choice([b for b in bases if b != query[idx]])
    return "".join(query), "".join(ref)

def test_correctness_all(debug=False):
    lengths = [100, 200, 500, 1000, 2000, 5000, 8000, 10000, 12000, 15000]
    query_list = []
    ref_list = []

    print(f"\nTesting {len(lengths)} sequence pairs (in one inter-query batch)...")

    for L in lengths:
        q, r = generate_mutated_pair(L)
        query_list.append(q)
        ref_list.append(r)

    print("Running reference (single-block, sequential)...")
    ref_results = []
    for i in range(len(query_list)):
        score, pos, _ = gpu_sw_single_block_packed.smith_waterman_gpu_single_block_packed(
            query_list[i], ref_list[i]
        )
        ref_results.append((score, pos))
        if debug:
            print(f"[Debug] Ref Score {i+1}: {score}, Pos: {pos}")

    print("Running inter-query packed (parallel across blocks)...")
    try:
        packed_results = gpu_sw_guided_inter_query.smith_waterman_gpu_guided_packed(
            query_list, ref_list
        )
    except Exception as e:
        print("Error running inter-query packed kernel:")
        traceback.print_exc()
        return

    for i in range(len(query_list)):
        ref_score, ref_pos = ref_results[i]
        packed_score, packed_pos = packed_results[i]

        print(f"\n=== Pair {i + 1} | Length = {lengths[i]} ===")
        print(f"Ref   : score = {ref_score}, pos = {ref_pos}")
        print(f"Query : score = {packed_score}, pos = {packed_pos}")

        if ref_score != packed_score:
            print("Mismatch in score!")
        else:
            print("Match")

        if debug:
            print(f"[Debug] Score diff = {ref_score - packed_score}")
            if ref_pos != packed_pos:
                print(f"[Debug] Pos diff = {ref_pos} vs {packed_pos}")

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    test_correctness_all(debug=True)