import re
import statistics

# === Stats containers ===
score_pos_mismatch = []             # Score ❌, Pos ❌
score_only_mismatch_diffs = []      # Score ❌, Pos ✅ - all diffs
large_score_diffs = []              # Score ❌, Pos ✅ - only diffs > 50

# === Output file ===
output_file = "affine_my_dataset_mismatch_analysis_output.txt"

# === Load file ===
with open("affine_my_dataset_mismatch_report.txt", "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    if lines[i].startswith("Pair"):
        header = lines[i].strip()
        mismatch_type = header.split(":")[1].strip()

        triton_line = lines[i+1].strip()
        agatha_line = lines[i+2].strip()

        def parse(line):
            score = int(re.search(r'score:\s*(\d+)', line).group(1))
            pos = tuple(map(int, re.search(r'pos:\s*\((\d+),\s*(\d+)\)', line).groups()))
            lengths = re.search(r'query_len:\s*(\d+),\s*ref_len:\s*(\d+)', line)
            q_len, r_len = (int(lengths.group(1)), int(lengths.group(2))) if lengths else (None, None)
            return score, pos, q_len, r_len

        triton_score, triton_pos, q_len, r_len = parse(triton_line)
        agatha_score, agatha_pos, *_ = parse(agatha_line)

        # Case 1: Score mismatch only
        if mismatch_type == "Score ❌, Pos ✅":
            score_diff = abs(triton_score - agatha_score)
            score_only_mismatch_diffs.append(score_diff)
            if score_diff > 50:
                pair_id = int(re.search(r'#(\d+)', header).group(1))
                large_score_diffs.append({
                    "pair": pair_id,
                    "triton_score": triton_score,
                    "agatha_score": agatha_score,
                    "pos": triton_pos,  # same for both
                    "score_diff": score_diff
                })

        # Case 2: Score and Pos both mismatch
        elif mismatch_type == "Score ❌, Pos ❌":
            pair_id = int(re.search(r'#(\d+)', header).group(1))
            delta_pos = abs(triton_pos[0] - agatha_pos[0])

            if agatha_score == 32767 and 11000 <= agatha_pos[0] <= 12000:
                category = "A"
            elif delta_pos > 200:
                category = "B"
            else:
                category = "C"

            score_pos_mismatch.append({
                "pair": pair_id,
                "category": category,
                "triton_score": triton_score,
                "agatha_score": agatha_score,
                "triton_pos": triton_pos,
                "agatha_pos": agatha_pos,
                "query_len": q_len,
                "ref_len": r_len,
                "score_diff": abs(triton_score - agatha_score),
                "delta_pos": delta_pos,
            })

        i += 3
    else:
        i += 1

# === Generate output ===
with open(output_file, "w") as f:
    f.write("=== Score ❌, Pos ✅ ===\n")
    f.write(f"Total: {len(score_only_mismatch_diffs)}\n")
    if score_only_mismatch_diffs:
        f.write(f"  Max score diff: {max(score_only_mismatch_diffs)}\n")
        f.write(f"  Mean score diff: {statistics.mean(score_only_mismatch_diffs):.2f}\n")
        f.write(f"  Median score diff: {statistics.median(score_only_mismatch_diffs)}\n")
        f.write(f"  Count with score diff > 50: {len(large_score_diffs)}\n")

    if large_score_diffs:
        f.write("\n--- Score ❌, Pos ✅ with diff > 50 ---\n")
        for item in large_score_diffs:
            f.write(f"Pair #{item['pair']}:\n")
            f.write(f"  Triton score: {item['triton_score']}, AGAThA score: {item['agatha_score']}\n")
            f.write(f"  Pos: {item['pos']}, Score diff: {item['score_diff']}\n\n")

    # Count and categorize Score ❌, Pos ❌
    class_counts = {"A": 0, "B": 0, "C": 0}
    class_C_details = []

    for item in score_pos_mismatch:
        class_counts[item["category"]] += 1
        if item["category"] == "C":
            class_C_details.append(item)

    f.write("\n=== Score ❌, Pos ❌ ===\n")
    f.write(f"Total: {len(score_pos_mismatch)}\n")
    f.write(f"  Class A: {class_counts['A']} (AGAThA = 32767 and pos in 11000~12000)\n")
    f.write(f"  Class B: {class_counts['B']} (pos diff > 200)\n")
    f.write(f"  Class C: {class_counts['C']} (neither A nor B)\n")

    # Write detailed Class C entries
    f.write("\n--- Class C Mismatches ---\n")
    for entry in class_C_details:
        f.write(f"Pair #{entry['pair']}:\n")
        f.write(f"  Triton score: {entry['triton_score']}, pos: {entry['triton_pos']}\n")
        f.write(f"  AGAThA score: {entry['agatha_score']}, pos: {entry['agatha_pos']}\n")
        f.write(f"  Score diff: {entry['score_diff']}, Pos diff: {entry['delta_pos']}\n\n")

print(f"✅ Updated analysis written to '{output_file}'")