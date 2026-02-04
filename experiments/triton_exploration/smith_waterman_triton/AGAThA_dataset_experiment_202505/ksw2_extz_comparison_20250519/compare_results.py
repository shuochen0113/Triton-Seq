import json
import re

# path
# triton_json_path = "../outputs/triton_affine_guided_alignment_results/triton_alignments_10000_pairs.json"
triton_json_path = "/root/shuochen/Cornell_intern/triton_experiment/smith_waterman_triton/outputs/guided_ksw2_gpu_20250521_231159.json"
agatha_scorelog_path = "/root/shuochen/Cornell_intern/triton_experiment/AGAThA/output/score.log"
output_report_path = "./ksw2_comparison_output_20250521_7.txt"

# read triton results
with open(triton_json_path) as f:
    triton_data = json.load(f)

# read agatha results
agatha_data = []
with open(agatha_scorelog_path) as f:
    for line in f:
        match = re.match(r"(\d+)\s+query_batch_end=(\d+)\s+target_batch_end=(\d+)", line.strip())
        if match:
            score = int(match.group(1))
            ref_pos = int(match.group(2)) + 1
            query_pos = int(match.group(3)) + 1
            agatha_data.append((score, query_pos, ref_pos))

assert len(triton_data) == len(agatha_data), "Mismatch in number of sequence pairs."

exact_match = 0
score_only_mismatch = 0
position_only_mismatch = 0
both_mismatch = 0
diffs = []

# compare results
for i, (triton_item, agatha_item) in enumerate(zip(triton_data, agatha_data)):
    t_score = triton_item["score"]
    t_qpos, t_rpos = triton_item["position"]
    a_score, a_qpos, a_rpos = agatha_item

    score_match = (t_score == a_score)
    position_match = (t_qpos == a_qpos and t_rpos == a_rpos)

    if score_match and position_match:
        exact_match += 1
    else:
        # query_seq = triton_item["query"]
        # ref_seq = triton_item["reference"]
        # q_len = len(query_seq)
        # r_len = len(ref_seq)

        if score_match and not position_match:
            position_only_mismatch += 1
            reason = "Score ✅, Pos ❌"
        elif not score_match and position_match:
            score_only_mismatch += 1
            reason = "Score ❌, Pos ✅"
        else:
            both_mismatch += 1
            reason = "Score ❌, Pos ❌"

        diffs.append((i, t_score, t_qpos, t_rpos, a_score, a_qpos, a_rpos, reason))

# write report
with open(output_report_path, "w") as f:
    f.write("==== Alignment Comparison Report ====\n")
    f.write(f"Total pairs: {len(triton_data)}\n")
    f.write(f"✔️  Exact match (score & position): {exact_match}\n")
    f.write(f"⚠️  Score match, but position mismatch: {position_only_mismatch}\n")
    f.write(f"⚠️  Position match, but score mismatch: {score_only_mismatch}\n")
    f.write(f"❌  Score & position both mismatch: {both_mismatch}\n\n")

    f.write("---- Mismatched Details ----\n")
    for pid, t_s, t_q, t_r, a_s, a_q, a_r, reason in diffs:
        f.write(f"Pair #{pid}: {reason}\n")
        f.write(f"  Triton → score: {t_s}, pos: ({t_q}, {t_r})\n")
        f.write(f"  AGAThA → score: {a_s}, pos: ({a_q}, {a_r})\n\n")

print(f"✅ Done. Report saved to: {output_report_path}")