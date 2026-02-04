import json
import matplotlib.pyplot as plt
from pathlib import Path

json_path = "../outputs/sw_guided_vs_agatha_20250505_193706.json"
with open(json_path, "r") as f:
    data = json.load(f)

num_pairs = [entry["num_pairs"] for entry in data]

triton_local = [entry["triton_local_kernel_time_ms"] for entry in data]
triton_total = [entry["triton_total_time_ms"] for entry in data]

agatha_local = [entry["agatha_wall_time_json_ms"] for entry in data]
agatha_total = [entry["total_ms"] for entry in data]

plt.figure(figsize=(10, 6))
plt.plot(num_pairs, triton_local, "o-", label="Triton: Local Kernel Time")
plt.plot(num_pairs, agatha_local, "o--", label="AGAThA: Local Kernel Time")

plt.plot(num_pairs, triton_total, "s-", label="Triton: Total Time")
plt.plot(num_pairs, agatha_total, "s--", label="AGAThA: Total Time")

plt.xlabel("Number of Sequence Pairs")
plt.ylabel("Time (ms)")
plt.title("Guided Triton vs AGAThA - Local & Total Time Comparison")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.savefig("guided_time_comparison_mydataset.png", dpi=300)
plt.show()