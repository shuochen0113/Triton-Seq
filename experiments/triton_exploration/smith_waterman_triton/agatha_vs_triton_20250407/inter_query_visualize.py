import matplotlib.pyplot as plt
import pandas as pd

# Define your actual benchmarking data
data = [
    {
        "num_pairs": 1,
        "triton_sequential_packed_wall_ms": 113.421,
        "triton_interquery_packed_wall_ms": 4143.485,
        "agatha_wall_ms": 245.701
    },
    {
        "num_pairs": 10,
        "triton_sequential_packed_wall_ms": 809.0939999999999,
        "triton_interquery_packed_wall_ms": 27640.264,
        "agatha_wall_ms": 351.745
    },
    {
        "num_pairs": 50,
        "triton_sequential_packed_wall_ms": 5018.250000000003,
        "triton_interquery_packed_wall_ms": 26574.943,
        "agatha_wall_ms": 430.844
    },
    {
        "num_pairs": 100,
        "triton_sequential_packed_wall_ms": 5582.77,
        "triton_interquery_packed_wall_ms": 21234.264,
        "agatha_wall_ms": 345.916
    },
    {
        "num_pairs": 500,
        "triton_sequential_packed_wall_ms": 42996.67600000004,
        "triton_interquery_packed_wall_ms": 28603.311,
        "agatha_wall_ms": 487.003
    },
    {
        "num_pairs": 1000,
        "triton_sequential_packed_wall_ms": 89228.86799999999,
        "triton_interquery_packed_wall_ms": 30853.15,
        "agatha_wall_ms": 502.333
    },
    {
        "num_pairs": 2000,
        "triton_sequential_packed_wall_ms": 163688.72499999966,
        "triton_interquery_packed_wall_ms": 38285.359,
        "agatha_wall_ms": 510.524
    },
    {
        "num_pairs": 5000,
        "triton_sequential_packed_wall_ms": 411909.5660000009,
        "triton_interquery_packed_wall_ms": 50120.703,
        "agatha_wall_ms": 796.974
    },
    {
        "num_pairs": 10000,
        "triton_sequential_packed_wall_ms": 828294.3559999972,
        "triton_interquery_packed_wall_ms": 75752.961,
        "agatha_wall_ms": 1506.216
    },
    {
        "num_pairs": 15000,
        "triton_sequential_packed_wall_ms": 1201055.7890000045,
        "triton_interquery_packed_wall_ms": 105465.719,
        "agatha_wall_ms": 1874.225
    }
]

# Load data into pandas DataFrame
df = pd.DataFrame(data)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(df["num_pairs"], df["triton_sequential_packed_wall_ms"], marker='o', label="Triton Sequential (Packed)")
plt.plot(df["num_pairs"], df["triton_interquery_packed_wall_ms"], marker='s', label="Triton Inter-query (Packed)")
plt.plot(df["num_pairs"], df["agatha_wall_ms"], marker='^', label="AGAThA")

# Log scale for better visualization
plt.xscale("log")
plt.yscale("log")

# Labels and title
plt.xlabel("Number of Sequence Pairs (log scale)")
plt.ylabel("Time (ms, log scale)")
plt.title("Kernel Time vs. Number of Sequence Pairs (log-log)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

# Save plot to local file
plt.savefig("kernel_time_vs_num_pairs_corrected.png", dpi=300)
