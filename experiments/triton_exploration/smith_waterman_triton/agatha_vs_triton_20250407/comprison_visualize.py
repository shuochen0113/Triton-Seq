import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Data
data = [
  {
    "sequence_length": 100,
    "triton_unpacked_wall_time_ms": 0.269,
    "triton_unpacked_wall_time1_ms": 0.8351802825927734,
    "triton_packed_wall_time_ms": 0.176,
    "triton_packed_wall_time1_ms": 0.5414485931396484,
    "agatha_kernel_time1_ms": 0.01024,
    "agatha_wall_time_ms": 0.219136
  },
  {
    "sequence_length": 500,
    "triton_unpacked_wall_time_ms": 0.663,
    "triton_unpacked_wall_time1_ms": 1.512289047241211,
    "triton_packed_wall_time_ms": 0.6,
    "triton_packed_wall_time1_ms": 1.833200454711914,
    "agatha_kernel_time1_ms": 0.011264,
    "agatha_wall_time_ms": 1.46534
  },
  {
    "sequence_length": 2000,
    "triton_unpacked_wall_time_ms": 7.839,
    "triton_unpacked_wall_time1_ms": 12.052297592163086,
    "triton_packed_wall_time_ms": 7.816,
    "triton_packed_wall_time1_ms": 12.229681015014648,
    "agatha_kernel_time1_ms": 0.011264,
    "agatha_wall_time_ms": 16.5519
  },
  {
    "sequence_length": 5000,
    "triton_unpacked_wall_time_ms": 43.345,
    "triton_unpacked_wall_time1_ms": 160.81905364990234,
    "triton_packed_wall_time_ms": 43.402,
    "triton_packed_wall_time1_ms": 160.0024700164795,
    "agatha_kernel_time1_ms": 0.011264,
    "agatha_wall_time_ms": 95.6088
  },
  {
    "sequence_length": 10000,
    "triton_unpacked_wall_time_ms": 162.198,
    "triton_unpacked_wall_time1_ms": 613.149881362915,
    "triton_packed_wall_time_ms": 162.94,
    "triton_packed_wall_time1_ms": 407.70864486694336,
    "agatha_kernel_time1_ms": 0.00704,
    "agatha_wall_time_ms": 343.612
  },
  {
    "sequence_length": 15000,
    "triton_unpacked_wall_time_ms": 340.005,
    "triton_unpacked_wall_time1_ms": 940.0787353515625,
    "triton_packed_wall_time_ms": 337.219,
    "triton_packed_wall_time1_ms": 913.8681888580322,
    "agatha_kernel_time1_ms": 0.006144,
    "agatha_wall_time_ms": 765.434
  },
  {
    "sequence_length": 20000,
    "triton_unpacked_wall_time_ms": 766.744,
    "triton_unpacked_wall_time1_ms": 1768.1422233581543,
    "triton_packed_wall_time_ms": 789.713,
    "triton_packed_wall_time1_ms": 1812.0672702789307,
    "agatha_kernel_time1_ms": 0.006912,
    "agatha_wall_time_ms": 1353.38
  },
  {
    "sequence_length": 30000,
    "triton_unpacked_wall_time_ms": 1407.862,
    "triton_unpacked_wall_time1_ms": 3629.1775703430176,
    "triton_packed_wall_time_ms": 1420.279,
    "triton_packed_wall_time1_ms": 3634.613513946533,
    "agatha_kernel_time1_ms": 0.008192,
    "agatha_wall_time_ms": 3048.13
  },
  {
    "sequence_length": 40000,
    "triton_unpacked_wall_time_ms": 3003.368,
    "triton_unpacked_wall_time1_ms": 6874.038457870483,
    "triton_packed_wall_time_ms": 3138.915,
    "triton_packed_wall_time1_ms": 6996.674060821533,
    "agatha_kernel_time1_ms": 0.006848,
    "agatha_wall_time_ms": 5387.54
  }
]
df = pd.DataFrame(data)

# Melt dataframe
df_melted = df.melt(id_vars="sequence_length", var_name="metric", value_name="time_ms")

# Helper: Extract kernel name and time type
def extract_info(metric):
    if "triton_unpacked" in metric:
        kernel = "Triton Unpacked"
    elif "triton_packed" in metric:
        kernel = "Triton Packed"
    elif "agatha" in metric:
        kernel = "AGAThA"
    else:
        kernel = "Unknown"
    time_type = "Kernel Time" if "kernel" in metric else "Wall Time"
    return pd.Series([kernel, time_type])

df_melted[["kernel", "time_type"]] = df_melted["metric"].apply(extract_info)

# Plot wall time
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_melted[df_melted["time_type"] == "Wall Time"],
    x="sequence_length",
    y="time_ms",
    hue="kernel",
    marker="o",
    errorbar=None
)
plt.title("Kernel Time vs. Sequence Length")
plt.xlabel("Sequence Length")
plt.ylabel("Time (ms)")
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

# Save figure with white background (avoid alpha/transparency issues)
plt.savefig("kernel_time_vs_sequence_length_corrected.png", dpi=300, facecolor='white')
plt.show()