import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------- DATA ----------------
quant_modes = ["4-bit", "8-bit", "16-bit", "32-bit"]

# Average inference time (seconds) from your logs
avg_time = [
    (11.30 + 10.66 + 11.57 + 10.43 + 10.16) / 5,  # 4bit
    (23.21 + 20.60 + 21.06 + 21.82 + 24.69) / 5,  # 8bit
    (334.61 + 324.21 + 321.86 + 327.10 + 325.51) / 5,  # 16bit
    (366.37 + 356.10 + 356.70 + 354.82 + 356.26) / 5,  # 32bit
]

# GPU memory (MB) before and after (nvidia-smi values)
gpu_mem_before = [3642, 4702, 6394, 6363]
gpu_mem_after  = [3730, 4990, 6978, 7102]

# ---------------- PLOT ----------------
sns.set(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot inference time (bar chart)
color_time = "tab:blue"
ax1.set_xlabel("Quantization Mode", fontsize=12)
ax1.set_ylabel("Avg Inference Time (s)", color=color_time, fontsize=12)
bars = ax1.bar(quant_modes, avg_time, color=color_time, alpha=0.7, label="Avg Inference Time (s)")
ax1.tick_params(axis="y", labelcolor=color_time)

# Annotate time values on bars
for bar, val in zip(bars, avg_time):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{val:.1f}s", ha="center", va="bottom", fontsize=10, color=color_time)

# Twin axis for GPU memory
ax2 = ax1.twinx()
color_mem = "tab:red"
ax2.set_ylabel("GPU Memory (MB)", color=color_mem, fontsize=12)
ax2.plot(quant_modes, gpu_mem_before, marker="o", linestyle="--", color="tab:orange", label="Memory Before")
ax2.plot(quant_modes, gpu_mem_after, marker="o", linestyle="-", color=color_mem, label="Memory After")
ax2.tick_params(axis="y", labelcolor=color_mem)

# Add legend
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9), bbox_transform=ax1.transAxes)

# Title
plt.title("Quantization Impact on Phi-3-mini-4k-instruct\n(Inference Latency vs GPU Memory)", fontsize=14)

plt.tight_layout()
plt.show()
