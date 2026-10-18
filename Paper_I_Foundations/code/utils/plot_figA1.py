# plot_figA1.py
"""
Generate figA1: complete PC1 heatmap for GPT-2 medium.
Data source: data/raw/cross_model_results.csv
Output: figures/appendix/figA1_gpt2_medium_pc1.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ Configuration ============
DATA_PATH = "data/raw/cross_model_results.csv"
OUTPUT_DIR = "figures/appendix"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "figA1_gpt2_medium_pc1.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})

# ============ Load data ============
df = pd.read_csv(DATA_PATH)
sub = df[df["model"] == "gpt2-medium"].copy()

n_layer = int(sub["layer"].max()) + 1
n_head = int(sub["head"].max()) + 1

print(f"GPT-2 medium: {n_layer} layers, {n_head} heads")

# ============ Build PC1 matrix ============
pc1_matrix = np.full((n_layer, n_head), np.nan)
for _, row in sub.iterrows():
    pc1_matrix[int(row["layer"]), int(row["head"])] = row["mean_conc"]

# ============ Plotting ============
fig, ax = plt.subplots(figsize=(7, 9))
im = ax.imshow(pc1_matrix, aspect="auto", cmap="viridis",
               vmin=0.2, vmax=0.8, origin="upper")

ax.set_xlabel("Head")
ax.set_ylabel("Layer")
ax.set_title("GPT-2 medium: PC1 concentration (layer × head)")

ax.set_xticks(range(n_head))
ax.set_yticks(range(0, n_layer, 2))
ax.set_yticklabels(range(0, n_layer, 2))

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Mean PC1 concentration")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {OUTPUT_PATH}")