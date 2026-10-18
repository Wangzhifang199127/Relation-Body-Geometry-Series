# code/analysis/verify_A1.py
"""
Empirical verification of Assumption A1: q_i · k_j ≈ λ v_i · v_j.

For each head of GPT-2 small, compute the Pearson correlation coefficient
between q_i · k_j and v_i · v_j.

Outputs:
  - data/raw/A1_verification.csv (per-head correlation coefficients)
  - figures/appendix/figA6_a1_verification.png (histogram)
  - Table A1 summary statistics printed to terminal
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============ Configuration ============
MODEL_NAME = "gpt2"
OUTPUT_CSV = "data/raw/A1_verification.csv"
OUTPUT_FIG = "figures/appendix/figA6_a1_verification.png"
os.makedirs("data/raw", exist_ok=True)
os.makedirs("figures/appendix", exist_ok=True)

TEXT = ("The cat sat on the mat. It was a sunny day and the cat wanted "
        "to play outside. The dog ran to the park and saw a bird. "
        "The bird flew over the house and landed on the roof. "
        "She watched the bird from the window and smiled.")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})


# ============ Load model ============
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model.eval()

inputs = tokenizer(TEXT, return_tensors="pt")
seq_len = inputs["input_ids"].shape[1]

n_layer = model.config.n_layer
n_head = model.config.n_head
n_embd = model.config.n_embd
head_dim = n_embd // n_head

print(f"Layers: {n_layer}, Heads: {n_head}, Seq: {seq_len}, head_dim: {head_dim}")


# ============ Extract QKV from all layers ============
qkv_store = {}

def make_hook(layer_idx):
    def hook_fn(module, inp, out):
        qkv_store[layer_idx] = out.detach()
    return hook_fn

hooks = []
for l in range(n_layer):
    h = model.transformer.h[l].attn.c_attn.register_forward_hook(make_hook(l))
    hooks.append(h)

with torch.no_grad():
    model(**inputs)

for h in hooks:
    h.remove()


# ============ Compute per-head correlation ============
records = []

for layer in range(n_layer):
    qkv = qkv_store[layer][0]  # [seq, 3*n_embd]
    q, k, v = qkv.split(n_embd, dim=-1)

    q = q.view(seq_len, n_head, head_dim).transpose(0, 1)  # [n_head, seq, head_dim]
    k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
    v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

    for head in range(n_head):
        q_h = q[head]  # [seq, head_dim]
        k_h = k[head]
        v_h = v[head]

        # Compute q_i · k_j and v_i · v_j, shape [seq, seq]
        qk = (q_h @ k_h.T).cpu().numpy()
        vv = (v_h @ v_h.T).cpu().numpy()

        # Take the causal lower triangle (including diagonal)
        mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        qk_flat = qk[mask]
        vv_flat = vv[mask]

        # Filter zero variance
        if np.std(qk_flat) < 1e-8 or np.std(vv_flat) < 1e-8:
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(qk_flat, vv_flat)

        records.append({
            "layer": layer,
            "head": head,
            "pearson_r": r,
            "p_value": p,
        })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")


# ============ Summary statistics (Table A1) ============
valid = df.dropna(subset=["pearson_r"])
r_vals = valid["pearson_r"].values
p_vals = valid["p_value"].values

mean_r = np.mean(r_vals)
median_r = np.median(r_vals)
positive_frac = np.mean(r_vals > 0)
significant_frac = np.mean((r_vals > 0) & (p_vals < 0.01))
max_r = np.max(r_vals)
min_r = np.min(r_vals)
max_idx = valid["pearson_r"].idxmax()
min_idx = valid["pearson_r"].idxmin()
max_layer = int(valid.loc[max_idx, "layer"])
max_head = int(valid.loc[max_idx, "head"])
min_layer = int(valid.loc[min_idx, "layer"])
min_head = int(valid.loc[min_idx, "head"])

print("\n===== Table A1: Verification of Assumption A1 =====")
print(f"Total heads:                       {len(valid)}")
print(f"Mean correlation:                  {mean_r:.4f}")
print(f"Median correlation:                {median_r:.4f}")
print(f"Positive correlation ratio (r>0):  {positive_frac:.2%}")
print(f"Significant positive ratio:        {significant_frac:.2%}")
print(f"Maximum correlation:               {max_r:.4f} (Layer {max_layer}, Head {max_head})")
print(f"Minimum correlation:               {min_r:.4f} (Layer {min_layer}, Head {min_head})")


# ============ Figure A6: histogram ============
fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(r_vals, bins=25, color="steelblue", alpha=0.75,
        edgecolor="white", linewidth=0.5)

ax.axvline(0, color="black", linestyle="--", linewidth=1, label="r = 0")
ax.axvline(mean_r, color="crimson", linestyle="--", linewidth=1.5,
           label=f"mean = {mean_r:.3f}")
ax.axvline(median_r, color="orange", linestyle=":", linewidth=1.5,
           label=f"median = {median_r:.3f}")

ax.set_xlabel(r"Pearson $r(q_i \cdot k_j,\ v_i \cdot v_j)$")
ax.set_ylabel("Count (layer, head) pairs")
ax.set_title("Verification of Assumption A1 (GPT-2 small)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved: {OUTPUT_FIG}")


# ============ Layer-wise mean (optional) ============
print("\n===== Layer-wise mean correlation =====")
layer_mean = valid.groupby("layer")["pearson_r"].mean()
for layer, val in layer_mean.items():
    print(f"  Layer {layer:2d}: {val:+.4f}")