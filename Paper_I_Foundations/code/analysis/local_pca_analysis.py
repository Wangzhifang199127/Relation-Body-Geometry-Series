# local_pca_analysis.py

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.stats import pearsonr, spearmanr

# ============ Configuration ============
MODEL_NAME = "gpt2"
TOP_K_LOCAL = 10        # top-k attended tokens per query
TEXT = ("The cat sat on the mat. It was a sunny day and the cat wanted "
        "to play outside. The dog ran to the park and saw a bird.")
OUTPUT_DIR = "attention_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

print(f"Layers: {n_layer}, Heads: {n_head}, Seq: {seq_len}")


# ============ Single forward pass, collect QKV from all layers ============
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


# ============ Local PCA concentration ============
def local_pca_concentration(v_h, p, top_k=TOP_K_LOCAL):
    """
    For each query i:
      1. Take the V vectors of the top-k tokens with highest attention probability;
      2. Perform PCA;
      3. Return the PC1 variance ratio (how much variance the first principal component explains).

    Returns: array concentration[i] of length seq_len.
    """
    concentration = np.full(seq_len, np.nan)
    for i in range(seq_len):
        k_actual = min(top_k, i + 1)
        if k_actual < 3:
            continue
        p_i = p[i, :i + 1]
        top_indices = torch.topk(p_i, k_actual).indices
        V_top = v_h[top_indices].detach().cpu().numpy()  # [k, head_dim]

        # Center
        V_c = V_top - V_top.mean(axis=0, keepdims=True)

        # SVD
        try:
            U, S, Vt = np.linalg.svd(V_c, full_matrices=False)
            energies = S ** 2
            total = energies.sum()
            if total < 1e-12:
                continue
            concentration[i] = energies[0] / total
        except Exception:
            continue
    return concentration


# ============ Batch scan over all layers and heads ============
print("\nScanning local PCA across all layers and heads...")
records = []
concentration_maps = np.full((n_layer, n_head, seq_len), np.nan)

for layer in range(n_layer):
    qkv = qkv_store[layer][0]
    q, k, v = qkv.split(n_embd, dim=-1)
    q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
    k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
    v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

    for head in range(n_head):
        q_h, k_h, v_h = q[head], k[head], v[head]

        # Attention probabilities
        scale = 1.0 / np.sqrt(head_dim)
        scores = (q_h @ k_h.T) * scale
        mask = torch.tril(torch.ones(seq_len, seq_len))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        p = torch.softmax(scores, dim=-1)

        # Local PCA concentration
        conc = local_pca_concentration(v_h, p, TOP_K_LOCAL)
        concentration_maps[layer, head] = conc

        # Attention entropy
        p_np = p.cpu().numpy()
        entropy = -np.nansum(
            np.where(p_np > 0, p_np * np.log(p_np + 1e-12), 0), axis=-1
        )

        # Correlation: concentration vs. entropy (causal alignment)
        valid = ~np.isnan(conc)
        if valid.sum() > 3 and np.std(conc[valid]) > 1e-8 and np.std(entropy[valid]) > 1e-8:
            r_conc_entropy, _ = pearsonr(conc[valid], entropy[valid])
        else:
            r_conc_entropy = np.nan

        # Mean concentration (over valid positions)
        mean_conc = np.nanmean(conc)

        records.append({
            "layer": layer,
            "head": head,
            "mean_conc": mean_conc,
            "r_conc_entropy": r_conc_entropy,
        })

    print(f"  Layer {layer} done")

df = pd.DataFrame(records)
csv_path = os.path.join(OUTPUT_DIR, "local_pca_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


# ============ Summary statistics ============
# Theoretical baseline: if V vectors are completely random in head_dim dimensions,
# the PC1 variance ratio is approximately 1/head_dim (very small).
random_baseline = 1.0 / head_dim

print("\n===== Local PCA Summary =====")
print(f"head_dim = {head_dim}, random baseline PC1 ratio ≈ {random_baseline:.4f}")
print(f"\nMean PC1 concentration across all (layer, head):")
print(f"  mean = {df['mean_conc'].mean():.4f}")
print(f"  min  = {df['mean_conc'].min():.4f}")
print(f"  max  = {df['mean_conc'].max():.4f}")
print(f"\nFraction with mean_conc > 0.5 (strong local trunk):")
print(f"  {(df['mean_conc'] > 0.5).mean():.2%}")
print(f"Fraction with mean_conc > 0.3:")
print(f"  {(df['mean_conc'] > 0.3).mean():.2%}")
print(f"\nCorrelation between concentration and attention entropy:")
print(f"  mean r = {df['r_conc_entropy'].mean():+.4f}")


# ============ Visualization ============
# Figure 1: heatmap of mean concentration per layer/head
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

mat_conc = df['mean_conc'].values.reshape(n_layer, n_head)
im0 = axes[0].imshow(mat_conc, aspect='auto', cmap='viridis', vmin=0, vmax=1)
axes[0].set_xlabel("Head")
axes[0].set_ylabel("Layer")
axes[0].set_title(f"Local PC1 concentration (top-{TOP_K_LOCAL} tokens)")
plt.colorbar(im0, ax=axes[0])

mat_r = df['r_conc_entropy'].values.reshape(n_layer, n_head)
im1 = axes[1].imshow(mat_r, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_xlabel("Head")
axes[1].set_ylabel("Layer")
axes[1].set_title("r(concentration, attention entropy)")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "local_pca_heatmaps.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Saved: {fig_path}")


# Figure 2: layer trend + comparison with random baseline
fig, ax = plt.subplots(figsize=(10, 6))
layer_mean = df.groupby('layer')['mean_conc'].mean()
ax.plot(layer_mean.index, layer_mean.values, 'o-', label='mean PC1 ratio')
ax.axhline(random_baseline, color='r', ls='--',
           label=f'random baseline = 1/{head_dim}')
ax.set_xlabel("Layer")
ax.set_ylabel("Mean PC1 concentration")
ax.set_title(f"Local trunk strength (top-{TOP_K_LOCAL} tokens)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
trend_path = os.path.join(OUTPUT_DIR, "local_pca_trend.png")
plt.savefig(trend_path, dpi=150)
plt.show()
print(f"Saved: {trend_path}")


# ============ Choose a representative head, plot per-query concentration curve ============
# Choose the head with the highest concentration as representative
best_row = df.loc[df['mean_conc'].idxmax()]
best_layer = int(best_row['layer'])
best_head = int(best_row['head'])
print(f"\nRepresentative head: layer {best_layer}, head {best_head} "
      f"(mean_conc = {best_row['mean_conc']:.4f})")

conc_curve = concentration_maps[best_layer, best_head]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(seq_len), conc_curve, 'o-', color='steelblue')
ax.axhline(0.5, color='orange', ls='--', label='0.5 threshold')
ax.axhline(random_baseline, color='r', ls='--',
           label=f'random baseline = 1/{head_dim}')
ax.set_xlabel("Query position")
ax.set_ylabel("PC1 concentration")
ax.set_title(f"PC1 concentration per query (layer {best_layer}, head {best_head})")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "local_pca_curve.png")
plt.savefig(curve_path, dpi=150)
plt.show()
print(f"Saved: {curve_path}")

print("\nDone.")