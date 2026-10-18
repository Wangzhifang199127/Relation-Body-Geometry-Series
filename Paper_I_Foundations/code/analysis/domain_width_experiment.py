# domain_width_experiment.py

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============ Configuration ============
MODEL_NAME = "gpt2"
TOP_K_LOCAL = 10
OUTPUT_DIR = "attention_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ Four semantic widths ============
DOMAINS = {
    "repeat": (
        "the the the the the the the the the the "
        "the the the the the the the the the the "
        "the the the the the the the the the the"
    ),
    "single": (
        "def factorial(n):\n"
        "    if n == 0:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n\n"
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
        "class Node:\n"
        "    def __init__(self, val):\n"
        "        self.val = val\n"
        "        self.left = None\n"
        "        self.right = None"
    ),
    "double": (
        "The integral of x squared is x cubed over three. "
        "Consider the equation y equals m x plus b. "
        "The derivative of sine is cosine. "
        "A rose is a flower that blooms in the garden. "
        "The poet wrote verses about love and loss. "
        "Beauty is truth, truth beauty, that is all. "
        "The matrix has eigenvalues that satisfy the characteristic polynomial. "
        "She walked through the meadow with grace and wonder."
    ),
    "multi": (
        "The stock market rose today. "
        "def hello(): print('world'). "
        "The integral of x squared is x cubed over three. "
        "She said, 'I love you.' "
        "The cat sat on the mat. "
        "for i in range(10): print(i). "
        "Quantum mechanics describes the behavior of particles. "
        "Once upon a time, there was a princess. "
        "The algorithm has O(n log n) complexity. "
        "He felt happy and sad at the same time. "
        "The protein folds into a complex structure. "
        "The artist painted a beautiful landscape."
    ),
}


# ============ Load model ============
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model.eval()

n_layer = model.config.n_layer
n_head = model.config.n_head
n_embd = model.config.n_embd
head_dim = n_embd // n_head


# ============ Analysis function ============
def analyze_domain(text, domain_name):
    """For a given input text, compute PC1 and rA for all layers/heads."""
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]

    qkv_store = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            qkv_store[layer_idx] = out.detach()
        return hook_fn

    hooks = []
    for l in range(n_layer):
        h = model.transformer.h[l].attn.c_attn.register_forward_hook(
            make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    records = []
    for layer in range(n_layer):
        qkv = qkv_store[layer][0]
        q, k, v = qkv.split(n_embd, dim=-1)
        q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
        k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
        v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

        for head in range(n_head):
            q_h, k_h, v_h = q[head], k[head], v[head]

            scale = 1.0 / np.sqrt(head_dim)
            scores = (q_h @ k_h.T) * scale
            mask = torch.tril(torch.ones(seq_len, seq_len))
            scores = scores.masked_fill(mask == 0, float('-inf'))
            p = torch.softmax(scores, dim=-1)
            mask_bool = mask.bool()

            # Scheme A: local redundancy
            V_bar = p @ v_h
            delta = torch.norm(v_h.unsqueeze(0) - V_bar.unsqueeze(1), dim=-1)
            p_flat = p[mask_bool].numpy()
            delta_flat = delta[mask_bool].numpy()
            if np.std(p_flat) > 1e-8 and np.std(delta_flat) > 1e-8:
                rA = float(np.corrcoef(p_flat, delta_flat)[0, 1])
            else:
                rA = np.nan

            # Local PCA
            conc = np.full(seq_len, np.nan)
            for i in range(seq_len):
                k_actual = min(TOP_K_LOCAL, i + 1)
                if k_actual < 3:
                    continue
                p_i = p[i, :i + 1]
                top_idx = torch.topk(p_i, k_actual).indices
                V_top = v_h[top_idx].detach().cpu().numpy()
                V_c = V_top - V_top.mean(axis=0, keepdims=True)
                try:
                    _, S, _ = np.linalg.svd(V_c, full_matrices=False)
                    e = S ** 2
                    if e.sum() > 1e-12:
                        conc[i] = e[0] / e.sum()
                except Exception:
                    pass

            mean_conc = float(np.nanmean(conc))

            records.append({
                "domain": domain_name,
                "layer": layer,
                "head": head,
                "rA": rA,
                "mean_conc": mean_conc,
            })

    return pd.DataFrame(records)


# ============ Run all domains ============
print("\n" + "=" * 60)
all_results = []
for domain_name, text in DOMAINS.items():
    print(f"\nAnalyzing domain: {domain_name}")
    inputs = tokenizer(text, return_tensors="pt")
    print(f"  Seq length: {inputs['input_ids'].shape[1]}")
    df = analyze_domain(text, domain_name)
    all_results.append(df)

df_all = pd.concat(all_results, ignore_index=True)
csv_path = os.path.join(OUTPUT_DIR, "domain_width_results.csv")
df_all.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


# ============ Summary ============
print("\n===== Summary =====")
summary = df_all.groupby('domain').agg({
    'mean_conc': ['mean', 'std', 'max'],
    'rA': ['mean'],
}).round(4)
print(summary)


# ============ Visualization ============
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

domain_order = ["repeat", "single", "double", "multi"]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

# Figure 1: mean PC1 per domain
ax = axes[0, 0]
means = [df_all[df_all['domain'] == d]['mean_conc'].mean()
         for d in domain_order]
stds = [df_all[df_all['domain'] == d]['mean_conc'].std()
        for d in domain_order]
ax.bar(domain_order, means, yerr=stds, capsize=5, color=colors)
ax.axhline(1/head_dim, color='r', ls='--', label=f'random = 1/{head_dim}')
ax.set_ylabel("Mean PC1 concentration")
ax.set_title("PC1 vs semantic width")
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Figure 2: layer trends
ax = axes[0, 1]
for d, c in zip(domain_order, colors):
    sub = df_all[df_all['domain'] == d]
    layer_mean = sub.groupby('layer')['mean_conc'].mean()
    ax.plot(layer_mean.index, layer_mean.values, 'o-',
            label=d, color=c)
ax.axhline(1/head_dim, color='r', ls='--', alpha=0.5,
           label=f'random = 1/{head_dim}')
ax.set_xlabel("Layer")
ax.set_ylabel("Mean PC1")
ax.set_title("Layer trend by domain")
ax.legend()
ax.grid(alpha=0.3)

# Figure 3: rA distribution
ax = axes[1, 0]
for d, c in zip(domain_order, colors):
    sub = df_all[df_all['domain'] == d]
    ax.hist(sub['rA'].dropna(), bins=20, alpha=0.5, label=d, color=c)
ax.axvline(0, color='k', ls='--')
ax.set_xlabel("rA")
ax.set_ylabel("Count")
ax.set_title("rA distribution by domain")
ax.legend()
ax.grid(alpha=0.3)

# Figure 4: PC1 heatmap by domain
ax = axes[1, 1]
pc1_matrix = np.zeros((len(domain_order), n_layer))
for i, d in enumerate(domain_order):
    sub = df_all[df_all['domain'] == d]
    layer_mean = sub.groupby('layer')['mean_conc'].mean()
    pc1_matrix[i, :] = layer_mean.values
im = ax.imshow(pc1_matrix, aspect='auto', cmap='viridis')
ax.set_yticks(range(len(domain_order)))
ax.set_yticklabels(domain_order)
ax.set_xlabel("Layer")
ax.set_title("PC1 heatmap: domain × layer")
plt.colorbar(im, ax=ax, label='Mean PC1')

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "domain_width_results.png")
plt.savefig(save_path, dpi=150)
plt.show()
print(f"\nSaved: {save_path}")


# ============ Key test ============
print("\n===== Key Test =====")
print("Hypothesis: narrower domain -> higher PC1")
print()
for d in domain_order:
    sub = df_all[df_all['domain'] == d]
    mean_pc1 = sub['mean_conc'].mean()
    print(f"  {d:<10}: mean PC1 = {mean_pc1:.4f}")

print()
print("Order check (should be decreasing):")
means_ordered = [df_all[df_all['domain'] == d]['mean_conc'].mean()
                 for d in domain_order]
is_decreasing = all(means_ordered[i] >= means_ordered[i+1]
                    for i in range(len(means_ordered) - 1))
print(f"  Monotonically decreasing: {is_decreasing}")
if is_decreasing:
    print("  Hypothesis SUPPORTED")
else:
    print("  Hypothesis not fully supported")

print("\nDone.")