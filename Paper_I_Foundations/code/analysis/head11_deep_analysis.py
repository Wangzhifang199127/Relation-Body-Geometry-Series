# head11_deep_analysis.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============ Configuration ============
MODEL_NAME = "gpt2"
LAYER = 1
HEAD = 11
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
input_ids = inputs["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
seq_len = input_ids.shape[1]

print(f"Sequence length: {seq_len}")
print(f"Tokens: {tokens}")


# ============ Hook QKV ============
qkv_store = {}

def hook_fn(module, inp, out):
    qkv_store['out'] = out.detach()

hook = model.transformer.h[LAYER].attn.c_attn.register_forward_hook(hook_fn)
with torch.no_grad():
    model(**inputs)
hook.remove()

# ============ Extract Q/K/V ============
qkv = qkv_store['out'][0]
n_embd = model.config.n_embd
n_head = model.config.n_head
head_dim = n_embd // n_head

q, k, v = qkv.split(n_embd, dim=-1)
q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

q_h, k_h, v_h = q[HEAD], k[HEAD], v[HEAD]

# ============ Attention matrix ============
scale = 1.0 / np.sqrt(head_dim)
scores = (q_h @ k_h.T) * scale
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float('-inf'))
p = torch.softmax(scores, dim=-1).detach().cpu().numpy()


# ============ Figure 1: attention matrix heatmap ============
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(p, xticklabels=tokens, yticklabels=tokens,
            cmap='viridis', ax=ax, cbar_kws={'label': 'attention'})
ax.set_xlabel("Key", fontsize=12)
ax.set_ylabel("Query", fontsize=12)
ax.set_title(f"Attention matrix: layer {LAYER}, head {HEAD}", fontsize=14)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
save1 = os.path.join(OUTPUT_DIR, f"head2_matrix_L{LAYER}_H{HEAD}.png")
plt.savefig(save1, dpi=150)
plt.show()
print(f"Saved: {save1}")


# ============ Figure 2: top-1 attended token per query ============
top1 = p.argmax(axis=1)
top1_names = [tokens[i] for i in top1]
top1_probs = p.max(axis=1)

fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.cm.viridis(top1_probs / top1_probs.max())
bars = ax.barh(range(seq_len), top1_probs, color=colors)
ax.set_yticks(range(seq_len))
ax.set_yticklabels([f"{i}: {t}" for i, t in enumerate(tokens)], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Max attention probability", fontsize=12)
ax.set_title(f"Top-1 attention target per query (L{LAYER} H{HEAD})", fontsize=14)

for i in range(seq_len):
    ax.text(top1_probs[i] + 0.01, i,
            f"-> {top1_names[i]}", va='center', fontsize=8)

plt.tight_layout()
save2 = os.path.join(OUTPUT_DIR, f"head2_top1_L{LAYER}_H{HEAD}.png")
plt.savefig(save2, dpi=150)
plt.show()
print(f"Saved: {save2}")


# ============ Figure 3: attention entropy and concentration ============
entropy = -np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: entropy curve
ax = axes[0]
ax.plot(range(seq_len), entropy, 'o-', color='steelblue')
ax.set_xlabel("Query position", fontsize=12)
ax.set_ylabel("Attention entropy", fontsize=12)
ax.set_title(f"Attention entropy per query (L{LAYER} H{HEAD})", fontsize=14)
ax.grid(alpha=0.3)

# Right: top-1 probability curve
ax = axes[1]
ax.plot(range(seq_len), top1_probs, 's-', color='coral')
ax.set_xlabel("Query position", fontsize=12)
ax.set_ylabel("Top-1 attention probability", fontsize=12)
ax.set_title(f"Max attention probability per query", fontsize=14)
ax.grid(alpha=0.3)

plt.tight_layout()
save3 = os.path.join(OUTPUT_DIR, f"head2_entropy_L{LAYER}_H{HEAD}.png")
plt.savefig(save3, dpi=150)
plt.show()
print(f"Saved: {save3}")


# ============ Text statistics ============
print("\n===== Top-1 target distribution =====")
cnt = Counter(top1_names)
for tok, c in cnt.most_common(15):
    print(f"  '{tok}': {c} times ({c/seq_len*100:.1f}%)")

print("\n===== Attention statistics =====")
print(f"  Mean entropy: {entropy.mean():.4f}")
print(f"  Mean top-1 prob: {top1_probs.mean():.4f}")
print(f"  Min top-1 prob: {top1_probs.min():.4f}")
print(f"  Max top-1 prob: {top1_probs.max():.4f}")


# ============ Comparison: high-PC1 head vs low-PC1 head ============
# Choose a low-concentration head as the comparison target
LOW_LAYER = 3
LOW_HEAD = 1

qkv_store2 = {}
def hook_fn2(module, inp, out):
    qkv_store2['out'] = out.detach()

hook2 = model.transformer.h[LOW_LAYER].attn.c_attn.register_forward_hook(hook_fn2)
with torch.no_grad():
    model(**inputs)
hook2.remove()

qkv2 = qkv_store2['out'][0]
q2, k2, v2 = qkv2.split(n_embd, dim=-1)
q2 = q2.view(seq_len, n_head, head_dim).transpose(0, 1)
k2 = k2.view(seq_len, n_head, head_dim).transpose(0, 1)
q_low, k_low = q2[LOW_HEAD], k2[LOW_HEAD]

scores2 = (q_low @ k_low.T) * scale
scores2 = scores2.masked_fill(mask == 0, float('-inf'))
p_low = torch.softmax(scores2, dim=-1).detach().cpu().numpy()

entropy_low = -np.sum(np.where(p_low > 0, p_low * np.log(p_low + 1e-12), 0), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: high-PC1 head
sns.heatmap(p, xticklabels=tokens, yticklabels=tokens,
            cmap='viridis', ax=axes[0], cbar_kws={'label': 'attention'})
axes[0].set_title(f"High PC1 head: L{LAYER} H{HEAD}", fontsize=12)
axes[0].set_xlabel("Key")
axes[0].set_ylabel("Query")
plt.sca(axes[0])
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)

# Right: low-PC1 head
sns.heatmap(p_low, xticklabels=tokens, yticklabels=tokens,
            cmap='viridis', ax=axes[1], cbar_kws={'label': 'attention'})
axes[1].set_title(f"Low PC1 head: L{LOW_LAYER} H{LOW_HEAD}", fontsize=12)
axes[1].set_xlabel("Key")
axes[1].set_ylabel("Query")
plt.sca(axes[1])
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)

plt.tight_layout()
save4 = os.path.join(OUTPUT_DIR, "head2_compare_high_low.png")
plt.savefig(save4, dpi=150)
plt.show()
print(f"Saved: {save4}")

print("\n===== Comparison =====")
print(f"High PC1 head (L{LAYER} H{HEAD}): mean entropy = {entropy.mean():.4f}")
print(f"Low PC1 head (L{LOW_LAYER} H{LOW_HEAD}): mean entropy = {entropy_low.mean():.4f}")
print(f"Entropy difference: {entropy_low.mean() - entropy.mean():.4f}")

print("\nDone.")