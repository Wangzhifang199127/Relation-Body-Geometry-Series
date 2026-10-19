import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"
MODEL_TAG = "gpt2_small"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
prompt = "The quick brown fox jumps over the lazy dog. The sun rises in the east because it"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

config = AutoConfig.from_pretrained(MODEL_NAME, attn_implementation="eager")
model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME, config=config, torch_dtype=torch.float16
).to(DEVICE).eval()

# ================= Extract Attention Weights =================
with torch.no_grad():
    outputs = model(inputs["input_ids"], output_attentions=True)

attentions = outputs.attentions  # tuple of (batch, num_heads, seq_len, seq_len)

results = []
for layer_idx, attn in enumerate(attentions):
    # attn shape: (1, num_heads, seq_len, seq_len)
    attn_np = attn[0].float().cpu().numpy()  # (num_heads, seq_len, seq_len)
    num_heads, seq_len, _ = attn_np.shape

    # Average entropy per layer (averaged over all heads and all query positions)
    eps = 1e-12
    entropy_per_head = -np.sum(attn_np * np.log(attn_np + eps), axis=-1)  # (num_heads, seq_len)
    mean_entropy = entropy_per_head.mean()

    # Average max probability per layer (measures concentration)
    max_prob_per_head = attn_np.max(axis=-1)  # (num_heads, seq_len)
    mean_max_prob = max_prob_per_head.mean()

    results.append({
        "layer": layer_idx,
        "mean_entropy": mean_entropy,
        "mean_max_prob": mean_max_prob
    })

df = pd.DataFrame(results)
os.makedirs(f"data/raw/{MODEL_TAG}", exist_ok=True)
df.to_csv(f"data/raw/{MODEL_TAG}/attention_entropy.csv", index=False)
print(df.to_string(index=False))