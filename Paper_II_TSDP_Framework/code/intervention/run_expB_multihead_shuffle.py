import torch
import torch.nn.functional as F
import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig
from modelscope import snapshot_download

# 1. Prepare environment
print("Preparing GPT-2...")
model_dir = snapshot_download('AI-ModelScope/gpt2')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir, attn_implementation="eager")
model = GPT2LMHeadModel.from_pretrained(model_dir, config=config).to(DEVICE).eval()

prompt = "The quick brown fox jumps over the lazy dog. The sun rises in the east because it"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

# 2. Control group: normal prediction
with torch.no_grad():
    normal_logits = model(inputs["input_ids"]).logits[0, -1, :]
    normal_probs = F.softmax(normal_logits, dim=-1)

# 3. Intervention group: shuffle the input feature dimensions of W_O (c_proj) in layer 9
# That is, shuffle the result after multi-head concatenation
def shuffle_pre_hook(module, input):
    hidden_states = input[0]
    # Fix random seed to ensure reproducibility
    torch.manual_seed(42)
    # Generate a random permutation of feature dimensions
    shuffle_idx = torch.randperm(hidden_states.shape[-1]).to(hidden_states.device)
    # Shuffle the last dimension (hidden_size)
    shuffled_states = hidden_states[..., shuffle_idx]
    return (shuffled_states,)

# Mount Hook to c_proj (W_O) of layer 9
hook = model.transformer.h[9].attn.c_proj.register_forward_pre_hook(shuffle_pre_hook)

with torch.no_grad():
    int_logits = model(inputs["input_ids"]).logits[0, -1, :]
    int_probs = F.softmax(int_logits, dim=-1)

hook.remove()

# 4. Compute data
top5_n = torch.topk(normal_probs, 5)
top5_i = torch.topk(int_probs, 5)
kl = F.kl_div(F.log_softmax(int_logits, dim=-1), normal_probs, reduction='sum').item()

# 5. Print and save CSV
print("\n=== Experiment B: Multi-Head Shuffle Intervention ===")
print(f"Normal Top-1:    {tokenizer.decode(top5_n.indices[0])} ({top5_n.values[0]:.4f})")
print(f"Shuffled Top-1:  {tokenizer.decode(top5_i.indices[0])} ({top5_i.values[0]:.4f})")
print(f"KL Divergence:   {kl:.4f}")

# Automatically save as CSV
data = {
    "intervention": ["Normal", "Shuffle W_O Input (Layer 9)"],
    "top1_token": [tokenizer.decode(top5_n.indices[0]), tokenizer.decode(top5_i.indices[0])],
    "top1_prob": [round(top5_n.values[0].item(), 4), round(top5_i.values[0].item(), 4)],
    "kl_divergence": [0.0, round(kl, 4)]
}
df = pd.DataFrame(data)
os.makedirs("data/raw/gpt2_small", exist_ok=True)
csv_path = "data/raw/gpt2_small/expB_multihead_shuffle.csv"
df.to_csv(csv_path, index=False)
print(f"\nData automatically saved to {csv_path}")