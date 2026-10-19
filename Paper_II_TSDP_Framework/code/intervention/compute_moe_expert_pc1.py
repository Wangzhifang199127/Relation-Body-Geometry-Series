import torch
import numpy as np
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ================= 1. Environment and path configuration =================
MODEL_PATH = "/root/autodl-tmp/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda"
OUTPUT_CSV = "data/raw/moe/expert_pc1_ratio.csv"

print("Loading MoE model in BFloat16 precision...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, attn_implementation="eager")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, config=config, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
).eval()

# ================= 2. Prepare long text input =================
# Use 2048-token text concatenated from 100 sentences to ensure a sufficiently rich router distribution
prompt = "The quick brown fox jumps over the lazy dog. The sun rises in the east because it is a new day. " * 100
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
inputs["input_ids"] = inputs["input_ids"][:, :2048]
seq_len = inputs["input_ids"].shape[1]
print(f"Sequence length: {seq_len} tokens")

target_layer = 12
num_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads 
head_dim = config.hidden_size // num_heads

# ================= 3. Extract V and Router Logits =================
v_store = {}
router_store = {}

def get_v_hook(layer_idx):
    def hook(module, input, output):
        # output shape: (1, seq_len, hidden_size)
        v_store[layer_idx] = output.detach().float().cpu()
    return hook

def get_router_hook(layer_idx):
    def hook(module, input, output):
        # gate output shape: (1, seq_len, num_experts) or (seq_len, num_experts)
        logits = output[0] if isinstance(output, tuple) else output
        router_store[layer_idx] = logits.detach().float().cpu()
    return hook

hooks = []
hooks.append(model.model.layers[target_layer].self_attn.v_proj.register_forward_hook(get_v_hook(target_layer)))
hooks.append(model.model.layers[target_layer].mlp.gate.register_forward_hook(get_router_hook(target_layer)))

with torch.no_grad():
    model(inputs["input_ids"])

for h in hooks: h.remove()

# ================= 4. Dimension repair and data extraction =================
V = v_store[target_layer]
Router_logits = router_store[target_layer]

# Remove batch dimension (1, seq_len, hidden) -> (seq_len, hidden)
if V.dim() == 3:
    V = V[0]
if Router_logits.dim() == 3:
    Router_logits = Router_logits[0]

# Find the most activated expert (Top-2 routing)
topk_vals, topk_indices = torch.topk(Router_logits, 2, dim=-1) # (seq_len, 2)
expert_counts = torch.bincount(topk_indices.flatten(), minlength=config.num_experts)
hottest_expert = torch.argmax(expert_counts).item()
print(f"\nHottest Expert ID: {hottest_expert}, routed {expert_counts[hottest_expert].item()} times.")

# Extract token indices routed to this expert
expert_mask = (topk_indices == hottest_expert).any(dim=-1) 
selected_indices = torch.where(expert_mask)[0]
print(f"Number of selected tokens for Expert {hottest_expert}: {len(selected_indices)}")

# ================= 5. Compute PC1 Ratio =================
def compute_pc1_ratio(matrix):
    """Compute the explained variance ratio of the first principal component of the matrix"""
    if matrix.shape[0] < 2:
        return 0.0
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    explained_variance = S ** 2
    pc1_ratio = (explained_variance[0] / explained_variance.sum()).item()
    return pc1_ratio

# Extract data
V_selected = V[selected_indices] # Data within the expert
V_selected_reshaped = V_selected.reshape(len(selected_indices), num_kv_heads, head_dim).transpose(0, 1)

V_global = V.reshape(seq_len, num_kv_heads, head_dim).transpose(0, 1) # Global data

expert_pc1_list = []
global_pc1_list = []

# Iterate over each KV head to compute average PC1
for h in range(num_kv_heads):
    expert_pc1_list.append(compute_pc1_ratio(V_selected_reshaped[h]))
    global_pc1_list.append(compute_pc1_ratio(V_global[h]))

mean_expert_pc1 = np.mean(expert_pc1_list)
mean_global_pc1 = np.mean(global_pc1_list)

print("\n=== PCA PC1 Ratio Results ===")
print(f"Expert {hottest_expert} Internal PC1 Ratio: {mean_expert_pc1:.4f}")
print(f"Global Dense PC1 Ratio: {mean_global_pc1:.4f}")

# ================= 6. Save data to CSV =================
os.makedirs("data/raw/moe", exist_ok=True)
df = pd.DataFrame({
    "structure": ["Global Dense", f"Expert {hottest_expert} Internal"],
    "pc1_ratio": [round(mean_global_pc1, 4), round(mean_expert_pc1, 4)]
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nData saved to {OUTPUT_CSV}")

# ================= 7. Conclusion output =================
if mean_expert_pc1 > mean_global_pc1:
    print("\nSUCCESS: Expert internal PC1 Ratio is HIGHER than Global.")
    print("This proves the Router divides the global Bush into localized Traditional Trees (Narrow Semantics)!")
else:
    print("\nWARNING: Expert internal PC1 is not higher. Check context.")