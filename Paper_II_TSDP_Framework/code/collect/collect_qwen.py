import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen1.5-0.5B"

# If already cached locally, you can replace MODEL_NAME with a local path, e.g.:
# MODEL_NAME = "/root/autodl-tmp/Qwen1.5-0.5B"

print(f"[Qwen] Loading model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True, attn_implementation="eager")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, config=config, torch_dtype=torch.float16,
    device_map={"": 0}, trust_remote_code=True
).eval()

prompt = "The quick brown fox jumps over the lazy dog. The sun rises in the east because it"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

# Find the token indices for "it" and "dog"
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
idx_it = None
idx_dog = None
for i, t in enumerate(tokens):
    if "it" in t.lower() and idx_it is None:
        idx_it = i
    if "dog" in t.lower() and idx_dog is None:
        idx_dog = i
if idx_it is None or idx_dog is None:
    print("[Qwen] Warning: 'it' or 'dog' not found, using last two positions as fallback.")
    idx_it = len(tokens) - 2
    idx_dog = len(tokens) - 3

num_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads
head_dim = config.hidden_size // num_heads

os.makedirs("data/raw/qwen", exist_ok=True)

# ========== 1. U-shape curve ==========
print("[Qwen] Extracting U-shape data...")
v_store = {}
def get_v_hook(layer_idx):
    def hook(module, input, output):
        v_store[layer_idx] = output.detach().float().cpu()
    return hook
hooks = [layer.self_attn.v_proj.register_forward_hook(get_v_hook(i))
         for i, layer in enumerate(model.model.layers)]
with torch.no_grad():
    model(inputs["input_ids"])
for h in hooks:
    h.remove()

u_shape_data = []
for layer_idx in range(len(model.model.layers)):
    v = v_store[layer_idx][0]  # (seq_len, num_kv_heads * head_dim)
    seq_len = v.shape[0]
    v = v.reshape(seq_len, num_kv_heads, head_dim).transpose(0, 1)  # (num_kv_heads, seq_len, head_dim)
    var_s_list = []
    for h in range(num_kv_heads):
        v_h = v[h]
        dot_v = torch.matmul(v_h, v_h.T)
        norm_v = torch.norm(v_h, dim=-1, keepdim=True)
        c_v = torch.clamp(dot_v / (norm_v * norm_v.T + 1e-8), -1.0, 1.0)
        s_v = torch.sqrt(1 - c_v**2 + 1e-8)
        var_s_list.append(s_v.var().item())
    u_shape_data.append({"layer": layer_idx, "s_variance": np.mean(var_s_list)})
pd.DataFrame(u_shape_data).to_csv("data/u_shape.csv", index=False)
print(f"[Qwen] U-shape saved: data/u_shape.csv ({len(u_shape_data)} layers)")

# ========== 2. QK asymmetry ==========
print("[Qwen] Extracting QK asymmetry...")
with torch.no_grad():
    outputs = model(inputs["input_ids"], output_attentions=True)
mid_layer = len(model.model.layers) // 2
attn = outputs.attentions[mid_layer][0].mean(dim=0).float().cpu().numpy()
qk_asym = np.abs(attn - attn.T).mean().item()
pd.DataFrame([{"model": "Qwen1.5-0.5B", "qk_asym": qk_asym}]).to_csv("data/qk_asym.csv", index=False)
print(f"[Qwen] QK Asymmetry: {qk_asym:.6f} -> data/qk_asym.csv")

# ========== 3. W_v principal angles ==========
print("[Qwen] Extracting W_v principal angles...")
W = model.model.layers[mid_layer].self_attn.v_proj.weight.detach().float().cpu()
W_v_reshaped = W.reshape(num_kv_heads, head_dim, config.hidden_size).permute(1, 0, 2)  # (head_dim, num_kv_heads, hidden_size) ? needs adjustment
# Correct handling: v_proj.weight shape is (num_kv_heads * head_dim, hidden_size)
# First reshape to (num_kv_heads, head_dim, hidden_size), then transpose to (num_kv_heads, hidden_size, head_dim)
W_v_reshaped = W.reshape(num_kv_heads, head_dim, config.hidden_size).permute(0, 2, 1)
bases = [torch.linalg.svd(W_v_reshaped[h], full_matrices=False)[0] for h in range(num_kv_heads)]
angle_matrix = np.zeros((num_kv_heads, num_kv_heads))
for i in range(num_kv_heads):
    for j in range(num_kv_heads):
        if i == j:
            angle_matrix[i, j] = 1.0
        else:
            M = torch.matmul(bases[i].T, bases[j])
            angle_matrix[i, j] = torch.clamp(torch.linalg.svdvals(M).mean(), -1.0, 1.0).item()
wv_angle = np.abs(angle_matrix[~np.eye(num_kv_heads, dtype=bool)]).mean()
pd.DataFrame([{"model": "Qwen1.5-0.5B", "wv_angle": wv_angle}]).to_csv("data/wv_angles.csv", index=False)
print(f"[Qwen] W_v Mean |cos|: {wv_angle:.6f} -> data/wv_angles.csv")

# ========== 4. KL divergence curve (layer-wise intervention) ==========
print("[Qwen] Extracting KL divergence curve...")
with torch.no_grad():
    logits_normal = model(inputs["input_ids"]).logits[0, -1, :].float()
    probs_normal = F.softmax(logits_normal, dim=-1)

def coordinate_replacement_hook(module, input, output):
    if isinstance(output, tuple):
        hidden_states = output[0]
        is_tuple = True
    else:
        hidden_states = output
        is_tuple = False
    if hidden_states.dim() == 3:
        hidden_states[:, idx_it, :] = hidden_states[:, idx_dog, :]
    return (hidden_states,) + output[1:] if is_tuple else hidden_states

kl_curve = []
for layer_idx in range(len(model.model.layers)):
    h = model.model.layers[layer_idx].register_forward_hook(coordinate_replacement_hook)
    with torch.no_grad():
        out = model(inputs["input_ids"])
        probs_int = F.softmax(out.logits[0, -1, :].float(), dim=-1)
    h.remove()
    kl = F.kl_div(F.log_softmax(out.logits[0, -1, :].float(), dim=-1), probs_normal, reduction='sum').item()
    kl_curve.append({"layer": layer_idx, "kl_divergence": kl})
pd.DataFrame(kl_curve).to_csv("data/kl_curve.csv", index=False)
print(f"[Qwen] KL curve saved: data/kl_curve.csv")
print("[Qwen] Done.")