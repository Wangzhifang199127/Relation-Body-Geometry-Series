import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"

print(f"[GPT-2 small] Loading model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME, attn_implementation="eager")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, torch_dtype=torch.float16).to(DEVICE).eval()

prompt = "The quick brown fox jumps over the lazy dog. The sun rises in the east because it"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
try:
    idx_it = tokens.index("Ġit")
    idx_dog = tokens.index("Ġdog")
except ValueError:
    idx_it, idx_dog = -2, -3

num_heads = config.n_head
head_dim = config.n_embd // num_heads

# ========== 1. U-shape curve ==========
print("[GPT-2 small] Extracting U-shape...")
v_store = {}
def get_v_hook(layer_idx):
    def hook(module, input, output):
        qkv = output[0] if isinstance(output, tuple) else output
        v_store[layer_idx] = qkv[..., 2*config.n_embd:].detach().float().cpu()
    return hook
hooks = [layer.attn.c_attn.register_forward_hook(get_v_hook(i)) 
         for i, layer in enumerate(model.transformer.h)]
with torch.no_grad():
    model(inputs["input_ids"])
for h in hooks: h.remove()

u_shape_data = []
for layer_idx in range(len(model.transformer.h)):
    v = v_store[layer_idx][0]
    v = v.reshape(v.shape[0], num_heads, head_dim).transpose(0, 1)
    var_s_list = []
    for h in range(num_heads):
        v_h = v[h]
        dot_v = torch.matmul(v_h, v_h.T)
        norm_v = torch.norm(v_h, dim=-1, keepdim=True)
        c_v = torch.clamp(dot_v / (norm_v * norm_v.T + 1e-8), -1.0, 1.0)
        s_v = torch.sqrt(1 - c_v**2 + 1e-8)
        var_s_list.append(s_v.var().item())
    u_shape_data.append({"layer": layer_idx, "s_variance": np.mean(var_s_list)})
pd.DataFrame(u_shape_data).to_csv("data/gpt2_small_u_shape.csv", index=False)
print(f"[GPT-2 small] U-shape saved: data/gpt2_small_u_shape.csv")

# ========== 2. QK asymmetry ==========
print("[GPT-2 small] Extracting QK asymmetry...")
with torch.no_grad():
    outputs = model(inputs["input_ids"], output_attentions=True)
mid_layer = len(model.transformer.h) // 2
attn = outputs.attentions[mid_layer][0].mean(dim=0).float().cpu().numpy()
qk_asym = np.abs(attn - attn.T).mean().item()
pd.DataFrame([{"model": "GPT-2 small", "qk_asym": qk_asym}]).to_csv("data/gpt2_small_qk_asym.csv", index=False)
print(f"[GPT-2 small] QK Asymmetry: {qk_asym:.6f}")

# ========== 3. W_v principal angles ==========
print("[GPT-2 small] Extracting W_v principal angles...")
W = model.transformer.h[mid_layer].attn.c_attn.weight[:, 2*config.n_embd:].detach().float().cpu()
W_v_reshaped = W.reshape(config.n_embd, num_heads, head_dim).permute(1, 0, 2)
bases = [torch.linalg.svd(W_v_reshaped[h], full_matrices=False)[0] for h in range(num_heads)]
angle_matrix = np.zeros((num_heads, num_heads))
for i in range(num_heads):
    for j in range(num_heads):
        if i == j: angle_matrix[i, j] = 1.0
        else:
            M = torch.matmul(bases[i].T, bases[j])
            angle_matrix[i, j] = torch.clamp(torch.linalg.svdvals(M).mean(), -1.0, 1.0).item()
wv_angle = np.abs(angle_matrix[~np.eye(num_heads, dtype=bool)]).mean()
pd.DataFrame([{"model": "GPT-2 small", "wv_angle": wv_angle}]).to_csv("data/gpt2_small_wv_angles.csv", index=False)
print(f"[GPT-2 small] W_v Mean |cos|: {wv_angle:.6f}")

# ========== 4. KL divergence curve (layer-wise intervention) ==========
print("[GPT-2 small] Extracting KL divergence curve...")
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
for layer_idx in range(len(model.transformer.h)):
    h = model.transformer.h[layer_idx].register_forward_hook(coordinate_replacement_hook)
    with torch.no_grad():
        out = model(inputs["input_ids"])
        probs_int = F.softmax(out.logits[0, -1, :].float(), dim=-1)
    h.remove()
    kl = F.kl_div(F.log_softmax(out.logits[0, -1, :].float(), dim=-1), probs_normal, reduction='sum').item()
    kl_curve.append({"layer": layer_idx, "kl_divergence": kl})
pd.DataFrame(kl_curve).to_csv("data/gpt2_small_kl_curve.csv", index=False)
print(f"[GPT-2 small] KL curve saved: data/gpt2_small_kl_curve.csv")
print("[GPT-2 small] Done.")