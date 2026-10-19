import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google-bert/bert-base-uncased"

print(f"[BERT] Loading model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME, attn_implementation="eager")
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, config=config, torch_dtype=torch.float16).to(DEVICE).eval()

prompt = "The quick brown fox jumps over the lazy dog. The sun rises in the east because it"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

num_heads = config.num_attention_heads
head_dim = config.hidden_size // num_heads

# ========== 1. U-shape curve ==========
print("[BERT] Extracting U-shape data...")
v_store = {}
def get_v_hook(layer_idx):
    def hook(module, input, output):
        v_store[layer_idx] = output.detach().float().cpu()
    return hook
hooks = [layer.attention.self.value.register_forward_hook(get_v_hook(i)) 
         for i, layer in enumerate(model.bert.encoder.layer)]
with torch.no_grad():
    model(inputs["input_ids"])
for h in hooks: h.remove()

u_shape_data = []
for layer_idx in range(len(model.bert.encoder.layer)):
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

pd.DataFrame(u_shape_data).to_csv("data/bert_u_shape.csv", index=False)
print(f"[BERT] U-shape saved: data/bert_u_shape.csv ({len(u_shape_data)} layers)")

# ========== 2. QK asymmetry ==========
print("[BERT] Extracting QK asymmetry...")
with torch.no_grad():
    outputs = model(inputs["input_ids"], output_attentions=True)
mid_layer = len(model.bert.encoder.layer) // 2
attn = outputs.attentions[mid_layer][0].mean(dim=0).float().cpu().numpy()
qk_asym = np.abs(attn - attn.T).mean().item()
pd.DataFrame([{"model": "BERT-base", "qk_asym": qk_asym}]).to_csv("data/bert_qk_asym.csv", index=False)
print(f"[BERT] QK Asymmetry: {qk_asym:.6f} -> data/bert_qk_asym.csv")

# ========== 3. W_v principal angles ==========
print("[BERT] Extracting W_v principal angles...")
W = model.bert.encoder.layer[mid_layer].attention.self.value.weight.detach().float().cpu()
W_v_reshaped = W.reshape(config.hidden_size, num_heads, head_dim).permute(1, 0, 2)
bases = [torch.linalg.svd(W_v_reshaped[h], full_matrices=False)[0] for h in range(num_heads)]
angle_matrix = np.zeros((num_heads, num_heads))
for i in range(num_heads):
    for j in range(num_heads):
        if i == j: angle_matrix[i, j] = 1.0
        else:
            M = torch.matmul(bases[i].T, bases[j])
            angle_matrix[i, j] = torch.clamp(torch.linalg.svdvals(M).mean(), -1.0, 1.0).item()
wv_angle = np.abs(angle_matrix[~np.eye(num_heads, dtype=bool)]).mean()
pd.DataFrame([{"model": "BERT-base", "wv_angle": wv_angle}]).to_csv("data/bert_wv_angles.csv", index=False)
print(f"[BERT] W_v Mean |cos|: {wv_angle:.6f} -> data/bert_wv_angles.csv")
print("[BERT] Done.")