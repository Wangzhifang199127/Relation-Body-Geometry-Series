import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig
from modelscope import snapshot_download

print("Preparing GPT-2...")
model_dir = snapshot_download('AI-ModelScope/gpt2')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

config = AutoConfig.from_pretrained(model_dir, attn_implementation="eager")
model = GPT2LMHeadModel.from_pretrained(model_dir, config=config).to(DEVICE).eval()

prompt = "The quick brown fox jumps over the lazy dog. The sun rises in the east because it"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
seq_len = inputs["input_ids"].shape[1]

idx_it = tokens.index("Ġit")
idx_sun = tokens.index("Ġsun")

print(f"Sequence length: {seq_len}, it index: {idx_it}, sun index: {idx_sun}")

# Control group: normal prediction
with torch.no_grad():
    normal_logits = model(inputs["input_ids"]).logits[0, -1, :]
    normal_probs = F.softmax(normal_logits, dim=-1)

# Intervention group: manually construct 4D attention_mask
# shape (batch, 1, seq_len, seq_len)
attn_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32).to(DEVICE)
# Set the attention score of 'it' towards 'sun' to a minimum value (equivalent to cutting the connection)
attn_mask[0, 0, idx_it, idx_sun] = torch.finfo(torch.float32).min

with torch.no_grad():
    int_logits = model(inputs["input_ids"], attention_mask=attn_mask).logits[0, -1, :]
    int_probs = F.softmax(int_logits, dim=-1)

top5_n = torch.topk(normal_probs, 5)
top5_i = torch.topk(int_probs, 5)
kl = F.kl_div(F.log_softmax(int_logits, dim=-1), normal_probs, reduction='sum').item()

print("\n=== Experiment A: QK Routing Intervention ===")
print(f"Normal Top-1:  {tokenizer.decode(top5_n.indices[0])} ({top5_n.values[0]:.4f})")
print(f"QK Cut Top-1:  {tokenizer.decode(top5_i.indices[0])} ({top5_i.values[0]:.4f})")
print(f"KL Divergence: {kl:.4f}")