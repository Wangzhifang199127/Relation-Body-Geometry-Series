# ablation_v2.py

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============ Configuration ============
MODEL_NAME = "gpt2"
OUTPUT_DIR = "attention_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ Extended test sets ============
SYNTAX_TESTS = [
    # Subject-verb agreement - singular
    ("The cat ___ on the mat.", "is", "are"),
    ("The dog ___ in the garden.", "is", "are"),
    ("The book ___ on the table.", "is", "are"),
    ("The boy ___ to school.", "goes", "go"),
    ("The girl ___ a song.", "sings", "sing"),
    ("The bird ___ in the tree.", "sits", "sit"),
    ("The man ___ a doctor.", "is", "are"),
    ("The woman ___ a teacher.", "is", "are"),
    ("The child ___ happy.", "is", "are"),
    ("The flower ___ beautiful.", "is", "are"),
    # Subject-verb agreement - plural
    ("The cats ___ on the mat.", "are", "is"),
    ("The dogs ___ in the garden.", "are", "is"),
    ("The books ___ on the table.", "are", "is"),
    ("The boys ___ to school.", "go", "goes"),
    ("The girls ___ a song.", "sing", "sings"),
    ("The birds ___ in the tree.", "sit", "sits"),
    ("The men ___ doctors.", "are", "is"),
    ("The women ___ teachers.", "are", "is"),
    ("The children ___ happy.", "are", "is"),
    ("The flowers ___ beautiful.", "are", "is"),
    # Tense
    ("Yesterday, she ___ to the store.", "went", "goes"),
    ("Tomorrow, he ___ to school.", "will go", "went"),
    ("Last week, they ___ the movie.", "watched", "watch"),
    ("Now, I ___ a book.", "am reading", "read"),
    ("Last night, it ___ heavily.", "rained", "rains"),
    ("Next year, we ___ to Japan.", "will go", "went"),
    ("Yesterday, the cat ___ the fish.", "ate", "eats"),
    ("Tomorrow, she ___ the cake.", "will bake", "baked"),
    ("Last month, he ___ a new car.", "bought", "buys"),
    ("Now, they ___ dinner.", "are eating", "ate"),
    # Pronouns
    ("John said ___ was tired.", "he", "they"),
    ("Mary said ___ was happy.", "she", "he"),
    ("The dog wagged ___ tail.", "its", "their"),
    ("The children lost ___ toys.", "their", "its"),
    ("I saw the man ___ was tall.", "who", "which"),
    ("I saw the book ___ was red.", "which", "who"),
    ("She gave ___ a gift.", "him", "he"),
    ("They invited ___ to the party.", "us", "we"),
    ("The cat licked ___ paws.", "its", "their"),
    ("The students raised ___ hands.", "their", "its"),
    # Articles
    ("___ apple a day keeps the doctor away.", "An", "A"),
    ("I saw ___ elephant at the zoo.", "an", "a"),
    ("She is ___ honest person.", "an", "a"),
    ("He is ___ university student.", "a", "an"),
    ("___ sun rises in the east.", "The", "A"),
    ("___ moon is bright tonight.", "The", "A"),
    ("I need ___ umbrella.", "an", "a"),
    ("She bought ___ new car.", "a", "an"),
    ("___ earth is round.", "The", "A"),
    ("He ate ___ orange.", "an", "a"),
    # Prepositions
    ("The book is ___ the table.", "on", "in"),
    ("The cat is ___ the box.", "in", "on"),
    ("She walked ___ the park.", "through", "on"),
    ("He arrived ___ 5 o'clock.", "at", "in"),
    ("They live ___ London.", "in", "at"),
    ("The picture is ___ the wall.", "on", "in"),
    ("She is good ___ math.", "at", "in"),
    ("He is interested ___ science.", "in", "at"),
    ("The plane flew ___ the clouds.", "above", "on"),
    ("She sat ___ the chair.", "on", "in"),
]

SEMANTIC_TESTS = [
    ("The cat sat on the ___.", "mat", "sky"),
    ("The fish swims in the ___.", "water", "tree"),
    ("The sun rises in the ___.", "east", "west"),
    ("She drank a cup of ___.", "coffee", "bread"),
    ("The bird flew over the ___.", "house", "water"),
    ("He wrote a letter with a ___.", "pen", "shoe"),
    ("The baby cried for ___.", "milk", "stone"),
    ("They planted flowers in the ___.", "garden", "ocean"),
    ("The chef cooked a delicious ___.", "meal", "book"),
    ("She wore a beautiful ___.", "dress", "car"),
    ("The dog chased the ___.", "cat", "cloud"),
    ("He drove his ___ to work.", "car", "tree"),
    ("The student studied for the ___.", "exam", "beach"),
    ("She painted a picture of a ___.", "sunset", "stone"),
    ("The farmer planted ___ in the field.", "corn", "metal"),
    ("He read a ___ before bed.", "book", "stone"),
    ("The doctor examined the ___.", "patient", "planet"),
    ("She baked a ___ for the party.", "cake", "table"),
    ("The athlete ran around the ___.", "track", "soup"),
    ("He bought a new ___ for his computer.", "keyboard", "cloud"),
    ("The teacher wrote on the ___.", "blackboard", "ocean"),
    ("She drank a glass of ___.", "water", "sand"),
    ("The musician played the ___.", "piano", "cake"),
    ("He climbed the tall ___.", "mountain", "soup"),
    ("The bird built a ___ in the tree.", "nest", "house"),
    ("She wore a warm ___ in winter.", "coat", "fan"),
    ("The ship sailed across the ___.", "ocean", "desert"),
    ("He ate a slice of ___.", "pizza", "paper"),
    ("The photographer took a ___.", "photo", "sandwich"),
    ("She solved the math ___.", "problem", "soup"),
    ("The gardener watered the ___.", "flowers", "stones"),
    ("He fixed the broken ___.", "car", "cloud"),
    ("The librarian organized the ___.", "books", "snacks"),
    ("She caught the ___ in the net.", "fish", "cloud"),
    ("The painter used a ___.", "brush", "spoon"),
    ("He drove through the ___.", "tunnel", "sky"),
    ("The bee collected ___ from the flower.", "nectar", "sand"),
    ("She wrote a ___ to her friend.", "letter", "stone"),
    ("The firefighter climbed the ___.", "ladder", "pillow"),
    ("He played ___ with his friends.", "soccer", "soup"),
    ("The scientist conducted an ___.", "experiment", "umbrella"),
    ("She cut the ___ with a knife.", "bread", "cloud"),
    ("The pilot flew the ___.", "airplane", "sofa"),
    ("He washed his ___ in the sink.", "hands", "hats"),
    ("The elephant has a long ___.", "trunk", "tail"),
    ("She wore ___ on her feet.", "shoes", "gloves"),
    ("The moon shines at ___.", "night", "noon"),
    ("He ate breakfast in the ___.", "morning", "evening"),
    ("The snow falls in ___.", "winter", "summer"),
    ("She put the keys in her ___.", "pocket", "hair"),
    ("The train arrived at the ___.", "station", "kitchen"),
    ("He swam in the ___.", "pool", "tree"),
    ("The clock on the wall shows the ___.", "time", "food"),
    ("She opened the ___ to enter the room.", "door", "book"),
    ("The cat licked its ___.", "paws", "wings"),
    ("He drank soup with a ___.", "spoon", "fork"),
    ("The tree has many ___.", "leaves", "wheels"),
    ("She lit a ___ in the dark.", "candle", "stone"),
    ("The phone rang, so he ___ it.", "answered", "cooked"),
    ("The teacher gave ___ to the students.", "homework", "clouds"),
    ("She cooked rice in a ___.", "pot", "cup"),
]


# ============ Evaluation functions ============
def get_choice_logprob(context, choice, model, tokenizer):
    """Compute log P(choice | context)."""
    full_text = context.replace("___", choice)
    if "___" in context:
        prefix = context.split("___")[0]
        if prefix.endswith(" "):
            prefix = prefix[:-1]
    else:
        prefix = context

    prefix_ids = tokenizer(prefix, return_tensors="pt",
                           add_special_tokens=False)["input_ids"]
    prefix_len = prefix_ids.shape[1]

    full_ids = tokenizer(full_text, return_tensors="pt",
                         add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    log_prob = 0.0
    for pos in range(prefix_len, full_ids.shape[1]):
        if pos == 0:
            continue
        log_probs = torch.log_softmax(logits[0, pos - 1], dim=-1)
        token_id = full_ids[0, pos].item()
        log_prob += log_probs[token_id].item()

    return log_prob


def evaluate_logprob(tests, model, tokenizer):
    """For each test, compute log P(correct) - log P(wrong), and return the mean."""
    scores = []
    for context, correct, wrong in tests:
        score_c = get_choice_logprob(context, correct, model, tokenizer)
        score_w = get_choice_logprob(context, wrong, model, tokenizer)
        scores.append(score_c - score_w)
    return float(np.mean(scores))


# ============ Load model ============
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model.eval()

n_layer = model.config.n_layer
n_head = model.config.n_head
n_embd = model.config.n_embd
head_dim = n_embd // n_head


# ============ Ablation hooks (with tuple fix) ============
class MultiHeadAblationHook:
    """Ablate a specified set of heads, or the whole layer.
    GPT-2 attn forward returns a tuple: (attn_output, attn_weights).
    We need to modify attn_output and return the full tuple.
    """
    def __init__(self, layer, heads=None):
        self.layer = layer
        self.heads = heads  # None means ablate the whole layer
        self.enabled = False

    def __call__(self, module, inp, out):
        if not self.enabled:
            return out

        # Handle tuple output
        if isinstance(out, tuple):
            attn_output = out[0]
            rest = out[1:]
        else:
            attn_output = out
            rest = None

        attn_output = attn_output.clone()

        if self.heads is None:
            attn_output[:, :, :] = 0.0
        else:
            for h in self.heads:
                start = h * head_dim
                end = start + head_dim
                attn_output[:, :, start:end] = 0.0

        if rest is not None:
            return (attn_output,) + rest
        return attn_output


class PatchHook:
    """Activation patching: replace the output of a specified head with its mean across all tokens."""
    def __init__(self, layer, head):
        self.layer = layer
        self.head = head
        self.enabled = False

    def __call__(self, module, inp, out):
        if not self.enabled:
            return out

        if isinstance(out, tuple):
            attn_output = out[0]
            rest = out[1:]
        else:
            attn_output = out
            rest = None

        attn_output = attn_output.clone()
        start = self.head * head_dim
        end = start + head_dim
        mean_val = attn_output[:, :, start:end].mean(dim=1, keepdim=True)
        attn_output[:, :, start:end] = mean_val

        if rest is not None:
            return (attn_output,) + rest
        return attn_output


# ============ Register hooks ============
single_hooks = {}
for layer in range(n_layer):
    for head in range(n_head):
        hook = MultiHeadAblationHook(layer, [head])
        handle = model.transformer.h[layer].attn.register_forward_hook(hook)
        single_hooks[(layer, head)] = (hook, handle)

layer_hooks = {}
for layer in range(n_layer):
    hook = MultiHeadAblationHook(layer, None)
    handle = model.transformer.h[layer].attn.register_forward_hook(hook)
    layer_hooks[layer] = (hook, handle)

patch_hooks = {}
for layer in range(n_layer):
    for head in range(n_head):
        hook = PatchHook(layer, head)
        handle = model.transformer.h[layer].attn.register_forward_hook(hook)
        patch_hooks[(layer, head)] = (hook, handle)

print(f"Registered hooks: {len(single_hooks)} single + "
      f"{len(layer_hooks)} layer + {len(patch_hooks)} patch")


# ============ Baseline ============
print("\n===== Baseline =====")
base_syntax = evaluate_logprob(SYNTAX_TESTS, model, tokenizer)
base_semantic = evaluate_logprob(SEMANTIC_TESTS, model, tokenizer)
print(f"Syntax log-prob diff:   {base_syntax:+.4f}")
print(f"Semantic log-prob diff: {base_semantic:+.4f}")


# ============ Experiment A: layer ablation ============
print("\n===== Layer Ablation =====")
layer_results = []
for layer in range(n_layer):
    hook, _ = layer_hooks[layer]
    hook.enabled = True
    syn = evaluate_logprob(SYNTAX_TESTS, model, tokenizer)
    sem = evaluate_logprob(SEMANTIC_TESTS, model, tokenizer)
    hook.enabled = False

    layer_results.append({
        "layer": layer,
        "syntax": syn,
        "semantic": sem,
        "syntax_delta": syn - base_syntax,
        "semantic_delta": sem - base_semantic,
    })
    print(f"  Layer {layer:2d}: syntax Δ={syn - base_syntax:+.4f}, "
          f"semantic Δ={sem - base_semantic:+.4f}")

df_layer = pd.DataFrame(layer_results)
df_layer.to_csv(os.path.join(OUTPUT_DIR, "ablation_layer.csv"), index=False)


# ============ Experiment B: single-head ablation ============
REPRESENTATIVE = [
    (1, 11),   # high-PC1 structural head
    (3, 1),    # low-PC1 semantic head
    (5, 5),    # intermediate
    (11, 8),   # deep high-PC1
]

print("\n===== Single Head Ablation =====")
single_results = []
for (layer, head) in REPRESENTATIVE:
    hook, _ = single_hooks[(layer, head)]
    hook.enabled = True
    syn = evaluate_logprob(SYNTAX_TESTS, model, tokenizer)
    sem = evaluate_logprob(SEMANTIC_TESTS, model, tokenizer)
    hook.enabled = False

    single_results.append({
        "layer": layer,
        "head": head,
        "syntax": syn,
        "semantic": sem,
        "syntax_delta": syn - base_syntax,
        "semantic_delta": sem - base_semantic,
    })
    print(f"  L{layer}_H{head:2d}: syntax Δ={syn - base_syntax:+.4f}, "
          f"semantic Δ={sem - base_semantic:+.4f}")

df_single = pd.DataFrame(single_results)
df_single.to_csv(os.path.join(OUTPUT_DIR, "ablation_single.csv"), index=False)


# ============ Experiment C: activation patching ============
print("\n===== Activation Patching =====")
patch_results = []
for (layer, head) in REPRESENTATIVE:
    hook, _ = patch_hooks[(layer, head)]
    hook.enabled = True
    syn = evaluate_logprob(SYNTAX_TESTS, model, tokenizer)
    sem = evaluate_logprob(SEMANTIC_TESTS, model, tokenizer)
    hook.enabled = False

    patch_results.append({
        "layer": layer,
        "head": head,
        "syntax": syn,
        "semantic": sem,
        "syntax_delta": syn - base_syntax,
        "semantic_delta": sem - base_semantic,
    })
    print(f"  L{layer}_H{head:2d}: syntax Δ={syn - base_syntax:+.4f}, "
          f"semantic Δ={sem - base_semantic:+.4f}")

df_patch = pd.DataFrame(patch_results)
df_patch.to_csv(os.path.join(OUTPUT_DIR, "ablation_patch.csv"), index=False)


# ============ Remove hooks ============
for hook, handle in single_hooks.values():
    handle.remove()
for hook, handle in layer_hooks.values():
    handle.remove()
for hook, handle in patch_hooks.values():
    handle.remove()


# ============ Visualization ============
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Figure 1: layer ablation effect
ax = axes[0, 0]
ax.plot(df_layer['layer'], df_layer['syntax_delta'], 'o-',
        label='Syntax Δ', color='steelblue')
ax.plot(df_layer['layer'], df_layer['semantic_delta'], 's-',
        label='Semantic Δ', color='coral')
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("Log-prob delta")
ax.set_title("Layer ablation effect")
ax.legend()
ax.grid(alpha=0.3)

# Figure 2: single-head ablation
ax = axes[0, 1]
x = np.arange(len(REPRESENTATIVE))
labels = [f"L{l}_H{h}" for l, h in REPRESENTATIVE]
width = 0.35

ax.bar(x - width/2, df_single['syntax_delta'], width,
       label='Syntax Δ', color='steelblue')
ax.bar(x + width/2, df_single['semantic_delta'], width,
       label='Semantic Δ', color='coral')
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Log-prob delta")
ax.set_title("Single head ablation")
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Figure 3: activation patching
ax = axes[1, 0]
ax.bar(x - width/2, df_patch['syntax_delta'], width,
       label='Syntax Δ', color='steelblue')
ax.bar(x + width/2, df_patch['semantic_delta'], width,
       label='Semantic Δ', color='coral')
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Log-prob delta")
ax.set_title("Activation patching")
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Figure 4: layer ablation syntax vs semantic
ax = axes[1, 1]
ax.scatter(df_layer['syntax_delta'], df_layer['semantic_delta'],
           c=df_layer['layer'], cmap='viridis', s=80)
for _, row in df_layer.iterrows():
    ax.annotate(f"L{int(row['layer'])}",
                (row['syntax_delta'], row['semantic_delta']),
                fontsize=8)
ax.axhline(0, color='k', ls='--', lw=0.5)
ax.axvline(0, color='k', ls='--', lw=0.5)
ax.set_xlabel("Syntax Δ")
ax.set_ylabel("Semantic Δ")
ax.set_title("Layer ablation: syntax vs semantic")
ax.grid(alpha=0.3)
plt.colorbar(ax.collections[0], ax=ax, label='Layer')

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "ablation_v2_results.png")
plt.savefig(save_path, dpi=150)
plt.show()
print(f"\nSaved: {save_path}")


# ============ Summary ============
print("\n===== Summary =====")
print(f"Baseline: syntax={base_syntax:+.4f}, semantic={base_semantic:+.4f}")

print("\nLayer ablation (top 3 most affected):")
df_layer_sorted = df_layer.reindex(
    df_layer[['syntax_delta', 'semantic_delta']].abs().max(axis=1).sort_values(
        ascending=False).index)
for _, row in df_layer_sorted.head(3).iterrows():
    print(f"  L{int(row['layer'])}: syntax Δ={row['syntax_delta']:+.4f}, "
          f"semantic Δ={row['semantic_delta']:+.4f}")

print("\nSingle head ablation:")
for _, row in df_single.iterrows():
    print(f"  L{int(row['layer'])}_H{int(row['head'])}: "
          f"syntax Δ={row['syntax_delta']:+.4f}, "
          f"semantic Δ={row['semantic_delta']:+.4f}")

print("\nActivation patching:")
for _, row in df_patch.iterrows():
    print(f"  L{int(row['layer'])}_H{int(row['head'])}: "
          f"syntax Δ={row['syntax_delta']:+.4f}, "
          f"semantic Δ={row['semantic_delta']:+.4f}")

print("\nDone.")