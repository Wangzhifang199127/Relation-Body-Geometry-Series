import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Global Configuration =================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

os.makedirs('figures/G_series', exist_ok=True)

# ================= Model Name Mapping =================
MODEL_DISPLAY = {
    'bert': 'BERT-base',
    'gpt2_small': 'GPT-2 small',
    'gpt2_medium': 'GPT-2 medium',
    'qwen': 'Qwen1.5-0.5B',
    'moe': 'MoE (Qwen-MoE-A2.7B)'
}

# Unified color palette
PALETTE = {
    'BERT-base': '#A23B72',
    'GPT-2 small': '#2E86AB',
    'GPT-2 medium': '#1B7F79',
    'Qwen1.5-0.5B': '#F18F01',
    'MoE (Qwen-MoE-A2.7B)': '#8E44AD'
}

# ================= Load U-shape curves for all models =================
def load_all_u_shape():
    records = []
    raw_dir = 'data/raw'
    for model_key in MODEL_DISPLAY:
        model_dir = os.path.join(raw_dir, model_key)
        u_path = os.path.join(model_dir, 'u_shape.csv')
        if not os.path.exists(u_path):
            print(f"[Warning] Missing {u_path}, skipping.")
            continue
        df = pd.read_csv(u_path)
        df['model'] = MODEL_DISPLAY[model_key]
        records.append(df)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

# ================= Load QK asymmetry (excluding MoE) =================
def load_all_qk_asym():
    records = []
    raw_dir = 'data/raw'
    for model_key in ['bert', 'gpt2_small', 'gpt2_medium', 'qwen']:
        qk_path = os.path.join(raw_dir, model_key, 'qk_asym.csv')
        if not os.path.exists(qk_path):
            print(f"[Warning] Missing {qk_path}, skipping.")
            continue
        df = pd.read_csv(qk_path)
        records.append(df)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

# ================= Load W_v principal angles (excluding MoE) =================
def load_all_wv_angles():
    records = []
    raw_dir = 'data/raw'
    for model_key in ['bert', 'gpt2_small', 'gpt2_medium', 'qwen']:
        wv_path = os.path.join(raw_dir, model_key, 'wv_angles.csv')
        if not os.path.exists(wv_path):
            print(f"[Warning] Missing {wv_path}, skipping.")
            continue
        df = pd.read_csv(wv_path)
        records.append(df)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

# ================= G-1: Cross-model U-shape =================
def plot_G1():
    df = load_all_u_shape()
    if df.empty:
        print("[G-1] No U-shape data found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='layer', y='s_variance', hue='model',
                 marker='o', linewidth=2, markersize=6,
                 palette=PALETTE, ax=ax)

    # Annotate the specificity of MoE
    moe_data = df[df['model'] == MODEL_DISPLAY['moe']]
    if not moe_data.empty:
        moe_layer6 = moe_data[moe_data['layer'] == 6]['s_variance'].values
        if len(moe_layer6) > 0:
            ax.annotate('MoE Router breaks global U-shape',
                        xy=(6, moe_layer6[0]),
                        xytext=(10, 0.056),
                        fontsize=10, color=PALETTE['MoE (Qwen-MoE-A2.7B)'],
                        arrowprops=dict(arrowstyle='->', color=PALETTE['MoE (Qwen-MoE-A2.7B)'], linewidth=1.5))

    ax.set_title('Cross-Model U-shape Comparison (Dense vs MoE)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Var(s) (Semantic Expansion)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig('figures/G_series/fig_G1_cross_model_u_shape.pdf', format='pdf')
    plt.show()
    print("[G-1] Saved to figures/G_series/fig_G1_cross_model_u_shape.pdf")

# ================= G-2: Cross-model QK asymmetry =================
def plot_G2():
    df = load_all_qk_asym()
    if df.empty:
        print("[G-2] No QK asymmetry data found, skipping.")
        return

    # Ensure consistent order
    order = ['BERT-base', 'GPT-2 small', 'GPT-2 medium', 'Qwen1.5-0.5B']
    df['model'] = pd.Categorical(df['model'], categories=order, ordered=True)
    df = df.sort_values('model')

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x='model', y='qk_asym', palette=PALETTE, ax=ax)

    for i, row in enumerate(df.itertuples()):
        ax.text(i, row.qk_asym + 0.005, f"{row.qk_asym:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Cross-Model QK Routing Asymmetry (Attention Probability)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean |A_ij - A_ji|', fontsize=12)
    ax.set_ylim(0, df['qk_asym'].max() * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/G_series/fig_G2_cross_model_qk_asym.pdf', format='pdf')
    plt.show()
    print("[G-2] Saved to figures/G_series/fig_G2_cross_model_qk_asym.pdf")

# ================= G-3: Cross-model W_v principal angles =================
def plot_G3():
    df = load_all_wv_angles()
    if df.empty:
        print("[G-3] No W_v principal angle data found, skipping.")
        return

    order = ['BERT-base', 'GPT-2 small', 'GPT-2 medium', 'Qwen1.5-0.5B']
    df['model'] = pd.Categorical(df['model'], categories=order, ordered=True)
    df = df.sort_values('model')

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x='model', y='wv_angle', palette=PALETTE, ax=ax)

    for i, row in enumerate(df.itertuples()):
        ax.text(i, row.wv_angle + 0.01, f"{row.wv_angle:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Cross-Model W_v Principal Angles (Mean Cosine Similarity)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean |cos| (Subspace Overlap)', fontsize=12)
    ax.set_ylim(0, df['wv_angle'].max() * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/G_series/fig_G3_cross_model_wv_angles.pdf', format='pdf')
    plt.show()
    print("[G-3] Saved to figures/G_series/fig_G3_cross_model_wv_angles.pdf")

# ================= Main entry =================
if __name__ == '__main__':
    print("Generating G-series figures from data/raw/ ...")
    plot_G1()
    plot_G2()
    plot_G3()
    print("✅ G-series figures completed.")