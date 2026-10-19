import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
})

os.makedirs('figures/T_series', exist_ok=True)

COLORS = {
    'c_axis': '#2E86AB',
    's_axis': '#A23B72',
    'chaos': '#6C757D',
    'qkv': '#3A7CA5',
    'mlp': '#D1495B',
    'router': '#2A9D8F',
    'frozen': '#4A4E69',
    'dynamic': '#E76F51',
    'highlight': '#FFD166',
}


# ==================== T-1: Relation Body Unit Circle ====================
def plot_T1_relation_circle():
    fig, ax = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='black', linewidth=1.8, zorder=2)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)

    ax.text(1.35, -0.10, r'\(c\) (Inner Product / Composition / Latitude)',
            color=COLORS['c_axis'], fontsize=11, ha='center', va='top')
    ax.text(-0.10, 1.35, r'\(s\) (Outer Product / Direction / Longitude)',
            color=COLORS['s_axis'], fontsize=11, ha='right', va='center', rotation=90)

    points = [
        (1.0, 0.0, r'\(c=1,\, s=0\)' + '\nFully Aligned', COLORS['c_axis'], 'right'),
        (0.0, 1.0, r'\(c=0,\, s=1\)' + '\nFully Orthogonal', COLORS['s_axis'], 'left'),
        (-1.0, 0.0, r'\(c=-1,\, s=0\)' + '\nFully Opposite', '#333333', 'left'),
        (0.707, 0.707, r'\(c=s=\frac{\sqrt{2}}{2}\)' + '\nPartially Aligned', '#666666', 'left'),
    ]
    for x, y, label, color, ha in points:
        ax.scatter([x], [y], s=80, color=color, zorder=5,
                   edgecolor='white', linewidth=1.5)
        offset_x = 0.15 if ha == 'left' else -0.15
        ax.text(x + offset_x, y + 0.06, label, fontsize=9, ha=ha, va='bottom', color=color)

    theta_ex = np.pi / 4
    ax.plot([0, np.cos(theta_ex)], [0, np.sin(theta_ex)],
            color=COLORS['highlight'], linewidth=2.5, zorder=3)
    ax.plot([np.cos(theta_ex), np.cos(theta_ex)], [0, np.sin(theta_ex)],
            color='gray', linewidth=1, linestyle=':', zorder=1)
    ax.plot([0, np.cos(theta_ex)], [np.sin(theta_ex), np.sin(theta_ex)],
            color='gray', linewidth=1, linestyle=':', zorder=1)

    ax.text(0.5, 0.5, r'\(c^2 + s^2 = 1\)', fontsize=16,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='black', linewidth=1.2, alpha=0.95),
            zorder=6)

    ax.text(0, -1.25, 'Relation Body Unit Circle (Local Tangent Space)',
            fontsize=11, ha='center', va='top', style='italic', color='#333333')

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig('figures/T_series/fig_T1_relation_circle.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()
    print("T-1 saved.")


# ==================== T-2: Multi-Head Independent Modules ====================
def plot_T2_multi_head_modules():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(7.5, 8.6, 'Multi-Head Independent Modules',
            fontsize=15, ha='center', fontweight='bold')
    ax.text(7.5, 8.2, 'Each Head is an Independent TSDP Module; '
            'Outputs are Concatenated along Feature Dimension',
            fontsize=11, ha='center', style='italic', color='#555555')

    ax.add_patch(mpatches.Rectangle((6.0, 7.3), 3.0, 0.55,
                                     facecolor='#F5F5F5', edgecolor='#333333', linewidth=1.5))
    ax.text(7.5, 7.575, 'Input X', fontsize=12,
            ha='center', va='center', fontweight='bold')

    head_colors = [('#2E86AB', 'Head 1'), ('#2A9D8F', 'Head 2'), ('#F18F01', 'Head 3')]
    head_positions = [2.0, 5.0, 8.0]

    for (color, name), x_pos in zip(head_colors, head_positions):
        ax.annotate('', xy=(x_pos + 1.0, 6.3), xytext=(7.5, 7.3),
                    arrowprops=dict(arrowstyle='->', color='#666666', linewidth=1.5))
        ax.text((7.5 + x_pos + 1.0)/2, 6.9, 'W_Q, W_K, W_V', fontsize=8,
                color='#666666', ha='center', style='italic')

        ax.add_patch(mpatches.Rectangle((x_pos, 4.0), 2.0, 2.3,
                                         facecolor=color, edgecolor='white',
                                         linewidth=2.0, alpha=0.85))
        ax.text(x_pos + 1.0, 5.95, name, fontsize=12, ha='center',
                va='center', color='white', fontweight='bold')
        ax.text(x_pos + 1.0, 5.30, 'Independent\nTSDP Module',
                fontsize=9, ha='center', va='center',
                color='white', style='italic')
        ax.text(x_pos + 1.0, 4.55, 'QK\(^T\) → Softmax → A·V',
                fontsize=8.5, ha='center', va='center', color='white')

        ax.annotate('', xy=(x_pos + 1.0, 3.0), xytext=(x_pos + 1.0, 3.95),
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=1.8))
        ax.text(x_pos + 1.0, 2.7, f'Output_{name[-1]}',
                fontsize=9, ha='center', color=color, fontweight='bold')

    concat_y = 1.9
    ax.add_patch(mpatches.Rectangle((1.5, concat_y), 9.0, 0.6,
                                     facecolor='#F0F0F0', edgecolor='#333333',
                                     linewidth=1.8, linestyle='--'))
    ax.text(6.0, concat_y + 0.3, 'Feature-Dimension Concatenation',
            fontsize=11, ha='center', va='center', fontweight='bold', color='#333333')

    ax.annotate('', xy=(6.0, 1.1), xytext=(6.0, 1.85),
                arrowprops=dict(arrowstyle='->', color='#333333', linewidth=2.2))

    ax.add_patch(mpatches.Rectangle((4.0, 0.3), 4.0, 0.7,
                                     facecolor='#E8E8E8', edgecolor='#333333', linewidth=1.8))
    ax.text(6.0, 0.65, 'W_O Projection + Residual Add-Back',
            fontsize=10, ha='center', va='center', fontweight='bold')

    ax.text(12.5, 5.0, 'Key Points',
            fontsize=12, fontweight='bold', color='#333333', ha='left')
    ax.text(12.5, 4.5,
            '• Each Head has independent\n   Q, K, V, and attention.',
            fontsize=9.5, ha='left', color='#555555', va='top')
    ax.text(12.5, 3.4,
            '• Heads run in parallel.',
            fontsize=9.5, ha='left', color='#555555', va='top')
    ax.text(12.5, 2.9,
            '• Outputs concatenated along\n   feature dimension.',
            fontsize=9.5, ha='left', color='#555555', va='top')
    ax.text(12.5, 1.8,
            '• W_O projects back to\n   residual stream.',
            fontsize=9.5, ha='left', color='#555555', va='top')

    plt.savefig('figures/T_series/fig_T2_multi_head_modules.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()
    print("T-2 saved.")


# ==================== T-3: U-shape Outline and Shoe-Foot Confrontation ====================
def plot_T3_u_shape_shoe_foot():
    fig, ax = plt.subplots(figsize=(11, 6))
    layers = np.arange(0, 13)

    u_shape_prior = 0.05 + 0.03 * np.exp(-layers/2) + 0.025 * np.exp(-(12-layers)/2.5)
    ax.plot(layers, u_shape_prior, 'o--', color=COLORS['chaos'],
            linewidth=2, markersize=7, label='Architecture Prior U-shape (Random)', zorder=3)

    u_shape_pretrained = u_shape_prior.copy()
    u_shape_pretrained[3:10] += 0.018
    ax.plot(layers, u_shape_pretrained, 'o-', color=COLORS['c_axis'],
            linewidth=2.5, markersize=8, label='Pretrained (Anti-gravity Ballooning)', zorder=4)

    ax.fill_between(layers[3:10], u_shape_prior[3:10], u_shape_pretrained[3:10],
                    color=COLORS['highlight'], alpha=0.3, zorder=2)

    ax.annotate('Pretraining "Ballooning"\n(Anti-gravity Widening Broad Semantics)',
                xy=(6.5, (u_shape_prior[6] + u_shape_pretrained[6])/2),
                xytext=(5.5, 0.088),
                fontsize=10, color='#D48806', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#D48806', linewidth=2))

    ax.annotate('Shoe: Architecture Constraint\n(Pole Convergence)',
                xy=(0.5, u_shape_prior[0]), xytext=(1.5, 0.090),
                fontsize=9, color=COLORS['chaos'],
                arrowprops=dict(arrowstyle='->', color=COLORS['chaos'], linewidth=1.5))
    ax.annotate('Shoe: Architecture Constraint\n(Deep Prediction Convergence)',
                xy=(11.5, u_shape_prior[11]), xytext=(9.5, 0.090),
                fontsize=9, color=COLORS['chaos'],
                arrowprops=dict(arrowstyle='->', color=COLORS['chaos'], linewidth=1.5))

    ax.axvspan(-0.5, 2.5, alpha=0.08, color='blue')
    ax.axvspan(2.5, 9.5, alpha=0.08, color='orange')
    ax.axvspan(9.5, 12.5, alpha=0.08, color='blue')
    ax.text(1, 0.098, 'Shallow\n(Lexical)', ha='center', fontsize=9, color='#333333')
    ax.text(6, 0.098, 'Middle\n(Semantic Expansion)', ha='center', fontsize=9, color='#333333')
    ax.text(11, 0.098, 'Deep\n(Prediction)', ha='center', fontsize=9, color='#333333')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Var(s) (Semantic Expansion)', fontsize=12)
    ax.set_title('U-shape Track and "Shoe-Foot Confrontation":\n'
                 'Architecture Prior vs Pretrained Anti-gravity',
                 fontsize=12, fontweight='bold', pad=20)

    ax.legend(loc='lower right', bbox_to_anchor=(1.0, -0.25),
              fontsize=9, framealpha=0.9, ncol=2)

    ax.grid(alpha=0.3)
    ax.set_xticks(layers)
    ax.set_ylim(0.045, 0.105)

    plt.tight_layout()
    plt.savefig('figures/T_series/fig_T3_u_shape_shoe_foot.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()
    print("T-3 saved.")


# ==================== T-4: Static-Dynamic Dichotomy ====================
def plot_T4_static_dynamic():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    np.random.seed(42)
    n_heads = 12
    angle_matrix = np.random.uniform(0.1, 0.3, (n_heads, n_heads))
    for block in [(0,4), (4,8), (8,12)]:
        for i in range(block[0], block[1]):
            for j in range(block[0], block[1]):
                if i != j:
                    angle_matrix[i, j] = np.random.uniform(0.5, 0.7)
    np.fill_diagonal(angle_matrix, 1.0)

    im1 = ax1.imshow(angle_matrix, cmap='coolwarm', vmin=0, vmax=0.8)
    ax1.set_title('Static: W_v (Longitude) Frozen\n'
                  'Base vs Chat Frobenius Norm = 0.0176',
                  fontsize=11, color=COLORS['frozen'], fontweight='bold')
    ax1.set_xlabel('Head'); ax1.set_ylabel('Head')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = axes[1]
    np.random.seed(1)
    A_base = np.random.uniform(0.05, 0.2, (n_heads, n_heads))
    A_chat = A_base.copy()
    for i in range(n_heads):
        for j in range(n_heads):
            if np.random.rand() > 0.7:
                A_chat[i, j] = np.random.uniform(0.0, 0.35)
    diff = A_base - A_chat
    im2 = ax2.imshow(diff, cmap='PiYG', vmin=-0.2, vmax=0.2)
    ax2.set_title('Dynamic: QK Routing (Latitude) Redrawn\n'
                  'KL Divergence = 0.2564',
                  fontsize=11, color=COLORS['dynamic'], fontweight='bold')
    ax2.set_xlabel('Key Token'); ax2.set_ylabel('Query Token')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    plt.suptitle('Static-Dynamic Dichotomy: W_v Frozen, QK Redrawn',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/T_series/fig_T4_static_dynamic.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()
    print("T-4 saved.")


# ==================== T-5: Tri-Space Dual-Projection Framework ====================
def plot_T5_tsdp_framework():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(8, 8.6, 'Tri-Space Dual-Projection (TSDP) Framework',
            fontsize=16, ha='center', fontweight='bold')
    ax.text(8, 8.2, 'Three Internal Relation Circles + Two Cross-Space Projections + Residual Integration',
            fontsize=11, ha='center', style='italic', color='#555555')

    ax.add_patch(mpatches.Rectangle((6.5, 7.3), 3.0, 0.55,
                                     facecolor='#F5F5F5', edgecolor='#333333', linewidth=1.5))
    ax.text(8, 7.575, 'Input X (Residual Stream)', fontsize=12,
            ha='center', va='center', fontweight='bold')

    for x_target, label in [(2.5, 'W_Q'), (8.0, 'W_K'), (13.5, 'W_V')]:
        ax.annotate('', xy=(x_target, 6.4), xytext=(8, 7.3),
                    arrowprops=dict(arrowstyle='->', color='#666666', linewidth=1.8))
        ax.text((8+x_target)/2 + 0.15, 6.9, label, fontsize=10,
                color='#666666', style='italic')

    def draw_space(center, radius, facecolor, edgecolor, label, sublabel, seed):
        cx, cy = center
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(cx + radius*np.cos(theta), cy + radius*np.sin(theta),
                color=edgecolor, linewidth=2, alpha=0.55)
        ax.fill(cx + radius*np.cos(theta), cy + radius*np.sin(theta),
                color=facecolor, alpha=0.4, zorder=1)
        r_inner = radius * 0.74
        ax.plot(cx + r_inner*np.cos(theta), cy + r_inner*np.sin(theta),
                color=edgecolor, linewidth=1.2, linestyle='--', alpha=0.7, zorder=2)
        np.random.seed(seed)
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pts = [(cx + r_inner*0.72*np.cos(a), cy + r_inner*0.72*np.sin(a)) for a in angles]
        for (px, py) in pts:
            ax.scatter([px], [py], s=50, color=edgecolor, zorder=4,
                       edgecolor='white', linewidth=1)
        for i in range(8):
            for j in range(i+1, 8):
                if abs(i-j) == 1 or abs(i-j) == 7:
                    ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                            color=edgecolor, linewidth=0.8, alpha=0.5, zorder=3)
        ax.text(cx, cy - radius - 0.55, label, fontsize=13, ha='center',
                fontweight='bold', color=edgecolor)
        ax.text(cx, cy - radius - 0.9, sublabel, fontsize=9,
                ha='center', color=edgecolor, style='italic')

    draw_space((2.5, 5.0), 1.15, '#D6E9F8', '#2E86AB',
               'Q Space', '(Q-Q Internal Relation Circle)', 42)
    draw_space((8.0, 5.0), 1.15, '#D5F0E3', '#2A9D8F',
               'K Space', '(K-K Internal Relation Circle)', 43)
    draw_space((13.5, 5.0), 1.15, '#FCE0C8', '#D35400',
               'V Space', '(V-V Internal Relation Circle)', 44)

    ax.annotate('', xy=(6.8, 5.0), xytext=(3.7, 5.0),
                arrowprops=dict(arrowstyle='<->', color='#8E44AD', linewidth=2.5))
    ax.text(5.25, 5.5, 'QK\(^T\)', fontsize=12, ha='center',
            fontweight='bold', color='#8E44AD')
    ax.text(5.25, 5.15, 'First Cross-Space (Absolute)', fontsize=9,
            ha='center', color='#8E44AD', style='italic')

    ax.add_patch(mpatches.Rectangle((9.2, 4.55), 2.6, 0.9,
                                     facecolor='#FFF3CD', edgecolor='#F18F01', linewidth=2))
    ax.text(10.5, 5.15, 'Softmax', fontsize=12, ha='center',
            fontweight='bold', color='#B8860B')
    ax.text(10.5, 4.75, 'Decouples Position & Intensity', fontsize=8.5,
            ha='center', color='#B8860B', style='italic')

    ax.annotate('', xy=(12.35, 5.0), xytext=(11.8, 5.0),
                arrowprops=dict(arrowstyle='->', color='#E76F51', linewidth=2.5))
    ax.text(12.05, 5.4, 'A·V', fontsize=12, ha='center',
            fontweight='bold', color='#E76F51')
    ax.text(10.5, 4.15, 'Second Cross-Space (Position-Aligned)', fontsize=9,
            ha='center', color='#E76F51', style='italic')

    head_data = [('#2E86AB', 'Head 1'), ('#2A9D8F', 'Head 2'), ('#F18F01', 'Head 3'),
                 ('#8E44AD', 'Head 4'), ('#E76F51', 'Head 5'), ('#D35400', 'Head H')]
    for i, (color, label) in enumerate(head_data):
        x_pos = 3.0 + i * 1.7
        ax.add_patch(mpatches.Rectangle((x_pos, 1.6), 1.4, 0.7,
                                         facecolor=color, edgecolor='white',
                                         linewidth=1.5, alpha=0.85))
        ax.text(x_pos + 0.7, 1.95, label, fontsize=9, ha='center',
                va='center', color='white', fontweight='bold')

    ax.text(8, 2.6, 'Multi-Head Independent Modules (Feature-Dimension Concatenation)',
            fontsize=10, ha='center', color='#333333', fontweight='bold')

    ax.annotate('', xy=(8, 1.15), xytext=(8, 1.55),
                arrowprops=dict(arrowstyle='->', color='#333333', linewidth=2))

    ax.add_patch(mpatches.Rectangle((6.5, 0.5), 3.0, 0.55,
                                     facecolor='#F5F5F5', edgecolor='#333333', linewidth=1.5))
    ax.text(8, 0.775, 'W_O Projection + Residual Add-Back',
            fontsize=10, ha='center', va='center', fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#D6E9F8', edgecolor='#2E86AB', label='Q Space (Internal Relation Circle)'),
        mpatches.Patch(facecolor='#D5F0E3', edgecolor='#2A9D8F', label='K Space (Internal Relation Circle)'),
        mpatches.Patch(facecolor='#FCE0C8', edgecolor='#D35400', label='V Space (Internal Relation Circle)'),
        mpatches.Patch(facecolor='#FFF3CD', edgecolor='#F18F01', label='Softmax (Position-Intensity Decoupling)'),
        mpatches.Patch(facecolor='#F5F5F5', edgecolor='#333333', label='Residual Stream Integration'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 0.0),
              fontsize=9, framealpha=0.9, ncol=2)

    plt.savefig('figures/T_series/fig_T5_tsdp_framework.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()
    print("T-5 saved.")


# ==================== T-6: Hierarchical State Progression ====================
def plot_T6_hierarchical_state_progression():
    fig, ax = plt.subplots(figsize=(18, 8.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    ax.text(9, 8.1, 'Hierarchical State Progression with Residual Integration',
            fontsize=15, ha='center', fontweight='bold')
    ax.text(9, 7.7, 'Each Layer is an Independent TSDP Module; '
            'Residual Stream Integrates Across Layers',
            fontsize=11, ha='center', style='italic', color='#555555')

    residual_y = 1.3
    ax.plot([0.5, 17.5], [residual_y, residual_y], color='#333333',
            linewidth=6, zorder=2, solid_capstyle='round')
    ax.text(0.3, residual_y - 0.55, 'Residual Stream\n(Integration Channel)',
            fontsize=11, color='#333333', fontweight='bold', va='top')

    n_layers_to_show = 5
    layer_indices = [0, 3, 6, 9, 11]
    x_positions = np.linspace(2.5, 15.5, n_layers_to_show)
    layer_states = ['Initial State', 'State 3', 'State 6', 'State 9', 'Final State']
    state_colors = ['#D6E9F8', '#B8DCF0', '#9ACEE8', '#7CBEE0', '#5EAED8']

    for i, (x_pos, layer_idx, state_label, color) in enumerate(
            zip(x_positions, layer_indices, layer_states, state_colors)):

        box_y = 3.3
        box_w = 2.4
        box_h = 2.5

        ax.add_patch(mpatches.Rectangle((x_pos - box_w/2, box_y), box_w, box_h,
                                         facecolor=color, edgecolor='#2E86AB',
                                         linewidth=1.8, alpha=0.85, zorder=3))

        ax.text(x_pos, box_y + box_h - 0.35, f'Layer {layer_idx}',
                fontsize=11, ha='center', fontweight='bold', color='#1A4C6E')

        ax.text(x_pos, box_y + box_h - 0.85, 'Q · K · V',
                fontsize=9, ha='center', color='#1A4C6E',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor='#2E86AB',
                          linewidth=0.8, alpha=0.8))

        ax.text(x_pos, box_y + 0.65, 'Independent\nTSDP Module',
                fontsize=8.5, ha='center', color='#1A4C6E', style='italic')

        ax.text(x_pos, box_y + box_h + 0.25, state_label,
                fontsize=10, ha='center', fontweight='bold', color='#333333')

        ax.annotate('', xy=(x_pos, box_y - 0.05), xytext=(x_pos, residual_y + 0.1),
                    arrowprops=dict(arrowstyle='->', color='#666666', linewidth=1.5))
        ax.annotate('', xy=(x_pos + 0.4, residual_y + 0.1), xytext=(x_pos + 0.4, box_y - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#E76F51', linewidth=1.5))

    for i in range(n_layers_to_show - 1):
        x1 = x_positions[i] + 1.25
        x2 = x_positions[i+1] - 1.25
        y_mid = 4.55
        ax.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle='->', color='#999999', linewidth=1.2,
                                    linestyle='dashed'))

    ax.text(9, 0.4,
            'Each state inherits all prior information via residual, but is an independent complete state',
            fontsize=10, ha='center', color='#333333', style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                      edgecolor='#CCCCCC', linewidth=1, alpha=0.95))

    plt.savefig('figures/T_series/fig_T6_hierarchical_state_progression.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()
    print("T-6 saved.")


# ==================== Batch Execution ====================
if __name__ == '__main__':
    print("Generating T-series figures (new numbering)...")
    plot_T1_relation_circle()
    plot_T2_multi_head_modules()
    plot_T3_u_shape_shoe_foot()
    plot_T4_static_dynamic()
    plot_T5_tsdp_framework()
    plot_T6_hierarchical_state_progression()
    print("\n✅ All 6 T-series figures generated in figures/T_series/")