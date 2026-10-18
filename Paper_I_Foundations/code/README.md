# Code Documentation (code/README.md)

This repository contains the complete implementation for the paper *From Pythagoras to Transformers: A Geometric Theory of Attention via the Relation Body*. All experiments are based on HuggingFace Transformers and can be fully reproduced on a single GPU.

## Directory Structure

```
code/
├── README.md                     # This file
├── analysis/                     # Core analysis scripts
│   ├── batch_scan.py
│   ├── local_pca_analysis.py
│   ├── cross_model_analysis.py
│   ├── head11_deep_analysis.py
│   ├── layer11_analysis.py
│   ├── ablation_v2.py
│   ├── cumulative_ablation.py
│   ├── domain_width_experiment.py
│   └── verify_A1.py              # Empirical verification of Assumption A1
├── utils/                        # Utility modules and helper scripts
│   ├── hooks.py                  # QKV extraction, ablation, patching
│   ├── metrics.py                # Schemes A/B/C, local PCA, entropy
│   ├── visualization.py          # Unified plotting style and functions
│   ├── plot_figA1.py             # Figure A1: GPT-2 medium PC1 heatmap
│   ├── plot_figA2.py             # Figure A2: rA distribution
│   ├── plot_figA3.py             # Figure A3: SVD spectrum vs MP
│   ├── plot_figA4.py             # Figure A4: multi-seed PC1 robustness
│   ├── crop_fig01_fig02.py       # Crop Figure 1 and Figure 2
└── reproduce/                    # One-click reproduction scripts
    └── run_all.sh
```

## Environment Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 |
| transformers | ≥ 4.40 |
| numpy | ≥ 1.26 |
| pandas | ≥ 2.2 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.13 |
| scipy | ≥ 1.12 |

**Hardware**: A single NVIDIA T4 (16 GB) or higher. All experiments can be completed within 1 hour.

## Installation

The `requirements.txt` file is located in the **project root directory**, one level above this `code/` folder.

From the project root, run:

```bash
pip install -r requirements.txt
```

If you are currently inside the `code/` directory, use:

```bash
pip install -r ../requirements.txt
```

## Quick Start

### Reproduce all experiments

From the project root:

```bash
bash code/reproduce/run_all.sh
```

### Run a single script

From the project root:

```bash
python code/analysis/batch_scan.py
```

### Use utility modules

```python
from utils.hooks import register_qkv_hooks, remove_hooks
from utils.metrics import compute_all_metrics
from utils.visualization import plot_pc1_heatmap, save_figure
```

## Script Details

### Analysis Scripts (analysis/)

#### batch_scan.py

- **Purpose**: Full layer-head scan, computing Schemes A/B/C correlation coefficients.
- **Input**: GPT-2 small + one English paragraph.
- **Output**: `data/raw/batch_scan_results.csv`
- **Runtime**: ~30 seconds.

#### local_pca_analysis.py

- **Purpose**: Compute local PCA concentration and its correlation with entropy for each head.
- **Input**: GPT-2 small + English text.
- **Output**: `data/raw/local_pca_results.csv`
- **Runtime**: ~1 minute.

#### cross_model_analysis.py

- **Purpose**: Repeat the analysis on GPT-2 small and medium.
- **Input**: Two models + English text.
- **Output**:
  - `data/raw/cross_model_results.csv`
  - `figures/appendix/figA5_cross_model_trend.png` (cross-model layer trend)
- **Runtime**: ~3 minutes.

#### head11_deep_analysis.py

- **Purpose**: Specialized analysis of a high-PC1 head (L1 H11): attention matrix, top-1 distribution, entropy curve.
- **Input**: GPT-2 small + English text.
- **Output**: `figures/fig03_high_low_pc1.png`
- **Runtime**: ~30 seconds.

#### layer11_analysis.py

- **Purpose**: Per-head ablation and geometric feature analysis of Layer 11.
- **Input**: GPT-2 small + semantic test set.
- **Output**:
  - `data/raw/layer11_head_ablation.csv`
  - `data/raw/layer11_head_features.csv`
  - `data/raw/layer_compare_features.csv`
  - `figures/fig04_layer11_head7.png`
  - `figures/appendix/figA7_layer0_vs_layer11.png` (Layer 0 vs Layer 11 scatter plot)
- **Runtime**: ~2 minutes.

#### ablation_v2.py

- **Purpose**: Single-head ablation, layer ablation, activation patching.
- **Input**: GPT-2 small + 60 syntactic + 60 semantic test questions.
- **Output**:
  - `data/raw/ablation_layer.csv`
  - `data/raw/ablation_single.csv`
  - `data/raw/ablation_patch.csv`
- **Runtime**: ~5 minutes.

#### cumulative_ablation.py

- **Purpose**: Cumulative ablation, finding the phase transition point.
- **Input**: GPT-2 small + test set.
- **Output**:
  - `data/raw/cumulative_ablation.csv`
  - `figures/fig05_cumulative_ablation.png`
- **Runtime**: ~3 minutes.

#### domain_width_experiment.py

- **Purpose**: Controlled-domain experiments (repeat/single/double/multi).
- **Input**: GPT-2 small + four semantic-width texts.
- **Output**:
  - `data/raw/domain_width_results.csv`
  - `figures/fig06_domain_width.png`
- **Runtime**: ~2 minutes.

#### verify_A1.py

- **Purpose**: Empirical verification of Assumption A1 (QKV alignment). Computes Pearson correlation between \(q_i \cdot k_j\) and \(v_i \cdot v_j\) across all heads of GPT-2 small.
- **Input**: GPT-2 small + English text.
- **Output**:
  - `data/raw/A1_verification.csv`
  - `figures/appendix/figA6_a1_verification.png`
  - Table A1 statistics printed to terminal.
- **Runtime**: ~1 minute.

### Utility Modules (utils/)

#### hooks.py

- **Function**: Encapsulates QKV extraction, ablation, and activation patching hooks.
- **Main classes**:
  - `QKVHook`: Extract Q/K/V from `c_attn` output.
  - `AttentionOutputHook`: Ablate specified heads or the whole layer.
  - `PatchHook`: Replace a specified head's output with its mean.
- **Main functions**:
  - `register_qkv_hooks(model, n_layer)`: Register QKV hooks for all layers.
  - `remove_hooks(handles)`: Remove all hooks.

#### metrics.py

- **Function**: Encapsulates relation body coordinates, Schemes A/B/C, local PCA, entropy, etc.
- **Main functions**:
  - `relation_coords(a, b)`: Compute relation body coordinates \((c, s)\).
  - `wedge_norm_squared(a, b)`: High-dimensional wedge norm squared.
  - `compute_attention(q_h, k_h, head_dim, causal=True)`: Attention probabilities.
  - `attention_entropy(p)`: Attention entropy.
  - `compute_scheme_A(p, v_h)`: Scheme A correlation.
  - `compute_scheme_BC(p, v_h, top_k=5)`: Scheme B/C correlation.
  - `compute_local_pca(p, v_h, top_k=10)`: Local PC1 concentration.
  - `compute_all_metrics(...)`: Compute all metrics at once.

#### visualization.py

- **Function**: Unified plotting style for the paper.
- **Main functions**:
  - `setup_style()`: Set global fonts, sizes, and colors.
  - `save_figure(fig, path)`: Save a figure.
  - `plot_pc1_heatmap(...)`: PC1 heatmap.
  - `plot_scheme_scatter(...)`: Scheme A/B/C scatter plots.
  - `plot_relation_body(...)`: Relation body unit circle.
  - `plot_layer_trend(...)`: Layer trend curves.
  - `plot_ra_histogram(...)`: \(r_A\) distribution histogram.
  - `plot_cumulative_ablation(...)`: Cumulative ablation curves.
  - `plot_domain_width(...)`: Controlled-domain bar chart.

#### plot_figA1.py

- **Purpose**: Generate GPT-2 medium PC1 heatmap.
- **Input**: `data/raw/cross_model_results.csv`
- **Output**: `figures/appendix/figA1_gpt2_medium_pc1.png`

#### plot_figA2.py

- **Purpose**: Generate \(r_A\) distribution histogram.
- **Input**: `data/raw/cross_model_results.csv`
- **Output**: `figures/appendix/figA2_ra_distribution.png`

#### plot_figA3.py

- **Purpose**: Generate SVD singular value spectrum vs. Marchenko-Pastur curve.
- **Input**: QKV extracted from GPT-2 in real time.
- **Output**: `figures/appendix/figA3_svd_spectrum.png`

#### plot_figA4.py

- **Purpose**: Generate multi-seed PC1 robustness check.
- **Input**: 5 different texts.
- **Output**:
  - `data/raw/multi_seed_results.csv`
  - `figures/appendix/figA4_multi_seed.png`

#### crop_fig01_fig02.py

- **Purpose**: Crop Figure 1 and Figure 2 from a four-panel figure.
- **Input**: `figures/raw/four_panel.png`
- **Output**:
  - `figures/fig01_relation_body.png`
  - `figures/fig02_scheme_abc.png`

### Reproduction (reproduce/)

#### run_all.sh

```bash
#!/bin/bash
set -e

# Move to project root (parent directory of this script's directory)
cd "$(dirname "$0")/.."

echo "=== Running all experiments ==="

python code/analysis/batch_scan.py
python code/analysis/local_pca_analysis.py
python code/analysis/cross_model_analysis.py
python code/analysis/head11_deep_analysis.py
python code/analysis/layer11_analysis.py
python code/analysis/ablation_v2.py
python code/analysis/cumulative_ablation.py
python code/analysis/domain_width_experiment.py
python code/analysis/verify_A1.py

echo "=== Generating appendix figures ==="

python code/utils/plot_figA1.py
python code/utils/plot_figA2.py
python code/utils/plot_figA3.py
python code/utils/plot_figA4.py
python code/utils/crop_fig01_fig02.py

echo "=== All done ==="
```

## Data Dependencies

All scripts depend on CSV files under `data/raw/`. See `data/README.md` for field descriptions and sources.

## Notes

1. **Model download**: GPT-2 is downloaded automatically on first run. Model cache is at `~/.cache/huggingface/`.
2. **GPU**: GPU is recommended but CPU works as well (slower).
3. **Randomness**: All experiments use `model.eval()`; no training is involved, so there are no random seed issues.
4. **Paths**: Scripts use relative paths. Please run from the project root, or use `bash code/reproduce/run_all.sh`.
5. **Data overwrite**: Re-running scripts will overwrite CSV files under `data/raw/`.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{anonymous2026relation,
  title={From Pythagoras to Transformers: A Geometric Theory of Attention via the Relation Body},
  author={Anonymous},
  year={2026}
}
```