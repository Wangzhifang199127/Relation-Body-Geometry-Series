# Code Directory: Paper II

This directory contains all the executable scripts for Paper II.

## 📂 Structure
- **`collect/`**: Scripts for extracting V-space activations, QK matrices, and hidden states from models.
- **`intervention/`**: ⭐️ Core causal intervention scripts.
  - `run_expA_qk_intervention.py`: Cuts specific QK routes to prove causal collapse.
  - `run_expB_multihead_shuffle.py`: Shuffles feature dimensions to test residual fallback.
  - `compute_moe_expert_pc1.py`: Slices MoE experts to verify narrow semantics (PC1 ratio).
- **`analysis/`**: Geometric metric calculations (e.g., attention entropy, temperature experiments).
- **`utils/`**: Shared utility modules (model loading, PyTorch hooks, geometric math).
- **`plotting/`**: Scripts to generate all figures (T-series, G-series).

## 🚀 Execution Example

To reproduce the QK routing intervention (Experiment A) on GPT-2:

```bash
python intervention/run_expA_qk_intervention.py
```
**Note: Running MoE scripts requires substantial VRAM. Ensure you have a GPU with at least 16GB (for 4-bit) or 96GB (for BF16).**