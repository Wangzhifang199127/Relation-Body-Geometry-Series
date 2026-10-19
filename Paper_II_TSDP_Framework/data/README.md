
# Data Dictionary: Paper II

All raw CSV data is stored in the `raw/` directory, organized by model.

## 📊 Dictionary

### `gpt2_small/`
*   `u_shape.csv`: Layer-wise semantic expansion variance.
*   `qk_asym.csv`: Mean QK routing asymmetry.
*   `wv_angles.csv`: W_v subspace principal angles.
*   `kl_curve.csv`: Layer-wise KL divergence under causal intervention.
*   `attention_entropy.csv`: Average softmax entropy across layers.
*   `temperature_experiment.csv`: Entropy/asymmetry under different temperatures.
*   **`expA_qk_intervention.csv`**: Top-1 probability and KL divergence before/after cutting QK routes.
*   **`expB_multihead_shuffle.csv`**: Top-1 probability and KL divergence after shuffling features.

### `moe/`
*   `u_shape.csv`: Router-broken U-shape variance.
*   `router_entropy.csv`: Expert activation entropy.
*   **`expert_pc1_ratio.csv`**: PC1 ratio comparison (Global Dense vs. Expert Internal).

### `qwen/` & `bert/`
*   Standard `u_shape.csv`, `qk_asym.csv`, `wv_angles.csv`, `kl_curve.csv`.