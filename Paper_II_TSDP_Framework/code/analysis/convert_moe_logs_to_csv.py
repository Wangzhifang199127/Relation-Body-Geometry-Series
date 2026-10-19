import pandas as pd

# MoE data manually extracted from AutoDL terminal output logs (cannot be rerun locally)
# Source: exp8_moe_geometry_analysis.png and terminal print output

moe_u_shape = [
    {"layer": 0, "s_variance": 0.0553},
    {"layer": 1, "s_variance": 0.0528},
    {"layer": 2, "s_variance": 0.0576},
    {"layer": 3, "s_variance": 0.0596},
    {"layer": 4, "s_variance": 0.0601},
    {"layer": 5, "s_variance": 0.0620},
    {"layer": 6, "s_variance": 0.0618},
    {"layer": 7, "s_variance": 0.0510},
    {"layer": 8, "s_variance": 0.0520},
    {"layer": 9, "s_variance": 0.0519},
    {"layer": 10, "s_variance": 0.0520},
    {"layer": 11, "s_variance": 0.0517},
    {"layer": 12, "s_variance": 0.0526},
    {"layer": 13, "s_variance": 0.0524},
    {"layer": 14, "s_variance": 0.0523},
    {"layer": 15, "s_variance": 0.0526},
    {"layer": 16, "s_variance": 0.0525},
    {"layer": 17, "s_variance": 0.0526},
    {"layer": 18, "s_variance": 0.0529},
    {"layer": 19, "s_variance": 0.0528},
    {"layer": 20, "s_variance": 0.0523},
    {"layer": 21, "s_variance": 0.0518},
    {"layer": 22, "s_variance": 0.0517},
    {"layer": 23, "s_variance": 0.0510},
]
pd.DataFrame(moe_u_shape).to_csv("data/moe_u_shape.csv", index=False)
print("MoE U-shape saved (source: manual extraction from AutoDL terminal log)")

# Router entropy (extracted from exp8 terminal output)
router_entropy = [
    {"layer": 0, "entropy": 3.72}, {"layer": 1, "entropy": 3.66},
    {"layer": 2, "entropy": 3.60}, {"layer": 3, "entropy": 3.61},
    {"layer": 4, "entropy": 3.55}, {"layer": 5, "entropy": 3.18},
    {"layer": 6, "entropy": 3.17}, {"layer": 7, "entropy": 3.19},
    {"layer": 8, "entropy": 3.30}, {"layer": 9, "entropy": 3.40},
    {"layer": 10, "entropy": 3.24}, {"layer": 11, "entropy": 3.07},
    {"layer": 12, "entropy": 3.29}, {"layer": 13, "entropy": 2.88},
    {"layer": 14, "entropy": 3.13}, {"layer": 15, "entropy": 3.04},
    {"layer": 16, "entropy": 2.70}, {"layer": 17, "entropy": 2.81},
    {"layer": 18, "entropy": 2.97}, {"layer": 19, "entropy": 3.13},
    {"layer": 20, "entropy": 2.98}, {"layer": 21, "entropy": 3.18},
    {"layer": 22, "entropy": 3.19}, {"layer": 23, "entropy": 3.24},
]
pd.DataFrame(router_entropy).to_csv("data/moe_router_entropy.csv", index=False)
print("MoE Router entropy saved (source: manual extraction from AutoDL terminal log)")