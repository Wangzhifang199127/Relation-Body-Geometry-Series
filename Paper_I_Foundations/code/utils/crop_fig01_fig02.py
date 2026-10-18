# crop_fig01_fig02.py
"""
Crop fig01 (Relation Body) and fig02 (Scheme A/B/C) from a four-panel figure.
Input: original four-panel figure (must be saved as PNG first)
Output: figures/fig01_relation_body.png, figures/fig02_scheme_abc.png
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

# ============ Configuration ============
SOURCE_IMAGE = "figures/raw/four_panel.png"  # original four-panel figure
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ Load source image ============
img = Image.open(SOURCE_IMAGE)
w, h = img.size
print(f"Source image size: {w} x {h}")

# Four panels: top-left, top-right, bottom-left, bottom-right
half_w = w // 2
half_h = h // 2

# ============ Cropping ============
# Bottom-right: Relation Body figure
fig01 = img.crop((half_w, half_h, w, h))
fig01_path = os.path.join(OUTPUT_DIR, "fig01_relation_body.png")
fig01.save(fig01_path, dpi=(300, 300))
print(f"Saved: {fig01_path}")

# Top-left + top-right + bottom-left: Scheme A/B/C
# Combine into a single figure
fig02 = img.crop((0, 0, w, half_h))  # top half (A + B)
fig02_lower = img.crop((0, half_h, half_w, h))  # bottom-left (C)

# Combine using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(fig02.crop((0, 0, half_w, half_h)))
axes[0].axis('off')
axes[0].set_title("(a) Scheme A: Local redundancy", fontsize=12)

axes[1].imshow(fig02.crop((half_w, 0, w, half_h)))
axes[1].axis('off')
axes[1].set_title("(b) Scheme B: Global PC energy", fontsize=12)

axes[2].imshow(fig02_lower)
axes[2].axis('off')
axes[2].set_title("(c) Scheme C: Global residual", fontsize=12)

plt.tight_layout()
fig02_path = os.path.join(OUTPUT_DIR, "fig02_scheme_abc.png")
plt.savefig(fig02_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {fig02_path}")