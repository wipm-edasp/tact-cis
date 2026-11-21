#!/usr/bin/env python
"""Generate the SI decision map (Figure S3)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# class, share of positives (%), total positives
pts = [
    ("H-bond", 18.7, 2828),
    ("hydrophobic", 43.7, 6602),
    ("π", 26.4, 3990),
    ("salt", 0.28, 42),
    ("halogen", 2.65, 400),
    ("water", 8.32, 1258),
]

fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.axvspan(0, 10, color="#F4D35E", alpha=0.18, lw=0)
ax.axhline(116, color="#B05A5A", ls="--", lw=1.2)
ax.axvline(10, color="#2E86AB", ls="--", lw=1.2)
for name, share, n in pts:
    ax.scatter(share, n, s=45, color="#2E86AB", zorder=3)
    dx = 0.6 if share < 30 else -6.5
    dy = 1.18 if name not in ("salt", "halogen") else 0.72
    ax.annotate(name, (share, n), xytext=(share + dx, n * dy),
                fontsize=8, color="#222222")
ax.set_xlim(0, 48)
ax.set_yscale("log")
ax.set_ylim(20, 12000)
ax.set_xlabel("class share of true positives (%)")
ax.set_ylabel("positive pairs in the dataset")
ax.set_title("Calibration-unit decision map (schematic)", fontsize=10)
ax.text(1.2, 7500, "share < α\nmarginal bound\ncan exclude", fontsize=7,
        color="#8A6D00")
ax.text(29, 150, "n < 116: per-type\nnot estimable", fontsize=7,
        color="#8A2B2B")
ax.text(13, 1150, "share ≥ α and n ≥ 116:\nglobal may suffice;\nscore alignment decides",
        fontsize=7, color="#1F5F7A")
fig.tight_layout()
fig.savefig(OUT / "figS3_decision_map.png", dpi=300)
fig.savefig(OUT / "figS3_decision_map.pdf")
plt.close(fig)
print("wrote", OUT / "figS3_decision_map.png")
