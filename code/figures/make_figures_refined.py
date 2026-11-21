# -*- coding: utf-8 -*-
"""Regenerate the merged figure set under a unified Chem. Sci. style.

Data are read from ../results (unchanged); only layout, palette, typography and
label placement are refined. Outputs PDF + PNG into ./figures, matching the
merged numbering used in manuscript.md.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Unified style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METHOD_COLORS = {
    "type": "#6FB3B8", "geo_partner": "#3B6FB6", "crc_global": "#E3B23C",
    "crc_geo_partner": "#5B8C5A", "top1": "#B05A5A",
}
METHOD_LABELS = {
    "type": "per-type", "geo_partner": "geo score", "crc_global": "CRC (p)",
    "crc_geo_partner": "CRC + geo", "top1": "top-1",
}
METHODS = ["type", "geo_partner", "crc_global", "crc_geo_partner", "top1"]

TYPES = ["H-bond", "hydrophobic", "pi", "salt", "halogen", "water"]
TYPE_COLORS = ["#3B6FB6", "#5B8C5A", "#9B59B6", "#B05A5A", "#E3B23C", "#2E86AB"]

BLUE = "#2E86AB"
RED = "#B05A5A"
GREEN = "#3A7D44"
INK = "#2C3E50"


def load(name):
    with open(BASE / name, encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def despine(ax, keep_left=True, keep_bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    if not keep_bottom:
        ax.spines["bottom"].set_visible(False)


def panel_tag(ax, tag):
    ax.text(-0.12, 1.03, f"({tag})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


# ----------------------------------------------------------------------------
# Figure 1 — six-type pipeline (flow diagram)
# ----------------------------------------------------------------------------
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 1.25, 1.95, 1.4, "ligand atom\n+ pocket"),
        (2.55, 1.25, 1.95, 1.4, "six types\n+ water"),
        (4.90, 1.25, 1.95, 1.4, "COD / EGNN\nmodel"),
        (7.25, 1.25, 1.95, 1.4, "per-type vs\nglobal CRC"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc="#EAF2F8", ec=INK, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
    for x in (2.15, 4.50, 6.85):
        ax.annotate("", xy=(x + 0.38, 1.95), xytext=(x, 1.95),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color=INK))
    ax.text(5.0, 0.55,
            "water bridge: L(polar)–W–P(polar), complementary, no direct contact",
            ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    save(fig, "fig1_six_type_pipeline")


# ----------------------------------------------------------------------------
# Figure 2 — water-bridge prevalence (family + target)
# ----------------------------------------------------------------------------
def fig2_prevalence():
    rows = load("water_bridge_labels.json")
    fam = defaultdict(list)
    tgt = defaultdict(list)
    for r in rows:
        frac = r["n_bridged_atoms"] / max(1, r["n_lig_atoms"])
        fam[r["family"]].append(frac)
        tgt[r["target"]].append(frac)
    fam_rate = {k: float(np.mean(v)) for k, v in fam.items()}
    tgt_rate = {k: float(np.mean(v)) for k, v in tgt.items()}
    top = sorted(tgt_rate.items(), key=lambda x: -x[1])[:10]
    bottom = sorted(tgt_rate.items(), key=lambda x: x[1])[:6]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    ax = axes[0]
    ax.bar(list(fam_rate), list(fam_rate.values()), color=BLUE,
           edgecolor="white", lw=0.6)
    ax.set_ylabel("water-bridged atom fraction")
    ax.tick_params(axis="x", rotation=30)
    despine(ax)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    panel_tag(ax, "a")

    ax = axes[1]
    names = [x[0] for x in top + bottom]
    vals = [x[1] for x in top + bottom]
    colors = [BLUE] * len(top) + [RED] * len(bottom)
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white", lw=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("water-bridged atom fraction")
    despine(ax)
    ax.grid(axis="x", color="#E6E6E6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    panel_tag(ax, "b")

    fig.tight_layout()
    save(fig, "fig2_water_prevalence")


# ----------------------------------------------------------------------------
# Figure 3 — five-type calibration coverage (2x2 bars with CI)
# ----------------------------------------------------------------------------
def fig3_coverage(data):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharey=True)
    tags = ["a", "b", "c", "d"]
    k = 0
    for row, alpha in enumerate(["0.1", "0.2"]):
        for col, split in enumerate(["target", "scaffold"]):
            ax = axes[row, col]
            vals, los, his = [], [], []
            for m in METHODS:
                e = data[split][alpha][m]
                vals.append(e["coverage_mean"])
                los.append(e["coverage_mean"] - e["coverage_ci"][0])
                his.append(e["coverage_ci"][1] - e["coverage_mean"])
            x = np.arange(len(METHODS))
            ax.bar(x, vals, yerr=[los, his], capsize=3,
                   color=[METHOD_COLORS[m] for m in METHODS],
                   edgecolor="black", linewidth=0.4, error_kw={"lw": 0.9})
            ax.axhline(1 - float(alpha), color="black", ls="--", lw=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS],
                               rotation=25, ha="right", fontsize=7.5)
            ax.set_ylim(0.74, 0.96)
            ax.set_title(f"{split}-held-out, FNR target {int(float(alpha) * 100)}%",
                         fontsize=9.5)
            if col == 0:
                ax.set_ylabel("Empirical coverage", fontsize=9)
            ax.tick_params(axis="y", labelsize=7.5)
            despine(ax)
            panel_tag(ax, tags[k])
            k += 1
    fig.tight_layout()
    save(fig, "fig3_five_type_coverage")


# ----------------------------------------------------------------------------
# Figure 4 — water coverage: per-type vs global CRC
# ----------------------------------------------------------------------------
def fig4_water_coverage():
    splits = [("target", "COD target"), ("scaffold", "COD scaffold"),
              ("temporal", "COD temporal"), ("temporal_strict", "COD strict"),
              ("egnn_target", "EGNN target"), ("egnn_scaffold", "EGNN scaffold")]
    w1 = {"target": "cis_water_w1_target_summary.json",
          "scaffold": "cis_water_w1_scaffold_summary.json",
          "temporal": "cis_water_w1_temporal_5s_summary.json",
          "temporal_strict": "cis_water_w1_temporal_strict_5s_summary.json"}
    per, glob, prob = [], [], []
    for key, _ in splits:
        d = load(w1.get(key, f"cis_water_{key}_summary.json"))["0.1"]
        per.append(d["crc_type"]["per_type_coverage"]["5"])
        glob.append(d["crc_geo_partner"]["per_type_coverage"]["5"])
        prob.append(d["crc_global"]["per_type_coverage"]["5"])
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.bar(x - 0.18, per, 0.36, label="per-type CRC", color=BLUE,
           edgecolor="white", lw=0.6)
    ax.bar(x + 0.18, glob, 0.36,
           label="global CRC + geo (COD W1; EGNN water-blind)",
           color=RED, edgecolor="white", lw=0.6)
    ax.plot(x, prob, "o-", color=GREEN, lw=1.4, ms=5,
            label="global CRC (probability)")
    ax.axhline(0.90, color="black", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in splits], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("water coverage")
    ax.set_ylim(0.5, 1.0)
    despine(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    save(fig, "fig4_water_coverage")


# ----------------------------------------------------------------------------
# Figure 5 — model architecture
# ----------------------------------------------------------------------------
def fig5_architecture():
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    facecolor=fc, edgecolor=ec, linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    blue, blue_e = "#D6E4F0", "#1A5276"
    green, green_e = "#EAF3EA", "#5B8C5A"
    box(0.15, 4.85, 2.25, 1.00, "Ligand atom\nfeatures (37)", blue, blue_e)
    box(2.75, 4.85, 2.25, 1.00, "GNN\n2 message-passing\nlayers $\\rightarrow h_i$", blue, blue_e)
    box(0.15, 2.55, 2.25, 1.00, "Pocket atom\nfeatures (24)", green, green_e)
    box(2.75, 2.55, 2.25, 1.00, "MLP\n2 layers $\\rightarrow p_j$", green, green_e)
    box(5.35, 3.15, 2.65, 1.65,
        "Pairwise\ncomplementarity\nMLP($[h_i, p_j,$\n$RBF(d_{ij})]) \\rightarrow z_{ij}$",
        "#FDF2DC", "#E3B23C", fs=6.4)
    box(5.35, 5.05, 2.65, 0.85, "Capability head\n$c_i$ (5)", blue, blue_e, fs=6.6)
    box(5.35, 1.15, 2.65, 0.85, "Opportunity head\n$o_j$ (5)", green, green_e, fs=6.6)
    box(8.35, 3.05, 2.95, 1.75,
        "$s_{ik} = c_{ik} + \\tau$\nlogsumexp$_j$\n$(o_{jk} + z_{ijk})/\\tau$",
        "#EDE7F6", "#6C3483", fs=6.6)
    box(11.65, 3.05, 2.75, 1.75, "Capability gate\n5 logits per\nligand atom",
        "#F2F2F2", "#333333", fs=6.6)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=1.0, color="#333333"))

    arrow(2.41, 5.35, 2.74, 5.35)
    arrow(2.41, 3.05, 2.74, 3.05)
    arrow(5.01, 5.15, 5.34, 4.35)
    arrow(5.01, 3.25, 5.34, 3.70)
    arrow(5.01, 5.55, 5.34, 5.45)
    arrow(5.01, 2.85, 5.34, 2.10)
    arrow(8.01, 3.95, 8.34, 3.95)
    arrow(8.01, 1.60, 8.34, 3.35)
    arrow(8.01, 5.45, 8.34, 4.25)
    arrow(11.31, 3.95, 11.64, 3.95)

    ax.add_patch(FancyBboxPatch((0.15, 0.15), 14.25, 0.70,
                               boxstyle="round,pad=0.03", facecolor="#FDF6E3",
                               edgecolor="#E6B800", lw=1.0))
    ax.text(0.35, 0.50,
            "Loss: class-weighted BCE on capable atoms + 0.1 $\\times$ opportunity "
            "perturbation\n(mask the true partner pocket atoms; require the interaction logit to drop)",
            fontsize=6.2, va="center")
    fig.tight_layout()
    save(fig, "fig7_architecture")


# ----------------------------------------------------------------------------
# Figure 6 — per-type water coverage heatmap
# ----------------------------------------------------------------------------
def fig6_heatmap():
    splits = [("target", "COD target"), ("scaffold", "COD scaffold"),
              ("temporal", "COD temporal"), ("temporal_strict", "COD strict"),
              ("egnn_target", "EGNN target"), ("egnn_scaffold", "EGNN scaffold")]
    w1 = {"target": "cis_water_w1_target_summary.json",
          "scaffold": "cis_water_w1_scaffold_summary.json",
          "temporal": "cis_water_w1_temporal_5s_summary.json",
          "temporal_strict": "cis_water_w1_temporal_strict_5s_summary.json"}
    mat = []
    for key, _ in splits:
        d = load(w1.get(key, f"cis_water_{key}_summary.json"))["0.1"]["crc_type"]
        mat.append([d["per_type_coverage"][str(k)] for k in range(6)])
    mat = np.asarray(mat)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    im = ax.imshow(mat, vmin=0.6, vmax=1.0, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(6))
    ax.set_xticklabels(TYPES, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([s[1] for s in splits], fontsize=8)
    for i in range(len(splits)):
        for j in range(6):
            v = mat[i, j]
            txt_color = "white" if v < 0.82 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=6.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, fraction=0.045, pad=0.03)
    cb.set_label("coverage", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    save(fig, "fig5_heatmap")


# ----------------------------------------------------------------------------
# Figure 7 — per-target conserved water-bridged residues
# ----------------------------------------------------------------------------
def fig7_case():
    d = load("water_network_case.json")
    targets = ["HSP90", "DHFR", "VEGFR2", "ERa"]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    for i, t in enumerate(targets):
        top = sorted(d[t]["top_residues"][:5], key=lambda r: r["fraction"])
        ys = i + np.linspace(-0.28, 0.28, len(top))
        for r, yy in zip(top, ys):
            ax.scatter(r["fraction"], yy, s=38, color=BLUE, edgecolor="white",
                       linewidth=0.5, zorder=3)
            ax.annotate(r["residue"], xy=(r["fraction"], yy), xytext=(1.03, yy),
                        fontsize=6.5, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="#999999"))
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=9)
    ax.set_xlabel("fraction of complexes with water-bridged residue")
    ax.set_xlim(0, 1.38)
    ax.invert_yaxis()
    despine(ax)
    fig.tight_layout()
    save(fig, "fig6_water_case")


# ----------------------------------------------------------------------------
# Figure S1 — coverage vs set-size frontier
# ----------------------------------------------------------------------------
def figS1_frontier(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True, sharex=True)
    for ax, split, tag in [(axes[0], "target", "a"), (axes[1], "scaffold", "b")]:
        for m in METHODS:
            e = data[split]["0.1"][m]
            ax.scatter(e["size_mean"], e["coverage_mean"], s=55,
                       color=METHOD_COLORS[m], edgecolor="black", linewidth=0.4, zorder=3)
        ax.axhline(0.9, color="black", ls="--", lw=0.9)
        ax.set_xlim(0.55, 0.9)
        ax.set_ylim(0.78, 0.96)
        ax.set_title(f"{split}-held-out, 90%", fontsize=9.5)
        ax.set_xlabel("Set size (types / atom)", fontsize=9)
        if split == "target":
            ax.set_ylabel("Coverage (FNR = 10%)", fontsize=9)
        ax.tick_params(labelsize=8)
        despine(ax)
        panel_tag(ax, tag)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=METHOD_COLORS[m],
                      label=METHOD_LABELS[m], markeredgecolor="black") for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "figS1_efficiency")


# ----------------------------------------------------------------------------
# Figure S2 — coverage heterogeneity (per-target + per-type)
# ----------------------------------------------------------------------------
def _per_target(split):
    agg = defaultdict(lambda: [0, 0])
    for k in range(4):
        p = BASE / split / f"shard_{k}.json"
        if not p.exists():
            continue
        for r in json.load(open(p, encoding="utf-8")):
            if r["method"] == "crc_geo_partner" and abs(r["alpha"] - 0.1) < 1e-9:
                for t, v in r.get("per_target", {}).items():
                    agg[t][0] += v["covered"]
                    agg[t][1] += v["n_true"]
    return {t: c / max(1, n) for t, (c, n) in agg.items()}


def figS2_heterogeneity(target_summary, scaffold_summary):
    pt = _per_target("cis_final_target")
    ps = _per_target("cis_final_scaffold")
    targets = sorted(pt, key=lambda t: 0.5 * (pt[t] + ps.get(t, 0)))
    mat = np.asarray([[pt[t], ps[t]] for t in targets])

    fig = plt.figure(figsize=(7.6, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0.6, vmax=1.0, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["target", "scaffold"], fontsize=8)
    ax.set_yticks(np.arange(len(targets)))
    ax.set_yticklabels(targets, fontsize=6.2)
    for i in range(len(targets)):
        for j in range(2):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.8, color="black" if v < 0.88 else "white")
    ax.set_title("a  Per-target coverage (global CRC + geo, 10%)", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("coverage", fontsize=7)

    ax2 = fig.add_subplot(gs[0, 1])
    type_keys = ["0", "1", "2", "3", "4"]
    type_labels = ["H-bond", "hydrophobic", "pi", "salt", "halogen"]
    grid = np.zeros((5, 2))
    for j, split in enumerate(["target", "scaffold"]):
        data = target_summary if split == "target" else scaffold_summary
        for k, t in enumerate(type_keys):
            grid[k, j] = data["0.1"]["crc_type"]["per_type_coverage"].get(t, np.nan)
    im2 = ax2.imshow(grid, cmap="YlOrRd", vmin=0.6, vmax=1.0, aspect="auto")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["target", "scaffold"], fontsize=8)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(type_labels, fontsize=7)
    for i in range(5):
        for j in range(2):
            v = grid[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=7, color="black" if v < 0.88 else "white")
    ax2.set_title("b  Per-type coverage (per-type CRC, 10%)", fontsize=9, loc="left")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.03)
    cb2.set_label("coverage", fontsize=7)

    fig.tight_layout()
    save(fig, "figS2_heterogeneity")


if __name__ == "__main__":
    data = {
        "target": load(r"cis_final_target\summary.json"),
        "scaffold": load(r"cis_final_scaffold\summary.json"),
    }
    fig1_pipeline()
    fig2_prevalence()
    fig3_coverage(data)
    fig4_water_coverage()
    fig5_architecture()
    fig6_heatmap()
    fig7_case()
    figS1_frontier(data)
    figS2_heterogeneity(data["target"], data["scaffold"])
    print("all 9 figures ->", OUT)
