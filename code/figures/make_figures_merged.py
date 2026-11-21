#!/usr/bin/env python
"""Assemble the merged Chem. Sci. figure set.

The data figures are generated in the two source packages:
  ../manuscript_cis/make_figures_cis.py
  ../manuscript_water/make_figures_water.py
This script copies the selected panels into `figures/` with merged numbering and
checks that every expected file exists. Run it after regenerating either source.
"""
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"

SOURCES = [
    ("manuscript_water/figures/fig1_pipeline", "fig1_six_type_pipeline"),
    ("manuscript_water/figures/fig2_prevalence", "fig2_water_prevalence"),
    ("manuscript_cis/figures/fig2_coverage", "fig3_five_type_coverage"),
    ("manuscript_water/figures/fig3_water_coverage", "fig4_water_coverage"),
    ("manuscript_cis/figures/fig5_architecture", "fig7_architecture"),
    ("manuscript_water/figures/fig6_heatmap", "fig5_heatmap"),
    ("manuscript_water/figures/fig5_case", "fig6_water_case"),
    ("manuscript_cis/figures/fig3_frontier", "figS1_efficiency"),
    ("manuscript_cis/figures/fig4_heterogeneity", "figS2_heterogeneity"),
]


def main():
    OUT.mkdir(exist_ok=True)
    missing = []
    for src_base, dst_base in SOURCES:
        for ext in (".png", ".pdf"):
            src = ROOT / (src_base + ext)
            dst = OUT / (dst_base + ext)
            if not src.exists():
                missing.append(str(src))
                continue
            shutil.copy2(src, dst)
    if missing:
        print("missing source files:")
        for m in missing:
            print(" ", m)
        return 2
    print(f"assembled {len(SOURCES)} figure pairs into {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
