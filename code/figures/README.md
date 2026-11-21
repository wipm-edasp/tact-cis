# Figure scripts

| script | output |
|---|---|
| `make_figures_refined.py` | Figures 1–7 (`fig1_six_type_pipeline`, `fig2_water_prevalence`, `fig3_five_type_coverage`, `fig4_water_coverage`, `fig5_heatmap`, `fig6_water_case`, `fig7_architecture`) as PDF + PNG |
| `make_figures_merged.py` | Figs. S1–S2 (`figS1_efficiency`, `figS2_heterogeneity`) |
| `make_figures_si.py` | Fig. S3 (`figS3_decision_map`) |
| `renumber_figures.py` | renumbering helper used while merging the five-type and six-type drafts |
| `apply_fig_citations.py` | writes the in-text figure citation keys |

Run from the repository root with `PYTHONPATH=code/src`:

```bash
python code/figures/make_figures_refined.py
python code/figures/make_figures_merged.py
python code/figures/make_figures_si.py
```

Inputs are `results/summary/*.json` and `data/*.json`; outputs land next to the scripts unless `OUT` is edited at the top of each file. These scripts produced the Figures 1–7 and S1–S3 files that were converted to 600 dpi TIFF for the RSC submission; the rendered PDF/PNG are not tracked in this repository.
