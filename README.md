# tact-cis 鈥?code and data for *Chemistry decides the calibration unit*


Typed protein鈥搇igand interaction models return one probability per (atom, interaction type) pair, but the field reports a single aggregate coverage number for the whole set. This repository supports the argument that the **calibration unit is chemistry-dependent**: five direct contact classes (H-bond, hydrophobic, 蟺, salt, halogen) cannot share one conformal risk-control threshold, and water must be treated as a sixth, water-mediated class with its own unit.

## Contents

| path | content |
|---|---|
| `code/src/xact/` | Python package: PDB parsing, label construction, splits, model, cross-conformal calibration, conformal risk control, metrics |
| `code/scripts/` | run scripts for every split (5-type, 6-type water-blind, W1 water-aware, EGNN) and the PLIP / ProLIF audits |
| `data/` | water bridge labels, PDB accessions and release dates, 58-complex PLIP audit, HSP90/DHFR/VEGFR2/ER伪 network case, checksums |
| `results/summary/` | one JSON per split, from which Tables 1鈥? and SI Tables S1鈥揝11 are generated |


## Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
export PYTHONPATH=code/src       # Windows: set PYTHONPATH=code\src
```

Python 3.10+, PyTorch 2.x, RDKit 2023.9+. All experiments ran on one NVIDIA RTX PRO 6000 (96 GB) per shard, four shards.

## Data

The crystal dataset is 750 complexes / 29 targets, with per-atom labels H-bond 2828, hydrophobic 6602, 蟺 3990, salt 42, halogen 400, and 2282 water bridges in 541 complexes.

The two dataset pickles are ~75 MB and are **not** committed (`*.pkl` is ignored). Rebuild them from the raw PDB directory and `ligands.csv`:

```bash
python -m xact.build --csv /path/to/ligands.csv --pdb-dir /path/to/pdb \
    --out data/dataset_cod_water.pkl --include-water
python -m xact.build --csv /path/to/ligands.csv --pdb-dir /path/to/pdb \
    --out data/dataset_cod_water_w1.pkl --include-water
sha256sum -c data/dataset_cod_water.sha256
```

Expected digests

```text
2d9f51f4173e4bdf294565c254e009fd8defe5124a673c2e3cb62b679712a320  dataset_cod_water.pkl
7a3e9bd7211f7b460129d9a336d2b7793e4322c198b1ab98c46b4a6119ae9f4b  dataset_cod_water_w1.pkl
```

The builder now stores water coordinates in both builds, so a fresh `dataset_cod_water.pkl` follows the W1 layout; the legacy checksum is kept for the water-blind runs. Use `dataset_cod_water_w1.pkl` for the water-aware (W1) score. PDB structures and `ligands.csv` are external inputs and are not redistributed; the accession list is in `data/pdb_accessions.txt`.

## Reproducing the tables

`results/summary/` holds the compact summaries behind the manuscript:

| file | experiment |
|---|---|
| `cis_b2_{target,scaffold}_summary.json` | five-type, target- / scaffold-held-out |
| `cis_5t_temporal{,_strict}_summary.json` | five-type, temporal / strict target-disjoint temporal |
| `cis_water_{target,scaffold,temporal,temporal_strict}_summary.json` | six-type, water-blind score |
| `cis_water_w1_{target,scaffold}_summary.json`, `cis_water_w1_temporal{,_strict}_5s_summary.json` | six-type, W1 water-aware score (temporal variants: 5 seeds) |
| `cis_water_w1_egnn_{target,scaffold}_summary.json` | EGNN backbone, W1 |

Headline numbers reproduced from these files: five-type coverage 0.888 (target) / 0.905 (scaffold); water coverage 0.735/0.751/0.750/0.635 (water-blind), 0.848/0.877/0.886/0.787 (W1), 0.923/0.937/0.928/0.857 (per-type CRC), and 0.905/0.937/0.931/0.853 (global CRC on the probability score).

The shell scripts still contain the original cluster paths (`/home/xiao/workspace/tact`); edit `DATA`, `OUT` and `PYTHONPATH` at the top of each script for your machine. `code/scripts/rebuild_datasets.sh` documents the rebuild order.




`CITATION.cff` carries the same metadata and is rendered by GitHub's "Cite this repository" button.

## Licenses

- Code (`code/`): MIT, see `LICENSE`.
- Derived data and result summaries (`data/`, `results/`, `figures/`): CC-BY-4.0, see `LICENSE-DATA`.
- PDB structures are not redistributed; obtain them from RCSB under the PDB terms of use.

## Contact

Xiongjie Xiao (xiaoxj007@outlook.com),  鈥?Key Laboratory of Magnetic Resonance in Biological Systems, Innovation Academy for Precision Measurement of Science and Technology, Chinese Academy of Sciences.
