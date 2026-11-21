# Data included in this snapshot

All files here are derived, small enough for GitHub, and licensed CC-BY-4.0
except PDB structures, which are not redistributed.

| file | content |
|---|---|
| `pdb_accessions.txt` | 669 PDB accessions used for release dates / splits |
| `pdb_release_dates.json` | PDB release date per accession |
| `water_bridge_labels.json` | per-complex water-bridged atom indices and counts |
| `water_direct_overlap.json` | 49% direct/water overlap counts |
| `water_network_case.json` | HSP90/DHFR/VEGFR2/ERa residue recurrence |
| `plip_water_audit_60.json` | 58-complex PLIP water-bridge audit |
| `dataset_cod_water.sha256` | SHA-256 for the two dataset pickles |

## Dataset pickles not committed

`dataset_cod_water.pkl` (~73 MB) and `dataset_cod_water_w1.pkl` (~75 MB) are
omitted to keep the repository small. Rebuild them with `xact.build`; verify:

```text
2d9f51f4173e4bdf294565c254e009fd8defe5124a673c2e3cb62b679712a320  dataset_cod_water.pkl
7a3e9bd7211f7b460129d9a336d2b7793e4322c198b1ab98c46b4a6119ae9f4b  dataset_cod_water_w1.pkl
```

The current builder stores water coordinates in both builds; the legacy
`dataset_cod_water.pkl` checksum is kept for the water-blind runs, while W1
uses `dataset_cod_water_w1.pkl`.

Raw PDB files and `ligands.csv` are from the parent benchmark and are not
included. The crystal structures are available from RCSB; the accession list is
in `pdb_accessions.txt`.
