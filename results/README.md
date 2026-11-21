# Result summaries

These are compact `summary.json` files behind the manuscript tables; full run
directories (shards, logs, per-target predictions) are not included.

| file | experiment |
|---|---|
| `cis_b2_target_summary.json` | five-type, target-held-out |
| `cis_b2_scaffold_summary.json` | five-type, scaffold-held-out |
| `cis_5t_temporal_summary.json` | five-type, temporal |
| `cis_5t_temporal_strict_summary.json` | five-type, strict target-disjoint temporal |
| `cis_water_target_summary.json` | six-type water-blind, target |
| `cis_water_scaffold_summary.json` | six-type water-blind, scaffold |
| `cis_water_temporal_summary.json` | six-type water-blind, temporal |
| `cis_water_temporal_strict_summary.json` | six-type water-blind, strict |
| `cis_water_w1_target_summary.json` | six-type W1, target |
| `cis_water_w1_scaffold_summary.json` | six-type W1, scaffold |
| `cis_water_w1_temporal_5s_summary.json` | six-type W1, temporal, 5 seeds |
| `cis_water_w1_temporal_strict_5s_summary.json` | six-type W1, strict, 5 seeds |
| `cis_water_w1_egnn_target_summary.json` | EGNN W1, target |
| `cis_water_w1_egnn_scaffold_summary.json` | EGNN W1, scaffold |

The W1 temporal/strict values used in the paper are the `*_5s` files, not the
earlier 3-seed runs.
