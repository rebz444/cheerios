# Data Overview

> Stable description of how raw neural + behavior data become the canonical analysis tables. Pipeline lives in the `0*` scripts. Numbers in this doc reflect the current state of `unit_properties_final.csv` and `sessions_all.csv` in `logs/`.

## 1. Pipeline at a glance

Eight numbered scripts run in order. Each consumes the previous output, so the pipeline must be re-run in sequence after any upstream change.

| Step | Script | Purpose | Key output |
| --- | --- | --- | --- |
| 0a | `0a_datajoint_processing_check.py` | Cross-check DataJoint processing log vs. Google Sheet recording log; report stuck stages | `sessions_cross_checked.csv`, `location_summary.csv` |
| 0b | `0b_neural_data_processing.py` | Turn the curated spike-sorting pickle into one cleaned, trial-aligned pickle per session | `{session_id}.pkl`, `sessions_official_raw.csv`, `region_cell_count.csv` |
| 0c | `0c_neural_data_quality_metrics.py` | Apply spike-sorting QC thresholds (FR, ISI, presence, contamination, amplitude cutoff) | `unit_properties_with_qc.csv` + QC plots |
| 0d | `0d_neural_data_examining.py` | Add group/cohort labels, per-session performance metrics, recording-level filters | `sessions_all.csv`, `sessions_vetted.csv`, `units_all.csv`, `units_vetted.csv` |
| 0e | `0e_track_qc.py` | Audit histological probe tracks (direction, region order, depth vs. insertion, outside-brain) | `track_qc.csv` |
| 0f | `0f_neuron_location_matching.py` | Assign atlas region to each unit using per-session tip-anchored depth + DV shrinkage | `unit_properties_with_qc_and_regions.csv` |
| 0g | `0g_waveform_diagnostic.py` | Waveform metrics; flag units that violate biological FR/PT priors for their region | `flagged_units*.csv`, `ccb_boundary_units.csv`, `rt_broad_boundary_check.csv` |
| 0h | `0h_cell_type_relabeling.py` | Boundary-recovery region corrections (R1–R5) + cell-type assignment (MSN / FSI / RS / TAN / high_FR) | `unit_properties_final.csv` ← **canonical for all downstream analysis** |
| 0i | `0i_cell_type_relabeling_by_depth.py` | Add depth-based MSN flags as a histology-independent fallback | adds `is_msn_depth` / `is_msn_depth_permissive` to `unit_properties_final.csv` |

Downstream loaders (`a_rescaling`, `population_decoder`, `v1_dms_crosscorrelation`, etc.) read `unit_properties_final.csv` and slice rows via the boolean membership columns it carries (`is_str_unit`, `is_str_msn`, `is_cortex`, `is_v1_cortical`, `is_mc_l5l6`, `is_msn_depth`).

### Why so many stages

Three independent things must be true before a unit can enter an analysis:

1. **Sorting is clean** — passes the spike-sorting QC gate (0c).
2. **The recording is usable** — session long enough, unit fires in enough trials (0d).
3. **The location is trustworthy** — the histological track passes audit (0e), the unit lands at a placeable depth (0f), and the waveform is consistent with that region's biology (0g/0h).

Each script's output is one of those gates. The split makes failures debuggable — when CP unit counts move, you can see whether the change came from sorting, the recording log, or histology.

## 2. Behavior task and trial structure

See `project _basics.md` for the full task description. In short:

- Three periods per trial: **background** (cue on, light) → **wait** (cue off, no licks until decision) → **consumption** (3 s after decision lick).
- Reward probability rises with wait time:  $p(t) = 0.9(1 - e^{-t/3})$.
- Two mouse groups with different mean background length:
  - **Short BG** (`group="s"`): mean 1 s background.
  - **Long BG** (`group="l"`): mean 5 s background.
- Four analysis anchors: `to_cue_on`, `to_cue_off`, `to_last_lick`, `to_decision`.

## 3. Per-session data format (`session_pickles/{id}.pkl`)

Each session pickle is a dict with these keys:

| Key | Type | Description |
| --- | --- | --- |
| `id` | str | `{mouse}_{date}_{region}`, e.g. `RZ034_2024-07-14_str` |
| `mouse` | str | Subject ID (`RZ034`, …) |
| `date` | str | YYYY-MM-DD |
| `region` | str | `str` or `v1` — refers to the **probe target**, not the unit assignment |
| `events` | DataFrame | All session events (trial starts, visual cue, licks, rewards) with trial-relative and anchor-relative times |
| `trials` | DataFrame | One row per trial, with `cue_on_time`, `cue_off_time`, `decision_time`, `wait_length`, `missed`, `rewarded`, `num_bg_repeat`, `good`, `last_lick_time` |
| `unit_ids` | list[int] | Sorted unit IDs in this session |
| `units` | dict[int → DataFrame] | One DataFrame of spike-aligned data per unit |

### Per-unit spike DataFrame

Each `units[unit_id]` row is one spike inside the [trial_start, trial_end] window of some trial (or a lookback spike for the `to_last_lick` anchor when the last lick came from a prior trial):

| Column | Description |
| --- | --- |
| `trial_id` | which trial the spike belongs to |
| `spike_time` | absolute session time |
| `trial_time` | spike_time − trial_start_time |
| `to_cue_on`, `to_cue_off`, `to_last_lick`, `to_decision` | spike time relative to each anchor |
| `period` | `background` / `wait` / `consumption` (None outside trial) |
| `lick_period` | secondary period column for `to_last_lick` raster: `lick_to_cue` / `wait` (allows negative times for lookback) |
| `cue_on_time`, `cue_off_time`, `last_lick_time`, `decision_time` | trial-level anchor times broadcast onto every spike (for fast per-spike filtering) |

Spikes outside any trial window are dropped.

## 4. Spike-sorting + QC pipeline

### Curated sessions (input to 0b)

`RZ_dj_progress.csv` + `neural_data_0519.pkl` come from the DataJoint pipeline after manual curation in Phy. Only units with `manual_label == "good"` are kept.

### QC thresholds (0c — `constants.QC_THRESHOLDS`)

| Metric | Operator | Threshold |
| --- | --- | --- |
| `firing_rate` | > | 0.1 Hz (STR override: 0.05 Hz — MSN floor) |
| `isi_violation` | < | 1.0 |
| `amplitude_cutoff` | < | 0.1 |
| `presence_ratio` | > | 0.8 |
| `contamination_rate` | < | 0.2 |

Secondary metrics (isolation_distance, l_ratio, d_prime, silhouette, drift, SNR) are reported but not gating. Units with uncomputable primary metrics (NaN) fail that metric.

### Recording-level filters (0d — `constants`)

- Session must have > 1 unit and length ≥ 1500 s (`MIN_SESSION_LENGTH`).
- Unit must fire in ≥ 80 % of trials (`MIN_PERCENT_TRIALS_WITH_SPIKES`).

### Track audit (0e)

Each histological track passes four independent gates:
1. **Direction** — DV sign matches population consensus.
2. **Region order** — track starts at cortical L1, ends deeper.
3. **Depth consistency** — track length within ±25 % of DV-shrinkage-corrected insertion depth.
4. **Outside-brain content** — < 1 % of trace labelled "Not found in brain".

Failing tracks are excluded from region assignment in 0f.

### Region assignment (0f)

Per-session tip-anchored mapping with DV-shrinkage correction:

```
peak_ch_fixed     = peak_channel_depth * DV_SCALE[mouse]
session_tip_fixed = min(insertion_depth_um * DV_SCALE[mouse], trace_max)
dist              = session_tip_fixed − peak_ch_fixed
```

`dist` is looked up in the track CSV to get an Allen-CCF atlas acronym. Sentinel labels (`above_surface`, `below_track`, `track_audit_failed`, `unknown_depth`, `no_session_match`) are filtered out before downstream stages.

### Cell-type relabeling (0h)

Stage 1 — region corrections using location + waveform:

| Rule | From | To | Trigger |
| --- | --- | --- | --- |
| R1 | GPe | CP | Already gated by 0g: PT ≥ 0.40 ms, FR ≤ 15 Hz |
| R2 | RT | CP | Broad waveform within 300 µm of CP depth range |
| R3 | ccb | CP | Nearest atlas region = CP + PT ≥ 0.40 ms |
| R4 | ccb | MOp6a | Nearest atlas region = MOp6a + gap ≤ 10 µm |
| R5 | ccb | `unplaceable` | All other ccb units |

Stage 2 — cell types (waveform-driven inside the corrected region):

| Cell type | Criterion |
| --- | --- |
| MSN | STR region + PT ≥ 0.40 ms (no FR bound — Berke 2004, Mello 2015) |
| FSI | PT < 0.35 ms, any region |
| RS  | Cortex region + PT ≥ 0.40 ms + FR < 30 Hz |
| TAN | STR region + PT 0.35–0.55 ms + FR 3–12 Hz |
| high_FR | FR > 20 Hz, not FSI, not in STR (STR-broad-high-FR → MSN) |
| ambiguous | none of the above |

Depth-based MSN gate (0i, STR probe only, ≥1500 µm from surface):
- `is_msn_depth` — PT ≥ 0.40 ms, FR < 5 Hz (strict).
- `is_msn_depth_permissive` — PT ≥ 0.40 ms, FR < 15 Hz.

## 5. Dataset summary (current state)

### Mice and sessions

| Group | Mice | Sessions (all) | Sessions (vetted) | Trials |
| --- | --- | --- | --- | --- |
| Short BG (`s`) | 10 | 52 | 49 | 21,385 |
| Long BG (`l`) | 10 | 44 | 44 | 12,550 |
| **Total** | **20** | **96** | **93** | **33,935** |

- **Short BG mice**: RZ034, RZ036, RZ047, RZ049, RZ050, RZ051, RZ061, RZ062, RZ063, RZ070.
- **Long BG mice**: RZ037, RZ038, RZ039, RZ052, RZ053, RZ055, RZ057, RZ058, RZ059, RZ065.

Sessions are excluded by 0d if they have ≤ 1 unit or < 1500 s of recording.

### Unit counts at each pipeline stage

| Stage | Units | Notes |
| --- | --- | --- |
| Good (manual label, post-sort) | ~3,800 | from `RZ_unit_properties.csv` |
| Recording-level pass (`units_all`) | **3,620** | session vetted + ≥80 % trials w/ spikes |
| Recording + sorting QC (`units_vetted`) | **2,737** | adds spike-sorting QC from 0c |
| Region-mapped + relabeled (`unit_properties_final`) | **3,489** | full superset for analyses that use boolean `is_*` filters |

`units_vetted` is the strictest list. `unit_properties_final` is the canonical analysis table — every downstream script slices it using `is_str_msn`, `is_cortex`, etc.

## 6. Unit assignment results

### By probe target × group (`unit_properties_final.csv`)

| Probe target | Long BG | Short BG | Total |
| --- | --- | --- | --- |
| STR probe | 1,011 | 1,635 | **2,646** |
| V1 probe | 436 | 407 | **843** |
| **Total** | **1,447** | **2,042** | **3,489** |

### By region group × BG group

| Region group | Long BG | Short BG | Total |
| --- | --- | --- | --- |
| Striatum | 125 | 192 | **317** |
| Motor cortex | 183 | 443 | **626** |
| Visual cortex | 67 | 163 | **230** |
| Thalamus | 599 | 762 | **1,361** |
| Hippocampus | 207 | 165 | **372** |
| Pallidum | 0 | 15 | **15** |
| Other / fiber | 183 | 168 | **351** |
| Excluded (fiber tracts, unplaceable) | 83 | 134 | **217** |
| **Total** | **1,447** | **2,042** | **3,489** |

### Detailed regions (top 20 acronyms × BG group)

| Region | Long BG | Short BG | Total |
| --- | --- | --- | --- |
| CP | 105 | 186 | **291** |
| VAL | 118 | 329 | **447** |
| PO | 175 | 208 | **383** |
| VPM | 204 | 8 | **212** |
| MOp6a | 101 | 283 | **384** |
| MOp5 | 37 | 107 | **144** |
| VISp5 | 19 | 67 | **86** |
| VISp6a | 21 | 26 | **47** |
| CA1 | 117 | 73 | **190** |
| ProS | 83 | 32 | **115** |
| SUB | 7 | 50 | **57** |
| DG-mo | 47 | 8 | **55** |
| RT | 11 | 118 | **129** |
| LD | 2 | 53 | **55** |
| VL | 28 | 16 | **44** |
| PCN | 44 | 0 | **44** |
| PAL | 31 | 52 | **83** |
| int | 18 | 30 | **48** |
| dhc | 6 | 31 | **37** |
| unplaceable | 22 | 51 | **73** |

### Thalamus subregions × BG group

Full breakdown of the 1,361 thalamic units in `unit_properties_final.csv` (rows where `region_group == "Thalamus"`):

| Subregion | Long BG | Short BG | Total |
| --- | --- | --- | --- |
| VAL | 118 | 329 | **447** |
| PO  | 175 | 208 | **383** |
| VPM | 204 | 8   | **212** |
| RT  | 11  | 118 | **129** |
| LD  | 2   | 53  | **55** |
| PCN | 44  | 0   | **44** |
| VL  | 28  | 16  | **44** |
| VPL | 11  | 19  | **30** |
| AV  | 6   | 3   | **9** |
| CL  | 0   | 6   | **6** |
| VM  | 0   | 2   | **2** |
| **Total** | **599** | **762** | **1,361** |

Several subregions are strongly group-asymmetric (VPM, PCN: mostly long; VAL, RT, LD: mostly short), reflecting different probe trajectories between cohorts rather than a true cell-density difference.

### Cell-type assignment × BG group

| Cell type | Long BG | Short BG | Total |
| --- | --- | --- | --- |
| MSN | 53 | 93 | **146** |
| FSI | 300 | 475 | **775** |
| RS | 169 | 438 | **607** |
| TAN | 5 | 11 | **16** |
| high_FR | 133 | 152 | **285** |
| ambiguous | 765 | 822 | **1,587** |
| unplaceable | 22 | 51 | **73** |

### Analysis-ready cell sets (`is_*` membership flags)

| Cell set | Long BG | Short BG | Total |
| --- | --- | --- | --- |
| `is_str_unit` (STR region label) | — | — | **317** |
| `is_str_msn` (CP/STR + broad waveform) | 53 | 93 | **146** |
| `is_cortex` (motor or visual cortex) | 250 | 606 | **856** |
| `is_v1_cortical` (V1 probe + visual cortex) | 67 | 163 | **230** |
| `is_mc_l5l6` (MOp/MOs layers 5–6) | 172 | 408 | **580** |
| `is_msn_depth` (depth-based, strict FR<5 Hz) | 116 | 168 | **284** |
| `is_msn_depth_permissive` (FR<15 Hz) | 328 | 458 | **786** |

The depth-based MSN sets are larger than the waveform-based one because they don't require trustworthy histology — they only need a valid `peak_channel_depth` and STR probe target.

## 7. Practical notes for analysis

- **Always load `unit_properties_final.csv`**, not `units_all` / `units_vetted` — the canonical table carries cell-type labels, depth-based fallback MSN flags, and all boolean filters used by the decoder/rescaling code.
- **Group assignment lives in `constants.GROUP_DICT`** (`s` / `l`). The same mouse never appears in both.
- **Region columns**: `region_acronym` is the post-correction label (= `corrected_region`). `region_acronym_atlas` preserves the pre-0h atlas label. `region` (lowercase) on session-level tables is the probe target, not the unit assignment.
- **STR MSN counts per session** drive `population_decoder` eligibility (≥ 15 units; see `0h_cell_type_relabeling.py` and `0i`).
- **Probe-track histology is the main loss source**: 0f and 0e together remove ~10 % of QC-good units that fall into fiber tracts, fail the track audit, or land outside the trace. Use `is_msn_depth_permissive` for analyses where you'd rather have more MSN candidates than perfect placement.
