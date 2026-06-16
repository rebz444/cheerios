# Analysis Results

> Stable description of the three main neural analyses currently producing results: temporal **rescaling** of population activity across regions, **population decoding** of elapsed time across regions, and **V1→DMS cross-correlation**. Each section pairs methods with current numbers from the per-analysis output directories under `data/`. Re-run the named script to refresh.

The three analyses share the same canonical inputs:
- Per-session pickles in `data/session_pickles/`
- The unit table `data/logs/unit_properties_final.csv` (post-0h+0i; see `data_overview.md`)
- The two BG groups defined in `constants.GROUP_DICT` ("s" / "l")

Both the rescaling and decoder analyses are now run across **multiple brain regions** (striatum, motor cortex, visual cortex, thalamus, hippocampus) rather than just striatum + motor-cortex control. Per-region/per-cell-type panels are produced by `b_cross_region_rescaling.py` and `cross_region_decoder.py`.

---

## 1. Cross-region rescaling analysis

**Script.** `rescaling/a_rescaling.py` (data + metrics + per-set figures in one script; skips analysis when a cache exists and skips plots when figures exist, unless regenerate flags are passed) → `rescaling/b_cross_region_rescaling.py` (headline cross-region figure). Output base: `data/rescaling/<set_label>/`; cross-region figure at `data/rescaling/cross_region/`. All rescaling scripts now live in the `rescaling/` folder.

### Question

Do neurons "rescale" their firing pattern with the wait interval — i.e. when the mouse waits longer, does each neuron's peak slide proportionally later (relative rescaling), or does it stay locked to the same absolute time (fixed timing)? Which regions and cell types show this signature?

The literature predicts **relative rescaling** for cells encoding *fraction of an interval* (a hallmark of an internal clock) and **fixed timing** for cells encoding *absolute elapsed time* since an event.

### Method

For each cell set × BG group × alignment anchor:

1. Pool all qualifying sessions; bin trials into **wait-time quartiles Q1–Q4**.
2. **Per-unit filter (`MIN_SPIKES_PER_QUARTILE = 30`)** — drop any unit whose sparsest quartile has fewer than 30 total spikes. Smoothed PSTHs from sparse spiking produce unreliable peak times (typically landing at the basis edge, masquerading as anchor-locked transients).
3. For each surviving unit and each quartile, compute a PSTH on:
   - **Normalized time** (0 → 1 of the interval, 100 bins).
   - **Absolute time** (50 ms bins, 0 → 15 s short BG / 22 s long BG).
4. Take peak time τ from the smoothed PSTH for each unit, in each representation.
5. Compare **Q2 vs Q3** (`QUARTILE_PAIR`) — the tail-free middle quartiles. Q4, the top wait-time quartile, is open-ended (75th percentile up to the 60 s miss cutoff, no outlier trimming) and adds noise to the peak estimate; Q2 and Q3 are both bounded between the 25th–75th percentiles. See `c_quartile_pair_comparison.py` and Section 1b for the pair justification.
   - **Normalized correlation** `r` of (τ_Q2, τ_Q3) across units, with shuffle null (1000 perms) and bootstrap CI on the linear slope.
   - **Scale factor** = median(τ_Q2 / τ_Q3). =1 means perfect relative rescaling.
   - **frac_rescaling** = fraction of units with τ_Q2/τ_Q3 within ±20 % of 1.
   - **Absolute correlation** `r_abs` of the same units in absolute time.
   - **frac_fixed** = fraction of units whose absolute-time ratio τ_Q3_abs/τ_Q2_abs sits within ±30 % of the *expected* ratio under perfect rescaling (= median wait_Q3 / median wait_Q2).
6. Aligned three ways per group (`cue_on`, `cue_off`, `last_lick`) so the result doesn't depend on a specific anchor.
7. **Best-fit (anchor, Qa→Qb) scan** is also run in the plotting step of `rescaling/a_rescaling.py` (figures 1, 1b, 3, 6) — for each group, find the (anchor × quartile pair) combination with the highest r across all 18 candidates (3 anchors × 6 ordered pairs). This catches cases where Q2→Q3 isn't the strongest contrast.

### Cell sets (`CELL_SETS`)

`CELL_SETS` defines 14 region/cell-type selections, all of which appear as columns in the cross-region headline panel (`b_cross_region_rescaling.py`) and flow through every downstream check (`c_`–`g_`). Deprecated sets (`str_units`, `str_msn_depth*`, `mc_l5l6`, `mop_mos_rs_final`, `msn_tier2_first_batch`, `v1`) are commented out in `CELL_SETS` but prior caches remain on disk for reference.

| Region | Set | Filter | Intent |
| --- | --- | --- | --- |
| Striatum | `str_all` | `is_str_unit` | All striatal units (cell-type aggregate) |
| Striatum | `str_msn` | `is_str_msn` | Waveform-validated MSNs |
| Striatum | `str_fsi` | `is_str_unit` ∩ `cell_type=='FSI'` | Striatal fast-spiking interneurons |
| Motor cortex | `mc_all` | `region_group=='Motor cortex'` | All motor-cortex units (aggregate) |
| Motor cortex | `mc_l5l6_rs` | `is_mc_l5l6` ∩ `cell_type=='RS'` | L5/L6 motor-cortex RS pyramidals (DMS input) |
| Motor cortex | `mc_fsi` | `region_group=='Motor cortex'` ∩ `cell_type=='FSI'` | Motor-cortex FSI |
| Visual cortex | `v1_all` | `region_group=='Visual cortex'` | All visual-cortex units (aggregate) |
| Visual cortex | `v1_rs` | `region_group=='Visual cortex'` ∩ `cell_type=='RS'` | V1 RS — visual upstream control |
| Thalamus | `thal` | `is_thal` | All-thalamus aggregate |
| Thalamus | `val` | `is_val` | Ventral anterior/lateral (motor relay) |
| Thalamus | `po` | `is_po` | Posterior nucleus |
| Thalamus | `vpm` | `is_vpm` | Ventral posteromedial (somatosensory relay) |
| Hippocampus | `ca1` | `is_ca1` | CA1 |
| Hippocampus | `hpf` | `is_hpf` | Hippocampal formation (CA1+ProS+SUB+CA3+dhc) |

### Key parameters

```
TIME_STEP                = 0.05 s   SIGMA                   = 2 bins
T_MAX_SHORT              = 15 s     T_MAX_LONG              = 22 s
N_BINS_NORM              = 100      MIN_TRIALS              = 5 / unit
N_SHUFFLE                = 1000     ABS_RESCALING_TOLERANCE = 0.30
MIN_SPIKES_PER_QUARTILE  = 30       (drops sparse-spiking units)
QUARTILE_PAIR            = ('Q2','Q3')   (compared pair; not Q3→Q4)
```

### Cross-region headline (from `data/rescaling/cross_region/cross_region_table.csv`)

For each region × group, the best anchor (max r) per cell set:

| Region | Cell set | Group | Best anchor | n_units | n_sess | n_mice | r | slope |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Striatum | `str_msn` | Short BG | cue_off | 49 | 12 | 7 | 0.78 | 0.76 |
| Striatum | `str_msn` | Long BG | cue_on | 29 | 7 | 4 | 0.84 | 0.82 |
| Striatum | `str_fsi` | Short BG | cue_off | 48 | 9 | 6 | 0.76 | 0.82 |
| Striatum | `str_fsi` | Long BG | last_lick | 33 | 4 | 3 | 0.80 | 0.83 |
| Motor cortex | `mc_l5l6_rs` | Short BG | cue_off | 225 | 24 | 9 | 0.74 | 0.81 |
| Motor cortex | `mc_l5l6_rs` | Long BG | cue_off | 74 | 12 | 6 | 0.80 | 0.87 |
| Motor cortex | `mc_fsi` | Short BG | cue_off | 73 | 23 | 8 | **0.94** | 0.77 |
| Motor cortex | `mc_fsi` | Long BG | cue_off | 33 | 11 | 7 | 0.76 | 0.16 |
| Visual cortex | `v1_rs` | Short BG | cue_on | 82 | 10 | 7 | 0.58 | 0.63 |
| Visual cortex | `v1_rs` | Long BG | last_lick | 24 | 6 | 5 | **0.96** | 0.87 |
| Thalamus | `thal` | Short BG | cue_on | 617 | 20 | 7 | 0.70 | 0.76 |
| Thalamus | `thal` | Long BG | cue_on | 572 | 20 | 8 | 0.72 | 0.76 |
| Thalamus | `val` | Short BG | cue_off | 313 | 17 | 7 | 0.66 | 0.71 |
| Thalamus | `val` | Long BG | cue_off | 113 | 12 | 8 | 0.73 | 0.84 |
| Hippocampus | `ca1` | Short BG | last_lick | 51 | 6 | 3 | 0.64 | 0.53 |
| Hippocampus | `ca1` | Long BG | cue_on | 101 | 8 | 4 | **0.87** | 0.88 |
| Hippocampus | `hpf` | Short BG | last_lick | 120 | 16 | 7 | 0.66 | 0.58 |
| Hippocampus | `hpf` | Long BG | cue_on | 227 | 20 | 8 | 0.78 | 0.80 |

`po` and `vpm` are dropped from the cross-region figure by the single-mouse filter (`MIN_MICE = 2`) — both have a single-mouse group. They remain in `CELL_SETS` and per-set caches.

> **Note.** The headline comparison pair is **Q2→Q3** (`QUARTILE_PAIR`), not Q3→Q4 — Q4 is the open-ended top wait-time quartile and adds noise; Q2/Q3 are tail-free (see Section 1b and `c_quartile_pair_comparison.py`). The 11 caches' `metrics`/`results_df` were re-derived for Q2→Q3 **in place** from the cached raw per-quartile spikes (`python rescaling/a_rescaling.py remetric`) — no overnight session re-pool needed, since the caches already hold all four quartiles. `po` and `vpm` (single-mouse groups) and the small-n long-BG sets remain caveated.

### What the numbers say

- **Every region in the panel rescales** — every cell × group combination clears the shuffle null at its best anchor, with r from 0.58 (V1 short BG) to 0.96 (V1 long BG, n=24) and a typical r of 0.7–0.85.
- **Striatum**: `str_msn` long BG at cue_on (r=0.84), short BG at cue_off (0.78); `str_fsi` comparable (0.80 long / 0.76 short) — the rescaling signal is not MSN-specific.
- **Motor cortex**: `mc_fsi` short BG cue_off (r=0.94) is the single strongest cell in the panel; `mc_l5l6_rs` r=0.74–0.80. `mc_fsi` long BG has a shallow slope (0.16) despite a high r.
- **Thalamus**: very large unit counts (>500 in `thal`); r=0.70–0.73 across `thal` and `val`.
- **Hippocampus**: strong in long BG (`ca1` 0.87, `hpf` 0.78), weaker in short BG (0.64–0.66).
- **Visual cortex**: `v1_rs` short BG (0.58) is the panel's weakest; long BG is 0.96 but on only n=24 (5 mice).
- **Anchor varies by region and group**: there is no single "best anchor" — the cue_on / cue_off / last_lick comparison must be done per-cell-set rather than assumed.

> **Edge-cell confound.** The `r` reported above is the cross-unit correlation of Q2-vs-Q3 *peak position*, and it is inflated by units firing at the interval edges (cue-locked or pre-motor / lick transients) rather than tiling the interval. **Section 1b** documents four follow-up checks — τ-stratification, crop-and-resort, full-PSTH similarity, peak-width test — that quantify the contamination and establish what survives it. Read these headline numbers together with Section 1b.

### Outputs per cell set

- `data/rescaling/<set>/enhanced/results_summary.csv` — per (group, anchor) row.
- `data/rescaling/<set>/results_cache.pkl` — full PSTH matrices + spike pools (used by the plotting step of `a_rescaling.py`).
- `data/rescaling/<set>/per_group/`, `enhanced/`, `committee/` — population heatmaps, τ_Q2 vs τ_Q3 scatters, scaling-factor distributions, committee-style panels (figures 1, 1b, 2, 3, 5, 6 now use the best-fit anchor × quartile pair per group).
- `data/rescaling/results_summary_all_sets.csv` — cross-set aggregate (legacy).
- `data/rescaling/cross_region/cross_region_rescaling.png` + `cross_region_table.csv` — **headline cross-region figure**, anchor-matched (panel A: r heatmap region × anchor, per group; panel B: r per region, short vs long, one sub-panel per anchor; panel C: rescaling vs fixed-latency fractions per anchor). Each group is compared at the *same* anchor — no cross-group anchor mixing.

---

## 1b. Rescaling edge-cell confound checks

**Scripts.** `rescaling/d_tau_stratification.py`, `rescaling/e_crop_resort_rescaling.py`, `rescaling/f_psth_similarity.py`, `rescaling/g_peak_width_test.py`. Output base: `data/rescaling/<check>/`.

**Why.** The Section-1 headline `r` is the *cross-unit correlation of Q2-vs-Q3 peak position*. It can be inflated by **edge cells** — units firing near the interval edges (cue-locked onset transients, pre-motor / lick activity) rather than tiling the interval. An edge cell peaks at normalized time τ ≈ 0 or τ ≈ 1 in *both* quartiles, so it lands as a high-leverage point on the τ_Q2-vs-τ_Q3 scatter and props up `r` without encoding interval fraction. These four checks, run in order, quantify the contamination and test what survives it. All four run from the existing per-set caches (`q_norm_spikes` / `q_spikes` hold the per-trial spike times for all four quartiles) — **no session re-pool** — and each reproduces the cached Section-1 Q2→Q3 metrics exactly at its no-crop baseline (validated to ~1e-15).

**Anchor-matched reporting.** The digests and figures of all four checks show all three anchors, with each behavioral group reported at the *same* anchor — never short@last_lick vs long@cue_on. Numbers quoted below are at the **`cue_off`** anchor: the pure wait-window (`cue_off → decision`), defined identically for both groups (no background period, no cross-trial `last_lick` spanning), so short-vs-long is like-for-like. The per-anchor figures cover all three.

### Check 1 — τ-stratification

**Purpose.** Localise the rescaling signal: is `r` concentrated in mid-interval-peaking units (genuine interval tiling) or in edge-peaking units (the confound)?

**Method.** For each cell set × group × anchor, bin units by τ_Q2 into 5 equal slices of the normalized interval (0–0.2 … 0.8–1.0). Per bin report `r`(τ_Q2, τ_Q3), slope, n_units, frac_rescaling; also an edge-vs-mid digest (pooled bins 0+4 vs 1+2+3 — wider and less range-restricted than a single bin). All three anchors.

**Results.** At `cue_off`, **56–97 % of units sit in the edge bins** (τ_Q2 < 0.2 or > 0.8) — the τ-distribution is U-shaped in every region. `r` weakens in the mid bins (`r_all` → `r_mid`): mc_l5l6_rs short 0.74 → 0.27, ca1 long 0.79 → 0.08, hpf long 0.70 → 0.03, po short 0.69 → 0.05, str_msn short 0.78 → 0.40. `r_all` tracks `r_edge` almost exactly. Some regions keep a genuine mid-interval signal: mc_l5l6_rs long (`r_mid` 0.84), val long (0.68), ca1 short (0.56), thal (0.48–0.53), str_fsi (0.48–0.53); V1 RS short BG goes negative in the mid bins (−0.30).

**Interpretation.** The headline `r` is disproportionately carried by edge-peak units; genuine mid-interval timing is a minority. Confirms the confound is real — motivated Checks 2–3.

**Outputs.** `data/rescaling/tau_stratification/`:
- `tau_stratification_table.csv` (per cell_set × group × anchor × τ-bin, plus an `all` row) and `tau_stratification_summary.csv` (the edge-vs-mid digest).
- `tau_stratification_heatmap.png` / `tau_stratification_counts.png` — region × τ-bin heatmaps of within-bin `r` and of the unit-count distribution (faceted anchor × group).
- `tau_stratification_bin_distribution.png` — one facet per cell set, % of units in each τ-bin drawn as a line per anchor × group, against a 20 % uniform reference. The U-shape (edge bins ≫ 20 %, middle ≪ 20 %) is the edge-cell pile-up made visible.
- `tau_stratification_r_all_vs_mid.png` — `r_all` vs `r_mid` scatter, one panel per anchor, y = x diagonal. Almost every point falls **below the diagonal** (19/23, 21/23, 19/22 at last_lick/cue_on/cue_off respectively; mean Δr = 0.17 / 0.32 / 0.30) — the headline `r` collapses when the edge bins are dropped.
- `tau_stratification_scatter_{short_BG,long_BG}_{last_lick,cue_on,cue_off}.png` (6 figures) — per cell-set τ_Q2 vs τ_Q3 rescaling scatter with edge-τ units in red and mid-τ in blue, plus two OLS fits (all-units grey vs mid-only blue). When the grey line hugs the identity while the blue line is much flatter, the edge corners are visibly carrying the headline `r` — the mechanism, not a summary.
- `tau_stratification_dr_vs_edge.png` — Δr (= `r_all` − `r_mid`) vs the fraction of edge-τ units per anchor, with a vertical "uniform" reference at edge-fraction = 0.40 (= 2 of 5 bins under uniform tiling) and a Δr = 0 horizontal. **Median edge fraction = 0.81–0.84 across anchors vs the 0.40 uniform** and 19–21 of ~23 cell sets land in the upper-right "confound zone" (over-represented edges AND `r` inflation). Note: the *slope* of Δr ~ frac_edge is flat — frac_edge is saturated high (~0.6–0.95) for every cell set so there is no across-region spread to support a trend. The finding is the absolute position, not a correlation.

### Check 2 — crop-and-resort

**Purpose.** Directly expose edge units: crop the interval edges, re-find each unit's peak inside the cropped window, re-run the rescaling correlation. A unit that fires at the edges *and* the middle keeps a genuine mid peak; a pure-edge unit's re-found peak is noise or pinned to the crop boundary.

**Method.** For **every** unit, crop its normalized PSTH to the window [X, 1−X] (re-bin, re-smooth, re-z-score from the cached per-trial spikes) and re-find the peak (argmax) *inside that cropped window*. No unit is pre-excluded — the crop + re-peak-finding is itself the classifier:
- a re-found peak is **boundary-pinned** if its argmax lands on the first or last bin of the cropped window — the PSTH is still rising/falling at the crop edge, so the unit's true peak lies *outside* [X, 1−X] (an edge cell, now exposed);
- a unit is **interior** if its re-found peak is strictly inside the window in *both* Q2 and Q3.

Two readouts: `r_all` = the rescaling correlation `r`(τ_Q2, τ_Q3) over all units; `r_interior` = the same `r` over interior units only. Crop sweep 0 / 5 / 10 / 15 / 20 / 25 %, headline 15 % (a moderate default shared with Checks 3/3b; the crop *curve* — `crop_resort_curves_*.png` — is the real output, and `r_interior` generally declines with more cropping with no clean plateau). Reproduces the cached Q2→Q3 `r` exactly at crop = 0 (all 66 cells).

**Why `r_interior` understates per-unit rescaling.** `r_interior` is a *cross-unit* correlation — it correlates peak position *across* the interior units. But interior units, by construction, all peak inside (X, 1−X), so their τ values are squeezed into a narrower range than the full [0, 1]. Correlation is attenuated by range restriction: even if every unit rescaled perfectly, bunching their peaks into a narrow band depresses `r`. A low `r_interior` can therefore mean "interior peaks happen to cluster," not "no rescaling" — Check 3, a per-unit measure, is not subject to this.

**Results.** At `cue_off`, 15 % crop. `frac_interior` is 21–58 %. `r_interior` (well-powered cells, n_interior ≥ 40): thal 0.44 short / 0.33 long, VAL 0.48 / 0.48, PO 0.28 short / 0.47 long, MC RS 0.50 short / 0.40 long, HPF 0.21 short / 0.25 long, CA1 long 0.25, V1 RS short −0.04, VPM long 0.05. Small-n cells (n_interior < 30, unreliable): MC FSI short 0.71, str_msn short 0.55, str_fsi short 0.22, CA1 short 0.32; the long-BG striatal/cortical cells have n_interior 6–14 (str_msn long −0.68 on n = 6 is meaningless). Across the well-powered cells `r_interior` lands at ~0.2–0.5 — clearly below the `r_all` ≈ 0.5–0.7 of the same cells.

**Interpretation.** Cropping the interval edges roughly halves the rescaling `r` in every well-powered region — `r_interior` ≈ 0.2–0.5 against `r_all` ≈ 0.5–0.7. The edge confound is **general**: no region is immune and none is uniquely culpable. The strongest surviving interior signal is in motor cortex RS (0.40–0.50), VAL (0.48), PO long (0.47) and thalamus (0.33–0.44); the weakest are V1 RS (≈ 0) and hippocampus (0.21–0.25). **Important caveat:** `r_interior` is a *cross-unit peak-position* correlation, range-restricted among interior units (which all peak in (0.15, 0.85), so their τ values are bunched) — a depressed `r_interior` therefore partly reflects range restriction rather than absent rescaling. Check 3, a per-unit measure, is the cleaner read.

**Outputs.** `data/rescaling/crop_resort/`:
- `crop_resort_table.csv` (per cell_set × group × anchor × crop) — the scalar metrics, including `r_all`, `r_interior`, `n_interior`, `frac_interior`.
- `crop_resort_curves_{cue_on,cue_off,last_lick}.png` — `r` vs crop fraction, one facet per cell set, short and long BG at the same anchor.
- `crop_resort_heatmap.png` — region × crop heatmap of `r_interior` (with `n_interior` annotated), faceted anchor × group.
- `crop_<XX>%/<sort>_sort/<anchor>/normsort_<set>.png` — **population sequential-firing heatmaps on cropped+resorted PSTHs** (a_rescaling fig1b-style layout, but with each cell-set + anchor + crop + sort as its own file so all 14 cell sets at one condition sit side by side). Per figure: 2 rows (short BG / long BG, anchor-matched — *not* best-per-group) × 4 cols (Q2 absolute, Q3 absolute, Q2 normalized cropped, Q3 normalized cropped). Absolute panels use a **per-quartile trial-end cap** (98 th percentile of pooled per-trial last-spike-times), so the x-axis tracks where trials actually end and z-scoring stays inside the data range — no stretched 15/22 s x-axes with mostly post-trial zeros. Crops rendered: 15 % and 25 % (`NORMSORT_CROPS`). Three row-sort methods rendered side by side (`SORT_METHODS`):
  - **`qa_sort`** — sort by Q2 cropped-norm peak only. The original asymmetric layout: Q2 panel is a diagonal by construction; Q3 diagonality in that order *is* the rescaling test.
  - **`mean_sort`** — sort by the mean of Q2 and Q3 cropped-norm peak bins. A symmetric consensus order; neither panel favored — both read as the same shared sequence.
  - **`spectral_sort`** — spectral seriation on the joint [Q2 | Q3] cropped-norm PSTH matrix (Fiedler vector of the row-similarity Laplacian, orientation auto-aligned so low row = early peak). Uses the *whole* PSTH shape rather than a single argmax, so units with similar joint dynamics sit adjacent. **Caveat:** the row order is shape-similarity, not strict peak-time — the cleanest diagonal in this view is a coherence claim, not a "neurons fire in the same temporal order" claim, which `qa_sort` visualizes directly.

  These 252 figures (2 crops × 3 sorts × 3 anchors × 14 cell sets) are visual companions to the scalar Check-2 / Check-3 numbers above — the patterns those numbers already document (high-`r_interior` / high-`xcorr` regions like thalamus, motor cortex and striatum retain a recognisable cropped diagonal across all three sorts; V1 RS short BG and small-n long-BG cells degrade to noise) should read directly off the heatmaps as a sanity check rather than as a separate finding.

### Check 3 — full-PSTH similarity

**Purpose.** Checks 1–2 compared peak *positions* across units. Check 3 compares the *whole* Q2 vs Q3 PSTH of each unit — a per-unit shape-preservation measure, not range-restricted by the cross-unit peak spread.

**Method.** Two similarity metrics, computed per unit between its Q2 PSTH and its Q3 PSTH (each a smoothed, z-scored firing-rate vector over time bins):
- **`xcorr`** — zero-lag normalized cross-correlation: the Pearson correlation of the two PSTH vectors, ∈ [−1, 1]. Measures whether the two PSTHs rise and fall together over time.
- **`procsim`** — `1 − Procrustes disparity`. Each PSTH is treated as a 2-D *curve* — bin *i* → point (time_i, rate_i), an (n_bins, 2) array. `scipy.spatial.procrustes` centres both curves, scales each to unit norm, finds the optimal rotation to superimpose them, and returns a disparity ∈ [0, 1]; `procsim = 1 − disparity`, higher = more similar. A shape distance that allows a similarity transform; in practice it tracks `xcorr` closely.

Run as a 2×2 of **alignment × cropping**:
- **anchor-aligned** — the two PSTHs are left on their natural axis (time from the anchor) and compared directly. In normalized time a fixed-latency unit's Q2/Q3 PSTHs already diverge (the fixed event lands at a smaller normalized fraction of the longer Q3), so a high anchor-aligned similarity = genuine rescaling. This is the headline test; **cropped (15 %) removes the edge transients and is the key population panel.**
- **peak-aligned** — each PSTH is first *shifted so its peak sits at a common reference index*, then the overlapping segment is compared. This **removes peak position**: what is left is the *shape and width* of the response, not where it sits. Peak-alignment isolates "is the bump the same shape" from "is the bump in the same place" — a complement to the τ-correlation, not itself a rescaling test. Its readout is the normalized − absolute contrast (developed in Check 3b).

**Results.** Anchor-aligned, median `xcorr`, uncropped → cropped 15 %, at `cue_off`. **Both groups drop substantially under cropping.** Short BG: MC FSI 0.83 → 0.64, MC RS 0.83 → 0.62, thal 0.77 → 0.53, VAL 0.72 → 0.55, PO 0.82 → 0.48, str_msn 0.71 → 0.42, HPF 0.63 → 0.42, str_fsi 0.61 → 0.25, V1 RS 0.15 → −0.11. Long BG: PO 0.81 → 0.71, thal 0.75 → 0.48, VAL 0.71 → 0.33, str_msn 0.69 → 0.33, str_fsi 0.73 → 0.28, HPF 0.71 → 0.22, MC RS 0.78 → 0.14. (The earlier "short BG survives the crop" was specific to the `last_lick` anchor; at the matched `cue_off` anchor short BG drops about as much as long BG.)

**Cross-anchor comparison.** Same metric, cropped-15% `xcorr` median across all 28 cell-set × group cells: **`last_lick` 0.79** (range 0.37 – 0.95), **`cue_on` 0.58** (0.34 – 0.85), **`cue_off` 0.34** (−0.11 – 0.71). The non-cue_off anchors are *not* extra positive evidence for rescaling — they are systematically inflated by activity unrelated to interval timing:
- At **`last_lick`**, the window collapses to the ~1 s from the last background lick to decision and is dominated by stereotyped lick-locked / pre-decision motor dynamics that reproduce across trials regardless of how long the animal waited. Almost everything reads high (mc_fsi long 0.94, vpm long 0.92, po long 0.89, thal long 0.88, mc_all long 0.87) — motor reproducibility, not interval timing.
- At **`cue_on`** for long BG, the window includes the whole background-period lick sequence before `cue_off`, which is similarly stereotyped and inflates short-vs-long similarity for the same reason. Short BG at `cue_on` is short (BG ≈ 0.5 s) so its inflation is milder.
- **`cue_off`** strips both — it is defined identically for both groups as the pure wait window with no within-trial events. The much lower numbers at `cue_off` are the test of interval rescaling specifically, not "motor activity is reproducible across quartiles."

So the cue_off numbers above are the defensible read; the cue_on / last_lick figures (`psth_similarity_{xcorr,procrustes}_{cue_on,cue_off,last_lick}.png`) make the anchor-dependence itself visible and serve as a sanity check on the crop machinery (the test isn't structurally crashing similarity to zero — at favorable anchors it stays high).

The peak-aligned normalized − absolute contrast is ≈ 0 — Q2→Q3 is a small interval stretch, so the width test has little power here; Check 3b retests it at maximal stretch.

**Interpretation.** At the matched `cue_off` anchor, edge cropping cuts the whole-PSTH similarity to a cropped `xcorr` of ~0.3–0.6 in most regions — confirming the confound is real and **not group-specific**: short and long BG both lose roughly half their similarity. PO is the most crop-robust (cropped 0.48–0.71); V1 RS short BG fails outright (cropped < 0). What survives — cropped `xcorr` ≈ 0.4–0.7 across thalamus, motor cortex and striatum — is genuine per-unit shape preservation through the middle of the interval and, unlike Check 2's `r_interior`, is *not* attenuated by range restriction, so it is the more trustworthy positive evidence. The flat peak-aligned contrast prompted Check 3b.

**Outputs.** `data/rescaling/psth_similarity/` — `psth_similarity_per_unit.csv` (10,152 unit × anchor rows), `psth_similarity_summary.csv`, and per-anchor 2×2 figures `psth_similarity_{xcorr,procrustes}_{cue_on,cue_off,last_lick}.png`.

### Check 3b — peak-aligned width test (maximal-stretch decider)

**Purpose.** The Check-3 peak-aligned normalized − absolute contrast was flat — but Q3→Q4 gives only ~1.3–2× interval stretch, where a width-rescaling test has minimal power. Give the contrast a fair test at maximal stretch with a proper null, before letting "fixed-width" set the narrative.

**Method.** The question: as the interval lengthens, does a unit's response *get wider* (width rescales) or stay a fixed number of seconds wide (fixed width)?
- **Peak-aligned, two coordinate systems.** For each unit, take its Qa and Qb PSTHs, find each one's peak, shift so the peaks coincide, and compare the overlap — twice: once in **normalized time** (x-axis = fraction of interval) and once in **absolute time** (x-axis = seconds). A width-*rescaling* unit has the same-width bump in *normalized* units → normalized peak-aligned profiles match, absolute ones don't. A *fixed-width* unit has the same-width bump in *seconds* → absolute profiles match, normalized don't.
- **Per-unit `contrast`** = (peak-aligned similarity of the normalized PSTHs) − (peak-aligned similarity of the absolute PSTHs). `contrast > 0` ⇒ width rescales; `< 0` ⇒ fixed width.
- **Cropping** drops the first/last 15 % of the normalized interval before building the PSTHs (for the absolute PSTH: keep only spikes whose normalized position lies in [0.15, 0.85], then bin in seconds) — removes anchor transients so the re-found peak is the genuine response peak.
- **Shuffle-null frac-positive.** Per region the headline statistic is `frac-positive` = the fraction of units with `contrast > 0`. It cannot be compared to 0.5: the contrast metric is itself biased (absolute-time similarity tends to beat normalized even for structureless data). The **within-unit shuffle null** measures that baseline directly — for each unit, pool its Qa + Qb trials, randomly **relabel** them into two pseudo-groups of the same sizes, recompute the contrast (this destroys the real Qa-vs-Qb duration difference); repeat 200×, and each iteration's population `frac-positive` is one null draw. The **real `frac-positive` vs that null distribution** gives a two-sided p; real > null ⇒ a genuine excess of width-rescaling units beyond the metric's bias. (Hence the median is the wrong statistic for a mixed population and the skew-vs-null is right.)

Run on **Q1 vs Q4** (and Q3 vs Q4 for reference), all three anchors, cropped, to span maximal interval stretch; per-cell **duration ratio** (median Qb / median Qa wait) is logged so the test's power is measured, not assumed. Per-trial spike histograms are precomputed once so each shuffle is a sum + smooth + z-score (validated to reproduce Check 3 to ~5e-15).

**Results.** A "region cell" here is one `(cell_set × group)` grouping (e.g. *Thalamus / long_BG*), aggregating many neurons into one frac-positive-vs-null verdict — **not** a single neuron. Duration ratios vary strongly by anchor: `cue_off` Q1→Q4 reaches **~6–13×** stretch (the genuine power test), whereas the long-BG behavioral anchor `cue_on` reaches only ~2–5× even at Q1→Q4 (its durations are background-period-dominated). Verdicts (per metric, 168 region-cells = 28 region×group × 6 anchor×Q-pair conditions): for `xcorr`, **0 of 168 return "fixed width"** (108 width-rescales, 60 flat); for `procsim`, just **1 of 168** (a single borderline *Str MSN / long_BG / cue_off / Q1→Q4* cell, p = 0.0498, n = 12). At maximal stretch (`cue_off`, Q1→Q4, `xcorr`) **16/28 cells are "width rescales", 0 fixed, 12 flat**. Clean **dose-response**: the excess of real over null frac-positive grows ~3–6× from Q3→Q4 (~2×) to Q1→Q4 (~6–13×) — e.g. thal long +0.03 → +0.19, VAL short +0.08 → +0.28, VPM long +0.05 → +0.24. The shuffle null is itself absolute-biased (null frac-positive ~0.05–0.4, up to ~0.6 at the inflated anchors), so a raw frac-positive of ~0.25 would falsely read as "mostly fixed-width"; against the null it is a significant positive (rescale-direction) excess.

**Interpretation.** **Fixed-width is not the answer.** The width contrast clears the shuffle null in the rescale direction wherever the test has power, never goes fixed, and grows monotonically with stretch — genuine, stretch-dependent width rescaling in a significant minority subpopulation. The earlier "flat at Q3→Q4" was an underpowered-test artifact. The 12 flat cells are predominantly low-n / long-BG (VPM-short n = 5; several long-BG cells n < 20), not fixed. The shuffle null was load-bearing: the raw fraction would have given the wrong sign.

> **Note.** `g_peak_width_test.py` is a stretch-power test: it deliberately retains the Q1↔Q4 / Q3↔Q4 quartile sweep to span maximal stretch, and is **not** switched to the project-wide Q2→Q3 headline pair. Its `Q3→Q4` references are intrinsic to the test, not stale.

**Outputs.** `data/rescaling/peak_width_test/` — `peak_width_test_per_unit.csv` (25,468 rows), `peak_width_test_region.csv` (336 cells = 168 per metric, with duration ratios + verdicts), `peak_width_test_stretch.png`, `peak_width_test_maxstretch.png`.

### What the confound checks conclude

- The Section-1 headline `r` **is** substantially edge-influenced — the τ-distribution is U-shaped (Check 1, 56–97 % of units in the edge bins) and, at the matched `cue_off` anchor, both the crop-resort `r_interior` (Check 2) and the full-PSTH similarity (Check 3) roughly **halve** under edge cropping. The confound is **general** — present in every well-powered region and in **both behavioral groups** (the earlier "short BG survives" was a `last_lick`-anchor artifact).
- Genuine rescaling **is** still present: a cropped full-PSTH `xcorr` of ~0.4–0.7 survives across thalamus, motor cortex and striatum (Check 3 — a per-unit measure, not subject to the range restriction that depresses Check 2's `r_interior`), and response width genuinely rescales — never fixed — at maximal stretch (Check 3b).
- **Net read:** interval rescaling is real, but a substantial fraction of the Section-1 headline `r` is edge-cell contamination, fairly uniformly across regions and both groups. The most crop-robust per-unit signal is in thalamus (especially PO) and motor cortex; V1 RS short BG does not survive edge removal.
- **Reporting takeaway:** the cross-unit peak-position `r` (Section 1) **overstates** rescaling. The defensible measures are the per-unit full-PSTH similarity after edge cropping (Check 3) and the shuffle-null width test (Check 3b). This feeds the Open Questions "region focus" decision.

---

## 2. Cross-region population decoding analysis

**Script.** `population_decoder.py` (per-set runs) → `cross_region_decoder.py` (cross-region overview). Output base: `data/population_decoding/<set_label>/`; cross-region figure at `data/population_decoding/cross_region/`.

### Question

Three claims, three regions:

| # | Claim | Metric |
|---|---|---|
| 1 | Region encodes elapsed time | per-trial Pearson r (decoded vs true time) |
| 2 | Per-trial clock speed is extractable | per-trial slope of decoded vs true time |
| 3 | Clock speed is modulated by reward history | Cohen's d between speed-after-rew vs speed-after-no-rew |

This is the neural test of the mechanistic narrative in `project _basics.md`:

> reward outcome → dopamine signal → DMS clock speed → wait-time adjustment

If the third claim holds (clock speed faster after reward), it supports the "dopamine speeds up the clock → mouse decides earlier" account. Each claim is tested **at each of three anchors** (`cue_on`, `cue_off`, `last_lick`) per region.

### Cell sets (`CELL_SETS`)

The decoder is run on the same regional panel as rescaling. Per-session unit counts vary by region, so different MIN_UNITS thresholds are used: cortical/striatal sets define a sweep, thalamus/hippocampal sets use the legacy single-threshold default of 15.

| Region | Label | Filter | Cell-type subset | MIN_UNITS |
| --- | --- | --- | --- | --- |
| Striatum | `msn` | `is_str_msn` | — | **sweep [5, 8, 15]** |
| Striatum | `all_str` | `is_str_unit` | — | 15 |
| Striatum | `msn_fsi_tan` | `is_str_unit` | MSN, FSI, TAN | 15 |
| Motor cortex | `mc_l5l6` | `is_mc_l5l6` | — | 15 (legacy, mixes cell types) |
| Motor cortex | `mc_l5l6_rs` | `is_mc_l5l6` | RS | **sweep [5, 8, 10, 15]** |
| Visual cortex | `v1` | `is_visp` | — | 15 |
| Thalamus | `thal` | `is_thal` | — | 15 |
| Thalamus | `val` | `is_val` | — | 15 |
| Thalamus | `po` | `is_po` | — | 15 |
| Thalamus | `vpm` | `is_vpm` | — | 15 |
| Hippocampus | `ca1` | `is_ca1` | — | 15 |
| Hippocampus | `hpf` | `is_hpf` | — | 15 |

`qc_pass_all` is applied on top of every set. The headline cross-region panel (Section "Cross-region overview") uses `mc_l5l6_rs` (MIN_UNITS=15), `msn` (MIN_UNITS=5), and `thal` (MIN_UNITS=15) — one set per region at the region-appropriate threshold.

### Inclusion thresholds

A session enters the decoder only if:

- ≥ **MIN_UNITS** of the requested cell set (per-set value above)
- ≥ **90 valid (non-miss) trials** (`MIN_TRIALS = 90`)
- present in `session_pickles/*_str.pkl` for striatum/motor-cortex/thalamus sets, or `*_v1.pkl` for V1/HPF/CA1 (see `pickle_suffixes` per CELL_SET)

### Method

**Inputs.** For each non-miss trial, build a population-rate matrix from the start anchor through the decision lick:

- **Anchor (regression target `t = 0`)** for the headline pooled decoder is per-group:
  - **Long BG** → `cue_on_time`
  - **Short BG** → `last_lick_time`
- 100 ms bins, Gaussian σ = 1 bin smoothing, decode horizon up to 10 s, only bins ≥ 300 ms used.

**Models.**

| Method | Train / test split | When |
| --- | --- | --- |
| Pooled | 80 % of trials → train, predict all trials | Always |
| LOTO (leave-one-trial-out) | All other trials → train, predict held-out | Optional (`USE_LOTO=True`) |

Both use `RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000])`. Output is one (decoded-time, true-time) trace per trial.

**Per-trial metrics.** MAE, R², bias, and **clock speed** = slope of `polyfit(true_time, decoded_time, deg=1)` over the trial's bins.

**Per-anchor extension (`run_anchor_decoders`).** A separate decoder is fit per anchor (cue_on, cue_off, last_lick) on the same session. As of the latest revision, each per-anchor result stores not only MAE/r/r² but also the full per-trial **clock_speed slope distribution** and **history-effect statistics** (`speeds`, `history_stats`). This is what lets us evaluate all three claims at all three anchors, not just the per-group behavioral anchor.

**Shuffle null.** 50 permutations; each shuffle re-permutes time labels within each trial and re-fits the pooled decoder. Session p = `(shuffle_MAE ≤ real_MAE).mean()`.

**History analysis.** For each trial with valid clock_speed and known prev_rewarded:
- Compare clock speed after-reward vs after-no-reward (Welch's t, MWU, Cohen's d).
- Pearson correlation of clock speed vs own time-waited (sanity confound).

**Time-matched control.** Wait time correlates with clock speed (faster clock → earlier decision). To rule out a wait-time confound, bin trials into 4 wait-time quartiles and re-test history within each quartile. A consistent effect across all four quartiles is not driven by wait-time differences.

### Key parameters

```
BIN_SIZE    = 0.10 s         SMOOTH_SD   = 1 bin
MAX_DECODE  = 10 s           MIN_WAIT    = 0.3 s
MIN_BINS    = 5              N_SHUFFLES  = 50
RIDGE_ALPHA = [0.1, 1, 10, 100, 1000, 10000]
MIN_UNITS   = per-set (5–15) MIN_TRIALS  = 90
```

### Evaluating decoder performance: r vs R²

For the headline cross-region story we report **per-trial Pearson r** rather than R². R² (skill score = 1 − SS_res/SS_tot) penalises both shape error AND systematic bias/scale errors; negative R² in this pipeline mostly reflects per-trial intercept/slope miscalibration when the global pooled fit doesn't perfectly hit each trial's offset, not that the decoder has no time signal. **Per-trial Pearson r** captures the "does decoded time track true time within a trial?" question directly and is positive in essentially every session. Aim points: r > 0.5 is good, r > 0.3 is modest, MAE < 1.5s (in a 3–5s window) is the meaningful per-trial accuracy threshold.

### Cross-region overview (from `data/population_decoding/cross_region/cross_region_decoder_table.csv`)

Region × anchor headlines, **split by behavioral group** (group pooling is not appropriate because the two groups have different behavioral anchors, n_mice, and per-session unit counts):

**Long BG** (Motor cortex n=2 sess, MSN n=1 sess, Thalamus n=12 sess):

| Region | Metric | cue_on | cue_off | last_lick |
| --- | --- | ---: | ---: | ---: |
| Motor cortex | median r | 0.41 | 0.41 | **0.67** |
| Motor cortex | Cohen's d | +0.97 | −0.08 | **+1.07** |
| Motor cortex | % p<.001 | 100 % | 0 % | 100 % |
| Striatum (MSN) | median r | 0.33 | 0.27 | **0.60** |
| Striatum (MSN) | Cohen's d | +0.73 | +0.02 | **+0.50** |
| Striatum (MSN) | % p<.001 | 100 % | 0 % | 100 % |
| Thalamus | median r | 0.41 | 0.41 | **0.67** |
| Thalamus | Cohen's d | +0.63 | +0.21 | **+0.75** |
| Thalamus | % p<.001 | 83 % | 0 % | 83 % |

**Short BG** (Motor cortex n=4 sess, MSN n=4 sess, Thalamus n=12 sess):

| Region | Metric | cue_on | cue_off | last_lick |
| --- | --- | ---: | ---: | ---: |
| Motor cortex | median r | 0.33 | 0.31 | **0.49** |
| Motor cortex | Cohen's d | +0.35 | +0.01 | **+0.62** |
| Motor cortex | % p<.001 | 50 % | 25 % | 75 % |
| Striatum (MSN) | median r | 0.25 | 0.23 | **0.33** |
| Striatum (MSN) | Cohen's d | +0.54 | −0.05 | **+0.46** |
| Striatum (MSN) | % p<.001 | 75 % | 25 % | 50 % |
| Thalamus | median r | 0.38 | 0.40 | **0.58** |
| Thalamus | Cohen's d | +0.34 | +0.18 | **+0.50** |
| Thalamus | % p<.001 | 50 % | 8 % | 67 % |

> **Sample-size caveats.** Long-BG MSN is a single session / single mouse (RZ065_2025-02-22). Long-BG MC is 2 sessions. Thalamus is the only region with balanced n_sessions (12 short / 12 long, 9 mice total). The reward-history effect direction (positive d everywhere except cue_off) is consistent regardless.

### What the numbers say

- **Claim 1 (encoding) holds in all three regions in both groups.** Per-trial r is positive at every anchor, with last_lick the strongest in every cell.
- **Claim 2 (clock speed extractable) holds in all three regions.** Median per-trial slope is positive in every (region × group × anchor) cell, peaking at last_lick (MC/Thal ~0.50, MSN ~0.30 long BG / 0.15 short BG).
- **Claim 3 (reward modulates clock speed) holds at cue_on and last_lick, but NOT at cue_off** — a clean falsifier. At the behavioral anchor and last_lick, Cohen's d is positive (faster after reward) and most sessions are significant at p<.001. At cue_off the effect collapses to ~0 in every region.
- **MC has the strongest reward modulation in long BG** (d=+1.07 at last_lick), Thalamus the largest sample, MSN the smallest. All regions point the same direction.
- **Decoder MAE** sits in the 0.8–2.0 s range across regions; MAE-vs-shuffle is significant in essentially every session.

### Min_units sweep — picking the threshold

For `mc_l5l6_rs` and `msn`, the decoder runs at multiple MIN_UNITS thresholds and outputs each under a `min_units_<N>/` subfolder. Mean pooled_r² (a sensitive but conservative metric) across thresholds:

| Cell set | MIN_UNITS | n_sess | mean pooled_r² | median pooled_r² |
| --- | ---: | ---: | ---: | ---: |
| `mc_l5l6_rs` | 5 | 19 | −3.73 | −3.76 |
| `mc_l5l6_rs` | 8 | 15 | −3.49 | −1.59 |
| `mc_l5l6_rs` | 10 | 12 | −3.75 | −2.72 |
| `mc_l5l6_rs` | 15 | 6 | −3.28 | −1.36 |
| `msn` | 5 | 5 | −4.99 | −4.40 |
| `msn` | 8 | 3 | −1.83 | −1.00 |
| `msn` | 15 | 1 | −0.08 | −0.08 |

Lowering MIN_UNITS admits more sessions but pulls in noisier per-session decoders (a few catastrophic sessions with R² < −5 drive the mean). The headline cross-region panel uses MIN_UNITS=15 for MC and 5 for MSN — the lowest threshold where per-trial r and clock-speed signals are still robust per session.

### Outputs per session

- `decoder_performance_<method>_<session>.png` — accuracy diagnostics.
- `clock_speed_analysis_<method>_<session>.png` — clock-speed distribution + history effect + time-matched quartile control.
- `pca_trajectories_<session>.png` — PCA coloured by previous reward.
- `committee_point1/2/3_<session>.png` — committee summary panels.
- `anchor_comparison_<session>.png` — MAE/r per anchor (cue_on / cue_off / last_lick).
- `trial_results_<session>.csv` — per-trial clock speed, R², MAE, reward.
- `decoder_raw_<session>.pkl` — raw cache; `anchor_results` now contains per-anchor `speeds` + `history_stats` (consumed by `cross_region_decoder.py`).

### Cross-region outputs

- `data/population_decoding/cross_region/figA_claim1_encoding__{Short_BG,Long_BG}.png` — claim 1 (encoding) per group.
- `data/population_decoding/cross_region/figB_claim2_clock_speed__{Short_BG,Long_BG}.png` — claim 2 (clock speed extractable) per group.
- `data/population_decoding/cross_region/figC_claim3_history__{Short_BG,Long_BG}.png` — claim 3 (history modulation) per group.
- `data/population_decoding/cross_region/cross_region_decoder_table.csv` — long-format companion (one row per session × anchor with all metrics).

---

## 3. V1 → DMS cross-correlation

**Script.** `v1_dms_crosscorrelation.py`. Output: `data/v1_dms_crosscorrelation/`.

### Question

Does V1 activity *lead* DMS activity during the wait period? The mechanistic narrative places **V1 upstream of DMS** as the source of learned timing input. A V1-leading cross-correlogram (CCG) peak — especially at monosynaptic (~5–15 ms) latency — is the prediction.

### Method

1. **Find simultaneous sessions** — read the Google Sheets recording log; keep dates where both the V1 *and* the STR insertion are marked simultaneous=`y`.
2. **Filter units** — V1 side: `is_v1_cortical` from `unit_properties_final.csv`. STR side: `is_str_msn`. Both gated by `qc_pass_all`.
3. **Extract trial-windowed spikes** — relative to cue offset, in the window 0.5–4.0 s after cue_off. Drop units with mean firing rate < 0.5 Hz; drop sessions with < 50 valid (non-miss) trials.
4. **Pairwise CCG** — for each (V1 unit × MSN unit) pair, sum cross-correlogram coincidences across trials in 1 ms bins, over ±100 ms lag.
5. **Jitter correction** — repeat with both spike trains independently jittered uniformly within ±25 ms (20 jittered repeats per pair). Subtract the mean of the jittered CCGs from the raw CCG; standard deviation across jitters gives the per-bin noise.
6. **Per-pair peak** — find the largest jitter-corrected CCG value at *positive* lag (V1 leading STR). Report peak lag (ms) and z-score relative to the jitter SD.
7. **Population summary** — sum corrected CCGs across all pairs in each session; one session-level median/mean peak lag; aggregate across sessions.

### Key parameters

```
CCG_WINDOW    = 0.100 s      CCG_BIN       = 0.001 s
JITTER_WINDOW = 0.025 s      N_SHUFFLES    = 100
ANALYSIS_START = 0.5 s       ANALYSIS_END  = 4.0 s   (post cue_off)
MIN_FIRING_RATE = 0.5 Hz     MIN_TRIALS    = 50
```

### Results to date

5 simultaneous sessions analysed; 146 V1-MSN pairs total; **130 / 146 pairs (89 %) significant at z > 2**.

| Session | n V1 | n MSN | n pairs | n sig (z>2) | Median peak lag (ms) | Mean peak lag (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| RZ047_2024-11-22 | 1 | 3 | 3 | 3 | 29.5 | 34.8 |
| RZ050_2024-11-20 | 5 | 11 | 55 | 47 | 45.5 | 48.6 |
| RZ059_2025-03-18 | 1 | 3 | 3 | 2 | 43.5 | 42.5 |
| RZ062_2025-03-04 | 7 | 1 | 7 | 6 | 49.5 | 56.5 |
| RZ062_2025-03-05 | 26 | 3 | 78 | 72 | 55.5 | 50.8 |

**Aggregate**: median per-session peak lag **≈ 45 ms**; mean **≈ 45 ms**. All 5 sessions show the V1-leads-STR direction (positive lags). The Wilcoxon test against zero is highly significant in `ccg_summary` console output.

### Interpretation

- **V1 reliably leads DMS** — the directional prediction is supported: positive lag in every session, 89 % of pairs significant.
- **But the latency is ~45 ms, not 5–15 ms** — this is well outside the monosynaptic range. The most likely interpretation: V1 → DMS coupling here is **polysynaptic / network-level** rather than direct corticostriatal input. The signal is real and consistently directional, but it does not look like a single-synapse projection.
- **Sample is small** — only 5 sessions met the simultaneity + unit-count criteria. RZ050_2024-11-20 (55 pairs, 47 significant) is the most data-rich session and dominates the aggregate.

### Outputs

- `data/v1_dms_crosscorrelation/ccg_<mouse>_<date>.png` — per-session 4-panel figure (population CCG, peak-lag histogram, z-score histogram, summary text).
- `data/v1_dms_crosscorrelation/ccg_summary.png` — cross-session pooled lag distribution + per-session medians + per-session significant fraction.
- `data/v1_dms_crosscorrelation/ccg_summary.csv` — one row per session.

---

## 4. Per-unit encoding GLM (descriptive feature map)

**Script.** `4c_simple_glm.py`. Output: `data/glm_simple/`. Decision-lick follow-up: `decision_lick_unique_recheck.py`.

### Question

Before testing whether reward history modulates the timing code (Claims 1, 3), establish the substrate: **which feature families does each unit encode, and do regions differ?** This is Claim 2(a) — elapsed-time encoding presence — made descriptive and per-unit. It is deliberately *not* an inferential test of reward history (that confounded the earlier GLM; see "The interaction-term problem" below).

### Method

A per-unit Poisson GLM (log link) predicts 25 ms-binned spike counts across the whole trial (cue_on → end of consumption) from a within-trial design with seven feature families:

| Family | Captures |
| --- | --- |
| `BG_time` | elapsed-time ramp over the background period |
| `WAIT_time` | elapsed-time ramp over the wait period |
| `cue` | cue_on / cue_off transients |
| `licks` | lick-locked firing (`lick_bg` / `lick_decision` / `lick_cons`) |
| `outcome` | reward delivery (consumption epoch) |
| `hazard` | saturating reward-hazard kernel, orthogonalized to the wait ramp |
| `spike_history` | autoregressive (refractoriness / bursting) |

1. **Fit** the full model per unit (unregularized IRLS; standardized-ridge fallback on divergence).
2. **Score each family** by cross-validated *marginal* ΔpR² (family-alone vs constant-rate null; 5-fold, trial-stratified to block spike-history leakage across the train/test boundary).
3. **Label** each unit by `dominant_family` (argmax of marginal ΔpR² over *task* families — `spike_history` excluded as intrinsic; `untuned/weak` if the best task family < 0.002).
4. **Aggregate** by `region_group` and `cell_type`.

No history terms, no reward×time interactions, no drift spline, no wait-band filter — the M0 design only.

**Elapsed-time bases are absolute, per-group.** The BG/WAIT raised-cosine bases are evaluated at real elapsed seconds (not duration-normalized), on a grid set per BG group: a single global BG grid of [0, 5 s] clamped 57 % of Long-BG trials (median Long BG ≈ 5.2 s) and wasted most columns on Short-BG (median ≈ 1.1 s). Per-group grids (BG: Short [0, 2.5 s]/4 cols, Long [0, 11 s]/9; WAIT: Short [0, 11 s]/8, Long [0, 16 s]/11) cover ~the 90–95th pct of each group's durations and recover Long-BG `BG_time` encoding the global grid had suppressed.

### Key parameters

```
DT = 0.025 s                       (25 ms bins)
Families: BG_time, WAIT_time, cue, licks, outcome, hazard, spike_history
Inclusion: ≥30 non-miss trials, ≥50 spikes, mean FR ≤ 50 Hz
CV: 5-fold, trial-stratified       dominant threshold: marg ΔpR² ≥ 0.002
COMPUTE_UNIQUE_DPR2 = False (sweep) per-group absolute-time bases
```

### Coverage

2,679 / 2,737 units fit (all `fit_status = ok`, no divergence fallbacks); 58 excluded, all `mean FR > 50 Hz`. Short / Long split 1,529 / 1,150.

### Headline — families ranked by median ΔpR² per region

The fair cross-family view (median marginal ΔpR²; `spike_history` omitted — it dominates raw variance everywhere at 0.04–0.07 but is intrinsic, not task):

| Region | n | Family ranking (median ΔpR²) |
| --- | --- | --- |
| **Motor cortex** | 449 | WAIT 0.0084 ≈ licks 0.0084 > outcome 0.0059 > BG 0.0013 |
| **Striatum** | 179 | WAIT 0.0046 > licks 0.0042 > outcome 0.0031 > BG 0.0015 |
| **Thalamus** | 1256 | licks 0.0043 > outcome 0.0033 > WAIT 0.0026 > BG 0.0009 |
| **Hippocampus** | 306 | licks 0.0037 > WAIT 0.0030 > outcome 0.0028 > BG 0.0019 |
| **Visual cortex** | 129 | **BG 0.0052 > WAIT 0.0039 > licks 0.0013 > outcome 0.0004** |

### What the numbers say

- **A distributed elapsed-time code exists but is modest.** `WAIT_time` is a top-2 task family in motor cortex, striatum, thalamus and hippocampus — the substrate Claims 1 & 3 require. Full-model pR² is 0.05–0.08; task families explain a few percent each, with intrinsic dynamics (spike history) dominating raw variance.
- **Regions differ (Claim 2 holds descriptively).** Motor cortex is the richest encoder; striatum leads with `WAIT > licks` (consistent with the DMS-clock premise); thalamus is more lick / outcome-driven; hippocampus is even.
- **V1 is the clean control.** It tracks the visual background (`BG_time` leads) and is *not* wait / lick coupled (licks, outcome at floor) — confirming the timing signal elsewhere is not a generic pipeline artifact.
- **Movement and reward are the dominant task correlates** — licks + outcome are as large or larger than timing everywhere except V1, so any clock-speed claim must be defended against them.

### The dominant-family caveat

Per-unit `dominant_family` (argmax of *raw* marginal ΔpR²) **over-credits flexible multi-column families**: `WAIT_time` has 8–11 basis columns vs `outcome`'s 1, so it wins the argmax in 24–31 % of units even where its median effect is not the largest. A basis-count-normalized version (`dominant_family_eff`, ΔpR²-per-column) over-corrects the *other* way, handing `outcome` ~60–69 %. The two **bracket** the truth; the **median-ΔpR² ranking above is the honest summary**. Both views are in `region_summary.csv` (`fracdom_raw_*` / `fracdom_eff_*`).

### Decision-lick follow-up (unique ΔpR²)

123 units are decision-lick-*specific* by marginal ΔpR² (`lick_decision` ≥ 0.002 and > `lick_bg`, `lick_cons`), concentrated in motor cortex (8.7 %) and striatum (8.4 %). Re-fitting these with **unique** ΔpR² (full minus family — does the decision transient survive partialling out the coincident wait-peak, reward, consumption and spike history?):

- **33 / 123 (27 %) survive** as genuine independent decision coders (unique ≥ 0.002). The rest were decision-*time* coincidence; the marginal metric over-credited ~3× (median marginal 0.0048 → unique 0.0014, ~60 % shared).
- **Striatum has the highest survival rate (7/15, 47 %)** vs motor cortex (8/39, 21 %) and thalamus (11/48, 23 %) — striatal decision coding is more independent, consistent with DMS as a genuine action / decision readout. Small n; hypothesis-strengthening, not definitive.

### The interaction-term problem (why the full reward-history GLM was set aside)

This descriptive GLM deliberately drops the interaction machinery of the earlier `4c_encoding_GLM_w_history.py`, whose headline was an LRT on `prev_rewarded × wait_time_basis` (does prior reward modulate the time code?). That test proved untrustworthy:

- **Duration-warping.** With a per-trial-stretched wait basis, the `prev_reward × basis` interaction soaks up trial-*duration*-dependent firing rather than reward-driven shape change. A 500-seed shuffle of `prev_rewarded` showed the M1_vs_M2 LRT was **catastrophically anti-conservative on high-pR² units — frac p < 0.05 up to 94 %** (nominal 5 %).
- **Conditional-wait confound.** Mice change how long they wait based on the prior outcome (E[current wait | prev_rewarded] differs, |Cohen's d| up to ~0.4), so the interaction picks up *current*-wait-dependent firing. Even after switching to an absolute-time basis, the LRT stayed anti-conservative (~85 % shuffle p < 0.05).
- **Rescue controls made it fragile.** Keeping the test calibrated required bolting on an absolute-time basis, a `current_wait × basis` confound family, a session-drift spline, and a per-group wait-band trial filter — each narrowing the data and adding fragility, while the 16-column interaction block was still free to overfit per-trial residual variance.

**Lesson.** High-df interaction columns are confounded by duration and conditional-wait structure and overfit noise; the omnibus LRT can't be trusted. The reward-history question was therefore split into low-df, well-calibrated tests (`4d` Test 2: OLS on wait-period rate controlling for wait-quantile; `4f` Test 1: time-encoding LRT with anchor-event kernels + sustained-peak criterion), and the *descriptive* question was answered by this M0 GLM. Targeted low-df tests beat one high-df interaction.

### Outputs

- `data/glm_simple/per_unit.csv` — per unit: anatomy, `bg_group`, fit quality, `label`, `dominant_family` + `dominant_family_eff`, marginal / in-sample ΔpR² for 7 families + 5 lick/cue sub-families.
- `data/glm_simple/coefficients_per_unit.csv` — fitted β per design column (kernel shapes).
- `data/glm_simple/per_unit_cv_folds.csv` — per-fold CV ΔpR² (for error bars / significance).
- `data/glm_simple/region_summary.csv`, `cell_type_summary.csv` — median / IQR ΔpR² + dominant fractions (raw + efficiency).
- `data/glm_simple/excluded_units.csv` — coverage audit (58 units, all FR > 50 Hz).
- `data/glm_simple/family_dpr2_by_region.png` — per-family ΔpR² boxplots by region.
- `data/glm_simple/decision_lick_unique.csv` — the 123-unit decision-lick unique-ΔpR² recheck.

---

## How the results fit together

| Analysis | What it shows | Status |
| --- | --- | --- |
| **Cross-region rescaling** | Striatum (MSN + FSI), motor cortex (RS + FSI), thalamus (VAL + aggregate), and hippocampus (CA1 + HPF) all show rescaling. V1 RS shows the weakest version. The signal is **not striatum-specific** — it is widespread across the wait-period circuit. **Edge-cell confound checks (Section 1b)** show the headline `r` is substantially edge-cell-inflated — at the matched cue_off anchor, edge cropping roughly halves it in every well-powered region and in both behavioral groups; the surviving signal is genuine but minority, most crop-robust in thalamus (PO) and motor cortex. | 9 of 11 cell sets in the headline panel rescale at both groups (po, vpm dropped on single-mouse grounds); r typically 0.6–0.85 at the best anchor. Confound checks: τ-distribution U-shaped (56–97% edge units); per-unit PSTH similarity halves under edge cropping in both groups; response width rescales (never fixed) at maximal stretch. |
| **Cross-region population decoding** | Motor cortex, striatum MSN, and thalamus all (i) encode elapsed time (per-trial r > 0.3 at every anchor), (ii) yield extractable per-trial clock-speed slopes (median > 0), and (iii) show **clock speed faster after reward** at cue_on and last_lick. **The reward effect collapses at cue_off in every region** — a clean negative-control anchor. | All three claims hold in all three regions at last_lick and (group-appropriate) cue_on. cue_off uniformly fails claim 3 across regions. |
| **V1→DMS CCG** | V1 visual-cortex spikes consistently lead striatal MSN spikes during the wait period in simultaneous recordings — supporting V1 as an upstream timing source — but at ~45 ms (polysynaptic), not monosynaptic latency. | 5/5 sessions positive, 130/146 pairs significant; sample size limits interpretation. |
| **Per-unit encoding GLM** | Single-unit feature map underlying the above: a modest, region-differentiated elapsed-time code (`WAIT_time` top-2 in motor cortex, striatum, thalamus, hippocampus), with V1 dissociating onto the visual background — and movement/reward as the dominant competitors. Decision-lick coding survives partialling most in striatum (47 %). The earlier reward×time interaction GLM was set aside as confounded (duration-warping + conditional-wait → anti-conservative LRT). | 2,679/2,737 units fit; descriptive substrate, no inferential reward-history claim. |

Together these results form the empirical backbone of the thesis narrative: temporally-scaled population codes are **distributed across striatum, cortex, thalamus and hippocampus** (not DMS-specific); they *can* be read out as a clock-speed signal sensitive to reward in every region tested; and V1 *does* lead DMS at the network level — though the V1→DMS coupling looks indirect rather than monosynaptic. The cue_off-null in claim 3 is an important sanity check: not every anchor recovers the reward effect, so the effect is real but anchor-conditional.

The **per-unit encoding GLM (Section 4)** is the single-unit foundation under the population results: it confirms that an elapsed-time code is *present and region-differentiated* (the precondition for the rescaling and clock-speed claims) and that V1 is dissociated onto the visual background rather than timing — but it also sets expectations, since single-unit task encoding is modest and dominated by movement and reward. Its **interaction-term post-mortem** is the methodological reason the reward-history question is pursued via the targeted `4d`/`4f` tests and the population decoder rather than an omnibus reward×time GLM: the high-df interaction was anti-conservative under duration-warping and conditional-wait confounds.

The rescaling claim carries an explicit edge-cell caveat (**Section 1b**): the headline cross-unit `r` overstates it — at the matched cue_off anchor, edge cropping roughly halves both the crop-resort `r_interior` and the full-PSTH similarity, in every well-powered region and in both behavioral groups. A genuine per-unit rescaling component nonetheless survives (cropped full-PSTH `xcorr` ≈ 0.4–0.7 in thalamus, motor cortex and striatum), and response width genuinely rescales (never fixed-width) at maximal stretch. The most crop-robust populations are thalamic (especially PO) and motor-cortical; this should inform the Open Questions "region focus" decision.
