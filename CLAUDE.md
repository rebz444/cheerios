# Per-unit reward-history & time-encoding analyses

## Background and current state

These three scripts replace the original per-unit Poisson GLM
(`4c_encoding_GLM_w_history.py`) as the primary tests for the reward-history
and time-encoding questions. The 4c GLM is retained for legacy reasons but
should **not** be used as the headline analysis: its M1_vs_M2 LRT is
confounded by duration-warping in the wait-time basis (validated by stratified
analysis + 500-seed shuffle calibration showing it can be catastrophically
anti-conservative on high-pR² units — frac p<0.05 up to 94% under shuffle).

The three replacement scripts split the original question into pieces that
each have clean, well-calibrated inferential machinery:

| Script | Test | Question | Per-unit statistic |
|---|---|---|---|
| `4d_simpler_reward_history_test.py` | Test 2 | Does reward shift baseline wait-period firing rate, controlling for current wait duration? | `beta_pr_cond` (Hz), `p_pr_cond` |
| `4e_time_encoding_test.py` | Test 1 (cue_off only) | Does this unit encode time since cue_off, beyond decision-locked and spike-history activity? | `chi2_wait`, `p_wait` (LRT, df=8) |
| `4f_time_encoding_two_anchors.py` | Test 1 (cue_off and cue_on) | As 4e, but tests both anchors. Headline is `sustained_sig_{anchor}` (LRT + peak_lag filter). | `chi2_wait_{anchor}`, `p_wait_{anchor}` (LRT, df=7), `peak_lag_s_{anchor}`, `sustained_sig_{anchor}` |

`4e` is largely superseded by `4f`. Use `4f` for the time-encoding test
going forward; keep `4e` available as a reference single-anchor version.
**4e has not received the 4f design improvements** (anchor-event kernels,
wait_onset t_min shift, rank-reduced basis). Don't compare 4e and 4f
headline numbers directly.

## CLI for all three scripts

Each script supports the same modes:

```bash
python <script>.py debug <session_id> <unit_id>     # single-unit smoke test
python <script>.py run [session_id ...]              # full run; all sessions if none specified
python <script>.py summary                           # region-level summary from existing per-unit CSV
python <script>.py plot                              # effect-size distribution plot
```

`run` automatically calls `summarize` and `plot` at the end. Outputs land in
the directory configured at the top of each script (`OUT_DIR`).

## What the tests do

### Test 2 (`4d_simpler_reward_history_test.py`)

For each unit:
1. Compute per-trial mean firing rate in the wait window (cue_off → decision)
2. Bin trials by current wait-duration into 4 quantiles
3. Fit OLS: `rate ~ prev_rewarded + C(wait_quantile_bin)`
4. Report `beta_pr_cond` (effect size in Hz), `p_pr_cond`, plus an
   unconditional version (`beta_pr_uncond`) for comparison and an
   interaction-diagnostic `p_interaction` (effect varies by quantile?)

The wait-quantile fixed effect controls for the duration-confound that
broke 4c. The test is intentionally low-df (1 df per unit for the headline),
robust, and easy to interpret.

### Test 1 (`4f_time_encoding_two_anchors.py`)

For each unit, fits **two separate** Poisson GLMs (one per anchor):

**cue_off anchor**:
- Window: `[cue_off_time, decision_time]` (wait period only)
- Design:
  - **wait_onset basis** (forward from cue_off, log-spaced raised cosines over
    `[T_MIN_WAIT_S=0.5s, 10s]`; 8 columns built, then constant-orthogonalized
    and last-column-dropped → 7 effective columns, exactly zero in `[0, 0.5s]`)
  - **cue_off_event kernel** (forward from cue_off, 5 linear, 0-500ms) —
    absorbs the brisk transient at the anchor so wait_onset measures only
    sustained encoding
  - decision_lock basis (backward from decision, 5 linear, 0-2s)
  - spike_history basis (5 log-spaced, 0-200ms)

**cue_on anchor**:
- Window: `[cue_on_time, decision_time]` (BG + wait combined)
- Design:
  - **wait_onset basis** (forward from cue_on, log-spaced over
    `[T_MIN_WAIT_S=0.5s, 15s]`; 7 effective columns after constant-orthog)
  - **cue_on_event kernel** (forward from cue_on, 5 linear, 0-500ms) —
    anchor transient absorber
  - cue_off event kernel (forward from cue_off, 5 linear, 0-500ms) —
    mid-window transient absorber
  - lick_bg kernel (forward from each BG lick, 5 log-spaced, 0-300ms, summed)
  - decision_lock basis (backward from decision, 5 linear, 0-2s)
  - spike_history basis (5 log-spaced, 0-200ms)

Per anchor, the LRT compares M_full (with wait_onset) vs M_null (without).
**df = 7** (the wait_onset basis is rank-reduced by one for identifiability
against the intercept — see "Design history" at the end of this doc).

**Headline statistic is `sustained_sig_{anchor}`, not raw `p_wait_{anchor}`.**
A unit is "sustained-significant" iff `p_wait < LRT_ALPHA` AND
`peak_lag_s >= SUSTAINED_PEAK_LAG_S` (default 1.0s). This filters out
units whose wait_onset kernel peaks right at the basis floor (0.5s) —
those are typically anchor-locked sensory tails extending past the event
kernel, not wait-period time encoders. Validated on RZ063 str+v1: V1 went
from 100% LRT-sig (with median χ²≈500 and all peaks at 0.5s) to 0%
sustained for cue_off, while Motor and Thalamus retained 40-50%.

**Why two anchors**: Long BG mice time from cue_on (their behavioral anchor);
Short BG mice time from last_lick (not implemented here yet; cue_off is the
closest proxy and is the cleanest "no events in window" window). Running both
captures different timing strategies. The cross-tabulation in the summary
shows which units pass which anchor — biologically informative even within
the same region.

### Per-unit kernel output

Both Test 1 scripts save fitted wait-kernel coefficients to a separate CSV
(`per_unit_kernels.csv`) so kernel shapes can be inspected after the run:

- For 4e: one row per (unit, basis_idx) — n_units × 8 rows (old N_BASIS_WAIT)
- For 4f: one row per (unit, anchor, basis_idx) — n_units × 2 × 7 rows
  (post-rank-reduction)

The fitted kernel function over time is `sum_j beta_j × B_j(t)`, evaluated
on the precomputed basis grid (use `precompute_all_bases()` and
`bases['<anchor>']['wait']`). The basis is exactly zero for `t < 0.5s`,
so plotted kernels show a flat zero segment at the start.

## How to validate the pipeline before scaling

These have all been done; document for context:

1. **Diagnostic debug on a known unit** — e.g. VAL u180 in
   RZ063_2025-03-05_str (Thalamus / high_FR per `corrected_region`).
   Time-encoding signal is real here; both anchors should fire with
   sensible kernel shapes (peak ~1.85s for cue_off, ~2.2s for cue_on;
   smooth ±0.06–0.07 log-rate units; basis-floor-zero for t<0.5s).
   Use **integer `id`** (180) as the CLI arg, not the string `unit_id`.

2. **Cross-region sanity check** — run on RZ063_2025-03-05 (str + v1
   probes). After applying anatomical labels via `corrected_region` /
   `region_group` (from `unit_properties_final.csv`), expect the
   dissociation pattern using **`frac_sustained_{anchor}` (not the raw
   LRT `frac_sig_fdr_{anchor}`)**: Motor cortex / Thalamus ~40-50%
   sustained; Visual cortex 0% sustained for cue_off (V1's high LRT-sig
   is the visual OFF tail extending past the event kernel — peak at the
   0.5s basis floor).

3. **Shuffle calibration** — sanity check that under shuffled prev_rewarded,
   the Test 2 LRT gives roughly-uniform p-values (or at least not
   anti-conservative). Test 1 isn't a prev_rewarded test, so shuffle isn't
   directly applicable; instead validate via the diversity panel.

If anything in the debug or cross-region check looks off, **fix before
scaling**. The earlier 4c pipeline had a 95% striatum hit rate that turned
out to be confound, which we only caught with within-pipeline shuffle and
stratification. Don't trust large effects until they've been stress-tested.
The 4f anchor-event-kernels + t_min shift + sustained-criterion design
emerged from exactly this kind of cross-region stress test — the original
naive design gave V1 100% sig with median χ²=496, all peak<0.5s.

## Gotchas and conventions

- **Anatomical labels**: probe-target labels in `units_vetted.csv` are
  unreliable (e.g. RZ063_2025-03-05_str probe contains 47 thalamus + 42 motor
  cortex + 0 striatum). Use `corrected_region` and `region_group` from
  `unit_properties_final.csv`. The latter has **no `session_id` column** —
  it's keyed by `(mouse, date_only, insertion_number, id)`. The 4d/4e/4f
  `load_anatomical_labels` helpers join it to `units_vetted.csv` to attach
  session_id; downstream code then keys on `(session_id, id)`. The summary
  functions default to `region_group` and fall back to `probe_region` only
  if labels are missing. Always exclude `region_group in ('Excluded',
  'Other')` from headline tables.

- **Unit inclusion**: min 30 valid (non-miss) trials, min 0.5 Hz mean firing
  rate, min 0.10 s wait duration per trial. Configurable at the top of each
  script.

- **Test 2 anchor**: currently only `cue_off → decision`. Group-specific
  anchors (last_lick for Short BG mice) is a known follow-up — not blocking.

- **Test 1 last_lick anchor**: not implemented in 4f. If you want it, it'd
  need to derive last_lick time from `events` (last lick_bg of trial, or
  previous trial's last consumption lick if no BG licks). For now, cue_off
  is the de-facto Short-BG-friendly anchor since it's the clean "no events
  in window" window.

- **FDR**: BH within `region_group`, q=0.05 by default. Don't apply FDR
  across regions globally; that would deflate fraction-significant in
  large-n regions disproportionately.

- **No regularization** in the LRT fits. Test 1 uses unregularized IRLS
  (Pillow-style). Wilks' theorem applies cleanly; no ridge bias. Test 2 uses
  OLS, also unregularized.

- **wait_onset basis identifiability**: the raised-cosine basis (50% overlap)
  satisfies `Σⱼ Bⱼ(t) ≈ 1` across the bulk of its support, which is
  collinear with the intercept. 4f handles this by (a) subtracting the
  column-mean over `[T_MIN_WAIT_S, t_max]` and (b) dropping the last column.
  Result: rank-7 (was 8) basis, no constant direction, LRT df=7. The fix
  doesn't change the LRT chi² for valid fits (the deviance difference is
  invariant to the constant absorbed by the intercept), but makes individual
  betas and the kernel plot physically interpretable.

- **Sustained-encoder criterion** (`sustained_sig_{anchor}` on
  `per_unit.csv`): `p_wait_{anchor} < LRT_ALPHA` AND
  `peak_lag_s_{anchor} >= SUSTAINED_PEAK_LAG_S` (default 1.0s). Report this
  as the headline fraction-significant, not the raw LRT. The peak-lag filter
  catches the residual "anchor-locked sensory tail past the event kernel"
  failure mode (visible in V1 even after event kernels are added).

- **Output paths** are configured per-script at the top:
  - 4d → `<DATA_DIR>/simpler_reward_history_test/`
  - 4e → `<DATA_DIR>/time_encoding_test/`
  - 4f → `<DATA_DIR>/time_encoding_two_anchors/`

  Each produces `per_unit.csv`, `region_summary.csv`, and a distribution
  plot. Test 1 scripts additionally save `per_unit_kernels.csv`. `INCREMENTAL_SAVE`
  is True by default; intermediate writes happen every 25 units within a
  session, so a crashed run keeps partial results.

## Suggested workflow

1. **Debug pass on a known unit** before any new analysis run:
   ```bash
   python 4f_time_encoding_two_anchors.py debug RZ063_2025-03-05_str 180
   ```
   Check that both anchors fit, kernel plots look reasonable, p-values are
   small. Same for `4d`:
   ```bash
   python 4d_simpler_reward_history_test.py debug RZ063_2025-03-05_str 180
   ```

2. **Cross-region session pair** to confirm dissociation:
   ```bash
   python 4f_time_encoding_two_anchors.py run RZ063_2025-03-05_str RZ063_2025-03-05_v1
   python 4d_simpler_reward_history_test.py run RZ063_2025-03-05_str RZ063_2025-03-05_v1
   ```
   Check both `region_summary.csv` outputs for expected anatomical dissociation
   patterns.

3. **Cross-tabulate Test 1 and Test 2** per unit using the helper script:
   ```bash
   python cross_tab_test1_test2.py
   ```
   Produces `cross_tab_{anchor}_{criterion}.csv` and a stacked-bar plot
   under `<DATA_DIR>/cross_tab_test1_test2/`. Reports the 2×2 (only-T1,
   both, only-T2, neither) per region for both anchors and both
   criteria (raw LRT and `sustained_sig`). The recommended headline is
   `cue_off / sustained` (cleanest Test 1 criterion, anchor uses the
   shortest event-free window).

4. **Scale to all sessions** only after the cross-region check is clean:
   ```bash
   python 4f_time_encoding_two_anchors.py run
   python 4d_simpler_reward_history_test.py run
   ```
   Expect overnight compute. `INCREMENTAL_SAVE` lets you resume after
   crashes — the loop skips units whose `(session_id, unit_id)` is already
   in the per-unit CSV if you re-run (note: the current scripts don't have
   explicit resume detection; check whether you need to add it for very long
   runs).

## When to ask the user before changing things

- Anything that changes the design matrix structure (adding/removing
  predictor families, changing basis parameters). Re-validation needed
  after design changes.
- Changing inclusion criteria. Can shift which units are in the headline
  analysis.
- Changing the LRT comparison (M_null vs M_full structure).
- Switching anchor definitions or window boundaries.

Lower-stakes changes that don't need permission:
- Bug fixes in helper functions.
- Output formatting / column reordering in CSVs.
- Plot styling.
- Adding diagnostic columns to per-unit CSV (as long as headline columns are
  preserved).

## Repository structure

```
4d_simpler_reward_history_test.py    # Test 2
4e_time_encoding_test.py             # Test 1 single-anchor (reference;
                                     # NOT updated with anchor-event /
                                     # t_min / sustained-criterion fixes)
4f_time_encoding_two_anchors.py      # Test 1 two-anchor (primary)
cross_tab_test1_test2.py             # CLAUDE.md step 3: 2x2 per-region
                                     # Test 1 × Test 2 cross-tabulation
4c_encoding_GLM_w_history.py         # Original GLM (deprecated as headline,
                                     # retained for descriptive shape analyses
                                     # on units passing Test 1 + Test 2)
```

## Design history

The 4f design above is the validated version. The naive starting design
(wait_onset on `[0, t_max]`, no anchor-event kernel, no peak-lag filter)
had three failure modes uncovered during the RZ063 str+v1 cross-region
validation:

1. **Raised cosines sum-to-constant** → collinear with intercept.
   Symptom: cue_on kernel plot flat at +112 log-rate units, intercept −112
   (they cancel). Fix: zero-mean basis columns over modeled support and
   drop one (rank-7).

2. **wait_onset(t≈0) absorbs the anchor transient** instead of measuring
   sustained encoding. Symptom: V1 came back 100% LRT-sig with median
   χ²≈500, all peak<0.5s — entirely visual-onset transients. Fix: add
   anchor-event kernels (cue_off_event for cue_off anchor; cue_on_event
   for cue_on anchor) over 0-500ms.

3. **Residual collinearity** between event kernels and the earliest
   log-spaced wait_onset bumps (which peak at 0–0.6s with the original
   `WAIT_BASIS_LOG_OFFSET=0.05` and `t_max=10s`). Symptom: V1 chi² stayed
   high after event kernels alone; wait_onset kernels showed unstable
   ±3-6 log-rate oscillation near t=0 (variance partitioning). Fix:
   shift wait_onset basis support to `[T_MIN_WAIT_S=0.5s, t_max]` so the
   event kernel exclusively owns `[0, 0.5s)`.

Even after (1-3), some regions still have sensory tails extending past
0.5s that the wait_onset basis catches (visible in V1 cue_on: 43%
sustained-sig with peak at 0.5s exactly). The `sustained_sig_{anchor}`
criterion (`peak_lag >= 1.0s`) gates these out at interpretation time.
The 4f cross-region check on RZ063 produces:

| Region | n | frac_sig_fdr cue_off | frac_sustained cue_off |
|---|---|---|---|
| Visual cortex | 7 | 100% | 0% |
| Motor cortex | 31 | 47% | 40% |
| Thalamus | 40 | 45% | 50% |

If your design changes alter these numbers materially, re-investigate.

External dependencies all three scripts use:
- `utils.get_session_data(session_id)` — returns (events, trials, units) tuple
- `paths` module — provides `LOGS_DIR`, `DATA_DIR`
- Standard packages: numpy, pandas, statsmodels, scipy, matplotlib
