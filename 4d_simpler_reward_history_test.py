#!/usr/bin/env python3
"""
4d_simpler_reward_history_test.py

Per-unit test: does previous-trial reward affect mean wait-period firing rate,
controlling for current-trial wait duration?

Motivation
----------
The GLM-based M1_vs_M2 test (4c_encoding_GLM_w_history.py) was contaminated by
duration-warping: the wait-time basis is stretched to each trial's wait window,
so `prev_rewarded × wait_basis` interactions covary with trial duration rather
than with reward-driven shape changes. Stratified and shuffle diagnostics
confirmed this.

This script replaces the GLM headline with a much simpler test that
sidesteps the duration confound directly:

  For each unit:
    1. Compute per-trial mean firing rate in the wait period (cue_off → decision)
    2. Bin trials by current-trial wait duration into 4 quantiles
    3. Fit OLS:  rate ~ prev_rewarded + C(wait_quantile_bin)
    4. Report the prev_rewarded coefficient and its t/p

The wait_quantile bins control for current-trial duration as a fixed effect.
The prev_rewarded coefficient is "mean firing-rate difference (Hz) between
prev-rewarded and prev-unrewarded trials, after accounting for which duration
bin the trial sits in."

Tradeoffs vs. the GLM
---------------------
WHAT THIS GIVES UP:
  - Cannot detect SHAPE modulation of the wait-period firing pattern by reward.
    If a unit's response shape (early vs late peak) shifts with reward, but
    the trial-averaged rate is the same, this test misses it.
  - Single statistic per unit. The richer kernel-level description requires
    a separate (descriptive, not inferential) analysis.

WHAT THIS GIVES BACK:
  - One degree of freedom for the headline test, not 16. Vastly less room
    to over-fit per-trial residual variance.
  - Standard inferential machinery (Wilks doesn't enter; we use OLS t-tests).
  - Direct interpretability: the effect size is in Hz, the sign tells you
    direction (positive = fires more after reward; negative = after no reward).
  - Robust to the duration confound: within a quantile bin, all trials have
    similar duration, so duration-correlated firing-rate variation is absorbed
    by the bin fixed effect.

Outputs
-------
  - per_unit.csv:           one row per included unit
  - region_summary.csv:     per-region fractions significant + effect size stats
  - per_unit_per_quartile.csv:   diagnostic; per-unit per-bin means
  - region_effect_distribution.png:   effect-size histograms by region group
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt

import utils
import paths as p


# =====================================================================
# Config
# =====================================================================
ANCHOR = 'cue_off'              # 'cue_off' or 'last_lick' (last_lick = follow-up)
N_QUANTILES = 4                 # wait-duration bins for confound control
MIN_VALID_TRIALS = 30           # min trials with valid history for inclusion
MIN_PER_PREV_REWARD = 10        # min trials each of prev_rewarded=0 and =1
MIN_PER_BIN_PER_CONDITION = 3   # for the per-quartile diagnostic (relaxed)
MIN_RATE_HZ = 0.5               # exclude very-low-FR units (sparse spike trains)
MIN_WAIT_S = 0.10               # exclude implausibly short trials
FDR_Q = 0.05                    # within-region FDR threshold
LRT_ALPHA = 0.05                # per-unit nominal significance

# Region grouping: drop anatomically-unplaceable units from headline tables
EXCLUDE_REGION_GROUPS = ('Excluded', 'Other')

# Paths
OUT_DIR = Path(p.DATA_DIR) / 'simpler_reward_history_test'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_UNIT_CSV = OUT_DIR / 'per_unit.csv'
REGION_SUMMARY_CSV = OUT_DIR / 'region_summary.csv'
QUARTILE_DETAIL_CSV = OUT_DIR / 'per_unit_per_quartile.csv'
EFFECT_PLOT_PATH = OUT_DIR / 'region_effect_distribution.png'

INCREMENTAL_SAVE = True


# =====================================================================
# Small helpers (self-contained — no import from 4c script needed)
# =====================================================================
def coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s != 0
    mapping = {
        "true": True, "t": True, "yes": True, "y": True, "1": True,
        "false": False, "f": False, "no": False, "n": False, "0": False,
    }
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(False)


def spikes_df_to_trial_map(spikes_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    if not {"trial_id", "trial_time"}.issubset(spikes_df.columns):
        raise ValueError("spikes_df must contain 'trial_id' and 'trial_time'.")
    out = {}
    for tid, sub in spikes_df.groupby("trial_id"):
        out[int(tid)] = sub["trial_time"].astype(float).to_numpy()
    return out


def get_anchor_time(tr: pd.Series, anchor: str) -> float:
    """Return the trial-time at which the wait window starts."""
    if anchor == 'cue_off':
        return float(tr['cue_off_time'])
    elif anchor == 'cue_on':
        return float(tr['cue_on_time'])
    elif anchor == 'last_lick':
        # The last lick before the decision lick — would need the events df.
        # For first-pass we don't support this; raise so it's explicit.
        raise NotImplementedError(
            "last_lick anchor requires events; not implemented in first pass"
        )
    else:
        raise ValueError(f"unknown anchor: {anchor}")


# =====================================================================
# Trial-history features (minimal version, only what this test needs)
# =====================================================================
def build_min_trial_features(trials_df: pd.DataFrame, anchor: str) -> pd.DataFrame:
    """
    Build per-trial features needed for the simpler test:
      - prev_rewarded:   bool (NaN for first trial of session)
      - wait_duration:   decision_time - anchor_time (current trial)
      - is_miss:         bool

    Returns a DataFrame indexed by trial_id.
    """
    trials = trials_df.sort_values('trial_id').copy().reset_index(drop=True)

    if 'rewarded' in trials.columns:
        trials['rewarded'] = coerce_bool_series(trials['rewarded']).astype(int)
    else:
        raise ValueError("trials_df must have 'rewarded' column")
    if 'missed' in trials.columns:
        trials['is_miss'] = coerce_bool_series(trials['missed'])
    else:
        trials['is_miss'] = False

    trials['prev_rewarded'] = trials['rewarded'].shift(1)

    # Current-trial wait window
    anchor_col = {
        'cue_off': 'cue_off_time',
        'cue_on': 'cue_on_time',
    }.get(anchor)
    if anchor_col is None or anchor_col not in trials.columns:
        raise ValueError(f"trials_df missing time column for anchor={anchor}")
    if 'decision_time' not in trials.columns:
        raise ValueError("trials_df must have 'decision_time'")
    trials['wait_duration'] = trials['decision_time'] - trials[anchor_col]
    trials['anchor_time'] = trials[anchor_col]

    return trials.set_index('trial_id', drop=False)


# =====================================================================
# Per-unit wait-period firing rates
# =====================================================================
def compute_trial_firing_rates(
    spikes_by_trial: Dict[int, np.ndarray],
    trial_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each trial in trial_features, compute mean firing rate in
    [anchor_time, decision_time].

    Returns a DataFrame of rows (one per trial) with columns:
        trial_id, prev_rewarded, wait_duration, rate_hz, n_spikes
    Only returns rows for trials that have valid spike data and valid history.
    """
    rows = []
    for tid, tr in trial_features.iterrows():
        if bool(tr['is_miss']):
            continue
        if pd.isna(tr['prev_rewarded']):
            continue
        if not np.isfinite(tr['wait_duration']) or tr['wait_duration'] < MIN_WAIT_S:
            continue
        if tid not in spikes_by_trial:
            continue

        sp = spikes_by_trial[tid]
        anchor_t = float(tr['anchor_time'])
        dec_t = float(tr['decision_time'])
        dur = float(tr['wait_duration'])

        # Spike times are in trial-local coords; count those in [anchor, decision]
        mask = (sp >= anchor_t) & (sp < dec_t)
        n_spikes = int(mask.sum())
        rate_hz = n_spikes / dur if dur > 0 else np.nan

        rows.append({
            'trial_id': int(tid),
            'prev_rewarded': int(tr['prev_rewarded']),
            'wait_duration': dur,
            'rate_hz': rate_hz,
            'n_spikes': n_spikes,
        })

    return pd.DataFrame(rows)


# =====================================================================
# Per-unit OLS test
# =====================================================================
def fit_per_unit(trial_rates: pd.DataFrame, n_quantiles: int = N_QUANTILES) -> dict:
    """
    Fit two regressions per unit and return a dict of statistics.

    Primary model:    rate ~ prev_rewarded + C(wait_quantile_bin)
    Unconditional:    rate ~ prev_rewarded     (for comparison)
    Interaction:      rate ~ prev_rewarded * C(wait_quantile_bin)
                        — test interaction for residual-confound diagnostic

    Returns dict with:
        n_trials, n_pr0, n_pr1,
        mean_rate, mean_rate_pr0, mean_rate_pr1,
        beta_pr_cond, t_pr_cond, p_pr_cond,
        beta_pr_uncond, t_pr_uncond, p_pr_uncond,
        p_interaction
        (or NaN values if fit failed / insufficient data)
    """
    out = {
        'n_trials': len(trial_rates),
        'n_pr0': int((trial_rates['prev_rewarded'] == 0).sum()),
        'n_pr1': int((trial_rates['prev_rewarded'] == 1).sum()),
        'mean_rate': float(trial_rates['rate_hz'].mean()),
        'mean_rate_pr0': np.nan, 'mean_rate_pr1': np.nan,
        'beta_pr_cond': np.nan, 't_pr_cond': np.nan, 'p_pr_cond': np.nan,
        'beta_pr_uncond': np.nan, 't_pr_uncond': np.nan, 'p_pr_uncond': np.nan,
        'p_interaction': np.nan,
        'fit_status': 'ok',
    }

    if out['n_pr0'] >= 1:
        out['mean_rate_pr0'] = float(
            trial_rates.loc[trial_rates['prev_rewarded'] == 0, 'rate_hz'].mean()
        )
    if out['n_pr1'] >= 1:
        out['mean_rate_pr1'] = float(
            trial_rates.loc[trial_rates['prev_rewarded'] == 1, 'rate_hz'].mean()
        )

    if (out['n_pr0'] < MIN_PER_PREV_REWARD or
        out['n_pr1'] < MIN_PER_PREV_REWARD or
        out['n_trials'] < MIN_VALID_TRIALS):
        out['fit_status'] = 'insufficient_trials'
        return out

    # Bin by wait quantile
    try:
        bins = pd.qcut(trial_rates['wait_duration'], q=n_quantiles,
                       labels=False, duplicates='drop')
    except Exception as e:
        out['fit_status'] = f'qcut_fail:{e}'
        return out
    df = trial_rates.copy()
    df['wait_q'] = bins.astype(int)
    df = df.dropna(subset=['rate_hz', 'wait_q'])
    if df['wait_q'].nunique() < 2:
        out['fit_status'] = 'single_quantile_bin'
        return out

    # --- Unconditional model: rate ~ prev_rewarded ---
    try:
        res_uncond = smf.ols('rate_hz ~ prev_rewarded', data=df).fit()
        out['beta_pr_uncond'] = float(res_uncond.params.get('prev_rewarded', np.nan))
        out['t_pr_uncond'] = float(res_uncond.tvalues.get('prev_rewarded', np.nan))
        out['p_pr_uncond'] = float(res_uncond.pvalues.get('prev_rewarded', np.nan))
    except Exception as e:
        out['fit_status'] = f'uncond_fit_fail:{e}'

    # --- Conditional model: rate ~ prev_rewarded + C(wait_q) ---
    try:
        res_cond = smf.ols('rate_hz ~ prev_rewarded + C(wait_q)', data=df).fit()
        out['beta_pr_cond'] = float(res_cond.params.get('prev_rewarded', np.nan))
        out['t_pr_cond'] = float(res_cond.tvalues.get('prev_rewarded', np.nan))
        out['p_pr_cond'] = float(res_cond.pvalues.get('prev_rewarded', np.nan))
    except Exception as e:
        out['fit_status'] = f'cond_fit_fail:{e}'
        return out

    # --- Interaction diagnostic: rate ~ prev_rewarded * C(wait_q) ---
    try:
        res_int = smf.ols('rate_hz ~ prev_rewarded * C(wait_q)', data=df).fit()
        # Joint test of all prev_rewarded:C(wait_q) interaction terms
        int_terms = [n for n in res_int.params.index
                     if 'prev_rewarded:C(wait_q)' in n]
        if int_terms:
            R = np.zeros((len(int_terms), len(res_int.params)))
            for i, name in enumerate(int_terms):
                R[i, list(res_int.params.index).index(name)] = 1.0
            ftest = res_int.f_test(R)
            out['p_interaction'] = float(ftest.pvalue)
    except Exception as e:
        # Interaction is a diagnostic — non-fatal
        pass

    return out


def per_quartile_means(trial_rates: pd.DataFrame, n_quantiles: int = N_QUANTILES) -> List[dict]:
    """
    Return a list of dicts, one per quartile bin × prev_rewarded condition,
    with mean firing rate. Used for the diagnostic CSV.
    """
    rows = []
    try:
        bins = pd.qcut(trial_rates['wait_duration'], q=n_quantiles,
                       labels=False, duplicates='drop')
    except Exception:
        return rows
    df = trial_rates.copy()
    df['wait_q'] = bins.astype(int)
    df = df.dropna(subset=['rate_hz', 'wait_q'])
    for q in sorted(df['wait_q'].unique()):
        for pr in (0, 1):
            sub = df[(df['wait_q'] == q) & (df['prev_rewarded'] == pr)]
            rows.append({
                'wait_q': int(q),
                'prev_rewarded': pr,
                'n_trials': int(len(sub)),
                'mean_rate': float(sub['rate_hz'].mean()) if len(sub) > 0 else np.nan,
                'wait_range_lo': float(sub['wait_duration'].min()) if len(sub) > 0 else np.nan,
                'wait_range_hi': float(sub['wait_duration'].max()) if len(sub) > 0 else np.nan,
            })
    return rows


# =====================================================================
# Anatomical labels (matches what 4c evaluate_models uses)
# =====================================================================
def load_anatomical_labels() -> pd.DataFrame:
    """Load unit-level anatomical labels keyed by (session_id, id).

    unit_properties_final.csv keys units by (mouse, date_only, insertion_number, id)
    and does not carry session_id. We join it to units_vetted.csv to attach session_id.
    """
    ana_path = Path(p.LOGS_DIR) / 'unit_properties_final.csv'
    uv_path = Path(p.LOGS_DIR) / 'units_vetted.csv'
    if not ana_path.exists() or not uv_path.exists():
        print(f"[warn] anatomy or units_vetted not found ({ana_path}, {uv_path})")
        return pd.DataFrame()
    try:
        ana = pd.read_csv(ana_path)
        uv = pd.read_csv(uv_path, index_col=0)
    except Exception as e:
        print(f"[warn] could not read anatomy/units_vetted: {e}")
        return pd.DataFrame()

    ana = ana.copy()
    ana['date'] = pd.to_datetime(ana['date_only']).dt.strftime('%Y-%m-%d')
    ana_keep = ana[['mouse', 'date', 'insertion_number', 'id',
                    'corrected_region', 'region_group', 'cell_type']]
    uv_keep = uv[['mouse', 'date', 'insertion_number', 'id', 'session_id']]
    merged = uv_keep.merge(ana_keep, on=['mouse', 'date', 'insertion_number', 'id'],
                           how='inner')
    return merged[['session_id', 'id',
                   'corrected_region', 'region_group', 'cell_type']]


# =====================================================================
# Main loop
# =====================================================================
def run(session_ids: Optional[List[str]] = None):
    """
    Fit the simpler test on all units in the given sessions (or all sessions
    in units_vetted if not specified). Writes per-unit and per-quartile CSVs.
    """
    units_vetted = pd.read_csv(
        os.path.join(p.LOGS_DIR, 'units_vetted.csv'), index_col=0
    ).sort_values('unit_id')
    all_session_ids = sorted(units_vetted['session_id'].unique().tolist())
    if session_ids is None:
        session_ids = all_session_ids
    else:
        missing = sorted(set(session_ids) - set(all_session_ids))
        if missing:
            print(f"[warn] sessions not in units_vetted: {missing}")
        session_ids = [s for s in session_ids if s in set(all_session_ids)]
        if not session_ids:
            print("[abort] no sessions to run after filtering.")
            return

    print(f"\nWill run {len(session_ids)} session(s) with anchor={ANCHOR}, "
          f"n_quantiles={N_QUANTILES}")

    anatomy = load_anatomical_labels()
    if not anatomy.empty:
        # Build (session_id, id) -> labels lookup
        anatomy_idx = anatomy.set_index(['session_id', 'id'], drop=False)
        print(f"[anat] loaded {len(anatomy)} unit labels")
    else:
        anatomy_idx = None
        print("[anat] no labels available; region columns will be probe-target only")

    # Open per-unit CSV for incremental writes
    if INCREMENTAL_SAVE and PER_UNIT_CSV.exists():
        PER_UNIT_CSV.unlink()
    if INCREMENTAL_SAVE and QUARTILE_DETAIL_CSV.exists():
        QUARTILE_DETAIL_CSV.unlink()

    all_unit_rows = []
    all_quartile_rows = []

    for sid in session_ids:
        print(f"\n=== Session {sid} ===")
        try:
            events, trials, units = utils.get_session_data(sid)
        except Exception as e:
            print(f"[skip session] {sid}: {e}")
            continue

        try:
            trial_features = build_min_trial_features(trials, anchor=ANCHOR)
        except Exception as e:
            print(f"[skip session] {sid}: {e}")
            continue

        session_units = units_vetted[units_vetted['session_id'] == sid]
        probe_region = (
            session_units['region'].iloc[0]
            if 'region' in session_units.columns and len(session_units) > 0
            else ''
        )

        n_units = len(session_units)
        n_ok = 0
        for ui, (_, unit_info) in enumerate(session_units.iterrows(), start=1):
            unit_id = unit_info['unit_id']
            unit_key = unit_info['id']

            try:
                spikes_df = units[unit_key]
            except KeyError:
                print(f"[skip] unit {unit_id}: not in session spikes dict")
                continue
            spikes_by_trial = spikes_df_to_trial_map(spikes_df)

            trial_rates = compute_trial_firing_rates(spikes_by_trial, trial_features)
            if len(trial_rates) == 0:
                continue

            # Per-unit firing-rate floor
            unit_mean_rate = float(trial_rates['rate_hz'].mean())
            if not np.isfinite(unit_mean_rate) or unit_mean_rate < MIN_RATE_HZ:
                continue

            # Fit
            stats_d = fit_per_unit(trial_rates, n_quantiles=N_QUANTILES)

            # Anatomy lookup
            corrected_region = ''
            region_group = ''
            cell_type = ''
            if anatomy_idx is not None:
                try:
                    arow = anatomy_idx.loc[(sid, unit_key)]
                    if isinstance(arow, pd.DataFrame):
                        arow = arow.iloc[0]
                    corrected_region = str(arow.get('corrected_region', ''))
                    region_group = str(arow.get('region_group', ''))
                    cell_type = str(arow.get('cell_type', ''))
                except KeyError:
                    pass

            row = {
                'session_id': sid,
                'unit_id': unit_id,
                'unit_key': unit_key,
                'probe_region': probe_region,
                'corrected_region': corrected_region,
                'region_group': region_group,
                'cell_type': cell_type,
                **stats_d,
            }
            all_unit_rows.append(row)
            n_ok += 1

            # Quartile diagnostic rows
            for qr in per_quartile_means(trial_rates, n_quantiles=N_QUANTILES):
                all_quartile_rows.append({
                    'session_id': sid,
                    'unit_id': unit_id,
                    'corrected_region': corrected_region,
                    'region_group': region_group,
                    **qr,
                })

            if INCREMENTAL_SAVE and (ui % 25 == 0 or ui == n_units):
                pd.DataFrame(all_unit_rows).to_csv(PER_UNIT_CSV, index=False)
                pd.DataFrame(all_quartile_rows).to_csv(QUARTILE_DETAIL_CSV, index=False)

        print(f"[session done] {sid}: {n_ok}/{n_units} units fit successfully")

    # Final save
    if all_unit_rows:
        pd.DataFrame(all_unit_rows).to_csv(PER_UNIT_CSV, index=False)
        print(f"\nSaved → {PER_UNIT_CSV}  ({len(all_unit_rows)} rows)")
    if all_quartile_rows:
        pd.DataFrame(all_quartile_rows).to_csv(QUARTILE_DETAIL_CSV, index=False)
        print(f"Saved → {QUARTILE_DETAIL_CSV}  ({len(all_quartile_rows)} rows)")


# =====================================================================
# Region-level summary
# =====================================================================
def bh_fdr(pvals: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns boolean rejection array at q."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = q * np.arange(1, n + 1) / n
    rej_sorted = ranked <= thresh
    if rej_sorted.any():
        kmax = np.where(rej_sorted)[0].max()
        rej_sorted[: kmax + 1] = True
    rej = np.zeros(n, dtype=bool)
    rej[order] = rej_sorted
    return rej


def summarize(per_unit_csv: Path = PER_UNIT_CSV,
              group_col: str = 'region_group',
              exclude_groups: Tuple[str, ...] = EXCLUDE_REGION_GROUPS) -> pd.DataFrame:
    """
    Aggregate per-unit results by region group. Reports nominal/FDR significance
    fractions and effect-size distribution stats.
    """
    if not per_unit_csv.exists():
        print(f"[summary warn] {per_unit_csv} not found.")
        return pd.DataFrame()
    df = pd.read_csv(per_unit_csv)
    if group_col not in df.columns or df[group_col].isna().all():
        print(f"[summary warn] group_col={group_col} missing or empty; "
              f"falling back to probe_region")
        group_col = 'probe_region'
    df = df.loc[df[group_col].notna() & (df[group_col] != '')]
    if exclude_groups:
        df = df.loc[~df[group_col].isin(exclude_groups)]
    df = df.loc[df['fit_status'] == 'ok'].copy()
    if df.empty:
        print("[summary] no usable rows after filters.")
        return pd.DataFrame()

    rows = []
    for region, sub in df.groupby(group_col):
        p = sub['p_pr_cond'].dropna().values
        if p.size == 0:
            continue
        rej_uncorr = p < LRT_ALPHA
        rej_fdr = bh_fdr(p, q=FDR_Q)
        betas = sub['beta_pr_cond'].dropna().values
        rows.append({
            group_col: region,
            'n_units': int(len(sub)),
            'n_with_pval': int(p.size),
            'frac_sig_uncorr': float(rej_uncorr.mean()),
            f'frac_sig_fdr_q{FDR_Q:.2f}': float(rej_fdr.mean()),
            'median_p': float(np.median(p)),
            'median_beta_hz': float(np.median(betas)) if betas.size else np.nan,
            'iqr_beta_hz': float(np.percentile(betas, 75) -
                                 np.percentile(betas, 25)) if betas.size else np.nan,
            'frac_positive_beta': float((betas > 0).mean()) if betas.size else np.nan,
            'frac_negative_beta': float((betas < 0).mean()) if betas.size else np.nan,
        })

    summary = pd.DataFrame(rows).sort_values(
        f'frac_sig_fdr_q{FDR_Q:.2f}', ascending=False
    )
    summary.to_csv(REGION_SUMMARY_CSV, index=False)
    print(f"\nSaved → {REGION_SUMMARY_CSV}")
    print(summary.to_string(index=False))
    return summary


# =====================================================================
# Plot
# =====================================================================
def plot_effect_distributions(per_unit_csv: Path = PER_UNIT_CSV,
                              group_col: str = 'region_group',
                              exclude_groups: Tuple[str, ...] = EXCLUDE_REGION_GROUPS):
    """Plot per-region effect-size (beta_pr_cond) distributions."""
    if not per_unit_csv.exists():
        print(f"[plot warn] {per_unit_csv} not found.")
        return
    df = pd.read_csv(per_unit_csv)
    if group_col not in df.columns or df[group_col].isna().all():
        group_col = 'probe_region'
    df = df.loc[df[group_col].notna() & (df[group_col] != '')]
    if exclude_groups:
        df = df.loc[~df[group_col].isin(exclude_groups)]
    df = df.loc[df['fit_status'] == 'ok'].copy()
    if df.empty:
        return

    regions = sorted(df[group_col].unique())
    n = len(regions)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, reg in zip(axes, regions):
        sub = df.loc[df[group_col] == reg]
        betas = sub['beta_pr_cond'].dropna().values
        pvals = sub['p_pr_cond'].dropna().values
        rej = bh_fdr(pvals, q=FDR_Q) if pvals.size else np.array([], dtype=bool)
        # Histogram of betas, colored by significance
        bins = np.linspace(-10, 10, 31)
        if betas.size:
            ax.hist(betas, bins=bins, color='lightgray', edgecolor='k',
                    alpha=0.6, label='all')
            sig_betas = sub.loc[sub['p_pr_cond'].notna(), 'beta_pr_cond'].values[rej]
            if sig_betas.size:
                ax.hist(sig_betas, bins=bins, color='C3', edgecolor='k',
                        alpha=0.7, label=f'FDR q={FDR_Q}')
        ax.axvline(0, color='k', linewidth=0.6)
        ax.set_title(f"{reg}\nn={len(sub)}", fontsize=10)
        ax.set_xlabel('β prev_rewarded (Hz)')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('# units')
    fig.suptitle('Per-unit reward effect, controlling for wait quantile',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(EFFECT_PLOT_PATH, dpi=150)
    print(f"Saved → {EFFECT_PLOT_PATH}")
    plt.close(fig)


# =====================================================================
# Sanity check on one unit (analogous to debug_example in the GLM script)
# =====================================================================
def debug_one(session_id: str, unit_id: int):
    """Run the full pipeline on a single (session, unit) and print verbose output."""
    units_vetted = pd.read_csv(
        os.path.join(p.LOGS_DIR, 'units_vetted.csv'), index_col=0
    )
    sub = units_vetted[(units_vetted['session_id'] == session_id) &
                       (units_vetted['id'] == unit_id)]
    if sub.empty:
        print(f"[abort] unit not found: {session_id} / id={unit_id}")
        return
    unit_info = sub.iloc[0]
    unit_key = unit_info['id']
    print(f"[debug] {session_id} / id={unit_key} / unit_id={unit_info['unit_id']}")

    events, trials, units = utils.get_session_data(session_id)
    trial_features = build_min_trial_features(trials, anchor=ANCHOR)
    spikes_df = units[unit_key]
    spikes_by_trial = spikes_df_to_trial_map(spikes_df)
    trial_rates = compute_trial_firing_rates(spikes_by_trial, trial_features)

    print(f"[debug] trials with valid rates: {len(trial_rates)}")
    print(f"[debug] n_pr=1: {(trial_rates['prev_rewarded']==1).sum()}, "
          f"n_pr=0: {(trial_rates['prev_rewarded']==0).sum()}")
    print(f"[debug] wait_duration range: "
          f"{trial_rates['wait_duration'].min():.2f} – "
          f"{trial_rates['wait_duration'].max():.2f} s")
    print(f"[debug] mean firing rate: {trial_rates['rate_hz'].mean():.2f} Hz")

    stats_d = fit_per_unit(trial_rates)
    print("\n[debug] Per-unit fit:")
    for k, v in stats_d.items():
        if isinstance(v, float):
            print(f"   {k:>22s}: {v:.4g}")
        else:
            print(f"   {k:>22s}: {v}")

    print("\n[debug] Per-quartile breakdown:")
    print(f"{'q':>2s} {'pr':>3s} {'n':>4s} {'rate_hz':>9s} {'wait_lo':>8s} {'wait_hi':>8s}")
    for qr in per_quartile_means(trial_rates):
        print(f"{qr['wait_q']:>2d} {qr['prev_rewarded']:>3d} {qr['n_trials']:>4d}  "
              f"{qr['mean_rate']:>8.3f}  {qr['wait_range_lo']:>7.3f}  {qr['wait_range_hi']:>7.3f}")


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage:")
        print(f"  python {sys.argv[0]} run [session_id ...]")
        print(f"  python {sys.argv[0]} summary")
        print(f"  python {sys.argv[0]} plot")
        print(f"  python {sys.argv[0]} debug <session_id> <unit_id>")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "run":
        sids = sys.argv[2:] if len(sys.argv) > 2 else None
        run(session_ids=sids)
        summarize()
        plot_effect_distributions()
    elif mode == "summary":
        summarize()
    elif mode == "plot":
        plot_effect_distributions()
    elif mode == "debug":
        if len(sys.argv) < 4:
            print(f"Usage: python {sys.argv[0]} debug <session_id> <unit_id>")
            sys.exit(1)
        debug_one(sys.argv[2], int(sys.argv[3]))
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
