#!/usr/bin/env python3
"""
4e_time_encoding_test.py

Test 1: Does this neuron encode time during the wait period?

Per-unit Poisson GLM with an absolute-time wait-onset kernel, plus controls
for decision-locked motor activity and spike history. LRT tests whether the
wait_onset kernel adds explanatory power beyond the controls.

Companion to 4d_simpler_reward_history_test.py (Test 2: reward main effect).
Run together to characterize each unit on two independent axes:
    - Time-encoding (this script)
    - Reward-modulation (4d)

Design matrix (within-wait-period bins):
    Intercept
    wait_onset_basis_1..8     ← log-spaced raised cosines, 0–10s post cue_off
    decision_lock_basis_1..5  ← linear raised cosines, 0–2s before decision
    spike_history_basis_1..5  ← log-spaced raised cosines, 0–200ms lookback

Headline LRT:
    M_null = intercept + decision_lock + spike_history
    M_full = M_null + wait_onset
    chi² (df=8) on whether wait_onset adds info

Notes on design choices:
    - Wait-onset kernel uses ABSOLUTE time, not warped, so the test isn't
      confounded by trial-duration variation (the failure mode that broke
      the M1_vs_M2 LRT in 4c_encoding_GLM_w_history.py).
    - Log-spaced wait basis gives fine resolution at cue_off (catches
      transient responses) and coarse coverage of late times (catches
      late ramping). 8 bases is plenty; df is manageable.
    - decision_lock kernel is a confound control: it absorbs ramp-to-action
      activity that would otherwise look like "late wait encoding." Without
      it, the test would over-attribute decision-lick motor preparation
      to time encoding.
    - Unregularized IRLS for the LRT (no ridge bias). Pillow-style.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

import utils
import paths as p


# =====================================================================
# Config
# =====================================================================
DT = 0.025                          # 25 ms bins

# Wait-onset kernel (the one we're testing)
N_BASIS_WAIT = 8
T_MAX_WAIT_S = 10.0                 # kernel coverage; late wait trials may extend past
WAIT_BASIS_LOG_OFFSET = 0.05        # c in log(t + c), avoids log(0)

# Decision-locked kernel (confound control)
N_BASIS_DECLOCK = 5
T_MAX_DECLOCK_S = 2.0                # 2s pre-decision motor preparation

# Spike history kernel (confound control)
N_BASIS_HIST = 5
T_MAX_HIST_S = 0.20                  # 200ms lookback
HIST_BASIS_LOG_OFFSET = 0.005
N_LAG_BINS_HIST = int(np.ceil(T_MAX_HIST_S / DT))  # how many past bins to include

# Anchor for the wait period
ANCHOR = 'cue_off'                  # only cue_off implemented for first pass

# Unit inclusion
MIN_RATE_HZ = 0.5
MIN_VALID_TRIALS = 30
MIN_WAIT_S = 0.10                   # ignore implausibly short trials

# FDR / significance
LRT_ALPHA = 0.05
FDR_Q = 0.05

# Region grouping
EXCLUDE_REGION_GROUPS = ('Excluded', 'Other')

# Paths
OUT_DIR = Path(p.DATA_DIR) / 'time_encoding_test'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_UNIT_CSV = OUT_DIR / 'per_unit.csv'
REGION_SUMMARY_CSV = OUT_DIR / 'region_summary.csv'
KERNEL_CSV = OUT_DIR / 'per_unit_kernels.csv'
EFFECT_PLOT_PATH = OUT_DIR / 'region_time_encoding_distribution.png'

INCREMENTAL_SAVE = True


# =====================================================================
# Small helpers (self-contained)
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


# =====================================================================
# Raised-cosine basis functions (Pillow-style)
# =====================================================================
def raised_cosine_log(t: np.ndarray, n_bases: int, t_max: float,
                     c: float = WAIT_BASIS_LOG_OFFSET) -> np.ndarray:
    """
    Log-spaced raised-cosine basis functions on [0, t_max].

    Peaks are evenly spaced in log(t + c). Each basis has half-width equal
    to the inter-peak spacing in log-space, giving 50%-overlap at midpoints.

    Args:
        t: array of times to evaluate at (any shape).
        n_bases: number of basis functions.
        t_max: maximum time covered.
        c: log offset to avoid log(0) singularity.

    Returns:
        Array of shape t.shape + (n_bases,) with basis values.
        Values are 0 outside [0, t_max + width_in_real_time].
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    log_t = np.log(t_arr + c)
    log_min = np.log(c)
    log_max = np.log(t_max + c)
    spacing = (log_max - log_min) / (n_bases - 1)
    peaks = log_min + np.arange(n_bases) * spacing
    width = spacing  # half-width in log-space

    delta = log_t[..., None] - peaks  # broadcast
    in_support = np.abs(delta) < width
    bases = np.zeros_like(delta)
    bases[in_support] = 0.5 * (np.cos(np.pi * delta[in_support] / width) + 1)
    # Mask out negative t (no response before event)
    bases[t_arr < 0] = 0
    return bases


def raised_cosine_linear(t: np.ndarray, n_bases: int, t_max: float) -> np.ndarray:
    """
    Linear-spaced raised-cosine basis functions on [0, t_max].
    Peaks evenly spaced from 0 to t_max; same half-overlap geometry.

    Args:
        t: array of times to evaluate at.
        n_bases: number of basis functions.
        t_max: maximum time covered.

    Returns:
        Array of shape t.shape + (n_bases,).
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    spacing = t_max / (n_bases - 1)
    peaks = np.arange(n_bases) * spacing
    width = spacing

    delta = t_arr[..., None] - peaks
    in_support = np.abs(delta) < width
    bases = np.zeros_like(delta)
    bases[in_support] = 0.5 * (np.cos(np.pi * delta[in_support] / width) + 1)
    bases[t_arr < 0] = 0
    return bases


def precompute_basis_grid(basis_type: str, n_bases: int, t_max: float, dt: float,
                          extra_t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute basis values on a discrete bin grid for fast lookup.

    Args:
        basis_type: 'log' or 'linear'
        n_bases: number of basis functions
        t_max: maximum time covered by the basis
        dt: bin size
        extra_t: extend evaluation a bit past t_max so late-time evals don't
                 silently clip to zero

    Returns:
        (grid_times, basis_values)
            grid_times shape: (n_grid,)
            basis_values shape: (n_grid, n_bases)
    """
    n_grid = int(np.ceil((t_max + extra_t) / dt)) + 1
    grid = np.arange(n_grid) * dt
    if basis_type == 'log':
        vals = raised_cosine_log(grid, n_bases, t_max)
    elif basis_type == 'linear':
        vals = raised_cosine_linear(grid, n_bases, t_max)
    else:
        raise ValueError(f"unknown basis_type: {basis_type}")
    return grid, vals


# =====================================================================
# Per-trial design matrix construction
# =====================================================================
def build_wait_onset_columns(n_bins_wait: int, dt: float,
                             basis_grid: np.ndarray, basis_vals: np.ndarray) -> np.ndarray:
    """
    Build the wait_onset_basis columns for one trial.

    The wait period starts at bin 0 (= cue_off). For each bin t = 0, dt, 2dt, ...,
    look up the basis values at time t since cue_off. Basis is zero past the
    coverage limit (so very long trials have flat-zero late-kernel contributions
    past t_max_wait).

    Args:
        n_bins_wait: number of bins in this trial's wait period
        dt: bin width
        basis_grid: time values where basis is precomputed (shape (G,))
        basis_vals: basis values (shape (G, n_bases))

    Returns:
        (n_bins_wait, n_bases) array
    """
    bin_times = np.arange(n_bins_wait) * dt
    # Look up basis values at each bin time
    # Use np.interp per basis, or index into precomputed grid
    G = len(basis_grid)
    n_bases = basis_vals.shape[1]
    out = np.zeros((n_bins_wait, n_bases))
    # For each bin, find nearest grid index
    indices = np.clip((bin_times / dt).astype(int), 0, G - 1)
    out = basis_vals[indices]
    # Anything past the grid extent is already 0 (basis is 0)
    return out


def build_decision_lock_columns(n_bins_wait: int, dt: float,
                                basis_grid: np.ndarray, basis_vals: np.ndarray) -> np.ndarray:
    """
    Build decision_lock_basis columns for one trial.

    decision_lock is backward-anchored: at bin t (in [0, T_wait)), the kernel
    looks at time-until-decision = T_wait - t. So early bins (small t) have
    large time-until-decision (kernel is ~0 unless very long T_wait), and
    late bins (t near T_wait) have small time-until-decision (kernel is ~1
    at the latest basis).

    Args:
        n_bins_wait: number of bins in wait period
        dt: bin width
        basis_grid: precomputed basis evaluation times (shape (G,))
        basis_vals: precomputed basis values (shape (G, n_bases))

    Returns:
        (n_bins_wait, n_bases) array
    """
    G = len(basis_grid)
    T_wait = n_bins_wait * dt
    bin_times = np.arange(n_bins_wait) * dt
    # time-until-decision for each bin: from T_wait down to dt
    time_until_dec = T_wait - bin_times - dt  # at last bin: 0; at first bin: T_wait - dt
    time_until_dec = np.clip(time_until_dec, 0, None)
    # Look up basis values at these times
    indices = np.clip((time_until_dec / dt).astype(int), 0, G - 1)
    return basis_vals[indices]


def build_spike_history_columns(spike_counts_full_trial: np.ndarray,
                                wait_start_bin: int,
                                n_bins_wait: int,
                                basis_grid: np.ndarray,
                                basis_vals: np.ndarray,
                                n_lag_bins: int) -> np.ndarray:
    """
    Build spike_history columns for the wait-period bins of one trial.

    The kernel is a causal convolution of past spike counts with each basis.
    For each wait-period bin t (absolute bin index t_abs = wait_start_bin + t),
    the design value at basis j is:
        sum_{l=1..n_lag_bins} basis_vals[l-1, j] * spike_counts[t_abs - l]

    We use the FULL trial spike train (including BG-period spikes) so early
    wait bins have proper history lookback.

    Args:
        spike_counts_full_trial: spike counts over the entire trial (shape (T_full,))
        wait_start_bin: index of the first wait-period bin in the full trial
        n_bins_wait: number of wait-period bins
        basis_grid: precomputed basis evaluation times (shape (G,))
        basis_vals: precomputed basis values at lags l*dt (shape (G, n_bases))
        n_lag_bins: number of past bins to consider

    Returns:
        (n_bins_wait, n_bases) array
    """
    n_bases = basis_vals.shape[1]
    T_full = len(spike_counts_full_trial)

    # We need basis values at lags 1*dt, 2*dt, ..., n_lag_bins*dt
    # basis_grid[0] is lag 0; basis_grid[k] is lag k*dt
    # Lag 1*dt corresponds to index 1 in basis_grid (assuming dt-stepped grid)
    if len(basis_grid) < n_lag_bins + 1:
        # Pad with zeros if basis grid is shorter than needed
        lag_basis = np.zeros((n_lag_bins, n_bases))
        n_avail = min(n_lag_bins, len(basis_grid) - 1)
        if n_avail > 0:
            lag_basis[:n_avail] = basis_vals[1:1 + n_avail]
    else:
        lag_basis = basis_vals[1:1 + n_lag_bins]  # shape (n_lag_bins, n_bases)

    out = np.zeros((n_bins_wait, n_bases))
    for t_local in range(n_bins_wait):
        t_abs = wait_start_bin + t_local
        for lag_idx in range(n_lag_bins):
            src_bin = t_abs - lag_idx - 1
            if 0 <= src_bin < T_full:
                out[t_local] += spike_counts_full_trial[src_bin] * lag_basis[lag_idx]
    return out


def build_per_trial_design(
    tr: pd.Series,
    spikes_trial: np.ndarray,
    wait_grid: np.ndarray, wait_vals: np.ndarray,
    declock_grid: np.ndarray, declock_vals: np.ndarray,
    hist_grid: np.ndarray, hist_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (X, y, names) for the wait-period bins of one trial.

    Args:
        tr: row of trials dataframe with cue_off_time, decision_time
        spikes_trial: spike times in trial-local coords (s)
        wait_grid, wait_vals: precomputed wait_onset basis
        declock_grid, declock_vals: precomputed decision_lock basis
        hist_grid, hist_vals: precomputed spike_history basis

    Returns:
        X: (n_bins_wait, n_total_cols) design matrix for this trial
        y: (n_bins_wait,) spike count per bin
        names: list of column names

    Raises:
        ValueError if the trial is invalid for inclusion.
    """
    cue_off = float(tr['cue_off_time'])
    dec = float(tr['decision_time'])
    if not (np.isfinite(cue_off) and np.isfinite(dec)) or dec <= cue_off:
        raise ValueError("invalid cue_off / decision times")
    wait_dur = dec - cue_off
    if wait_dur < MIN_WAIT_S:
        raise ValueError(f"wait too short: {wait_dur:.3f}s")

    # For spike history, we need the full trial spike train, including BG period
    bg_start = float(tr.get('cue_on_time', 0.0))
    cons_len = float(tr.get('consumption_length', 3.0))
    if not np.isfinite(cons_len) or cons_len <= 0:
        cons_len = 3.0
    t_min = min(bg_start, 0.0)
    t_max_trial = dec + cons_len
    n_bins_full = int(np.ceil((t_max_trial - t_min) / DT))
    bin_edges_full = t_min + np.arange(n_bins_full + 1) * DT
    spike_counts_full, _ = np.histogram(spikes_trial, bins=bin_edges_full)

    # Wait-period bin range within the full trial
    wait_start_bin = int(round((cue_off - t_min) / DT))
    wait_end_bin = int(round((dec - t_min) / DT))
    n_bins_wait = wait_end_bin - wait_start_bin
    if n_bins_wait < 2:
        raise ValueError("fewer than 2 wait-period bins")

    y = spike_counts_full[wait_start_bin:wait_end_bin].astype(float)

    # Build design column families
    wait_cols = build_wait_onset_columns(n_bins_wait, DT, wait_grid, wait_vals)
    declock_cols = build_decision_lock_columns(n_bins_wait, DT, declock_grid, declock_vals)
    hist_cols = build_spike_history_columns(
        spike_counts_full, wait_start_bin, n_bins_wait,
        hist_grid, hist_vals, N_LAG_BINS_HIST,
    )

    X = np.hstack([wait_cols, declock_cols, hist_cols])
    names = (
        [f'wait_onset_{j}' for j in range(wait_cols.shape[1])]
        + [f'decision_lock_{j}' for j in range(declock_cols.shape[1])]
        + [f'spike_hist_{j}' for j in range(hist_cols.shape[1])]
    )
    return X, y, names


# =====================================================================
# Per-unit fit
# =====================================================================
def fit_glm_unreg(X: np.ndarray, y: np.ndarray) -> Tuple[Optional[object], float]:
    """
    Fit unregularized Poisson GLM via IRLS. Return (result, log-likelihood).

    For intercept-only (X has 0 columns), returns the closed-form null LL
    using mean(y) as the constant predicted rate.
    """
    if X.shape[1] == 0:
        mu = max(float(y.mean()), 1e-9)
        ll = float(np.sum(y * np.log(mu) - mu))
        return None, ll
    Xc = sm.add_constant(X, has_constant='add')
    try:
        res = sm.GLM(y, Xc, family=sm.families.Poisson()).fit(maxiter=200)
        if not getattr(res, 'converged', True):
            raise RuntimeError("IRLS did not converge")
        mu = np.clip(res.predict(Xc), 1e-9, None)
        ll = float(np.sum(y * np.log(mu) - mu))
        if not np.isfinite(ll):
            raise RuntimeError("non-finite ll")
        return res, ll
    except Exception as e:
        print(f"[fit warn] {e}")
        return None, np.nan


def lrt_pvalue(ll_reduced: float, ll_full: float, df_diff: int) -> float:
    if not np.isfinite(ll_reduced) or not np.isfinite(ll_full) or df_diff <= 0:
        return np.nan
    chi2 = -2.0 * (ll_reduced - ll_full)
    if chi2 < 0:
        chi2 = 0.0
    return float(stats.chi2.sf(chi2, df_diff))


def pseudo_R2_mcfadden(y: np.ndarray, mu_full: np.ndarray, mu_null: float) -> float:
    mu_full = np.clip(mu_full, 1e-9, None)
    ll_full = np.sum(y * np.log(mu_full) - mu_full)
    mu_null = max(mu_null, 1e-9)
    ll_null = np.sum(y * np.log(mu_null) - mu_null)
    return 1.0 - (ll_full / ll_null) if ll_null != 0 else np.nan


def fit_per_unit(X_all: np.ndarray, y_all: np.ndarray, names: List[str]) -> dict:
    """
    Fit M_null and M_full, compute LRT and descriptive stats.

    M_null:  decision_lock + spike_history + intercept   (no wait_onset)
    M_full:  wait_onset + decision_lock + spike_history + intercept

    Returns dict with chi2/df/p for the headline test, plus descriptive
    full-vs-null and full-vs-intercept stats, and the fitted wait kernel.
    """
    cols_arr = np.array(names)
    wait_idx = np.where(np.char.startswith(cols_arr, 'wait_onset_'))[0]
    null_idx = np.where(~np.char.startswith(cols_arr, 'wait_onset_'))[0]

    # Fit M_full
    res_full, ll_full = fit_glm_unreg(X_all, y_all)
    if res_full is None:
        return _empty_fit_result()
    Xc_full = sm.add_constant(X_all, has_constant='add')
    mu_full = np.clip(res_full.predict(Xc_full), 1e-9, None)
    pr2_full = float(pseudo_R2_mcfadden(y_all, mu_full, y_all.mean()))

    # Fit M_null (drop wait_onset columns)
    X_null = X_all[:, null_idx]
    _, ll_null = fit_glm_unreg(X_null, y_all)

    # Fit intercept-only
    _, ll_int = fit_glm_unreg(np.zeros((len(y_all), 0)), y_all)

    df_wait = int(len(wait_idx))
    df_full_vs_int = int(X_all.shape[1])

    chi2_wait = -2.0 * (ll_null - ll_full)
    chi2_full_vs_int = -2.0 * (ll_int - ll_full)
    p_wait = lrt_pvalue(ll_null, ll_full, df_wait)
    p_full = lrt_pvalue(ll_int, ll_full, df_full_vs_int)

    # Extract wait kernel coefficients (excluding intercept)
    params = np.asarray(res_full.params)
    # add_constant places intercept first; the rest are in column order of X_all
    beta_intercept = float(params[0])
    beta_rest = params[1:]  # length = X_all.shape[1]
    beta_wait = beta_rest[wait_idx]

    return {
        'fit_status': 'ok',
        'n_bins': int(len(y_all)),
        'mean_rate_hz': float(y_all.sum() / (len(y_all) * DT)),
        'beta_intercept': beta_intercept,
        'beta_wait': beta_wait.tolist(),
        'll_full': float(ll_full),
        'll_null': float(ll_null),
        'll_intercept': float(ll_int),
        'df_wait': df_wait,
        'df_full_vs_int': df_full_vs_int,
        'chi2_wait': float(chi2_wait),
        'chi2_full_vs_int': float(chi2_full_vs_int),
        'p_wait': float(p_wait),
        'p_full_vs_int': float(p_full),
        'pseudo_r2_full': pr2_full,
    }


def _empty_fit_result() -> dict:
    return {
        'fit_status': 'failed',
        'n_bins': 0,
        'mean_rate_hz': np.nan,
        'beta_intercept': np.nan,
        'beta_wait': [np.nan] * N_BASIS_WAIT,
        'll_full': np.nan, 'll_null': np.nan, 'll_intercept': np.nan,
        'df_wait': N_BASIS_WAIT, 'df_full_vs_int': np.nan,
        'chi2_wait': np.nan, 'chi2_full_vs_int': np.nan,
        'p_wait': np.nan, 'p_full_vs_int': np.nan,
        'pseudo_r2_full': np.nan,
    }


def compute_wait_kernel_peak(beta_wait: List[float], wait_grid: np.ndarray,
                             wait_vals: np.ndarray, t_max: float = T_MAX_WAIT_S) -> Tuple[float, float]:
    """
    Evaluate the fitted wait kernel on a dense grid and return (peak_lag_s, peak_value).
    Peak is the time where the wait kernel sum reaches its maximum.
    """
    beta = np.asarray(beta_wait)
    # Wait kernel as a function of time: sum_j beta_j * B_j(t)
    # wait_vals: shape (G, n_bases); contribution: wait_vals @ beta = shape (G,)
    kernel_vals = wait_vals @ beta
    # Restrict to t in [0, t_max]
    mask = wait_grid <= t_max
    if not np.any(mask):
        return np.nan, np.nan
    t_grid = wait_grid[mask]
    k = kernel_vals[mask]
    peak_idx = int(np.argmax(k))
    return float(t_grid[peak_idx]), float(k[peak_idx])


# =====================================================================
# Build full design for one unit
# =====================================================================
def build_design_for_unit(
    spikes_by_trial: Dict[int, np.ndarray],
    trials_idx: pd.DataFrame,
    wait_grid: np.ndarray, wait_vals: np.ndarray,
    declock_grid: np.ndarray, declock_vals: np.ndarray,
    hist_grid: np.ndarray, hist_vals: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], List[int], List[Tuple[int, str]]]:
    """
    Build the wait-period design matrix for one unit, stacked across trials.

    Returns (X_all, y_all, names, used_tids, skipped) or
            (None, None, None, [], skipped) if no trials qualify.
    """
    X_blocks, y_blocks = [], []
    names: Optional[List[str]] = None
    used: List[int] = []
    skipped: List[Tuple[int, str]] = []

    for tid, sp in spikes_by_trial.items():
        if tid not in trials_idx.index:
            skipped.append((tid, 'not in trials_idx'))
            continue
        tr = trials_idx.loc[tid]
        if bool(coerce_bool_series(pd.Series([tr.get('missed', False)])).iloc[0]):
            skipped.append((tid, 'missed'))
            continue
        try:
            X, y, nm = build_per_trial_design(
                tr, sp,
                wait_grid, wait_vals,
                declock_grid, declock_vals,
                hist_grid, hist_vals,
            )
        except Exception as e:
            skipped.append((tid, str(e)))
            continue
        X_blocks.append(X)
        y_blocks.append(y)
        if names is None:
            names = nm
        used.append(tid)

    if not X_blocks:
        return None, None, names, used, skipped
    X_all = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    return X_all, y_all, names, used, skipped


# =====================================================================
# Anatomical labels (matches 4d / 4c convention)
# =====================================================================
def load_anatomical_labels() -> pd.DataFrame:
    """Load anatomy keyed by (session_id, id). unit_properties_final has no
    session_id column, so we join it to units_vetted on (mouse, date, insertion_number, id)."""
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
    """Run Test 1 on the given sessions (or all of units_vetted)."""
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
            print("[abort] no sessions to run.")
            return

    print(f"\nWill run {len(session_ids)} session(s) with anchor={ANCHOR}")
    print(f"Wait kernel: {N_BASIS_WAIT} log-spaced bases over 0-{T_MAX_WAIT_S}s")
    print(f"Decision-lock kernel: {N_BASIS_DECLOCK} linear bases over 0-{T_MAX_DECLOCK_S}s")
    print(f"Spike history kernel: {N_BASIS_HIST} log-spaced bases over 0-{T_MAX_HIST_S}s\n")

    # Precompute bases once
    wait_grid, wait_vals = precompute_basis_grid(
        'log', N_BASIS_WAIT, T_MAX_WAIT_S, DT, extra_t=5.0
    )
    declock_grid, declock_vals = precompute_basis_grid(
        'linear', N_BASIS_DECLOCK, T_MAX_DECLOCK_S, DT
    )
    hist_grid, hist_vals = precompute_basis_grid(
        'log', N_BASIS_HIST, T_MAX_HIST_S, DT
    )

    anatomy = load_anatomical_labels()
    anatomy_idx = (
        anatomy.set_index(['session_id', 'id'], drop=False) if not anatomy.empty else None
    )
    if anatomy_idx is not None:
        print(f"[anat] loaded {len(anatomy)} unit labels")

    # Reset outputs if incremental
    if INCREMENTAL_SAVE and PER_UNIT_CSV.exists():
        PER_UNIT_CSV.unlink()
    if INCREMENTAL_SAVE and KERNEL_CSV.exists():
        KERNEL_CSV.unlink()

    all_rows = []
    all_kernel_rows = []

    for sid in session_ids:
        print(f"\n=== Session {sid} ===")
        try:
            events, trials, units = utils.get_session_data(sid)
        except Exception as e:
            print(f"[skip] {sid}: {e}")
            continue

        # Filter miss trials
        if 'missed' in trials.columns:
            try:
                trials = trials.loc[~coerce_bool_series(trials['missed'])].copy()
            except Exception:
                pass
        trials_idx = trials.set_index('trial_id', drop=False)

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
                continue
            spikes_by_trial = spikes_df_to_trial_map(spikes_df)

            X_all, y_all, names, used, _ = build_design_for_unit(
                spikes_by_trial, trials_idx,
                wait_grid, wait_vals,
                declock_grid, declock_vals,
                hist_grid, hist_vals,
            )
            if X_all is None or len(used) < MIN_VALID_TRIALS:
                continue

            mean_rate = float(y_all.sum() / (len(y_all) * DT))
            if mean_rate < MIN_RATE_HZ:
                continue

            stats_d = fit_per_unit(X_all, y_all, names)
            if stats_d['fit_status'] != 'ok':
                continue

            # Compute kernel peak descriptively
            peak_lag, peak_val = compute_wait_kernel_peak(
                stats_d['beta_wait'], wait_grid, wait_vals
            )
            stats_d['peak_lag_s'] = peak_lag
            stats_d['peak_value'] = peak_val

            # Anatomy
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

            # Strip beta_wait from main row (saved separately)
            beta_wait = stats_d.pop('beta_wait')

            row = {
                'session_id': sid,
                'unit_id': unit_id,
                'unit_key': unit_key,
                'probe_region': probe_region,
                'corrected_region': corrected_region,
                'region_group': region_group,
                'cell_type': cell_type,
                'n_trials_used': int(len(used)),
                **stats_d,
            }
            all_rows.append(row)

            for j, b in enumerate(beta_wait):
                all_kernel_rows.append({
                    'session_id': sid,
                    'unit_id': unit_id,
                    'corrected_region': corrected_region,
                    'region_group': region_group,
                    'basis_idx': j,
                    'beta': float(b),
                })

            n_ok += 1

            if INCREMENTAL_SAVE and (ui % 25 == 0 or ui == n_units):
                pd.DataFrame(all_rows).to_csv(PER_UNIT_CSV, index=False)
                pd.DataFrame(all_kernel_rows).to_csv(KERNEL_CSV, index=False)

        print(f"[session done] {sid}: {n_ok}/{n_units} units fit")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(PER_UNIT_CSV, index=False)
        print(f"\nSaved → {PER_UNIT_CSV}  ({len(all_rows)} rows)")
    if all_kernel_rows:
        pd.DataFrame(all_kernel_rows).to_csv(KERNEL_CSV, index=False)
        print(f"Saved → {KERNEL_CSV}  ({len(all_kernel_rows)} rows)")


# =====================================================================
# Summary
# =====================================================================
def bh_fdr(pvals: np.ndarray, q: float = FDR_Q) -> np.ndarray:
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
    if not per_unit_csv.exists():
        print(f"[summary warn] {per_unit_csv} not found.")
        return pd.DataFrame()
    df = pd.read_csv(per_unit_csv)
    if group_col not in df.columns or df[group_col].isna().all():
        print(f"[summary warn] falling back to probe_region")
        group_col = 'probe_region'
    df = df.loc[df[group_col].notna() & (df[group_col] != '')]
    if exclude_groups:
        df = df.loc[~df[group_col].isin(exclude_groups)]
    df = df.loc[df['fit_status'] == 'ok']
    if df.empty:
        print("[summary] no usable rows after filters.")
        return pd.DataFrame()

    rows = []
    for region, sub in df.groupby(group_col):
        p = sub['p_wait'].dropna().values
        if p.size == 0:
            continue
        rej_uncorr = p < LRT_ALPHA
        rej_fdr = bh_fdr(p, q=FDR_Q)
        chi2_vals = sub['chi2_wait'].dropna().values
        peaks = sub['peak_lag_s'].dropna().values
        rows.append({
            group_col: region,
            'n_units': int(len(sub)),
            'n_with_pval': int(p.size),
            'frac_sig_uncorr': float(rej_uncorr.mean()),
            f'frac_sig_fdr_q{FDR_Q:.2f}': float(rej_fdr.mean()),
            'median_p': float(np.median(p)),
            'median_chi2': float(np.median(chi2_vals)) if chi2_vals.size else np.nan,
            'median_pseudo_r2': float(np.median(sub['pseudo_r2_full'].dropna())),
            'median_peak_lag_s': float(np.median(peaks)) if peaks.size else np.nan,
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
def plot_distributions(per_unit_csv: Path = PER_UNIT_CSV,
                       group_col: str = 'region_group',
                       exclude_groups: Tuple[str, ...] = EXCLUDE_REGION_GROUPS):
    if not per_unit_csv.exists():
        return
    df = pd.read_csv(per_unit_csv)
    if group_col not in df.columns or df[group_col].isna().all():
        group_col = 'probe_region'
    df = df.loc[df[group_col].notna() & (df[group_col] != '')]
    if exclude_groups:
        df = df.loc[~df[group_col].isin(exclude_groups)]
    df = df.loc[df['fit_status'] == 'ok']
    if df.empty:
        return

    regions = sorted(df[group_col].unique())
    n = len(regions)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 5.5), sharex='row')
    if n == 1:
        axes = axes[:, None]

    for col, reg in enumerate(regions):
        sub = df.loc[df[group_col] == reg]

        # Top: chi2 distribution
        ax = axes[0, col]
        chi2_vals = sub['chi2_wait'].dropna().values
        if chi2_vals.size:
            ax.hist(chi2_vals, bins=30, color='lightgray', edgecolor='k', alpha=0.7)
            ax.axvline(N_BASIS_WAIT, color='r', ls='--', linewidth=0.6,
                       label=f'df={N_BASIS_WAIT}')
        ax.set_title(f"{reg} (n={len(sub)})", fontsize=10)
        if col == 0:
            ax.set_ylabel('# units')
        ax.set_xlabel('χ² (wait LRT)')
        if col == 0:
            ax.legend(fontsize=8)

        # Bottom: peak lag distribution
        ax = axes[1, col]
        peaks = sub['peak_lag_s'].dropna().values
        if peaks.size:
            ax.hist(peaks, bins=20, range=(0, T_MAX_WAIT_S),
                    color='steelblue', edgecolor='k', alpha=0.7)
        if col == 0:
            ax.set_ylabel('# units')
        ax.set_xlabel('peak lag (s)')

    fig.suptitle('Test 1: time encoding during wait period', fontsize=11)
    fig.tight_layout()
    fig.savefig(EFFECT_PLOT_PATH, dpi=150)
    print(f"Saved → {EFFECT_PLOT_PATH}")
    plt.close(fig)


# =====================================================================
# Debug
# =====================================================================
def debug_one(session_id: str, unit_id: int):
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

    # Precompute bases
    wait_grid, wait_vals = precompute_basis_grid(
        'log', N_BASIS_WAIT, T_MAX_WAIT_S, DT, extra_t=5.0
    )
    declock_grid, declock_vals = precompute_basis_grid(
        'linear', N_BASIS_DECLOCK, T_MAX_DECLOCK_S, DT
    )
    hist_grid, hist_vals = precompute_basis_grid(
        'log', N_BASIS_HIST, T_MAX_HIST_S, DT
    )

    events, trials, units = utils.get_session_data(session_id)
    if 'missed' in trials.columns:
        trials = trials.loc[~coerce_bool_series(trials['missed'])].copy()
    trials_idx = trials.set_index('trial_id', drop=False)
    spikes_df = units[unit_key]
    spikes_by_trial = spikes_df_to_trial_map(spikes_df)

    X_all, y_all, names, used, skipped = build_design_for_unit(
        spikes_by_trial, trials_idx,
        wait_grid, wait_vals,
        declock_grid, declock_vals,
        hist_grid, hist_vals,
    )
    print(f"[debug] trials used: {len(used)}, skipped: {len(skipped)}")
    if X_all is None:
        print("[debug] no design built")
        return
    print(f"[debug] X shape: {X_all.shape}, y shape: {y_all.shape}")
    print(f"[debug] mean rate: {y_all.sum() / (len(y_all) * DT):.2f} Hz")
    print(f"[debug] column names ({len(names)}): "
          f"{names[:3]}...{names[-3:]}")

    stats_d = fit_per_unit(X_all, y_all, names)
    print("\n[debug] Fit results:")
    for k, v in stats_d.items():
        if isinstance(v, list):
            v_str = "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
        elif isinstance(v, float):
            v_str = f"{v:.4g}"
        else:
            v_str = str(v)
        print(f"   {k:>22s}: {v_str}")

    peak_lag, peak_val = compute_wait_kernel_peak(
        stats_d['beta_wait'], wait_grid, wait_vals
    )
    print(f"\n[debug] Wait kernel peak: t={peak_lag:.2f}s, value={peak_val:.3f}")

    # Plot fitted kernel
    fig, ax = plt.subplots(figsize=(6, 3))
    beta = np.asarray(stats_d['beta_wait'])
    kernel_vals = wait_vals @ beta
    mask = wait_grid <= T_MAX_WAIT_S
    ax.plot(wait_grid[mask], kernel_vals[mask], color='C0', lw=1.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(peak_lag, color='r', ls='--', lw=0.6, label=f'peak {peak_lag:.2f}s')
    ax.set_xlabel('time since cue_off (s)')
    ax.set_ylabel('fitted wait kernel (log-rate units)')
    ax.set_title(f"{session_id} u{unit_id}: wait kernel  "
                 f"χ²={stats_d['chi2_wait']:.1f}, p={stats_d['p_wait']:.3g}")
    ax.legend()
    fig.tight_layout()
    plot_path = OUT_DIR / f'debug_kernel_{session_id}_u{unit_id}.png'
    fig.savefig(plot_path, dpi=150)
    print(f"\n[debug] kernel plot → {plot_path}")
    plt.close(fig)


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
        plot_distributions()
    elif mode == "summary":
        summarize()
    elif mode == "plot":
        plot_distributions()
    elif mode == "debug":
        if len(sys.argv) < 4:
            print(f"Usage: python {sys.argv[0]} debug <session_id> <unit_id>")
            sys.exit(1)
        debug_one(sys.argv[2], int(sys.argv[3]))
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
