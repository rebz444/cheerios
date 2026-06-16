#!/usr/bin/env python3
"""
4f_time_encoding_two_anchors.py

Test 1, two-anchor version: per-unit Poisson GLM testing wait-period time
encoding from BOTH cue_on and cue_off anchors.

For each unit, fits two separate GLMs:

  ANCHOR = cue_off:
    Window:           [cue_off_time, decision_time]
    Design columns:   wait_onset_basis (forward from cue_off, 8 log-spaced),
                      decision_lock_basis (backward from decision, 5 linear),
                      spike_history_basis (causal self-history, 5 log-spaced)
    Headline LRT:     wait_onset vs no-wait_onset, df=8

  ANCHOR = cue_on:
    Window:           [cue_on_time, decision_time]   (BG + wait combined)
    Design columns:   wait_onset_basis (forward from cue_on, 8 log-spaced),
                      cue_off_event_basis (forward from cue_off, 5 linear, 500ms),
                      lick_bg_basis (forward from each BG lick, 5 log-spaced, 300ms),
                      decision_lock_basis (backward from decision, 5 linear),
                      spike_history_basis (5 log-spaced)
    Headline LRT:     wait_onset vs no-wait_onset, df=8

Each anchor's LRT tests: "after accounting for sensory/motor events within
this window, does the kernel anchored to this event add explanatory power?"

PER-UNIT OUTPUT (one row per unit):
  Columns suffixed by anchor:  chi2_wait_{anchor}, p_wait_{anchor},
                               pseudo_r2_{anchor}, peak_lag_{anchor},
                               n_trials_used_{anchor}, n_bins_{anchor}, ...
  Plus session/unit/region identifiers.

PER-REGION SUMMARY:
  Reports both anchors separately + a cross-tabulation of which units pass
  which anchor (only-cue_off, only-cue_on, both, neither) per region.

Companion script to 4d_simpler_reward_history_test.py (Test 2).
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
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

# --- Wait-onset kernel (per anchor) ---
N_BASIS_WAIT = 8
T_MIN_WAIT_S = 0.5                  # basis support starts at 0.5s post-anchor;
                                    # the anchor-event kernel (0-500ms) owns [0, t_min)
T_MAX_WAIT_CUE_OFF_S = 10.0         # cue_off window is shorter (wait only)
T_MAX_WAIT_CUE_ON_S = 15.0          # cue_on window covers BG + wait
WAIT_BASIS_LOG_OFFSET = 0.05

# --- Decision-locked backward kernel (both anchors) ---
N_BASIS_DECLOCK = 5
T_MAX_DECLOCK_S = 2.0

# --- Cue_off event kernel (cue_on anchor only) ---
N_BASIS_CUE_OFF = 5
T_MAX_CUE_OFF_S = 0.5               # 500 ms post cue_off transient

# --- Lick_bg event kernel (cue_on anchor only) ---
N_BASIS_LICK = 5
T_MAX_LICK_S = 0.3                  # 300 ms post-lick response
LICK_BASIS_LOG_OFFSET = 0.005

# --- Spike history kernel (both anchors) ---
N_BASIS_HIST = 5
T_MAX_HIST_S = 0.20
HIST_BASIS_LOG_OFFSET = 0.005
N_LAG_BINS_HIST = int(np.ceil(T_MAX_HIST_S / DT))

# --- Unit inclusion ---
MIN_RATE_HZ = 0.5
MIN_VALID_TRIALS = 30
MIN_WAIT_S = 0.10                   # min window length for trial inclusion

# --- FDR / significance ---
LRT_ALPHA = 0.05
FDR_Q = 0.05

# A unit is a "sustained encoder" iff p_wait < LRT_ALPHA AND peak_lag >= this.
# Filters out units whose kernel peaks immediately past the basis floor
# (likely anchor-locked sensory tail extending past the event kernel rather
# than wait-period time encoding).
SUSTAINED_PEAK_LAG_S = 1.0

# --- Region grouping ---
EXCLUDE_REGION_GROUPS = ('Excluded', 'Other')

# --- Anchor specs ---
# Anchor-aligned event kernels (cue_off/cue_on, 0-500ms transients) absorb
# the brisk onset response at the anchor so wait_onset only tests the residual
# (sustained) time-encoding signal. Without these, visual onset transients in
# V1 / sensory-driven responses in any region inflate the wait_onset LRT
# because wait_onset at t≈0 captures the transient. Validated on RZ063
# str+v1: removing this caused V1 to come back at 100% sig with median χ²=496
# and 100% of high-χ² units having peak_lag<0.5s.
ANCHOR_SPECS: Dict[str, Dict[str, Any]] = {
    'cue_off': {
        'window_start_col': 'cue_off_time',
        'window_end_col':   'decision_time',
        'wait_t_max':       T_MAX_WAIT_CUE_OFF_S,
        'include_cue_off_event': True,     # cue_off is the anchor — absorb transient
        'include_cue_on_event':  False,    # cue_on is before window
        'include_lick_bg':       False,    # no licks in wait period
    },
    'cue_on': {
        'window_start_col': 'cue_on_time',
        'window_end_col':   'decision_time',
        'wait_t_max':       T_MAX_WAIT_CUE_ON_S,
        'include_cue_off_event': True,     # cue_off is mid-window event
        'include_cue_on_event':  True,     # cue_on is the anchor — absorb transient
        'include_lick_bg':       True,     # BG licks happen mid-window
    },
}

# --- Paths ---
OUT_DIR = Path(p.DATA_DIR) / 'time_encoding_two_anchors'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_UNIT_CSV = OUT_DIR / 'per_unit.csv'
REGION_SUMMARY_CSV = OUT_DIR / 'region_summary.csv'
KERNEL_CSV = OUT_DIR / 'per_unit_kernels.csv'
EFFECT_PLOT_PATH = OUT_DIR / 'region_two_anchor_distribution.png'

INCREMENTAL_SAVE = True


# =====================================================================
# Small helpers
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


def get_lick_bg_times_for_trial(events: pd.DataFrame, trial_id: int) -> np.ndarray:
    """Return BG-period lick times (trial-local seconds) for one trial."""
    ev = events.loc[events["trial_id"] == trial_id]
    if "event_start_trial_time" not in ev.columns:
        return np.array([])
    licks = ev.loc[ev["event_type"] == "lick_bg",
                   "event_start_trial_time"].astype(float).to_numpy()
    return licks


# =====================================================================
# Raised-cosine basis functions (Pillow-style)
# =====================================================================
def raised_cosine_log(t: np.ndarray, n_bases: int, t_max: float,
                     c: float = WAIT_BASIS_LOG_OFFSET,
                     t_min: float = 0.0) -> np.ndarray:
    """Log-spaced raised-cosine basis with 50% overlap.

    Peaks are placed in log space over [t_min, t_max]. Values are forced to 0
    for t < t_min (lets the wait basis avoid overlap with a separate event
    kernel that owns the [0, t_min) transient window).
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    log_t = np.log(np.maximum(t_arr, t_min) + c)
    log_min = np.log(t_min + c)
    log_max = np.log(t_max + c)
    spacing = (log_max - log_min) / (n_bases - 1)
    peaks = log_min + np.arange(n_bases) * spacing
    width = spacing
    delta = log_t[..., None] - peaks
    in_support = np.abs(delta) < width
    bases = np.zeros_like(delta)
    bases[in_support] = 0.5 * (np.cos(np.pi * delta[in_support] / width) + 1)
    bases[t_arr < t_min] = 0
    return bases


def raised_cosine_linear(t: np.ndarray, n_bases: int, t_max: float) -> np.ndarray:
    """Linear-spaced raised-cosine basis with 50% overlap."""
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
                          extra_t: float = 0.0,
                          log_offset: float = WAIT_BASIS_LOG_OFFSET,
                          t_min: float = 0.0,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute basis values on a discrete bin grid for fast lookup.

    `t_min` shifts the basis support so peaks span [t_min, t_max]; values are
    0 for t < t_min. Only used for the wait_onset basis.
    """
    n_grid = int(np.ceil((t_max + extra_t) / dt)) + 1
    grid = np.arange(n_grid) * dt
    if basis_type == 'log':
        vals = raised_cosine_log(grid, n_bases, t_max, c=log_offset, t_min=t_min)
    elif basis_type == 'linear':
        vals = raised_cosine_linear(grid, n_bases, t_max)
    else:
        raise ValueError(f"unknown basis_type: {basis_type}")
    return grid, vals


def lookup_basis_at_times(times: np.ndarray, basis_grid: np.ndarray,
                          basis_vals: np.ndarray) -> np.ndarray:
    """
    Look up basis values at `times` via nearest-bin indexing. Negative times
    return 0. Times past the grid extent return 0.

    Args:
        times: array of times to evaluate at (shape (n,))
        basis_grid: precomputed times where basis was evaluated (shape (G,))
        basis_vals: basis values (shape (G, n_bases))

    Returns:
        Array of shape (n, n_bases) with basis values.
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    n_bases = basis_vals.shape[1]
    out = np.zeros((n, n_bases))
    valid = (times >= 0) & (times < basis_grid[-1] + DT * 0.5)
    if not np.any(valid):
        return out
    grid_dt = basis_grid[1] - basis_grid[0] if len(basis_grid) > 1 else DT
    indices = np.clip((times[valid] / grid_dt).astype(int), 0, basis_vals.shape[0] - 1)
    out[valid] = basis_vals[indices]
    return out


# =====================================================================
# Per-trial design construction (anchor-aware)
# =====================================================================
def build_forward_kernel_at_anchor(bin_times: np.ndarray, anchor_offset: float,
                                   basis_grid: np.ndarray,
                                   basis_vals: np.ndarray) -> np.ndarray:
    """
    Build forward-kernel columns triggered by a single event at `anchor_offset`
    (within the window). For each bin, evaluate the basis at (bin_time - anchor_offset).

    Returns: (n_bins, n_bases)
    """
    time_since_event = bin_times - anchor_offset
    return lookup_basis_at_times(time_since_event, basis_grid, basis_vals)


def build_multi_event_kernel(bin_times: np.ndarray, event_offsets: np.ndarray,
                            basis_grid: np.ndarray,
                            basis_vals: np.ndarray) -> np.ndarray:
    """
    Build kernel columns summing over multiple events (e.g., multiple BG licks).
    Each event contributes a forward kernel; columns are the sum.

    Args:
        bin_times: bin times relative to window start (shape (n_bins,))
        event_offsets: event times relative to window start (shape (k,))
        basis_grid, basis_vals: precomputed basis

    Returns:
        (n_bins, n_bases) summed kernel
    """
    n_bins = len(bin_times)
    n_bases = basis_vals.shape[1]
    out = np.zeros((n_bins, n_bases))
    for E in event_offsets:
        if E < 0:
            continue
        # Contribution from this event
        time_since = bin_times - E
        out += lookup_basis_at_times(time_since, basis_grid, basis_vals)
    return out


def build_decision_lock_cols(bin_times: np.ndarray, T_window: float,
                            basis_grid: np.ndarray,
                            basis_vals: np.ndarray) -> np.ndarray:
    """
    Build backward-kernel columns anchored to the decision_lick (end of window).
    For each bin, evaluate the basis at (T_window - bin_time - DT).

    Returns: (n_bins, n_bases)
    """
    time_until_decision = T_window - bin_times - DT
    time_until_decision = np.clip(time_until_decision, 0, None)
    return lookup_basis_at_times(time_until_decision, basis_grid, basis_vals)


def build_spike_history_cols(spike_counts_full_trial: np.ndarray,
                            window_start_bin: int, n_bins_window: int,
                            basis_grid: np.ndarray, basis_vals: np.ndarray,
                            n_lag_bins: int) -> np.ndarray:
    """
    Causal convolution of past spike counts with the history basis.

    The full-trial spike train is used (including pre-anchor spikes), so early
    window bins still get proper history lookback.

    Returns: (n_bins_window, n_bases)
    """
    n_bases = basis_vals.shape[1]
    T_full = len(spike_counts_full_trial)

    # Basis values at lags 1*dt, 2*dt, ..., n_lag_bins*dt
    if len(basis_grid) < n_lag_bins + 1:
        lag_basis = np.zeros((n_lag_bins, n_bases))
        n_avail = min(n_lag_bins, len(basis_grid) - 1)
        if n_avail > 0:
            lag_basis[:n_avail] = basis_vals[1:1 + n_avail]
    else:
        lag_basis = basis_vals[1:1 + n_lag_bins]

    out = np.zeros((n_bins_window, n_bases))
    for t_local in range(n_bins_window):
        t_abs = window_start_bin + t_local
        for lag_idx in range(n_lag_bins):
            src_bin = t_abs - lag_idx - 1
            if 0 <= src_bin < T_full:
                out[t_local] += spike_counts_full_trial[src_bin] * lag_basis[lag_idx]
    return out


def build_per_trial_design(
    tr: pd.Series,
    spikes_trial: np.ndarray,
    lick_bg_times_trial: np.ndarray,
    anchor_spec: Dict[str, Any],
    bases: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (X, y, names) for one trial, given the anchor spec.

    Args:
        tr: trials-dataframe row
        spikes_trial: spike times in trial-local seconds
        lick_bg_times_trial: BG-period lick times (trial-local seconds)
        anchor_spec: dict with window_start_col, window_end_col, etc.
        bases: dict of precomputed basis arrays:
            'wait': {'grid': ..., 'vals': ...}
            'declock': {...}
            'cue_off': {...}
            'lick_bg': {...}
            'hist': {...}

    Returns:
        X: (n_bins_window, n_total_cols) design matrix
        y: (n_bins_window,) spike count per bin
        names: list of column names

    Raises:
        ValueError if the trial cannot be processed.
    """
    win_start_col = anchor_spec['window_start_col']
    win_end_col   = anchor_spec['window_end_col']
    include_cue_off_evt = anchor_spec['include_cue_off_event']
    include_cue_on_evt  = anchor_spec.get('include_cue_on_event', False)
    include_lick_bg     = anchor_spec['include_lick_bg']

    win_start_trial = float(tr[win_start_col])
    win_end_trial   = float(tr[win_end_col])
    if not (np.isfinite(win_start_trial) and np.isfinite(win_end_trial)):
        raise ValueError(f"non-finite window times: {win_start_trial}, {win_end_trial}")
    T_window = win_end_trial - win_start_trial
    if T_window < MIN_WAIT_S:
        raise ValueError(f"window too short: {T_window:.3f}s")

    # Build full-trial spike counts for spike-history lookback
    bg_start = float(tr.get('cue_on_time', 0.0))
    cons_len = float(tr.get('consumption_length', 3.0))
    if not np.isfinite(cons_len) or cons_len <= 0:
        cons_len = 3.0
    t_min = min(bg_start, 0.0)
    t_max_trial = win_end_trial + cons_len
    n_bins_full = int(np.ceil((t_max_trial - t_min) / DT))
    bin_edges_full = t_min + np.arange(n_bins_full + 1) * DT
    spike_counts_full, _ = np.histogram(spikes_trial, bins=bin_edges_full)

    # Window-relative bin times
    window_start_bin = int(round((win_start_trial - t_min) / DT))
    window_end_bin   = int(round((win_end_trial   - t_min) / DT))
    n_bins_window = window_end_bin - window_start_bin
    if n_bins_window < 2:
        raise ValueError("fewer than 2 window bins")
    bin_times = np.arange(n_bins_window) * DT  # relative to window start

    y = spike_counts_full[window_start_bin:window_end_bin].astype(float)

    col_blocks = []
    col_names: List[str] = []

    # --- Wait-onset kernel (forward from window start = anchor event) ---
    wait_cols = build_forward_kernel_at_anchor(
        bin_times, 0.0, bases['wait']['grid'], bases['wait']['vals']
    )
    col_blocks.append(wait_cols)
    col_names += [f'wait_onset_{j}' for j in range(wait_cols.shape[1])]

    # --- Cue_off event kernel (anchor for cue_off-anchor; mid-window for cue_on-anchor) ---
    if include_cue_off_evt:
        cue_off_offset = float(tr['cue_off_time']) - win_start_trial
        cue_off_cols = build_forward_kernel_at_anchor(
            bin_times, cue_off_offset,
            bases['cue_off']['grid'], bases['cue_off']['vals']
        )
        col_blocks.append(cue_off_cols)
        col_names += [f'cue_off_event_{j}' for j in range(cue_off_cols.shape[1])]

    # --- Cue_on event kernel (anchor for cue_on-anchor only) ---
    if include_cue_on_evt:
        cue_on_offset = float(tr['cue_on_time']) - win_start_trial
        cue_on_cols = build_forward_kernel_at_anchor(
            bin_times, cue_on_offset,
            bases['cue_on']['grid'], bases['cue_on']['vals']
        )
        col_blocks.append(cue_on_cols)
        col_names += [f'cue_on_event_{j}' for j in range(cue_on_cols.shape[1])]

    # --- Lick_bg event kernels (only for cue_on anchor) ---
    if include_lick_bg:
        # Convert lick times (trial-local) to window-relative offsets
        if len(lick_bg_times_trial) > 0:
            lick_offsets = lick_bg_times_trial - win_start_trial
            # Keep only licks that fall within the window
            lick_offsets = lick_offsets[
                (lick_offsets >= 0) & (lick_offsets < T_window)
            ]
        else:
            lick_offsets = np.array([])
        lick_cols = build_multi_event_kernel(
            bin_times, lick_offsets,
            bases['lick_bg']['grid'], bases['lick_bg']['vals']
        )
        col_blocks.append(lick_cols)
        col_names += [f'lick_bg_{j}' for j in range(lick_cols.shape[1])]

    # --- Decision-locked backward kernel (always) ---
    declock_cols = build_decision_lock_cols(
        bin_times, T_window,
        bases['declock']['grid'], bases['declock']['vals']
    )
    col_blocks.append(declock_cols)
    col_names += [f'decision_lock_{j}' for j in range(declock_cols.shape[1])]

    # --- Spike history (always) ---
    hist_cols = build_spike_history_cols(
        spike_counts_full, window_start_bin, n_bins_window,
        bases['hist']['grid'], bases['hist']['vals'], N_LAG_BINS_HIST
    )
    col_blocks.append(hist_cols)
    col_names += [f'spike_hist_{j}' for j in range(hist_cols.shape[1])]

    X = np.hstack(col_blocks)
    return X, y, col_names


# =====================================================================
# Per-unit GLM fit
# =====================================================================
def fit_glm_unreg(X: np.ndarray, y: np.ndarray) -> Tuple[Optional[Any], float]:
    """Unregularized IRLS Poisson GLM. Returns (result, training log-likelihood)."""
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


def fit_per_unit(X_all: np.ndarray, y_all: np.ndarray, names: List[str]) -> Dict[str, Any]:
    """
    Fit M_full and M_null, compute LRT for the wait_onset family.

    M_null:  all columns EXCEPT wait_onset (decision_lock + spike_history +
             cue_off_event + lick_bg, depending on anchor)
    M_full:  M_null + wait_onset
    """
    cols_arr = np.array(names)
    wait_idx = np.where(np.char.startswith(cols_arr, 'wait_onset_'))[0]
    null_idx = np.where(~np.char.startswith(cols_arr, 'wait_onset_'))[0]

    res_full, ll_full = fit_glm_unreg(X_all, y_all)
    if res_full is None:
        return _empty_fit_result()
    Xc_full = sm.add_constant(X_all, has_constant='add')
    mu_full = np.clip(res_full.predict(Xc_full), 1e-9, None)
    pr2_full = float(pseudo_R2_mcfadden(y_all, mu_full, y_all.mean()))

    X_null = X_all[:, null_idx] if len(null_idx) else np.zeros((len(y_all), 0))
    _, ll_null = fit_glm_unreg(X_null, y_all)
    _, ll_int = fit_glm_unreg(np.zeros((len(y_all), 0)), y_all)

    df_wait = int(len(wait_idx))
    df_full_vs_int = int(X_all.shape[1])
    chi2_wait = -2.0 * (ll_null - ll_full) if np.isfinite(ll_null) else np.nan
    chi2_full_vs_int = -2.0 * (ll_int - ll_full) if np.isfinite(ll_int) else np.nan
    p_wait = lrt_pvalue(ll_null, ll_full, df_wait)
    p_full = lrt_pvalue(ll_int, ll_full, df_full_vs_int)

    params = np.asarray(res_full.params)
    beta_intercept = float(params[0])
    beta_rest = params[1:]
    beta_wait = beta_rest[wait_idx].tolist()

    return {
        'fit_status': 'ok',
        'n_bins': int(len(y_all)),
        'mean_rate_hz': float(y_all.sum() / (len(y_all) * DT)),
        'beta_intercept': beta_intercept,
        'beta_wait': beta_wait,
        'll_full': float(ll_full),
        'll_null': float(ll_null) if np.isfinite(ll_null) else np.nan,
        'll_intercept': float(ll_int) if np.isfinite(ll_int) else np.nan,
        'df_wait': df_wait,
        'df_full_vs_int': df_full_vs_int,
        'chi2_wait': float(chi2_wait) if np.isfinite(chi2_wait) else np.nan,
        'chi2_full_vs_int': float(chi2_full_vs_int) if np.isfinite(chi2_full_vs_int) else np.nan,
        'p_wait': float(p_wait) if np.isfinite(p_wait) else np.nan,
        'p_full_vs_int': float(p_full) if np.isfinite(p_full) else np.nan,
        'pseudo_r2': pr2_full,
    }


def _empty_fit_result() -> Dict[str, Any]:
    return {
        'fit_status': 'failed',
        'n_bins': 0,
        'mean_rate_hz': np.nan,
        'beta_intercept': np.nan,
        'beta_wait': [np.nan] * (N_BASIS_WAIT - 1),
        'll_full': np.nan, 'll_null': np.nan, 'll_intercept': np.nan,
        'df_wait': N_BASIS_WAIT - 1, 'df_full_vs_int': np.nan,
        'chi2_wait': np.nan, 'chi2_full_vs_int': np.nan,
        'p_wait': np.nan, 'p_full_vs_int': np.nan,
        'pseudo_r2': np.nan,
    }


def compute_wait_kernel_peak(beta_wait: List[float], wait_grid: np.ndarray,
                            wait_vals: np.ndarray, t_max: float) -> Tuple[float, float]:
    """Evaluate fitted wait kernel and return (peak_lag_s, peak_value)."""
    beta = np.asarray(beta_wait)
    if np.any(~np.isfinite(beta)):
        return np.nan, np.nan
    kernel_vals = wait_vals @ beta
    mask = wait_grid <= t_max
    if not np.any(mask):
        return np.nan, np.nan
    t_grid = wait_grid[mask]
    k = kernel_vals[mask]
    peak_idx = int(np.argmax(k))
    return float(t_grid[peak_idx]), float(k[peak_idx])


# =====================================================================
# Build full design for one unit (stacked across trials), per anchor
# =====================================================================
def build_design_for_unit_one_anchor(
    spikes_by_trial: Dict[int, np.ndarray],
    trials_idx: pd.DataFrame,
    events: pd.DataFrame,
    anchor_spec: Dict[str, Any],
    bases: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
           Optional[List[str]], List[int]]:
    """
    Build (X_all, y_all, names, used_tids) for one unit by stacking trials.
    Returns (None, None, None, []) if no trials qualify.
    """
    X_blocks, y_blocks = [], []
    names: Optional[List[str]] = None
    used: List[int] = []

    for tid, sp in spikes_by_trial.items():
        if tid not in trials_idx.index:
            continue
        tr = trials_idx.loc[tid]
        if bool(coerce_bool_series(pd.Series([tr.get('missed', False)])).iloc[0]):
            continue
        try:
            lick_bg = (get_lick_bg_times_for_trial(events, int(tid))
                       if anchor_spec['include_lick_bg'] else np.array([]))
            X, y, nm = build_per_trial_design(
                tr, sp, lick_bg, anchor_spec, bases
            )
        except Exception:
            continue
        X_blocks.append(X)
        y_blocks.append(y)
        if names is None:
            names = nm
        used.append(int(tid))

    if not X_blocks:
        return None, None, names, used
    X_all = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    return X_all, y_all, names, used


# =====================================================================
# Anatomical labels (matches 4c/4d/4e convention)
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
# Precompute all bases once
# =====================================================================
def _orthog_constant_and_drop(vals: np.ndarray, t_max_s: float,
                              t_min_s: float = 0.0) -> np.ndarray:
    """
    Make a raised-cosine basis identifiable in the presence of an intercept.

    Tiled raised cosines with 50% overlap satisfy sum_j B_j(t) ~= 1 across the
    bulk of their support, which is collinear with the intercept column in a
    GLM. The fitter then parks the baseline in the basis betas (kernel plot
    becomes a flat constant offset) and the LRT df is over-counted by one.

    Fix: subtract the column-mean (computed only over the active modeled
    support [t_min_s, t_max_s], not the pre-window or extra_t tail) so each
    column is zero-mean inside the active range, then drop the last column.
    Outside [t_min_s, t_max_s] the values are forced back to 0 so the basis
    remains exactly zero in the [0, t_min_s) window that the event kernel
    owns. Result: rank-(n-1) basis with no constant direction.
    """
    n_bin_lo = int(np.floor(t_min_s / DT))
    n_bin_hi = min(vals.shape[0], max(n_bin_lo + 1, int(np.ceil(t_max_s / DT))))
    means = vals[n_bin_lo:n_bin_hi].mean(axis=0, keepdims=True)
    out = vals - means
    if n_bin_lo > 0:
        out[:n_bin_lo] = 0.0
    return out[:, :-1]


def precompute_all_bases() -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Precompute basis arrays for both anchors. Some bases (declock, hist) are
    shared; wait basis differs by anchor (different t_max).

    The wait basis is orthogonalized against a constant and rank-reduced to
    N_BASIS_WAIT-1 to break collinearity with the intercept. See
    _orthog_constant_and_drop docstring.

    Returns nested dict: bases_by_anchor[anchor_name][family] = {'grid', 'vals'}
    """
    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    # Wait basis: differs per anchor (different t_max). Support shifted to
    # [T_MIN_WAIT_S, t_max] so the anchor-event kernel can own the [0, T_MIN_WAIT_S)
    # transient window without collinear competition from wait_onset's earliest
    # log-spaced bumps.
    wait_cue_off_grid, wait_cue_off_vals = precompute_basis_grid(
        'log', N_BASIS_WAIT, T_MAX_WAIT_CUE_OFF_S, DT, extra_t=5.0,
        t_min=T_MIN_WAIT_S,
    )
    wait_cue_on_grid, wait_cue_on_vals = precompute_basis_grid(
        'log', N_BASIS_WAIT, T_MAX_WAIT_CUE_ON_S, DT, extra_t=5.0,
        t_min=T_MIN_WAIT_S,
    )
    wait_cue_off_vals = _orthog_constant_and_drop(
        wait_cue_off_vals, T_MAX_WAIT_CUE_OFF_S, t_min_s=T_MIN_WAIT_S,
    )
    wait_cue_on_vals = _orthog_constant_and_drop(
        wait_cue_on_vals, T_MAX_WAIT_CUE_ON_S, t_min_s=T_MIN_WAIT_S,
    )

    # Shared bases
    declock_grid, declock_vals = precompute_basis_grid(
        'linear', N_BASIS_DECLOCK, T_MAX_DECLOCK_S, DT
    )
    hist_grid, hist_vals = precompute_basis_grid(
        'log', N_BASIS_HIST, T_MAX_HIST_S, DT, log_offset=HIST_BASIS_LOG_OFFSET
    )
    cue_off_grid, cue_off_vals = precompute_basis_grid(
        'linear', N_BASIS_CUE_OFF, T_MAX_CUE_OFF_S, DT
    )
    lick_grid, lick_vals = precompute_basis_grid(
        'log', N_BASIS_LICK, T_MAX_LICK_S, DT, log_offset=LICK_BASIS_LOG_OFFSET
    )

    # cue_on event basis: same shape as cue_off (short transient, 5 linear, 0-500ms).
    cue_on_grid, cue_on_vals = precompute_basis_grid(
        'linear', N_BASIS_CUE_OFF, T_MAX_CUE_OFF_S, DT
    )

    out['cue_off'] = {
        'wait':    {'grid': wait_cue_off_grid, 'vals': wait_cue_off_vals},
        'declock': {'grid': declock_grid,      'vals': declock_vals},
        'hist':    {'grid': hist_grid,         'vals': hist_vals},
        'cue_off': {'grid': cue_off_grid,      'vals': cue_off_vals},
    }
    out['cue_on'] = {
        'wait':    {'grid': wait_cue_on_grid,  'vals': wait_cue_on_vals},
        'declock': {'grid': declock_grid,      'vals': declock_vals},
        'hist':    {'grid': hist_grid,         'vals': hist_vals},
        'cue_off': {'grid': cue_off_grid,      'vals': cue_off_vals},
        'cue_on':  {'grid': cue_on_grid,       'vals': cue_on_vals},
        'lick_bg': {'grid': lick_grid,         'vals': lick_vals},
    }
    return out


# =====================================================================
# Main run loop
# =====================================================================
def run(session_ids: Optional[List[str]] = None):
    """Fit Test 1 on the given sessions, both anchors per unit."""
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

    print(f"\nWill run {len(session_ids)} session(s) with anchors: "
          f"{list(ANCHOR_SPECS.keys())}")
    print(f"Wait kernel cue_off: {N_BASIS_WAIT - 1} log bases over 0-{T_MAX_WAIT_CUE_OFF_S}s "
          f"(constant orthogonalized; df={N_BASIS_WAIT - 1})")
    print(f"Wait kernel cue_on:  {N_BASIS_WAIT - 1} log bases over 0-{T_MAX_WAIT_CUE_ON_S}s "
          f"(constant orthogonalized; df={N_BASIS_WAIT - 1})\n")

    bases_by_anchor = precompute_all_bases()

    anatomy = load_anatomical_labels()
    anatomy_idx = (
        anatomy.set_index(['session_id', 'id'], drop=False) if not anatomy.empty else None
    )

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
        n_ok_both = 0
        for ui, (_, unit_info) in enumerate(session_units.iterrows(), start=1):
            unit_id = unit_info['unit_id']
            unit_key = unit_info['id']

            try:
                spikes_df = units[unit_key]
            except KeyError:
                continue
            spikes_by_trial = spikes_df_to_trial_map(spikes_df)

            # Run both anchors
            results_per_anchor: Dict[str, Dict[str, Any]] = {}
            beta_wait_per_anchor: Dict[str, List[float]] = {}
            peak_lag_per_anchor: Dict[str, float] = {}

            for anchor_name, anchor_spec in ANCHOR_SPECS.items():
                X_all, y_all, names, used = build_design_for_unit_one_anchor(
                    spikes_by_trial, trials_idx, events,
                    anchor_spec, bases_by_anchor[anchor_name]
                )
                if X_all is None or len(used) < MIN_VALID_TRIALS:
                    results_per_anchor[anchor_name] = _empty_fit_result()
                    results_per_anchor[anchor_name]['n_trials_used'] = len(used)
                    beta_wait_per_anchor[anchor_name] = [np.nan] * (N_BASIS_WAIT - 1)
                    peak_lag_per_anchor[anchor_name] = np.nan
                    continue

                mean_rate = float(y_all.sum() / (len(y_all) * DT))
                if mean_rate < MIN_RATE_HZ:
                    results_per_anchor[anchor_name] = _empty_fit_result()
                    results_per_anchor[anchor_name]['n_trials_used'] = len(used)
                    results_per_anchor[anchor_name]['mean_rate_hz'] = mean_rate
                    results_per_anchor[anchor_name]['fit_status'] = 'below_min_rate'
                    beta_wait_per_anchor[anchor_name] = [np.nan] * (N_BASIS_WAIT - 1)
                    peak_lag_per_anchor[anchor_name] = np.nan
                    continue

                stats_d = fit_per_unit(X_all, y_all, names)
                stats_d['n_trials_used'] = int(len(used))

                # Compute peak descriptively
                wait_grid = bases_by_anchor[anchor_name]['wait']['grid']
                wait_vals = bases_by_anchor[anchor_name]['wait']['vals']
                wait_t_max = anchor_spec['wait_t_max']
                peak_lag, peak_val = compute_wait_kernel_peak(
                    stats_d['beta_wait'], wait_grid, wait_vals, wait_t_max
                )
                stats_d['peak_lag_s'] = peak_lag
                stats_d['peak_value'] = peak_val

                # Save beta_wait separately; drop from main stats dict
                beta_wait_per_anchor[anchor_name] = stats_d.pop('beta_wait')
                peak_lag_per_anchor[anchor_name] = peak_lag
                results_per_anchor[anchor_name] = stats_d

            # Skip unit entirely if BOTH anchors failed
            if all(r.get('fit_status') != 'ok' for r in results_per_anchor.values()):
                continue

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

            # Build per-unit row with suffixed columns per anchor
            row = {
                'session_id': sid,
                'unit_id': unit_id,
                'unit_key': unit_key,
                'probe_region': probe_region,
                'corrected_region': corrected_region,
                'region_group': region_group,
                'cell_type': cell_type,
            }
            for anchor_name in ANCHOR_SPECS.keys():
                for k, v in results_per_anchor[anchor_name].items():
                    row[f'{k}_{anchor_name}'] = v
                # Sustained-encoder flag: significant LRT AND late peak.
                # See SUSTAINED_PEAK_LAG_S docstring for rationale.
                p_val = results_per_anchor[anchor_name].get('p_wait', np.nan)
                peak = results_per_anchor[anchor_name].get('peak_lag_s', np.nan)
                row[f'sustained_sig_{anchor_name}'] = bool(
                    np.isfinite(p_val) and p_val < LRT_ALPHA
                    and np.isfinite(peak) and peak >= SUSTAINED_PEAK_LAG_S
                )

            all_rows.append(row)
            n_ok_both += 1

            # Kernel coefficients for plotting
            for anchor_name, beta_wait in beta_wait_per_anchor.items():
                for j, b in enumerate(beta_wait):
                    all_kernel_rows.append({
                        'session_id': sid,
                        'unit_id': unit_id,
                        'corrected_region': corrected_region,
                        'region_group': region_group,
                        'anchor': anchor_name,
                        'basis_idx': j,
                        'beta': float(b) if np.isfinite(b) else np.nan,
                    })

            if INCREMENTAL_SAVE and (ui % 25 == 0 or ui == n_units):
                pd.DataFrame(all_rows).to_csv(PER_UNIT_CSV, index=False)
                pd.DataFrame(all_kernel_rows).to_csv(KERNEL_CSV, index=False)

        print(f"[session done] {sid}: {n_ok_both}/{n_units} units fit")

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
    """
    Per-region summary, reporting both anchors and a cross-tabulation.
    """
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
    if df.empty:
        print("[summary] no usable rows after filters.")
        return pd.DataFrame()

    rows = []
    for region, sub in df.groupby(group_col):
        row = {group_col: region, 'n_units': int(len(sub))}

        for anchor in ANCHOR_SPECS.keys():
            p_col = f'p_wait_{anchor}'
            chi2_col = f'chi2_wait_{anchor}'
            peak_col = f'peak_lag_s_{anchor}'
            status_col = f'fit_status_{anchor}'

            sub_ok = sub.loc[sub[status_col] == 'ok'] if status_col in sub.columns else sub
            p = sub_ok[p_col].dropna().values if p_col in sub_ok.columns else np.array([])

            n_ok = int(len(sub_ok))
            if p.size == 0:
                row[f'n_ok_{anchor}'] = n_ok
                row[f'frac_sig_uncorr_{anchor}'] = np.nan
                row[f'frac_sig_fdr_{anchor}'] = np.nan
                row[f'median_chi2_{anchor}'] = np.nan
                row[f'median_peak_lag_{anchor}'] = np.nan
                continue

            rej_uncorr = p < LRT_ALPHA
            rej_fdr = bh_fdr(p, q=FDR_Q)
            chi2_vals = sub_ok[chi2_col].dropna().values
            peaks = sub_ok[peak_col].dropna().values

            row[f'n_ok_{anchor}'] = n_ok
            row[f'frac_sig_uncorr_{anchor}'] = float(rej_uncorr.mean())
            row[f'frac_sig_fdr_{anchor}'] = float(rej_fdr.mean())
            row[f'median_chi2_{anchor}'] = (
                float(np.median(chi2_vals)) if chi2_vals.size else np.nan
            )
            row[f'median_peak_lag_{anchor}'] = (
                float(np.median(peaks)) if peaks.size else np.nan
            )

            # Sustained encoders (precomputed column on per_unit.csv).
            sustained_col = f'sustained_sig_{anchor}'
            if sustained_col in sub_ok.columns:
                n_sust = int(sub_ok[sustained_col].fillna(False).astype(bool).sum())
                row[f'n_sustained_{anchor}'] = n_sust
                row[f'frac_sustained_{anchor}'] = n_sust / n_ok if n_ok > 0 else np.nan

        # Cross-tabulation: which anchor(s) pass per unit (FDR within region)
        sig_co = (sub['fit_status_cue_off'] == 'ok') & sub.get('p_wait_cue_off', pd.Series()).notna()
        sig_cn = (sub['fit_status_cue_on']  == 'ok') & sub.get('p_wait_cue_on',  pd.Series()).notna()
        if sig_co.any() and sig_cn.any():
            p_co = sub.loc[sig_co, 'p_wait_cue_off'].values
            p_cn = sub.loc[sig_cn, 'p_wait_cue_on'].values
            rej_co = pd.Series(False, index=sub.index)
            rej_cn = pd.Series(False, index=sub.index)
            rej_co.loc[sig_co] = bh_fdr(p_co, q=FDR_Q)
            rej_cn.loc[sig_cn] = bh_fdr(p_cn, q=FDR_Q)
            n_both    = int((rej_co & rej_cn).sum())
            n_only_co = int((rej_co & ~rej_cn).sum())
            n_only_cn = int((~rej_co & rej_cn).sum())
            n_neither = int((~rej_co & ~rej_cn).sum())
            row['n_sig_both']       = n_both
            row['n_sig_only_cue_off'] = n_only_co
            row['n_sig_only_cue_on']  = n_only_cn
            row['n_sig_neither']    = n_neither

        rows.append(row)

    summary = pd.DataFrame(rows)
    if 'frac_sig_fdr_cue_off' in summary.columns:
        summary = summary.sort_values('frac_sig_fdr_cue_off', ascending=False)
    summary.to_csv(REGION_SUMMARY_CSV, index=False)
    print(f"\nSaved → {REGION_SUMMARY_CSV}")
    # Print key columns
    key_cols = [c for c in summary.columns
                if c in (group_col, 'n_units')
                or 'frac_sig_fdr' in c
                or 'frac_sustained' in c
                or 'median_chi2' in c
                or c.startswith('n_sig_')]
    print(summary[key_cols].to_string(index=False))
    return summary


# =====================================================================
# Plot
# =====================================================================
def plot_distributions(per_unit_csv: Path = PER_UNIT_CSV,
                      group_col: str = 'region_group',
                      exclude_groups: Tuple[str, ...] = EXCLUDE_REGION_GROUPS):
    """Plot chi² distributions per region, side-by-side for both anchors."""
    if not per_unit_csv.exists():
        return
    df = pd.read_csv(per_unit_csv)
    if group_col not in df.columns or df[group_col].isna().all():
        group_col = 'probe_region'
    df = df.loc[df[group_col].notna() & (df[group_col] != '')]
    if exclude_groups:
        df = df.loc[~df[group_col].isin(exclude_groups)]
    if df.empty:
        return

    regions = sorted(df[group_col].unique())
    n = len(regions)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 5.5), sharex='row')
    if n == 1:
        axes = axes[:, None]

    for col, reg in enumerate(regions):
        sub = df.loc[df[group_col] == reg]

        # Top: chi² distribution per anchor
        ax = axes[0, col]
        for anchor, color in zip(['cue_off', 'cue_on'], ['C0', 'C1']):
            chi2_col = f'chi2_wait_{anchor}'
            if chi2_col not in sub.columns:
                continue
            vals = sub[chi2_col].dropna().values
            if vals.size == 0:
                continue
            ax.hist(vals, bins=30, color=color, edgecolor='k', alpha=0.5,
                    label=anchor)
        ax.axvline(N_BASIS_WAIT - 1, color='r', ls='--', lw=0.6,
                   label=f'df={N_BASIS_WAIT - 1}')
        ax.set_title(f"{reg} (n={len(sub)})", fontsize=10)
        if col == 0:
            ax.set_ylabel('# units')
        ax.set_xlabel('χ² (wait LRT)')
        ax.legend(fontsize=7)

        # Bottom: peak lag per anchor
        ax = axes[1, col]
        for anchor, color in zip(['cue_off', 'cue_on'], ['C0', 'C1']):
            peak_col = f'peak_lag_s_{anchor}'
            if peak_col not in sub.columns:
                continue
            vals = sub[peak_col].dropna().values
            if vals.size == 0:
                continue
            ax.hist(vals, bins=20, color=color, edgecolor='k', alpha=0.5,
                    label=anchor)
        if col == 0:
            ax.set_ylabel('# units')
        ax.set_xlabel('peak lag (s)')
        ax.legend(fontsize=7)

    fig.suptitle('Test 1 (two-anchor): time encoding during wait period', fontsize=11)
    fig.tight_layout()
    fig.savefig(EFFECT_PLOT_PATH, dpi=150)
    print(f"Saved → {EFFECT_PLOT_PATH}")
    plt.close(fig)


# =====================================================================
# Debug
# =====================================================================
def debug_one(session_id: str, unit_id: int):
    """Run both anchors on one unit; print details and plot fitted kernels."""
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

    bases_by_anchor = precompute_all_bases()

    events, trials, units = utils.get_session_data(session_id)
    if 'missed' in trials.columns:
        trials = trials.loc[~coerce_bool_series(trials['missed'])].copy()
    trials_idx = trials.set_index('trial_id', drop=False)
    spikes_df = units[unit_key]
    spikes_by_trial = spikes_df_to_trial_map(spikes_df)

    fig, axes = plt.subplots(1, len(ANCHOR_SPECS), figsize=(6 * len(ANCHOR_SPECS), 3),
                            squeeze=False)

    for ax_idx, (anchor_name, anchor_spec) in enumerate(ANCHOR_SPECS.items()):
        print(f"\n--- Anchor: {anchor_name} ---")
        X_all, y_all, names, used = build_design_for_unit_one_anchor(
            spikes_by_trial, trials_idx, events,
            anchor_spec, bases_by_anchor[anchor_name]
        )
        if X_all is None:
            print(f"[debug] no design built for {anchor_name}")
            continue
        print(f"[debug] trials used: {len(used)}")
        print(f"[debug] X shape: {X_all.shape}, y shape: {y_all.shape}")
        print(f"[debug] mean rate: {y_all.sum() / (len(y_all) * DT):.2f} Hz")
        print(f"[debug] column families: {set(n.rsplit('_', 1)[0] for n in names)}")

        stats_d = fit_per_unit(X_all, y_all, names)
        for k, v in stats_d.items():
            if k == 'beta_wait':
                continue
            if isinstance(v, float):
                v_str = f"{v:.4g}"
            else:
                v_str = str(v)
            print(f"   {k:>22s}: {v_str}")

        # Plot fitted kernel
        wait_grid = bases_by_anchor[anchor_name]['wait']['grid']
        wait_vals = bases_by_anchor[anchor_name]['wait']['vals']
        wait_t_max = anchor_spec['wait_t_max']
        peak_lag, peak_val = compute_wait_kernel_peak(
            stats_d['beta_wait'], wait_grid, wait_vals, wait_t_max
        )
        beta = np.asarray(stats_d['beta_wait'])
        kernel_vals = wait_vals @ beta
        mask = wait_grid <= wait_t_max
        ax = axes[0, ax_idx]
        ax.plot(wait_grid[mask], kernel_vals[mask], color='C0', lw=1.5)
        ax.axhline(0, color='k', lw=0.5)
        if np.isfinite(peak_lag):
            ax.axvline(peak_lag, color='r', ls='--', lw=0.6,
                       label=f'peak {peak_lag:.2f}s')
        ax.set_xlabel(f'time since {anchor_name} (s)')
        ax.set_ylabel('fitted wait kernel (log-rate units)')
        ax.set_title(f"{anchor_name}: χ²={stats_d['chi2_wait']:.1f}, "
                     f"p={stats_d['p_wait']:.3g}")
        ax.legend()

    fig.tight_layout()
    plot_path = OUT_DIR / f'debug_{session_id}_u{unit_id}.png'
    fig.savefig(plot_path, dpi=150)
    print(f"\n[debug] plot → {plot_path}")
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
