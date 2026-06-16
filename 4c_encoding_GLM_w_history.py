#!/usr/bin/env python3
"""
4c_encoding_GLM_w_history.py

Per-unit Poisson GLM for encoding analysis with full trial-history block,
nested model comparison via LRT, ridge regularization, and trial-stratified CV.

This is a refactor of 4c_encoding_GLM_w_convolution_take2.py with the
following changes per the GLM design plan:

ADDITIONS (load-bearing for the reward-history analysis):
  - Trial-history block: prev_rewarded, prev_missed, prev_wait_z,
    num_bg_repeats_z, trial_number_z (constant within trial, broadcast).
  - Interaction columns: prev_rewarded × wait_time_basis,
    prev_wait_z × wait_time_basis  ← the HEADLINE predictors.
  - Optional multi-step history (M3) and exponentially-weighted reward rate (M4).
  - Nested model comparison framework (M0 → M1 → M2 → M3 → M4) with
    formal likelihood-ratio tests.
  - Unit-inclusion criteria (min spikes, max FR, min trials per prev_reward).

MODIFICATIONS:
  - L1_wt = 0 (pure ridge) instead of 0.5 (elastic net), so all coefficients
    are interpretable.
  - Trial-stratified CV (splits by trial_id) instead of sequential time-bin
    CV — avoids leakage through spike-history kernels.

KEPT FROM ORIGINAL (these designs are good):
  - Period-masked time bases (bg_time_*, wait_time_*).
  - Hazard kernel orthogonalized to wait-time basis.
  - Per-context lick kernels (lick_bg, lick_decision, lick_cons).
  - Spike-history kernel.
  - Two-stage fit → evaluate workflow with artifact saving.
  - Per-family ΔpseudoR² for descriptive output.
  - Rule-based unit classification (kept as secondary descriptive output).

NESTED MODELS:
  M0  baseline:    time + within-trial events + spike history
  M1  + reward main effects:   prev_rewarded, prev_missed, prev_wait_z,
                                 num_bg_repeats_z, trial_number_z
  M2  + reward × time:          prev_rewarded × wait_time_basis,
                                 prev_wait_z × wait_time_basis  ← HEADLINE
  M3  + multi-step history:     prev_rewarded_2back × wait_time_basis,
                                 prev_rewarded_3back × wait_time_basis
  M4  + EW reward rate:         ew_reward_rate_tau{N} × wait_time_basis

Headline test per unit: LRT M2 vs M1 — does reward history modulate the
time code beyond a simple main-effect shift?

EXTERNAL DEPENDENCIES (not modified by this script):
  - utils:               get_session_data(), get_data_for_debugging()
  - paths (as p):        LOGS_DIR, DATA_DIR
  - glm_design_helpers:  assemble_design(), make_time_bins()
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
from patsy import dmatrix

import utils
import paths as p
from glm_design_helpers import assemble_design, make_time_bins


# =====================================================================
# Config
# =====================================================================
# --- Bin size and basis ---
DT = 0.025                # 25 ms bin size (per design plan; was 0.020 in v2)
N_BASIS_BG = 8            # ramp within BG
N_BASIS_WAIT = 8          # ramp within WAIT
N_BASIS_BGWAIT = 0        # combined BG+WAIT basis disabled
HAZARD_TAU = 3.0          # reward hazard time constant
INCLUDE_SPIKE_HISTORY = True

# --- Absolute-time wait/BG basis ---
# True (default): basis built on a fixed [0, T_MAX] grid and evaluated at
# each bin's local time since epoch onset. basis_j(t) takes the same value
# across trials at the same absolute t. Bins past T_MAX get clamped (rare).
#
# False: legacy duration-warped basis (per-trial basis stretched to fit
# that trial's wait window). Documented as the source of the M1_vs_M2
# anti-conservatism on high-pR² units (see 4c_shuffle_control on MOp6a u224
# and the wait-quartile stratified diagnostic).
ABSOLUTE_TIME_BASIS = True
T_MAX_BG = 5.0            # fixed-grid upper bound for BG basis (seconds)
T_MAX_WAIT = 10.0         # fixed-grid upper bound for WAIT basis (seconds)

# --- Wait-band trial filter (per bg_group) ---
# Restricts each session's analysis to trials whose current wait duration
# falls within a group-appropriate band. Reduces the conditional-wait-
# distribution confound (prev_rewarded → current wait coupling, |Cohen's d|
# up to ~0.4 across sessions) that survives the absolute-time basis fix.
# Without this, the M1_vs_M2 LRT remains anti-conservative on high-pR²
# units even after the duration-warping fix (~85% shuffle p<0.05 instead
# of nominal 5%).
#
# Bands chosen as ~Q25–Q75 of each group's per-session wait distribution
# (median over 52 Short BG / 44 Long BG sessions); within-band SD ≈ 0.6–0.7s
# in each group so the residual conditional-wait shift is small in ratio.
APPLY_WAIT_BAND_FILTER = True
WAIT_BAND_SHORT_BG = (1.0, 3.0)   # (lo, hi) seconds; Short BG mice
WAIT_BAND_LONG_BG  = (2.0, 4.5)   # (lo, hi) seconds; Long BG mice

# --- Regularization ---
ALPHA = 0.1               # ridge strength (consider CV per region in future)
L1_WT = 0.0               # pure ridge (was 0.5 = elastic net in v2)

# --- Trial-history block ---
INCLUDE_TRIAL_HISTORY = True       # M1 predictors
INCLUDE_INTERACTIONS = True        # M2 interaction kernels (the headline)
INCLUDE_MULTISTEP_HISTORY = False  # M3 — off for first pass
INCLUDE_REWARD_RATE = False        # M4 — off for first pass
REWARD_RATE_TAUS = [3.0, 10.0]     # EW half-lives (in trials) for M4
MAX_HISTORY_BACK = 3               # M3 lookback depth (2-back, 3-back)

# --- Drift control ---
# Natural cubic regression spline on trial_number_z, broadcast per-trial.
# Sits in every nested model (M0 → M4) so each subsequent test asks
# "beyond smooth session-long drift" rather than including it as signal.
# Without this, slow trial-to-trial firing-rate changes (motivation drift,
# satiation) can be soaked up by the M1 history block (which is constant
# within trial — see compute_nested_lrts docstring on M0_vs_M1).
N_DRIFT_BASIS = 5

# --- Current-wait-duration confound control ---
# E[current wait | prev_rewarded] differs from E[current wait | not prev_rewarded]
# (mice change how long they wait based on prior outcome — confirmed
# behaviorally; |Cohen's d| up to ~0.4 in some sessions). When the design's
# `prev_rewarded × wait_time_basis_j` interaction is formed against this,
# it picks up CURRENT-trial wait-duration-dependent firing rather than a
# genuine "prev outcome modulates time code" signal — and the same is true
# for prev_wait_z (which is mechanically autocorrelated with current wait).
#
# Fix: include current_wait_z × wait_time_basis_j as a CONFOUND family in
# every nested model (M0 through M4). It's never tested by any LRT —
# it's there only to absorb current-wait-dependent structure so the M2
# interaction block has to explain something beyond it.
INCLUDE_CURRENT_WAIT_CONTROL = True

# --- Unit inclusion ---
MIN_SPIKE_COUNT = 50
MAX_MEAN_FR_HZ = 50.0
MIN_TRIALS_PER_PREV_REWARD = 30
MIN_TRIALS_PREV_NOT_MISS = 30

# --- GLM noise floor (soft inclusion) ---
# Units with in-sample pR² below this are still fit and have valid LRTs in
# principle, but are flagged `below_glm_noise_floor=True` in the metrics CSV
# so the headline analysis can subset cleanly. Empirical floor: on the
# diversity panel, units with pR² ~ 0.01 (e.g., RZ049 u103) showed flat
# Poisson likelihood landscapes where reward history × time can't be
# distinguished from noise even when the unit rescales by other metrics.
MIN_PR2_HEADLINE = 0.02

# --- Classification (descriptive only; LRT is the formal test) ---
PR2_THR_BG   = 0.002
PR2_THR_WAIT = 0.002
PR2_THR_HAZ  = 0.0005

# --- LRT / FDR ---
LRT_ALPHA = 0.05          # per-unit nominal significance
FDR_Q = 0.05              # within-region FDR threshold

# --- CV ---
N_CV_FOLDS = 5
CV_BY_TRIAL = True        # trial-stratified (was sequential time-bin in v2)
CV_SEED = 42

# --- Fit mode ---
USE_STABLE_PRODUCTION_FIT = True  # standardized X + regularized Poisson

# --- Plotting ---
SHOW_PLOTS = True

# --- Paths ---
OUT_DIR = Path(p.DATA_DIR) / 'glm_encoding_w_history'
OUT_DIR.mkdir(parents=True, exist_ok=True)
COEF_CSV = OUT_DIR / "glm_coefficients_per_unit.csv"
METRICS_CSV = OUT_DIR / "glm_fit_metrics_per_unit.csv"
LRT_CSV = OUT_DIR / "glm_lrt_per_unit.csv"
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_SAVE_DIR = OUT_DIR / "debug_plots"
PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

INCREMENTAL_SAVE = True
RESUME_PREVIOUS = False


# =====================================================================
# Original helpers (kept)
# =====================================================================
def spikes_df_to_trial_map(spikes_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    if not {"trial_id", "trial_time"}.issubset(spikes_df.columns):
        raise ValueError("spikes_df must contain 'trial_id' and 'trial_time'.")
    out = {}
    for tid, sub in spikes_df.groupby("trial_id"):
        out[int(tid)] = sub["trial_time"].astype(float).to_numpy()
    return out


def get_lick_times_for_trial(events: pd.DataFrame, trial_id: int) -> Tuple[List[float], List[float]]:
    ev = events.loc[events["trial_id"] == trial_id]
    if "event_start_trial_time" not in ev.columns:
        raise ValueError("events must have 'event_start_trial_time' (trial-relative)")
    lick_times_bg = ev.loc[ev["event_type"] == "lick_bg", "event_start_trial_time"].astype(float).to_list()
    mask_cons = ev["event_type"].isin(["lick_cons", "lick_cons_reward", "lick_cons_no_reward"])
    lick_times_cons = ev.loc[mask_cons, "event_start_trial_time"].astype(float).to_list()
    return lick_times_bg, lick_times_cons


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


def pseudo_R2_mcfadden(y_true: np.ndarray, mu_full: np.ndarray, mu_null: float) -> float:
    mu_full = np.clip(mu_full, 1e-9, None)
    ll_full = np.sum(y_true * np.log(mu_full) - mu_full)
    mu_null = np.clip(mu_null, 1e-9, None)
    ll_null = np.sum(y_true * np.log(mu_null) - mu_null)
    return 1.0 - (ll_full / ll_null) if ll_null != 0 else np.nan


def standardize_columns(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def drop_zero_variance_columns(
    X: np.ndarray, names: List[str], tol: float = 1e-10,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Drop columns with zero (or near-zero) variance.

    Returns (X_filtered, names_filtered, kept_mask).
      - kept_mask has length X.shape[1]; True where column was retained.

    Rationale: trial-history features can be constant in a given session
    (e.g., a session with no miss trials → prev_missed = 0 everywhere).
    Such columns are linearly dependent with the intercept, inflate the
    nominal parameter count, and cause rank deficiency in the design matrix.

    Dropping them at fit time keeps LRT df correct: family_indices() uses
    name-based lookup, so a dropped column simply doesn't appear in the
    feature group, and the df_diff between nested models adjusts automatically.
    """
    if X.shape[1] == 0:
        return X, list(names), np.array([], dtype=bool)
    col_std = X.std(axis=0)
    kept_mask = col_std > tol
    if kept_mask.all():
        return X, list(names), kept_mask
    X_filtered = X[:, kept_mask]
    names_filtered = [n for n, k in zip(names, kept_mask) if k]
    return X_filtered, names_filtered, kept_mask


# =====================================================================
# NEW: Trial-history feature engineering
# =====================================================================
def build_trial_history_features(trials_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-trial history feature columns from the session's trials dataframe.

    Returns a DataFrame indexed by trial_id, adding columns:
      Single-step history (always built):
        - prev_rewarded:    int {0,1}; NaN for first trial
        - prev_missed:      int {0,1}; NaN for first trial
        - prev_wait_z:      float, z-scored within session (0 where undefined)
      Current-trial structural (always built):
        - num_bg_repeats_z: float, z-scored num_bg_repeat from `trials`
        - trial_number_z:   float, z-scored position in session
      Multi-step history (built if MAX_HISTORY_BACK > 1):
        - prev_rewarded_{k}back, prev_missed_{k}back for k in [2, MAX_HISTORY_BACK]
      Reward rate (built for each tau in REWARD_RATE_TAUS):
        - ew_reward_rate_tau{tau}: causal exponentially-weighted reward rate,
          shifted by 1 trial so the current trial uses history strictly before it.

    Notes:
      - rewarded/missed are coerced to int via coerce_bool_series before shifting.
      - z-scoring uses session-level mean/std; constant columns become 0.
      - Trials with miss=True have an undefined wait time; treated as NaN
        when computing prev_wait, then filled with 0 after z-scoring.
    """
    trials = trials_df.sort_values('trial_id').copy().reset_index(drop=True)

    if 'rewarded' in trials.columns:
        trials['rewarded'] = coerce_bool_series(trials['rewarded']).astype(int)
    if 'missed' in trials.columns:
        trials['missed'] = coerce_bool_series(trials['missed']).astype(int)
    else:
        trials['missed'] = 0

    # ---- Single-step history ----
    trials['prev_rewarded'] = trials['rewarded'].shift(1)
    trials['prev_missed'] = trials['missed'].shift(1)

    # Previous-trial wait time (undefined for miss → NaN, then z-score on valid)
    # Also compute CURRENT-trial wait z-score for the confound control —
    # see INCLUDE_CURRENT_WAIT_CONTROL.
    if {'decision_time', 'cue_off_time'}.issubset(trials.columns):
        wait_time = trials['decision_time'] - trials['cue_off_time']
        wait_time = wait_time.where(trials['missed'] == 0, np.nan)
        trials['prev_wait'] = wait_time.shift(1)
        pw_valid = trials['prev_wait'].dropna()
        if len(pw_valid) > 1 and pw_valid.std() > 0:
            trials['prev_wait_z'] = (trials['prev_wait'] - pw_valid.mean()) / pw_valid.std()
        else:
            trials['prev_wait_z'] = 0.0
        trials['prev_wait_z'] = trials['prev_wait_z'].fillna(0.0)
        # Current-trial wait z-score (for current_wait × wait_time_basis confound)
        cw_valid = wait_time.dropna()
        if len(cw_valid) > 1 and cw_valid.std() > 0:
            trials['current_wait_z'] = (wait_time - cw_valid.mean()) / cw_valid.std()
        else:
            trials['current_wait_z'] = 0.0
        trials['current_wait_z'] = trials['current_wait_z'].fillna(0.0)
    else:
        trials['prev_wait_z'] = 0.0
        trials['current_wait_z'] = 0.0

    # ---- Current-trial structural ----
    if 'num_bg_repeat' in trials.columns:
        bg = trials['num_bg_repeat'].astype(float)
        if bg.std() > 0:
            trials['num_bg_repeats_z'] = (bg - bg.mean()) / bg.std()
        else:
            trials['num_bg_repeats_z'] = 0.0
    else:
        trials['num_bg_repeats_z'] = 0.0

    tn = np.arange(len(trials), dtype=float)
    if tn.std() > 0:
        trials['trial_number_z'] = (tn - tn.mean()) / tn.std()
    else:
        trials['trial_number_z'] = 0.0

    # ---- Smooth drift basis on trial_number_z (for M0 absorption) ----
    # Natural cubic regression spline (patsy `cr`), df=N_DRIFT_BASIS columns.
    # Without the `0 +` patsy would add its own intercept column.
    if len(trials) >= N_DRIFT_BASIS + 1:
        drift = np.asarray(
            dmatrix(f"0 + cr(tn, df={N_DRIFT_BASIS})",
                    {"tn": trials['trial_number_z'].values})
        )
    else:
        # Degenerate session (very few trials) — fall back to zeros so the
        # design remains shape-stable; these columns will be dropped as
        # zero-variance at fit time.
        drift = np.zeros((len(trials), N_DRIFT_BASIS))
    for i in range(N_DRIFT_BASIS):
        trials[f'trial_drift_b{i}'] = drift[:, i]

    # ---- Multi-step history (M3) ----
    for k in range(2, MAX_HISTORY_BACK + 1):
        trials[f'prev_rewarded_{k}back'] = trials['rewarded'].shift(k)
        trials[f'prev_missed_{k}back'] = trials['missed'].shift(k)

    # ---- EW reward rate (M4) ----
    # Use halflife semantics in pandas .ewm to match REWARD_RATE_TAUS as half-lives in trials.
    # Shift by 1 so current trial uses history strictly before it.
    rew_series = trials['rewarded'].astype(float)
    for tau in REWARD_RATE_TAUS:
        ew = rew_series.ewm(halflife=tau, adjust=False).mean()
        trials[f'ew_reward_rate_tau{int(tau)}'] = ew.shift(1)

    return trials.set_index('trial_id', drop=False)


def trial_has_valid_history(hist_row, require_multistep: bool = False,
                            require_reward_rate: bool = False) -> bool:
    """Return True if a trial has all history features needed for the configured model."""
    if pd.isna(hist_row.get('prev_rewarded', np.nan)):
        return False
    if pd.isna(hist_row.get('prev_missed', np.nan)):
        return False
    if require_multistep:
        for k in range(2, MAX_HISTORY_BACK + 1):
            if pd.isna(hist_row.get(f'prev_rewarded_{k}back', np.nan)):
                return False
    if require_reward_rate:
        for tau in REWARD_RATE_TAUS:
            if pd.isna(hist_row.get(f'ew_reward_rate_tau{int(tau)}', np.nan)):
                return False
    return True


# =====================================================================
# NEW: Augment within-trial design with trial-history columns
# =====================================================================
def augment_design_with_history(
    X: np.ndarray, names: List[str], trial_hist: pd.Series, n_bins: int,
    include_history: bool = True,
    include_interactions: bool = True,
    include_multistep: bool = False,
    include_reward_rate: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Append trial-history columns to within-trial design matrix X.

    All history main effects are constant within a trial → broadcast as
    n_bins identical values per column.

    Interaction columns are constructed by element-wise multiplication of
    the history feature (broadcast to n_bins) with the wait_time_basis
    columns of X. Only the wait_time basis is interacted (the BG basis
    interactions don't have a strong a priori biological motivation; we
    are testing whether reward modulates the *wait-period* time code).
    """
    extra_cols: List[np.ndarray] = []
    extra_names: List[str] = []

    # ---- Drift basis (always added — used by M0 too, not part of history) ----
    # Constant within trial; per-trial spline value from build_trial_history_features.
    for i in range(N_DRIFT_BASIS):
        col = f'trial_drift_b{i}'
        extra_cols.append(np.full(n_bins, float(trial_hist[col])))
        extra_names.append(col)

    # ---- Current-wait × wait_time_basis confound control (M0 too) ----
    # See INCLUDE_CURRENT_WAIT_CONTROL: absorbs current-trial wait-duration
    # dependent firing structure that would otherwise be misattributed to
    # the prev_rewarded / prev_wait_z interaction columns.
    if INCLUDE_CURRENT_WAIT_CONTROL:
        cols_arr_cw = np.array(names)
        wait_idx_cw = np.where(np.char.startswith(cols_arr_cw, 'wait_time_'))[0]
        cw = float(trial_hist['current_wait_z'])
        for j in wait_idx_cw:
            extra_cols.append(X[:, j] * cw)
            extra_names.append(f'current_wait_z_x_{names[j]}')

    if not include_history:
        if extra_cols:
            X_new = np.hstack([X, np.column_stack(extra_cols)])
            return X_new, list(names) + extra_names
        return X, names

    # ---- Main effects (M1 block) ----
    # trial_number_z is intentionally NOT included: its linear trend is a
    # subset of the trial_drift natural cubic spline basis (added above and
    # present in M0). Keeping both made the design rank-deficient by 1.
    main_features = [
        ('prev_rewarded',     float(trial_hist['prev_rewarded'])),
        ('prev_missed',       float(trial_hist['prev_missed'])),
        ('prev_wait_z',       float(trial_hist['prev_wait_z'])),
        ('num_bg_repeats_z',  float(trial_hist['num_bg_repeats_z'])),
    ]
    for fname, fval in main_features:
        extra_cols.append(np.full(n_bins, fval))
        extra_names.append(fname)

    if include_multistep:
        for k in range(2, MAX_HISTORY_BACK + 1):
            r = float(trial_hist[f'prev_rewarded_{k}back'])
            extra_cols.append(np.full(n_bins, r))
            extra_names.append(f'prev_rewarded_{k}back')

    if include_reward_rate:
        for tau in REWARD_RATE_TAUS:
            r = float(trial_hist[f'ew_reward_rate_tau{int(tau)}'])
            extra_cols.append(np.full(n_bins, r))
            extra_names.append(f'ew_reward_rate_tau{int(tau)}')

    # ---- Interaction columns (M2/M3/M4 block) ----
    if include_interactions:
        cols_arr = np.array(names)
        wait_idx = np.where(np.char.startswith(cols_arr, 'wait_time_'))[0]

        pr = float(trial_hist['prev_rewarded'])
        pw = float(trial_hist['prev_wait_z'])

        for j in wait_idx:
            extra_cols.append(X[:, j] * pr)
            extra_names.append(f'prev_rewarded_x_{names[j]}')
        for j in wait_idx:
            extra_cols.append(X[:, j] * pw)
            extra_names.append(f'prev_wait_z_x_{names[j]}')

        if include_multistep:
            for k in range(2, MAX_HISTORY_BACK + 1):
                rk = float(trial_hist[f'prev_rewarded_{k}back'])
                for j in wait_idx:
                    extra_cols.append(X[:, j] * rk)
                    extra_names.append(f'prev_rewarded_{k}back_x_{names[j]}')

        if include_reward_rate:
            for tau in REWARD_RATE_TAUS:
                rr = float(trial_hist[f'ew_reward_rate_tau{int(tau)}'])
                for j in wait_idx:
                    extra_cols.append(X[:, j] * rr)
                    extra_names.append(f'ew_reward_rate_tau{int(tau)}_x_{names[j]}')

    if not extra_cols:
        return X, names

    X_new = np.hstack([X, np.column_stack(extra_cols)])
    names_new = list(names) + extra_names
    return X_new, names_new


# =====================================================================
# Modified: build per-trial X/y
# =====================================================================
def build_Xy_for_trial(
    tr: pd.Series, events: pd.DataFrame, spikes_trial: np.ndarray,
    trial_history_idx: Optional[pd.DataFrame] = None,
    include_history: bool = INCLUDE_TRIAL_HISTORY,
    include_interactions: bool = INCLUDE_INTERACTIONS,
    include_multistep: bool = INCLUDE_MULTISTEP_HISTORY,
    include_reward_rate: bool = INCLUDE_REWARD_RATE,
):
    """
    Build (X, y, names) for a single trial.

    The within-trial design is delegated to glm_design_helpers.assemble_design.
    This function then optionally appends the trial-history block.

    Raises ValueError if the trial cannot be processed (e.g. missing times,
    no valid history when history is required).
    """
    bg_start = float(tr["cue_on_time"])
    bg_end = float(tr["cue_off_time"])
    wait_end = float(tr["decision_time"])
    outcome_rewarded = int(tr["rewarded"])
    cons_len = float(tr.get("consumption_length", 3.0))

    if not np.isfinite(wait_end):
        raise ValueError("decision_time not finite")
    if not np.isfinite(bg_start) or not np.isfinite(bg_end):
        raise ValueError("cue times not finite")
    if not np.isfinite(cons_len) or cons_len <= 0:
        cons_len = 3.0
    if wait_end + cons_len <= 0:
        raise ValueError("invalid trial duration")

    bin_edges = make_time_bins(trial_start=0.0, trial_end=wait_end + cons_len, dt=DT)
    lick_times_bg, lick_times_cons = get_lick_times_for_trial(events, int(tr["trial_id"]))
    lick_time_decision = wait_end

    spike_counts, _ = np.histogram(spikes_trial, bins=bin_edges)

    X, names, _ = assemble_design(
        bin_edges=bin_edges,
        bg_start=bg_start, bg_end=bg_end, wait_end=wait_end,
        lick_times_bg=lick_times_bg,
        lick_time_decision=lick_time_decision,
        lick_times_cons=lick_times_cons,
        outcome_rewarded=outcome_rewarded,
        spike_counts_for_history=spike_counts if INCLUDE_SPIKE_HISTORY else None,
        dt=DT,
        n_basis_bg=N_BASIS_BG, n_basis_wait=N_BASIS_WAIT,
        n_basis_bgwait=N_BASIS_BGWAIT,
        absolute_time_basis=ABSOLUTE_TIME_BASIS,
        t_max_bg=T_MAX_BG, t_max_wait=T_MAX_WAIT,
        hazard_tau=HAZARD_TAU,
        include_spike_history=INCLUDE_SPIKE_HISTORY,
        drop_cue_box=True,
    )

    # Augment with trial-history block (the M1+ predictors)
    if trial_history_idx is not None and include_history:
        tid = int(tr['trial_id'])
        if tid not in trial_history_idx.index:
            raise ValueError(f"trial {tid} not in history index")
        hist = trial_history_idx.loc[tid]
        if not trial_has_valid_history(hist,
                                       require_multistep=include_multistep,
                                       require_reward_rate=include_reward_rate):
            raise ValueError(f"trial {tid} has incomplete history features")
        X, names = augment_design_with_history(
            X, names, hist, X.shape[0],
            include_history=include_history,
            include_interactions=include_interactions,
            include_multistep=include_multistep,
            include_reward_rate=include_reward_rate,
        )

    return X, spike_counts, names


# =====================================================================
# Feature groups (extended to include trial-history block)
# =====================================================================
def family_indices(names: List[str]) -> Dict[str, np.ndarray]:
    """Return index arrays for predictor families present in names."""
    cols = np.array(names)
    return {
        "BG_time":      np.where(np.char.startswith(cols, "bg_time_"))[0],
        "WAIT_time":    np.where(np.char.startswith(cols, "wait_time_"))[0],
        "BGWAIT_time":  np.where(np.char.startswith(cols, "bgwait_time_"))[0],
        "hazard":       np.where(cols == "hazard_ortho")[0],
        "cue":          np.where(np.isin(cols, ["cue_on", "cue_off"]))[0],
        "licks":        np.where(np.isin(cols, ["lick_bg", "lick_decision", "lick_cons"]))[0],
        "outcome":      np.where(cols == "outcome_rewarded")[0],
        "spike_history": np.where(np.char.startswith(cols, "hist_"))[0],
        # Smooth session-long drift control (in every model from M0 up):
        "trial_drift":   np.where(np.char.startswith(cols, "trial_drift_b"))[0],
        # Current-wait-duration confound control (in every model from M0 up):
        "current_wait_x_wait": np.where(
            np.char.startswith(cols, "current_wait_z_x_wait_time_")
        )[0],
        # NEW history families:
        # trial_number_z removed — subsumed by trial_drift spline basis.
        "hist_main": np.where(np.isin(cols, [
            "prev_rewarded", "prev_missed", "prev_wait_z",
            "num_bg_repeats_z",
        ]))[0],
        "hist_multistep_main": np.array([
            i for i, n in enumerate(cols)
            if n.endswith("back") and "_x_" not in n
        ], dtype=int),
        "hist_rrate_main": np.array([
            i for i, n in enumerate(cols)
            if n.startswith("ew_reward_rate_tau") and "_x_" not in n
        ], dtype=int),
        "int_pr_x_wait": np.array([
            i for i, n in enumerate(cols) if n.startswith("prev_rewarded_x_wait_time_")
        ], dtype=int),
        "int_pw_x_wait": np.array([
            i for i, n in enumerate(cols) if n.startswith("prev_wait_z_x_wait_time_")
        ], dtype=int),
        "int_multistep_x_wait": np.array([
            i for i, n in enumerate(cols)
            if ("back_x_wait_time_" in n)
        ], dtype=int),
        "int_rrate_x_wait": np.array([
            i for i, n in enumerate(cols)
            if n.startswith("ew_reward_rate_tau") and "_x_wait_time_" in n
        ], dtype=int),
    }


# =====================================================================
# Nested model definitions (M0 → M1 → M2 → M3 → M4)
# =====================================================================
MODEL_DEFINITIONS = {
    'M0': ['BG_time', 'WAIT_time', 'hazard', 'cue', 'licks',
           'outcome', 'spike_history', 'trial_drift', 'current_wait_x_wait'],
    'M1': ['BG_time', 'WAIT_time', 'hazard', 'cue', 'licks',
           'outcome', 'spike_history', 'trial_drift', 'current_wait_x_wait',
           'hist_main'],
    'M2': ['BG_time', 'WAIT_time', 'hazard', 'cue', 'licks',
           'outcome', 'spike_history', 'trial_drift', 'current_wait_x_wait',
           'hist_main',
           'int_pr_x_wait', 'int_pw_x_wait'],
    'M3': ['BG_time', 'WAIT_time', 'hazard', 'cue', 'licks',
           'outcome', 'spike_history', 'trial_drift', 'current_wait_x_wait',
           'hist_main',
           'int_pr_x_wait', 'int_pw_x_wait',
           'hist_multistep_main', 'int_multistep_x_wait'],
    'M4': ['BG_time', 'WAIT_time', 'hazard', 'cue', 'licks',
           'outcome', 'spike_history', 'trial_drift', 'current_wait_x_wait',
           'hist_main',
           'int_pr_x_wait', 'int_pw_x_wait',
           'hist_rrate_main', 'int_rrate_x_wait'],
}


def get_nested_model_columns(feature_groups: Dict[str, np.ndarray],
                             model_name: str) -> np.ndarray:
    """Return column indices included in the named nested model."""
    if model_name not in MODEL_DEFINITIONS:
        raise ValueError(f"unknown model {model_name}")
    indices: List[np.ndarray] = []
    for group in MODEL_DEFINITIONS[model_name]:
        if group in feature_groups and feature_groups[group].size > 0:
            indices.append(feature_groups[group])
    if not indices:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(indices))


# =====================================================================
# Trial-stratified k-fold CV
# =====================================================================
def trial_stratified_kfold_indices(trial_id_per_row: np.ndarray,
                                   n_splits: int = N_CV_FOLDS,
                                   seed: int = CV_SEED):
    """
    Yield (train_idx, test_idx) where each fold contains complete trials.
    Trials are assigned to folds at random (no temporal stratification).

    This eliminates the leakage that occurs with sequential time-bin folds
    when spike-history kernels span the train/test boundary.
    """
    rng = np.random.RandomState(seed)
    unique_trials = np.unique(trial_id_per_row)
    rng.shuffle(unique_trials)
    fold_assignments = np.array_split(unique_trials, n_splits)
    for fold_trials in fold_assignments:
        test_mask = np.isin(trial_id_per_row, fold_trials)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        yield train_idx, test_idx


def sequential_kfold_indices(n_samples: int, n_splits: int = N_CV_FOLDS):
    """Backup time-bin folds (kept for compatibility / explicit comparison)."""
    n_splits = max(2, int(n_splits))
    indices = np.arange(n_samples)
    fold_sizes = [n_samples // n_splits + (1 if i < (n_samples % n_splits) else 0)
                  for i in range(n_splits)]
    start = 0
    for fs in fold_sizes:
        stop = start + fs
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]]) if fs > 0 else indices
        yield train_idx, test_idx
        start = stop


def get_cv_splits(X_all, y_all, trial_id_per_row=None):
    """Return list of (train_idx, test_idx); trial-stratified if CV_BY_TRIAL and trial_id_per_row given."""
    if CV_BY_TRIAL and trial_id_per_row is not None:
        return list(trial_stratified_kfold_indices(trial_id_per_row, n_splits=N_CV_FOLDS))
    else:
        return list(sequential_kfold_indices(len(y_all), n_splits=N_CV_FOLDS))


# =====================================================================
# GLM fitting + LRT
# =====================================================================
# Decoupled from the production fit: the LRT path uses unregularized IRLS
# (with a tiny-ridge fallback) so the chi-squared df is honest. The
# production fit in fit_models keeps ALPHA=0.1 for interpretable coefficients.
LRT_RIDGE_FALLBACK = 1e-4   # used only when IRLS fails to converge


def fit_glm_for_lrt(X: np.ndarray, y: np.ndarray,
                    standardize: bool = True) -> Tuple[Optional[object], float]:
    """
    Fit a Poisson GLM with *minimal* regularization for use in the LRT.

    Path:
      1. Unregularized IRLS (`sm.GLM(...).fit`). If it converges and returns
         a finite log-likelihood, use that.
      2. Otherwise fall back to a tiny ridge (α=LRT_RIDGE_FALLBACK) so we
         still get an answer on units where IRLS diverges (low-rate units,
         near-collinear designs, etc.).

    Why decouple from the production fit:
      The production fit uses ALPHA=0.1 for stable, interpretable coefficient
      estimates. But ridge shrinkage reduces the *effective* df below the
      nominal df, which biases the chi-squared LRT (empirically we observed
      median shuffled p ≈ 0.94 instead of 0.5 — strongly conservative).
      The LRT only needs honest log-likelihoods, not interpretable betas,
      so use a near-unbiased estimator here.

    Returns (result_or_None, log_likelihood).
    """
    if X.shape[1] == 0:
        # Intercept-only — closed-form ll for Poisson with rate = mean(y)
        mu_const = max(float(y.mean()), 1e-9)
        ll = float(np.sum(y * np.log(mu_const) - mu_const))
        return None, ll

    if standardize:
        Xs, _, _ = standardize_columns(X)
    else:
        Xs = X
    Xc = sm.add_constant(Xs, has_constant='add')

    # --- 1. unregularized IRLS ---
    try:
        res = sm.GLM(y, Xc, family=sm.families.Poisson()).fit(maxiter=200)
        if not getattr(res, "converged", True):
            raise RuntimeError("IRLS did not converge")
        mu = np.clip(res.predict(Xc), 1e-9, None)
        if not np.all(np.isfinite(mu)):
            raise RuntimeError("non-finite mu from IRLS")
        ll = float(np.sum(y * np.log(mu) - mu))
        if not np.isfinite(ll):
            raise RuntimeError("non-finite ll from IRLS")
        return res, ll
    except Exception as e_irls:
        # --- 2. tiny-ridge fallback ---
        try:
            # Per-coefficient alpha with 0 on intercept (same rationale as
            # the production fit: don't shrink the baseline rate). The bias
            # at α=1e-4 is small but nonzero; this keeps the LRT path clean.
            alpha_vec = np.full(Xc.shape[1], float(LRT_RIDGE_FALLBACK))
            alpha_vec[0] = 0.0
            res = sm.GLM(y, Xc, family=sm.families.Poisson()).fit_regularized(
                alpha=alpha_vec, L1_wt=0.0, maxiter=1000
            )
            mu = np.clip(res.predict(Xc), 1e-9, None)
            ll = float(np.sum(y * np.log(mu) - mu))
            if not np.isfinite(ll):
                raise RuntimeError("non-finite ll from ridge fallback")
            return res, ll
        except Exception as e_reg:
            print(f"[lrt fit warn] IRLS=({e_irls}); ridge fallback=({e_reg})")
            return None, np.nan


def lrt_pvalue(ll_reduced: float, ll_full: float, df_diff: int) -> float:
    """Likelihood ratio test p-value (chi-squared with df_diff)."""
    if not np.isfinite(ll_reduced) or not np.isfinite(ll_full) or df_diff <= 0:
        return np.nan
    chi2 = -2.0 * (ll_reduced - ll_full)
    if chi2 < 0:
        chi2 = 0.0  # numerical edge case
    return float(stats.chi2.sf(chi2, df_diff))


HEADLINE_COMPARISONS = ('M1_vs_M2',)   # the only LRT we report for inference
DIAGNOSTIC_COMPARISONS = ('M0_vs_M1',)  # kept in CSV for shuffle/diagnostics only


def compute_nested_lrts(X_all: np.ndarray, y_all: np.ndarray,
                       feature_groups: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    Fit M0, M1, M2 (and optionally M3, M4); return LL per model and LRT p-values
    for the standard nested comparisons.

    Headline test: M1_vs_M2 — does reward history modulate the wait-time CODE
    beyond a main-effect shift? Empirically conservative under shuffle null
    (median shuffled p ≈ 0.81 instead of 0.5 — see 4c_shuffle_control.py),
    so positive findings are trustworthy with some power loss.

    Diagnostic only: M0_vs_M1 — anti-conservative under shuffle null
    (median ≈ 0.18, frac p<0.05 ≈ 26%) because the M1 main-effect columns
    are constant within trial and act like per-trial offsets, soaking up
    real trial-to-trial firing-rate drift that M0 doesn't model. Computed
    here for the shuffle-control diagnostic script, but downstream code
    (summarize_regions) only consumes M1_vs_M2.

    Uses fit_glm_for_lrt — unregularized IRLS with tiny-ridge fallback — so
    the chi-squared df is honest. (The production fit in fit_models still
    uses ALPHA=0.1 for interpretable coefficients; see fit_glm_for_lrt for
    rationale.)
    """
    results: Dict[str, dict] = {'models': {}, 'lrts': {}}

    models_to_fit = ['M0', 'M1', 'M2']
    if INCLUDE_MULTISTEP_HISTORY:
        models_to_fit.append('M3')
    if INCLUDE_REWARD_RATE:
        models_to_fit.append('M4')

    for model_name in models_to_fit:
        cols = get_nested_model_columns(feature_groups, model_name)
        if len(cols) == 0:
            print(f"[lrt warn] no columns for model {model_name}; skipping.")
            continue
        X_sub = X_all[:, cols]
        _, ll = fit_glm_for_lrt(X_sub, y_all)
        results['models'][model_name] = {
            'll': ll,
            'n_params': int(len(cols) + 1),  # +1 intercept
        }

    comparisons = [('M0', 'M1'), ('M1', 'M2')]
    if 'M3' in results['models']:
        comparisons.append(('M2', 'M3'))
    if 'M4' in results['models']:
        comparisons.append(('M2', 'M4'))

    for red, full in comparisons:
        if red not in results['models'] or full not in results['models']:
            continue
        ll_red = results['models'][red]['ll']
        ll_full = results['models'][full]['ll']
        df = results['models'][full]['n_params'] - results['models'][red]['n_params']
        p = lrt_pvalue(ll_red, ll_full, df)
        results['lrts'][f'{red}_vs_{full}'] = {
            'll_reduced': ll_red, 'll_full': ll_full, 'df': df, 'p_value': p,
        }
    return results


# =====================================================================
# CV-based ΔpseudoR² per family (modified: trial-stratified)
# =====================================================================
def fit_and_pr2_unreg(X: np.ndarray, y: np.ndarray) -> float:
    """Unregularized IRLS Poisson fit, in-sample pseudo-R²."""
    if X.shape[1] == 0:
        return 0.0
    Xc = sm.add_constant(X, has_constant='add')
    try:
        res = sm.GLM(y, Xc, family=sm.families.Poisson()).fit(maxiter=200)
        mu = res.predict(Xc)
        return pseudo_R2_mcfadden(y, mu, y.mean())
    except Exception:
        return np.nan


def glm_fit_predict_unreg(X_train, y_train, X_test):
    X_train_c = sm.add_constant(X_train, has_constant='add')
    X_test_c = sm.add_constant(X_test, has_constant='add')
    res = sm.GLM(y_train, X_train_c, family=sm.families.Poisson()).fit(maxiter=200)
    return res, res.predict(X_test_c)


def cv_delta_pr2_family_vs_null(
    X_all: np.ndarray, y_all: np.ndarray, fam_idx: np.ndarray,
    trial_id_per_row: Optional[np.ndarray] = None,
    n_splits: int = N_CV_FOLDS,
) -> float:
    """
    Cross-validated pseudo-R² for one covariate family vs intercept-only null.
    Uses trial-stratified folds when trial_id_per_row is provided and CV_BY_TRIAL is True.
    """
    if fam_idx.size == 0:
        return 0.0
    scores: List[float] = []
    splits = get_cv_splits(X_all, y_all, trial_id_per_row=trial_id_per_row)
    for train_idx, test_idx in splits:
        if len(train_idx) < 5 or len(test_idx) < 5:
            continue
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        Xf_train = X_all[train_idx][:, fam_idx]
        Xf_test = X_all[test_idx][:, fam_idx]
        mu_null_train = float(y_train.mean())
        try:
            _, mu_f_test = glm_fit_predict_unreg(Xf_train, y_train, Xf_test)
            pr2 = pseudo_R2_mcfadden(y_test, mu_f_test, mu_null_train)
        except Exception:
            pr2 = np.nan
        if np.isfinite(pr2):
            scores.append(pr2)
    return float(np.mean(scores)) if scores else 0.0


# =====================================================================
# Unit inclusion criteria
# =====================================================================
def check_unit_inclusion(
    spikes_by_trial: Dict[int, np.ndarray],
    trials_idx: pd.DataFrame,
    trial_history_idx: pd.DataFrame,
    min_spikes: int = MIN_SPIKE_COUNT,
    max_mean_fr: float = MAX_MEAN_FR_HZ,
    min_prev_reward: int = MIN_TRIALS_PER_PREV_REWARD,
    min_prev_not_miss: int = MIN_TRIALS_PREV_NOT_MISS,
) -> Tuple[bool, str]:
    """
    Check if a unit meets inclusion criteria. Returns (passes, reason).

    Criteria:
      - ≥ min_spikes total spikes across valid trials
      - mean FR ≤ max_mean_fr Hz (Poisson approximation breaks for very high FR)
      - ≥ min_prev_reward trials of each prev_rewarded value (interaction identified)
      - ≥ min_prev_not_miss trials with prev_missed=0 (avoid bias from
        too few non-miss-prev trials, which dominate the headline contrast)

    Trials counted as "valid" require: not missed, in trial_history_idx,
    and have non-NaN single-step history features (per trial_has_valid_history).
    """
    valid_tids: List[int] = []
    for tid in spikes_by_trial.keys():
        if tid not in trials_idx.index:
            continue
        tr = trials_idx.loc[tid]
        if int(coerce_bool_series(pd.Series([tr.get('missed', False)])).iloc[0]):
            continue
        if tid not in trial_history_idx.index:
            continue
        hist = trial_history_idx.loc[tid]
        if not trial_has_valid_history(hist):
            continue
        valid_tids.append(tid)

    if not valid_tids:
        return False, "no valid trials"

    total_spikes = sum(len(spikes_by_trial[tid]) for tid in valid_tids)
    if total_spikes < min_spikes:
        return False, f"only {total_spikes} spikes (need {min_spikes})"

    total_dur = 0.0
    for tid in valid_tids:
        tr = trials_idx.loc[tid]
        total_dur += (float(tr.get('decision_time', 0.0)) +
                      float(tr.get('consumption_length', 3.0)))
    if total_dur > 0:
        mean_fr = total_spikes / total_dur
        if mean_fr > max_mean_fr:
            return False, f"mean FR {mean_fr:.1f} Hz > {max_mean_fr}"

    n_prev_rew_1 = 0
    n_prev_rew_0 = 0
    n_prev_not_miss = 0
    for tid in valid_tids:
        hist = trial_history_idx.loc[tid]
        pr = hist.get('prev_rewarded', np.nan)
        pm = hist.get('prev_missed', np.nan)
        if pr == 1:
            n_prev_rew_1 += 1
        elif pr == 0:
            n_prev_rew_0 += 1
        if pm == 0:
            n_prev_not_miss += 1

    if n_prev_rew_1 < min_prev_reward:
        return False, f"only {n_prev_rew_1} prev_rewarded=1 trials (need {min_prev_reward})"
    if n_prev_rew_0 < min_prev_reward:
        return False, f"only {n_prev_rew_0} prev_rewarded=0 trials (need {min_prev_reward})"
    if n_prev_not_miss < min_prev_not_miss:
        return False, f"only {n_prev_not_miss} prev_not_miss trials (need {min_prev_not_miss})"

    return True, "ok"


# =====================================================================
# Classification (descriptive output; LRT is the formal test)
# =====================================================================
def classify_unit(delta_pr2: dict, names: List[str], params: np.ndarray) -> str:
    """Rule-based classification using ΔpseudoR² thresholds. Kept for descriptive output."""
    dBG = float(delta_pr2.get("delta_pr2_BG_time", 0.0))
    dWAIT = float(delta_pr2.get("delta_pr2_WAIT_time", 0.0))
    dHAZ = float(delta_pr2.get("delta_pr2_hazard", 0.0))
    dCUE = float(delta_pr2.get("delta_pr2_cue", 0.0))
    dLCK = float(delta_pr2.get("delta_pr2_licks", 0.0))
    dOUT = float(delta_pr2.get("delta_pr2_outcome", 0.0))

    name_to_coef = {n: float(b) for n, b in zip(["const"] + names, params)}
    b_dec = name_to_coef.get("lick_decision", 0.0)
    b_cons = name_to_coef.get("lick_cons", 0.0)
    b_out = name_to_coef.get("outcome_rewarded", 0.0)

    if dBG >= PR2_THR_BG and dWAIT >= PR2_THR_WAIT:
        return "bg+wait ramping"
    if dHAZ >= PR2_THR_HAZ and dWAIT < PR2_THR_WAIT:
        return "hazard-ramping"
    if dWAIT >= PR2_THR_WAIT and dBG < PR2_THR_BG:
        return "wait-only ramping"
    if dBG >= PR2_THR_BG and dWAIT < PR2_THR_WAIT:
        return "bg-only ramping"
    if dLCK > max(dBG, dWAIT, dCUE, dOUT, dHAZ) and abs(b_dec) > 0:
        return "decision-locked"
    if dOUT >= 0.001 or abs(b_cons) > 0 or abs(b_out) > 0:
        return "consumption/reward-locked"
    if dCUE >= 0.001:
        return "cue-locked"
    return "mixed/weak"


# =====================================================================
# Plotting
# =====================================================================
def plot_one_trial(tr, events, spikes_trial, trial_history_idx=None,
                   title="Example trial fit", save_path=None):
    X, y, names = build_Xy_for_trial(tr, events, spikes_trial,
                                      trial_history_idx=trial_history_idx)
    t = np.arange(len(y)) * DT
    Xc = sm.add_constant(X, has_constant='add')
    res = sm.GLM(y, Xc, family=sm.families.Poisson()).fit(maxiter=200)
    mu = res.predict(Xc) / DT

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    ax.plot(t, mu, label="Predicted rate (Hz)")
    if len(spikes_trial) > 0:
        ax.vlines(spikes_trial, ymin=0, ymax=max(mu) * 0.3 + 1e-9, linewidth=0.5)
    bg_start = float(tr["cue_on_time"]); bg_end = float(tr["cue_off_time"])
    wait_end = float(tr["decision_time"]); cons_len = float(tr.get("consumption_length", 3.0))
    ax.axvspan(bg_start, bg_end, alpha=0.1, label="BG")
    ax.axvspan(bg_end, wait_end, alpha=0.1, label="WAIT")
    ax.axvspan(wait_end, wait_end + cons_len, alpha=0.1, label="CONS")
    ax.set_xlabel("Time in trial (s)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"[plot] saved → {save_path}")
    if SHOW_PLOTS:
        try:
            plt.show()
        except Exception as e:
            print(f"[plot] show() skipped: {e}")
    plt.close(fig)


# =====================================================================
# bg_group lookup + wait-band trial filter
# =====================================================================
def load_bg_group_lookup() -> Dict[str, str]:
    """Build (session_id) → bg_group ('Short BG' | 'Long BG') from
    unit_properties_final. Returns {} on failure."""
    try:
        upf = pd.read_csv(os.path.join(p.LOGS_DIR, "unit_properties_final.csv"))
        if "bg_group" not in upf.columns:
            return {}
        upf["session_id"] = (
            upf["mouse"].astype(str) + "_"
            + upf["date_only"].astype(str) + "_"
            + upf["probe_region"].astype(str)
        )
        # Session group = most common non-null bg_group for the session
        agg = (upf.dropna(subset=["bg_group"])
                  .groupby("session_id")["bg_group"]
                  .agg(lambda s: s.mode().iat[0] if len(s) else None))
        return agg.to_dict()
    except Exception as e:
        print(f"[bg_group warn] could not load: {e}")
        return {}


def wait_band_for_group(bg_group: Optional[str]) -> Optional[Tuple[float, float]]:
    """Return (lo, hi) seconds for the group, or None if unknown."""
    if bg_group == "Short BG":
        return WAIT_BAND_SHORT_BG
    if bg_group == "Long BG":
        return WAIT_BAND_LONG_BG
    return None


def filter_trials_by_wait_band(trials_idx: pd.DataFrame, session_id: str,
                               bg_group_lookup: Dict[str, str]
                               ) -> Tuple[pd.DataFrame, Optional[Tuple[float, float]]]:
    """Restrict trials_idx to those with current wait in the group's band.

    Returns (filtered_trials_idx, band_used). band_used is None if filter
    is off or the session's group is unknown — original trials returned.
    """
    if not APPLY_WAIT_BAND_FILTER:
        return trials_idx, None
    group = bg_group_lookup.get(session_id)
    band = wait_band_for_group(group)
    if band is None:
        return trials_idx, None
    lo, hi = band
    wt = trials_idx["decision_time"] - trials_idx["cue_off_time"]
    keep = (wt >= lo) & (wt <= hi)
    return trials_idx.loc[keep].copy(), band


# =====================================================================
# IO helpers
# =====================================================================
def _load_processed_units() -> set:
    processed = set()
    for path in [METRICS_CSV, COEF_CSV, LRT_CSV]:
        if not path.exists():
            continue
        try:
            df_ids = pd.read_csv(path, usecols=["session_id", "unit_id"])
            for r in df_ids.itertuples(index=False):
                processed.add((r.session_id, str(r.unit_id)))
        except Exception as e:
            print(f"[resume warn] could not read {path.name}: {e}")
    return processed


def _append_row(path: Path, row: dict):
    df = pd.DataFrame([row])
    write_header = (not path.exists()) or path.stat().st_size == 0
    try:
        df.to_csv(path, mode='a', header=write_header, index=False)
    except Exception as e:
        print(f"[io warn] failed to append to {path.name}: {e}")


def _artifact_path(session_id, unit_id) -> Path:
    return MODEL_DIR / f"{session_id}__{unit_id}.json"


def _save_model_artifact(session_id, unit_id, region, names, params,
                         x_mean, x_std, extra: Optional[dict] = None):
    art = {
        "version": 2,                # bumped from v1
        "session_id": session_id,
        "unit_id": unit_id,
        "region": region,
        "names": list(names),
        "params": [float(v) for v in params],
        "x_mean": [float(v) for v in x_mean],
        "x_std": [float(v) for v in x_std],
        "alpha": ALPHA,
        "l1_wt": L1_WT,
        "dt": DT,
        "hazard_tau": HAZARD_TAU,
        "include_spike_history": INCLUDE_SPIKE_HISTORY,
        "include_trial_history": INCLUDE_TRIAL_HISTORY,
        "include_interactions": INCLUDE_INTERACTIONS,
        "include_multistep_history": INCLUDE_MULTISTEP_HISTORY,
        "include_reward_rate": INCLUDE_REWARD_RATE,
        "n_basis_bg": N_BASIS_BG,
        "n_basis_wait": N_BASIS_WAIT,
        "n_basis_bgwait": N_BASIS_BGWAIT,
        "reward_rate_taus": REWARD_RATE_TAUS,
        "max_history_back": MAX_HISTORY_BACK,
    }
    if extra:
        art.update(extra)
    path = _artifact_path(session_id, unit_id)
    try:
        with open(path, "w") as f:
            json.dump(art, f)
        return path
    except Exception as e:
        print(f"[io warn] failed to save artifact for unit {unit_id}: {e}")
        return None


def _load_model_artifact(session_id, unit_id):
    path = _artifact_path(session_id, unit_id)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[io warn] failed to load artifact for unit {unit_id}: {e}")
        return None


# =====================================================================
# Per-session: build design once, then iterate units
# =====================================================================
def build_session_design_for_unit(
    spikes_by_trial: Dict[int, np.ndarray],
    trials_idx: pd.DataFrame,
    trial_history_idx: pd.DataFrame,
    events: pd.DataFrame,
):
    """
    For one unit's spike trains, build (X_all, y_all, names, trial_id_per_row)
    by concatenating per-trial designs across all valid trials.

    Returns (X_all, y_all, names, trial_id_per_row, used_trials, skipped_trials).
    """
    X_blocks, y_blocks, tid_blocks = [], [], []
    names: Optional[List[str]] = None
    skipped: List[Tuple[int, str]] = []
    used: List[int] = []

    for tid, sp in spikes_by_trial.items():
        if tid not in trials_idx.index:
            skipped.append((tid, "not in trials_idx"))
            continue
        tr = trials_idx.loc[tid]
        try:
            X, y, nm = build_Xy_for_trial(
                tr, events, sp,
                trial_history_idx=trial_history_idx,
                include_history=INCLUDE_TRIAL_HISTORY,
                include_interactions=INCLUDE_INTERACTIONS,
                include_multistep=INCLUDE_MULTISTEP_HISTORY,
                include_reward_rate=INCLUDE_REWARD_RATE,
            )
        except Exception as e:
            skipped.append((tid, str(e)))
            continue
        X_blocks.append(X)
        y_blocks.append(y)
        tid_blocks.append(np.full(X.shape[0], tid, dtype=int))
        if names is None:
            names = nm
        elif nm != names:
            # Column order mismatch — should be rare but guard against it
            skipped.append((tid, "design names mismatch"))
            X_blocks.pop()
            y_blocks.pop()
            tid_blocks.pop()
            continue
        used.append(tid)

    if not X_blocks:
        return None, None, names, None, used, skipped

    X_all = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks).astype(float)
    trial_id_per_row = np.concatenate(tid_blocks)
    return X_all, y_all, names, trial_id_per_row, used, skipped


# =====================================================================
# Main fit loop
# =====================================================================
def fit_models(session_ids: Optional[List[str]] = None):
    """
    For each (session, unit), build the full design with trial-history block,
    apply unit-inclusion criteria, fit the M2 regularized Poisson GLM, save
    artifact for later evaluation.

    session_ids: if given, restrict the run to these session_ids (intersected
    with units_vetted). Useful for the one-session runtime check before
    launching the full overnight pass.
    """
    units_vetted = pd.read_csv(
        os.path.join(p.LOGS_DIR, "units_vetted.csv"), index_col=0
    ).sort_values("unit_id")
    all_session_ids = sorted(units_vetted["session_id"].unique().tolist())
    if session_ids is None:
        session_ids = all_session_ids
    else:
        missing = sorted(set(session_ids) - set(all_session_ids))
        if missing:
            print(f"[warn] sessions not in units_vetted: {missing}")
        session_ids = [s for s in session_ids if s in set(all_session_ids)]
        if not session_ids:
            print("[abort] no sessions to fit after filtering.")
            return

    bg_group_lookup = load_bg_group_lookup() if APPLY_WAIT_BAND_FILTER else {}

    for session_id in session_ids:
        print(f"\n=== Session {session_id} ===")
        session_units = units_vetted[units_vetted["session_id"] == session_id]
        events, trials, units = utils.get_session_data(session_id)

        # Build trial-history features BEFORE filtering miss trials
        # (so we can compute prev_missed and prev_wait correctly for ALL trials).
        trial_history_idx = build_trial_history_features(trials)

        # Now filter miss trials for the analysis loop
        if "missed" in trials.columns:
            try:
                missed_bool = coerce_bool_series(trials["missed"])
                trials = trials.loc[~missed_bool].copy()
            except Exception as e:
                print(f"[warn] could not filter missed trials: {e}")
        trials_idx = trials.set_index("trial_id", drop=False)

        n_pre_band = len(trials_idx)
        trials_idx, band = filter_trials_by_wait_band(
            trials_idx, session_id, bg_group_lookup
        )
        if band is not None:
            print(f"[wait-band] {bg_group_lookup.get(session_id)} band={band}: "
                  f"{len(trials_idx)}/{n_pre_band} trials retained")

        session_region = (
            session_units["region"].iloc[0]
            if "region" in session_units.columns and len(session_units) > 0
            else ""
        )

        n_units = len(session_units)
        for ui, (_, unit_info) in enumerate(session_units.iterrows(), start=1):
            unit_id = unit_info["unit_id"]
            region = unit_info.get("region", session_region)
            unit_key = unit_info["id"]

            art_path = _artifact_path(session_id, unit_id)
            if RESUME_PREVIOUS and art_path.exists():
                print(f"[skip] session {session_id} • unit {unit_id} artifact exists.")
                continue

            spikes_df = units[unit_key]
            spikes_by_trial = spikes_df_to_trial_map(spikes_df)

            # Unit-inclusion check
            ok, reason = check_unit_inclusion(
                spikes_by_trial, trials_idx, trial_history_idx,
            )
            if not ok:
                print(f"[exclude] session {session_id} • unit {unit_id}: {reason}")
                continue

            # Build design across all valid trials
            X_raw, y_all, names_raw, trial_id_per_row, used, skipped = (
                build_session_design_for_unit(
                    spikes_by_trial, trials_idx, trial_history_idx, events
                )
            )
            if X_raw is None or names_raw is None:
                print(f"[skip] session {session_id} • unit {unit_id}: no valid trials")
                continue

            # Drop zero-variance columns (e.g., prev_missed in a no-miss session).
            # This keeps LRT df honest downstream; family_indices() works by name.
            X_all, names, kept_mask = drop_zero_variance_columns(X_raw, names_raw)
            dropped_names = [n for n, k in zip(names_raw, kept_mask) if not k]
            if dropped_names:
                print(f"[fit] unit {unit_id}: dropped {len(dropped_names)} "
                      f"zero-var cols: {dropped_names}")

            # Production fit (full design = M2 or whichever models are toggled on)
            if USE_STABLE_PRODUCTION_FIT:
                Xs, mean, std = standardize_columns(X_all)
                X_design = sm.add_constant(Xs, has_constant='add')
                try:
                    # Build per-coefficient ridge penalty: 0 on the intercept
                    # (first column after add_constant) so the baseline rate
                    # isn't shrunk toward 0. Without this, low-FR units get
                    # their intercept pulled away from log(true rate × dt),
                    # inflating predicted rate above true rate and producing
                    # negative pR² values at evaluate time. The LRT path is
                    # unaffected (fit_glm_for_lrt uses unregularized IRLS).
                    alpha_vec = np.full(X_design.shape[1], float(ALPHA))
                    alpha_vec[0] = 0.0
                    res = sm.GLM(y_all, X_design, family=sm.families.Poisson()).fit_regularized(
                        alpha=alpha_vec, L1_wt=L1_WT, maxiter=1000
                    )
                    params = res.params
                except Exception as e:
                    print(f"[fit warn] unit {unit_id}: {e}; skipping.")
                    continue
                _save_model_artifact(
                    session_id, unit_id, region, names, params,
                    x_mean=mean.ravel().tolist(),
                    x_std=std.ravel().tolist(),
                    extra={
                        "n_timepoints": int(y_all.shape[0]),
                        "n_trials_used": int(len(used)),
                        "n_trials_skipped": int(len(skipped)),
                        "mean_rate_hz": float(y_all.sum() / (y_all.shape[0] * DT)),
                        "n_dropped_zero_var_cols": int(len(dropped_names)),
                        "dropped_zero_var_cols": dropped_names,
                    }
                )
            else:
                X_design = sm.add_constant(X_all, has_constant='add')
                try:
                    res = sm.GLM(y_all, X_design, family=sm.families.Poisson()).fit(maxiter=200)
                    params = res.params
                except Exception as e:
                    print(f"[fit warn] unit {unit_id}: {e}; skipping.")
                    continue
                _save_model_artifact(
                    session_id, unit_id, region, names, params,
                    x_mean=np.zeros(X_all.shape[1]).tolist(),
                    x_std=np.ones(X_all.shape[1]).tolist(),
                    extra={
                        "n_timepoints": int(y_all.shape[0]),
                        "n_trials_used": int(len(used)),
                        "n_trials_skipped": int(len(skipped)),
                        "mean_rate_hz": float(y_all.sum() / (y_all.shape[0] * DT)),
                        "n_dropped_zero_var_cols": int(len(dropped_names)),
                        "dropped_zero_var_cols": dropped_names,
                    }
                )

            print(f"[fit] saved artifact for unit {unit_id}  {ui}/{n_units}  "
                  f"trials_used={len(used)}")

    print("\nDone fitting models.")


# =====================================================================
# Main evaluate loop
# =====================================================================
def evaluate_models(session_ids: Optional[List[str]] = None):
    """
    For each unit with a saved artifact, rebuild the design and compute:
      - Full-model in-sample pseudo-R²
      - Per-family in-sample and CV ΔpseudoR²
      - Nested-model LRTs (M0 → M1 → M2 [→ M3 → M4])
      - Rule-based classification (descriptive)
    Writes one row per unit to coefficients, metrics, and LRT CSVs.

    session_ids: if given, restrict to these sessions (intersected with
    units_vetted). Pair with the same argument to fit_models for the
    one-session runtime check.
    """
    units_vetted = pd.read_csv(
        os.path.join(p.LOGS_DIR, "units_vetted.csv"), index_col=0
    ).sort_values("unit_id")
    all_session_ids = sorted(units_vetted["session_id"].unique().tolist())
    if session_ids is None:
        session_ids = all_session_ids
    else:
        missing = sorted(set(session_ids) - set(all_session_ids))
        if missing:
            print(f"[warn] sessions not in units_vetted: {missing}")
        session_ids = [s for s in session_ids if s in set(all_session_ids)]
        if not session_ids:
            print("[abort] no sessions to evaluate after filtering.")
            return

    # Load anatomical labels once.
    # units_vetted only carries the probe-target ("str"/"v1"). A probe aimed
    # at striatum may record from motor cortex (MOp) and VAL on the way
    # down, etc., so the probe target is NOT the anatomical region. Join
    # unit_properties_final on (session_id, id) for the unit-level labels.
    anat_lookup: Dict[Tuple[str, int], dict] = {}
    try:
        upf = pd.read_csv(os.path.join(p.LOGS_DIR, "unit_properties_final.csv"))
        upf["session_id"] = (
            upf["mouse"].astype(str) + "_"
            + upf["date_only"].astype(str) + "_"
            + upf["probe_region"].astype(str)
        )
        keep = ["session_id", "id", "corrected_region", "region_group", "cell_type"]
        keep = [c for c in keep if c in upf.columns]
        upf_sub = upf[keep].drop_duplicates(subset=["session_id", "id"])
        for r in upf_sub.itertuples(index=False):
            anat_lookup[(r.session_id, int(r.id))] = {
                "corrected_region": getattr(r, "corrected_region", None),
                "region_group": getattr(r, "region_group", None),
                "cell_type": getattr(r, "cell_type", None),
            }
        print(f"[anat] loaded labels for {len(anat_lookup)} (session, id) pairs "
              f"from unit_properties_final.csv")
    except Exception as e:
        print(f"[anat warn] could not load anatomical labels: {e}")

    processed_units = (
        _load_processed_units() if (INCREMENTAL_SAVE and RESUME_PREVIOUS) else set()
    )
    if processed_units:
        print(f"[resume] {len(processed_units)} units already evaluated; will skip.")

    bg_group_lookup = load_bg_group_lookup() if APPLY_WAIT_BAND_FILTER else {}

    if not INCREMENTAL_SAVE:
        all_coef_rows, all_metrics_rows, all_lrt_rows = [], [], []

    for session_id in session_ids:
        print(f"\n=== Evaluate session {session_id} ===")
        session_units = units_vetted[units_vetted["session_id"] == session_id]
        events, trials, units = utils.get_session_data(session_id)

        # Build history features before filtering
        trial_history_idx = build_trial_history_features(trials)

        if "missed" in trials.columns:
            try:
                trials = trials.loc[~coerce_bool_series(trials["missed"])].copy()
            except Exception as e:
                print(f"[warn] could not filter missed trials: {e}")
        trials_idx = trials.set_index("trial_id", drop=False)

        n_pre_band = len(trials_idx)
        trials_idx, band = filter_trials_by_wait_band(
            trials_idx, session_id, bg_group_lookup
        )
        if band is not None:
            print(f"[wait-band] {bg_group_lookup.get(session_id)} band={band}: "
                  f"{len(trials_idx)}/{n_pre_band} trials retained")

        session_region = (
            session_units["region"].iloc[0]
            if "region" in session_units.columns and len(session_units) > 0
            else ""
        )

        n_units = len(session_units)
        for ui, (_, unit_info) in enumerate(session_units.iterrows(), start=1):
            unit_id = unit_info["unit_id"]
            region = unit_info.get("region", session_region)
            unit_key_tuple = (session_id, str(unit_id))
            if unit_key_tuple in processed_units:
                print(f"[skip] {session_id} • unit {unit_id} already evaluated.")
                continue

            art = _load_model_artifact(session_id, unit_id)
            if art is None:
                print(f"[warn] no artifact for unit {unit_id}; skipping.")
                continue

            spikes_df = units[unit_info["id"]]
            spikes_by_trial = spikes_df_to_trial_map(spikes_df)

            # Rebuild raw design, then apply the same column filter that was
            # used at fit time. The artifact's `names` is the source of truth.
            X_raw, y_all, names_raw, trial_id_per_row, used, _ = (
                build_session_design_for_unit(
                    spikes_by_trial, trials_idx, trial_history_idx, events
                )
            )
            if X_raw is None or names_raw is None:
                print(f"[skip] {session_id} • unit {unit_id}: no design.")
                continue

            art_names = art.get("names", [])
            # Match by name to honor whatever fit-time filtering happened
            name_to_idx = {n: i for i, n in enumerate(names_raw)}
            try:
                kept_idx = np.array([name_to_idx[n] for n in art_names], dtype=int)
            except KeyError as e:
                print(f"[warn] {session_id} • unit {unit_id}: artifact name "
                      f"{e} not in current design; skipping.")
                continue
            X_all = X_raw[:, kept_idx]
            names = list(art_names)

            # Sanity: confirm any column the artifact dropped is also dropped here.
            # If a column has nonzero variance now but was dropped at fit time
            # (or vice versa), warn — could indicate data drift.
            current_drop = set(names_raw) - set(art_names)
            recorded_drop = set(art.get("dropped_zero_var_cols", []))
            if current_drop != recorded_drop:
                only_current = current_drop - recorded_drop
                only_recorded = recorded_drop - current_drop
                if only_current or only_recorded:
                    print(f"[warn] {session_id} • unit {unit_id}: filter mismatch. "
                          f"current_only={only_current}  artifact_only={only_recorded}")

            # Apply artifact's standardization → produce full-model prediction
            mean = np.array(art.get("x_mean", [0.0] * X_all.shape[1])).reshape(1, -1)
            std = np.array(art.get("x_std", [1.0] * X_all.shape[1])).reshape(1, -1)
            std[std == 0] = 1.0
            Xs = (X_all - mean) / std
            X_design = sm.add_constant(Xs, has_constant='add')
            params = np.array(art["params"], dtype=float)
            mu = np.exp(X_design @ params)
            pr2 = pseudo_R2_mcfadden(y_all, mu, y_all.mean())

            # Per-family in-sample ΔpseudoR² (using raw X)
            fam = family_indices(names)
            delta_pr2 = {
                f"delta_pr2_{k}": float(fit_and_pr2_unreg(X_all[:, idx], y_all))
                for k, idx in fam.items() if idx.size > 0
            }

            # Per-family CV ΔpseudoR² (trial-stratified)
            delta_pr2_cvnull = {}
            for fam_name, idx in fam.items():
                if idx.size == 0:
                    continue
                delta_cv = cv_delta_pr2_family_vs_null(
                    X_all, y_all, idx, trial_id_per_row=trial_id_per_row,
                    n_splits=N_CV_FOLDS,
                )
                delta_pr2_cvnull[f"delta_pr2_cvnull_{fam_name}"] = float(delta_cv)

            # ---- Nested model LRTs ----
            lrt_results = compute_nested_lrts(X_all, y_all, fam)

            # Classification (descriptive)
            coef_names = ["const"] + names
            label = classify_unit(delta_pr2, names, params)

            # ---- Rows ----
            coef_row = {"session_id": session_id, "unit_id": unit_id, "region": region}
            coef_row.update({f"beta_{n}": float(v) for n, v in zip(coef_names, params)})

            # Look up anatomical labels (None if missing).
            try:
                anat_key = (session_id, int(unit_info["id"]))
            except Exception:
                anat_key = None
            anat = anat_lookup.get(anat_key, {}) if anat_key is not None else {}

            metrics_row = {
                "session_id": session_id,
                "unit_id": unit_id,
                "region": region,                              # probe target
                "corrected_region": anat.get("corrected_region"),
                "region_group": anat.get("region_group"),
                "cell_type": anat.get("cell_type"),
                "n_timepoints": int(y_all.shape[0]),
                "n_trials_used": int(len(used)),
                "mean_rate_hz": float(y_all.sum() / (y_all.shape[0] * DT)),
                "pseudoR2_mcfadden": float(pr2),
                "below_glm_noise_floor": bool(pr2 < MIN_PR2_HEADLINE),
                "alpha": float(art.get("alpha", ALPHA)),
                "l1_wt": float(art.get("l1_wt", L1_WT)),
                "label": label,
                **delta_pr2,
                **delta_pr2_cvnull,
            }

            lrt_row = {
                "session_id": session_id,
                "unit_id": unit_id,
                "region": region,
            }
            for mname, info in lrt_results.get('models', {}).items():
                lrt_row[f'll_{mname}'] = info['ll']
                lrt_row[f'n_params_{mname}'] = info['n_params']
            for cmp_name, info in lrt_results.get('lrts', {}).items():
                lrt_row[f'lrt_{cmp_name}_chi2_p'] = info['p_value']
                lrt_row[f'lrt_{cmp_name}_df'] = info['df']

            if INCREMENTAL_SAVE:
                _append_row(COEF_CSV, coef_row)
                _append_row(METRICS_CSV, metrics_row)
                _append_row(LRT_CSV, lrt_row)
                processed_units.add(unit_key_tuple)
            else:
                all_coef_rows.append(coef_row)
                all_metrics_rows.append(metrics_row)
                all_lrt_rows.append(lrt_row)

            headline_p = lrt_results.get('lrts', {}).get('M1_vs_M2', {}).get('p_value', np.nan)
            print(f"[eval] unit {unit_id}  {ui}/{n_units}  PR2={pr2:.3f}  "
                  f"label={label}  M1_vs_M2 p={headline_p:.3g}")

    if not INCREMENTAL_SAVE:
        if all_coef_rows:
            pd.DataFrame(all_coef_rows).to_csv(COEF_CSV, index=False)
            print(f"\nSaved coefficients → {COEF_CSV}")
        if all_metrics_rows:
            pd.DataFrame(all_metrics_rows).to_csv(METRICS_CSV, index=False)
            print(f"Saved metrics → {METRICS_CSV}")
        if all_lrt_rows:
            pd.DataFrame(all_lrt_rows).to_csv(LRT_CSV, index=False)
            print(f"Saved LRTs → {LRT_CSV}")
    print("\nDone evaluating models.")


# =====================================================================
# Region-level FDR + headline summary
# =====================================================================
def bh_fdr(pvals: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns boolean array of rejection at q."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = q * np.arange(1, n + 1) / n
    rej_sorted = ranked <= thresholds
    if rej_sorted.any():
        max_k = np.where(rej_sorted)[0].max()
        rej_sorted[: max_k + 1] = True
    rej = np.zeros(n, dtype=bool)
    rej[order] = rej_sorted
    return rej


def summarize_regions(lrt_csv: Path = LRT_CSV, metrics_csv: Path = METRICS_CSV,
                      q: float = FDR_Q,
                      apply_noise_floor: bool = True,
                      group_col: str = "region_group",
                      exclude_groups: Tuple[str, ...] = ("Excluded", "Other")) -> pd.DataFrame:
    """
    Aggregate per-unit M1_vs_M2 LRT results to anatomical-region level.

    Defaults to `region_group` from unit_properties_final (Motor cortex,
    Thalamus, Visual cortex, …) — NOT the probe-target `region` column,
    which mixes structures the probe passed through. Falls back to `region`
    if `region_group` is unavailable.

    Reports effect-size distribution stats (median χ² / −log10 p, fraction
    above thresholds) in addition to nominal/FDR significance fractions,
    because the M1_vs_M2 design appears to fire at high rates in any
    region with wait-period firing structure — the meaningful cross-region
    contrast is the *magnitude* of the effect, not the fraction passing
    a binary threshold.

    M0_vs_M1 is excluded — anti-conservative under shuffle null
    (see compute_nested_lrts docstring) so it should not enter inference.

    Units flagged below_glm_noise_floor=True are dropped if apply_noise_floor.
    """
    if not lrt_csv.exists():
        print(f"[summary warn] {lrt_csv} not found.")
        return pd.DataFrame()
    lrt = pd.read_csv(lrt_csv)

    # Pull anatomical labels + noise-floor flag from metrics if available.
    metrics_cols = {"session_id", "unit_id", "below_glm_noise_floor",
                    "pseudoR2_mcfadden", "mean_rate_hz",
                    "region_group", "corrected_region"}
    floor_mask_col = None
    if metrics_csv.exists():
        try:
            metrics = pd.read_csv(
                metrics_csv,
                usecols=lambda c: c in metrics_cols,
            )
            join_cols = [c for c in
                         ["below_glm_noise_floor", "pseudoR2_mcfadden",
                          "mean_rate_hz", "region_group", "corrected_region"]
                         if c in metrics.columns]
            lrt = lrt.merge(
                metrics[["session_id", "unit_id"] + join_cols],
                on=["session_id", "unit_id"], how="left",
            )
            if "below_glm_noise_floor" in lrt.columns:
                floor_mask_col = "below_glm_noise_floor"
        except Exception as e:
            print(f"[summary warn] could not load metrics CSV: {e}")

    # Pick group column with fallback
    if group_col not in lrt.columns or lrt[group_col].isna().all():
        if "region" in lrt.columns:
            print(f"[summary warn] {group_col} not available; falling back "
                  f"to probe-target `region`. Re-run evaluate_models to "
                  f"populate anatomical labels.")
            group_col = "region"
        else:
            print("[summary warn] no group column available.")
            return pd.DataFrame()

    # Drop diagnostic-only anatomical groups (white-matter / unplaceable /
    # off-target) — these aren't valid biological regions for inference.
    if exclude_groups:
        lrt = lrt[~lrt[group_col].isin(exclude_groups)]

    summary_rows = []
    for group_val, sub in lrt.groupby(group_col, dropna=False):
        sub_total = len(sub)
        if floor_mask_col is not None:
            kept = sub[~sub[floor_mask_col].fillna(False).astype(bool)]
        else:
            kept = sub
        pvals = kept['lrt_M1_vs_M2_chi2_p'].values
        finite_mask = np.isfinite(pvals)
        if not finite_mask.any():
            continue
        pvals_f = pvals[finite_mask]
        dfs = kept['lrt_M1_vs_M2_df'].values[finite_mask]
        # χ² from p and df (vectorized via list comprehension; finite by construction)
        chi2_vals = np.array([
            stats.chi2.isf(p_, d_) if (p_ > 0 and np.isfinite(d_) and d_ > 0)
            else np.inf
            for p_, d_ in zip(pvals_f, dfs)
        ])
        neglog10p = -np.log10(np.clip(pvals_f, 1e-300, None))
        rej_uncorr = pvals_f < LRT_ALPHA
        rej_fdr = bh_fdr(pvals_f, q=q)

        summary_rows.append({
            group_col: group_val,
            'n_units_total': int(sub_total),
            'n_units_above_floor': int(len(kept)),
            'n_units_with_pval': int(finite_mask.sum()),
            'frac_sig_uncorr': float(rej_uncorr.mean()),
            'frac_sig_fdr_q{:.2f}'.format(q): float(rej_fdr.mean()),
            'median_p': float(np.median(pvals_f)),
            'median_neglog10p': float(np.median(neglog10p)),
            'median_chi2': float(np.median(chi2_vals)),
            'frac_chi2_gt_50': float(np.mean(chi2_vals > 50)),
            'frac_chi2_gt_200': float(np.mean(chi2_vals > 200)),
        })

    return pd.DataFrame(summary_rows).sort_values('median_chi2', ascending=False)


# =====================================================================
# Debug example (smoke test on one unit)
# =====================================================================
def debug_example(session_id: Optional[str] = None, unit_id: Optional[int] = None):
    """End-to-end smoke test on ONE example unit.

    Optional session_id / unit_id (the dict key into the session pickle's `units`)
    let you target a known-good unit (e.g. a strong rescaler from VAL or M1/M2)
    that need not be in units_vetted.csv.
    """
    units_vetted = pd.read_csv(
        os.path.join(p.LOGS_DIR, "units_vetted.csv"), index_col=0
    ).sort_values("unit_id")
    if session_id is None and unit_id is None:
        events, trials, spikes = utils.get_data_for_debugging(units_vetted)
        print("[debug] using utils.get_data_for_debugging defaults")
    else:
        if session_id is None or unit_id is None:
            raise ValueError("must provide both session_id and unit_id, or neither")
        events, trials, units = utils.get_session_data(session_id)
        if unit_id not in units:
            raise KeyError(
                f"unit_id {unit_id} not in session {session_id}; "
                f"available keys (first 10): {sorted(list(units.keys()))[:10]}"
            )
        spikes = units[unit_id]
        print(f"[debug] session={session_id}  unit_id={unit_id}  "
              f"spikes_df rows={len(spikes)}")

    # Build history BEFORE filtering miss
    trial_history_idx = build_trial_history_features(trials)

    if "missed" in trials.columns:
        trials = trials.loc[~coerce_bool_series(trials["missed"])].copy()
    trials_idx = trials.set_index("trial_id", drop=False)

    bg_group_lookup = load_bg_group_lookup() if APPLY_WAIT_BAND_FILTER else {}
    n_pre_band = len(trials_idx)
    trials_idx, band = filter_trials_by_wait_band(
        trials_idx, session_id or "", bg_group_lookup
    )
    if band is not None:
        print(f"[debug] wait-band {bg_group_lookup.get(session_id)} {band}: "
              f"{len(trials_idx)}/{n_pre_band} trials retained")

    spikes_by_trial = spikes_df_to_trial_map(spikes)

    # Unit inclusion
    ok, reason = check_unit_inclusion(spikes_by_trial, trials_idx, trial_history_idx)
    print(f"[debug] unit inclusion: {ok} ({reason})")
    if not ok:
        return

    X_raw, y_all, names_raw, trial_id_per_row, used, skipped = (
        build_session_design_for_unit(
            spikes_by_trial, trials_idx, trial_history_idx, events
        )
    )
    if X_raw is None:
        print("[debug] no design built.")
        return

    # Drop zero-variance columns (same as fit_models would)
    X_all, names, kept_mask = drop_zero_variance_columns(X_raw, names_raw)
    dropped = [n for n, k in zip(names_raw, kept_mask) if not k]
    if dropped:
        print(f"[debug] dropped {len(dropped)} zero-var cols: {dropped}")

    print(f"[debug] trials_used={len(used)}  X_raw={X_raw.shape}  X_kept={X_all.shape}  y={y_all.shape}")
    print(f"[debug] n_history_cols = {sum(1 for n in names if 'prev_' in n or 'trial_number' in n or 'num_bg' in n or 'ew_reward' in n)}")

    # Matrix diagnostics
    try:
        rank_full = np.linalg.matrix_rank(X_all)
        cond_full = np.linalg.cond(X_all)
        print(f"[diag] rank(X) = {rank_full} / {X_all.shape[1]} cols | cond(X) = {cond_full:.3e}")

        # Zero-variance columns: contribute no information; ridge will set
        # their coefficient from the prior, so flag them explicitly.
        col_std = X_all.std(axis=0)
        zero_var = np.where(col_std == 0)[0]
        if zero_var.size > 0:
            print(f"[diag] zero-variance columns ({zero_var.size}): "
                  f"{[names[i] for i in zero_var]}")

        # SVD on standardized X (matches the production fit's geometry) to
        # surface near-collinear directions beyond literal zero columns.
        Xs_diag, _, _ = standardize_columns(X_all)
        sv = np.linalg.svd(Xs_diag, compute_uv=False)
        if sv.size > 0 and sv.max() > 0:
            ratio = sv.min() / sv.max()
            print(f"[diag] standardized X: min(σ)/max(σ) = {ratio:.3e}")
            tiny_thresh = 1e-10 * sv.max()
            n_tiny = int((sv < tiny_thresh).sum())
            if n_tiny > 0:
                print(f"[diag] {n_tiny} singular values near zero "
                      f"(< 1e-10 · max) → effective rank "
                      f"{X_all.shape[1] - n_tiny} / {X_all.shape[1]}")
    except Exception as e:
        print(f"[diag] full matrix diagnostics failed: {e}")

    # Full-model fit (unregularized for sanity)
    Xc = sm.add_constant(X_all, has_constant='add')
    res = sm.GLM(y_all, Xc, family=sm.families.Poisson()).fit(maxiter=200)
    mu = res.predict(Xc)
    pr2 = pseudo_R2_mcfadden(y_all, mu, y_all.mean())
    mean_rate = y_all.sum() / (len(y_all) * DT)
    print(f"[debug] full-model pseudoR2={pr2:.3f}  mean_rate={mean_rate:.2f} Hz")

    # Per-family ΔpseudoR² (in-sample)
    fam = family_indices(names)
    print(f"\n[debug] in-sample ΔpseudoR² per family:")
    for k, idx in fam.items():
        if idx.size == 0:
            continue
        d = fit_and_pr2_unreg(X_all[:, idx], y_all)
        print(f"   {k:>22s}: {d:+.4f}  (n_cols={idx.size})")

    # Nested LRTs
    print(f"\n[debug] Nested model LRTs:")
    lrt = compute_nested_lrts(X_all, y_all, fam)
    for mname, info in lrt.get('models', {}).items():
        print(f"   {mname}: ll={info['ll']:.2f}  n_params={info['n_params']}")
    for cmp_name, info in lrt.get('lrts', {}).items():
        print(f"   LRT {cmp_name}: chi2_df={info['df']}  p={info['p_value']:.3g}")

    # Plot one example trial — pick from `used` (trials we already built
    # X/y for), so it's guaranteed to have valid history features and be
    # past the first trial of the session.
    try:
        tid_plot = next(
            (tid for tid in used if len(spikes_by_trial.get(tid, [])) > 0),
            None,
        )
        if tid_plot is not None:
            plot_one_trial(
                trials_idx.loc[tid_plot], events, spikes_by_trial[tid_plot],
                trial_history_idx=trial_history_idx,
                title=f"Debug plot • trial {tid_plot}",
                save_path=PLOT_SAVE_DIR / f"debug_trial_{tid_plot}.png",
            )
        else:
            print("[debug] no spiking trial in `used` to plot.")
    except Exception as e:
        print(f"[debug] plotting failed: {e}")


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    # For fit/evaluate, extra args are session_ids to restrict the run
    extra_sessions = sys.argv[2:] if len(sys.argv) > 2 else None
    if mode == "fit":
        fit_models(session_ids=extra_sessions)
    elif mode == "evaluate":
        evaluate_models(session_ids=extra_sessions)
    elif mode == "debug":
        sid = sys.argv[2] if len(sys.argv) > 2 else None
        uid = int(sys.argv[3]) if len(sys.argv) > 3 else None
        debug_example(session_id=sid, unit_id=uid)
    elif mode == "summary":
        s = summarize_regions()
        if not s.empty:
            print(s.to_string(index=False))
            summary_csv = OUT_DIR / "region_summary_M1_vs_M2.csv"
            s.to_csv(summary_csv, index=False)
            print(f"\nSaved → {summary_csv}")
    else:
        print(f"Usage: python {sys.argv[0]} [fit|evaluate [session_ids...]|debug [session_id unit_id]|summary]")
