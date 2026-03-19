#!/usr/bin/env python3
# fit_glm_per_unit.py
#
# Trains one Poisson GLM per unit using:
#   - BG time basis (masked to BG)
#   - WAIT time basis (masked to WAIT)
#   - (Optional combined BG+WAIT basis is disabled by config)
#   - Hazard(t) = 1 - exp(-t/3), orthogonalized to WAIT time basis
#   - Cue-on (delta), Cue-off (delta)  (cue_box removed to reduce collinearity)
#   - Lick kernels: BG mistake, decision, consumption
#   - Outcome during consumption
#   - (optional) Spike history taps
#
# Outputs:
#   - CSV with coefficients per unit, column names, and fit metrics
#   - CSV with per-family Δmetrics (optional)
#
# Assumptions:
#   - You have trial/event structures and unit spike-times available.
#   - If your project exposes loaders in `utils` and `paths` like your notebook,
#     this script will try to use them. Otherwise, fill the TODOs in the
#     "Project-specific loaders" section to point to your data.

#!/usr/bin/env python3
# fit_glm_per_unit_min.py
# Minimal-adaptation GLM training using your original column names.

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt

import utils
import paths as p
from glm_design_helpers import assemble_design, make_time_bins

# -------------------
# Config (simple)
# -------------------
DT = 0.020          # 20 ms bin size (change if you prefer)
N_BASIS_BG = 8       # ramp within BG
N_BASIS_WAIT = 8     # ramp within WAIT
N_BASIS_BGWAIT = 0   # disable combined BG+WAIT basis (set >0 to re-enable)
HAZARD_TAU = 3.0
INCLUDE_SPIKE_HISTORY = True  # enable inclusion of post-spike history taps
ALPHA = 0.1         # regularization strength
L1_WT = 0.5         # 0=ridge, 1=lasso

# === Classification / model options ===
PR2_THR_BG   = 0.002
PR2_THR_WAIT = 0.002
PR2_THR_HAZ  = 0.0005

# Use standardized X + regularized Poisson for production coefficients
USE_STABLE_PRODUCTION_FIT = True  # recommended for full runs

# Plotting options
SHOW_PLOTS = True                    # set False on headless runs

# OUT_DIR = Path("/Volumes/T7 Shield/glm_encoding")
OUT_DIR = Path(p.DATA_DIR) / 'glm_encoding_w_history'
OUT_DIR.mkdir(parents=True, exist_ok=True)
COEF_CSV = OUT_DIR / "glm_coefficients_per_unit.csv"
METRICS_CSV = OUT_DIR / "glm_fit_metrics_per_unit.csv"
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Incremental saving / resume options
INCREMENTAL_SAVE = True   # if True, append each unit's results immediately
RESUME_PREVIOUS = False   # if True, skip units already present in existing CSV(s)

PLOT_SAVE_DIR = OUT_DIR / "debug_plots"
PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# Small helpers
# -------------------
def spikes_df_to_trial_map(spikes_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Your spikes CSVs have columns like: trial_id, trial_time, (period ...).
    Convert to {trial_id: np.array(trial_time)}.
    """
    if not {"trial_id", "trial_time"}.issubset(spikes_df.columns):
        raise ValueError("spikes_df must contain 'trial_id' and 'trial_time'.")
    out = {}
    for tid, sub in spikes_df.groupby("trial_id"):
        out[int(tid)] = sub["trial_time"].astype(float).to_numpy()
    return out

def get_lick_times_for_trial(events: pd.DataFrame, trial_id: int) -> Tuple[List[float], List[float]]:
    """
    Return (lick_times_bg, lick_times_cons) in *trial* coordinates.
    Supports either:
      - 'lick_cons' (unsplit)
      - or 'lick_cons_reward' / 'lick_cons_no_reward' (already split)
    """
    ev = events.loc[events["trial_id"] == trial_id]
    if "event_start_trial_time" not in ev.columns:
        raise ValueError("events must have 'event_start_trial_time' (trial-relative)")

    # BG licks
    lick_times_bg = ev.loc[ev["event_type"] == "lick_bg", "event_start_trial_time"].astype(float).to_list()

    # Consumption licks (handle either naming scheme)
    mask_cons = ev["event_type"].isin(["lick_cons", "lick_cons_reward", "lick_cons_no_reward"])
    lick_times_cons = ev.loc[mask_cons, "event_start_trial_time"].astype(float).to_list()

    return lick_times_bg, lick_times_cons

def coerce_bool_series(s: pd.Series) -> pd.Series:
    """
    Robustly convert a Series to boolean:
      - bool dtype: returned as-is
      - numeric: nonzero → True
      - strings: 'true','t','yes','y','1' → True; 'false','f','no','n','0' → False
      - other/unrecognized → False
    """
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s != 0
    mapping = {
        "true": True, "t": True, "yes": True, "y": True, "1": True,
        "false": False, "f": False, "no": False, "n": False, "0": False,
    }
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(False)

def build_Xy_for_trial(tr: pd.Series, events: pd.DataFrame, spikes_trial: np.ndarray):
    """
    Use original trial columns:
      cue_on_time, cue_off_time, decision_time, rewarded, consumption_length
    """
    # Times in trial reference (your CSVs already are in trial coordinates)
    bg_start = float(tr["cue_on_time"]) 
    bg_end   = float(tr["cue_off_time"]) 
    wait_end = float(tr["decision_time"]) 
    outcome_rewarded = int(tr["rewarded"]) 
    cons_len = float(tr.get("consumption_length", 3.0))

    # Basic sanity checks to avoid invalid binning
    if not np.isfinite(wait_end):
        raise ValueError("decision_time is not finite")
    if not np.isfinite(cons_len) or cons_len <= 0:
        cons_len = 3.0  # fallback
    if not np.isfinite(bg_start) or not np.isfinite(bg_end):
        raise ValueError("cue_on/off times are not finite")
    if wait_end + cons_len <= 0:
        raise ValueError("trial_end (decision + consumption_length) <= 0")

    # Bin from 0 (trial start) to decision + consumption window
    bin_edges = make_time_bins(trial_start=0.0, trial_end=wait_end + cons_len, dt=DT)

    # Lick streams
    lick_times_bg, lick_times_cons = get_lick_times_for_trial(events, int(tr["trial_id"]))
    lick_time_decision = wait_end  # decision lick time = decision_time

    # Design matrix (hazard starts at WAIT onset = bg_end)
    # Precompute spike counts per bin (response) so we can feed them to spike history
    spike_counts, _ = np.histogram(spikes_trial, bins=bin_edges)

    X, names, _ = assemble_design(
        bin_edges=bin_edges,
        bg_start=bg_start,
        bg_end=bg_end,
        wait_end=wait_end,
        lick_times_bg=lick_times_bg,
        lick_time_decision=lick_time_decision,
        lick_times_cons=lick_times_cons,
        outcome_rewarded=outcome_rewarded,
        spike_counts_for_history=spike_counts if INCLUDE_SPIKE_HISTORY else None,
        dt=DT,
        n_basis_bg=N_BASIS_BG,
        n_basis_wait=N_BASIS_WAIT,
        n_basis_bgwait=N_BASIS_BGWAIT,
        hazard_tau=HAZARD_TAU,
        include_spike_history=INCLUDE_SPIKE_HISTORY,
        drop_cue_box=True
    )

    return X, spike_counts, names

# Removed unused fit_poisson_elasticnet (not referenced)

def pseudo_R2_mcfadden(y_true: np.ndarray, mu_full: np.ndarray, mu_null: float) -> float:
    mu_full = np.clip(mu_full, 1e-9, None)
    ll_full = np.sum(y_true * np.log(mu_full) - mu_full)
    mu_null = np.clip(mu_null, 1e-9, None)
    ll_null = np.sum(y_true * np.log(mu_null) - mu_null)
    return 1.0 - (ll_full / ll_null) if ll_null != 0 else np.nan

def standardize_columns(X):
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std

# ---------- Family helpers & quick PR2 (unregularized IRLS) ----------
def family_indices(names: List[str]):
    """Return index arrays for predictor families present in names."""
    cols = np.array(names)
    return {
        "BG_time":   np.where(np.char.startswith(cols, "bg_time_"))[0],
        "WAIT_time": np.where(np.char.startswith(cols, "wait_time_"))[0],
        # BGWAIT_time kept for forward compatibility; will be empty if basis disabled.
        "BGWAIT_time": np.where(np.char.startswith(cols, "bgwait_time_"))[0],
        "hazard":    np.where(cols == "hazard_ortho")[0],
        "cue":       np.where(np.isin(cols, ["cue_on","cue_off"]))[0],
        "licks":     np.where(np.isin(cols, ["lick_bg","lick_decision","lick_cons"]))[0],
        "outcome":   np.where(cols == "outcome_rewarded")[0],
        "history":   np.where(np.char.startswith(cols, "hist_"))[0],
    }

def fit_and_pr2(X: np.ndarray, y: np.ndarray) -> float:
    Xc = sm.add_constant(X, has_constant='add')
    res_tmp = sm.GLM(y, Xc, family=sm.families.Poisson()).fit(maxiter=200)
    mu_tmp  = res_tmp.predict(Xc)
    return pseudo_R2_mcfadden(y, mu_tmp, y.mean())

def glm_fit_predict_unreg(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    """
    Fit an unregularized Poisson GLM (IRLS) and predict on test.
    This mirrors fit_and_pr2 but splits train/test explicitly.
    """
    X_train_c = sm.add_constant(X_train, has_constant='add')
    X_test_c  = sm.add_constant(X_test, has_constant='add')
    res = sm.GLM(y_train, X_train_c, family=sm.families.Poisson()).fit(maxiter=200)
    mu_test = res.predict(X_test_c)
    return res, mu_test

def _sequential_kfold_indices(n_samples: int, n_splits: int):
    """
    Yield (train_idx, test_idx) where each test_idx is a contiguous block.
    No shuffling; preserves order. Dependency-free alternative to sklearn KFold.
    """
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

def cv_delta_pr2_family_vs_null(
    X_all: np.ndarray,
    y_all: np.ndarray,
    fam_idx: np.ndarray,
    n_splits: int = 5,
) -> float:
    """
    Cross-validated pseudoR² for ONE covariate family vs a null (intercept-only) model.

    For each fold:
      - Fit family-only GLM on train, compute PR² on test vs null(mean of train).
    Returns mean PR² across folds.
    """
    if fam_idx.size == 0:
        return 0.0

    scores = []
    for train_idx, test_idx in _sequential_kfold_indices(len(y_all), n_splits):
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        Xf_train = X_all[train_idx][:, fam_idx]
        Xf_test  = X_all[test_idx][:, fam_idx]

        mu_null_train = float(y_train.mean())
        _, mu_f_test = glm_fit_predict_unreg(Xf_train, y_train, Xf_test)
        pr2_fam = pseudo_R2_mcfadden(y_test, mu_f_test, mu_null_train)
        scores.append(pr2_fam)

    return float(np.mean(scores)) if scores else 0.0
    
# ---------- Unit classification from ΔpseudoR² + key betas ----------
def classify_unit(delta_pr2: dict, names: List[str], params: np.ndarray) -> str:
    """Simple rule-based classification using ΔpseudoR² thresholds.

    With BG+WAIT basis disabled, 'bg+wait ramping' = both BG and WAIT exceed thresholds.
    """
    dBG   = float(delta_pr2.get("delta_pr2_BG_time", 0.0))
    dWAIT = float(delta_pr2.get("delta_pr2_WAIT_time", 0.0))
    dHAZ  = float(delta_pr2.get("delta_pr2_hazard", 0.0))
    dCUE  = float(delta_pr2.get("delta_pr2_cue", 0.0))
    dLCK  = float(delta_pr2.get("delta_pr2_licks", 0.0))
    dOUT  = float(delta_pr2.get("delta_pr2_outcome", 0.0))

    name_to_coef = {n: float(b) for n, b in zip(["const"] + names, params)}
    b_dec  = name_to_coef.get("lick_decision", 0.0)
    b_cons = name_to_coef.get("lick_cons", 0.0)
    b_out  = name_to_coef.get("outcome_rewarded", 0.0)

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

def plot_one_trial(tr: pd.Series, events: pd.DataFrame, spikes_trial: np.ndarray, title: str = "Example trial fit", save_path: Path = None):
    X, y, _ = build_Xy_for_trial(tr, events, spikes_trial)
    t = np.arange(len(y)) * DT
    Xc = sm.add_constant(X, has_constant='add')
    res = sm.GLM(y, Xc, family=sm.families.Poisson()).fit(maxiter=200)
    mu = res.predict(Xc) / DT  # counts/bin → Hz

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    ax.plot(t, mu, label="Predicted rate (Hz)")
    if len(spikes_trial) > 0:
        ax.vlines(spikes_trial, ymin=0, ymax=max(mu)*0.3, linewidth=0.5)
    bg_start = float(tr["cue_on_time"]); bg_end = float(tr["cue_off_time"])
    wait_end = float(tr["decision_time"]); cons_len = float(tr.get("consumption_length", 3.0))
    ax.axvspan(bg_start, bg_end, alpha=0.1, label="BG")
    ax.axvspan(bg_end, wait_end, alpha=0.1, label="WAIT")
    ax.axvspan(wait_end, wait_end+cons_len, alpha=0.1, label="CONS")
    ax.set_xlabel("Time in trial (s)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_title(title)
    fig.tight_layout()

    # Save image if a path is provided
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"[plot] saved → {save_path}")

    # Show conditionally (won't crash on headless backends)
    if SHOW_PLOTS:
        try:
            plt.show()
        except Exception as e:
            print(f"[plot] show() skipped: {e}")
    plt.close(fig)

# -------------------
# Main loop
# -------------------
def _load_processed_units() -> set:
    """Return a set of (session_id, unit_id) that already exist in either metrics or coef CSVs.
    Used when RESUME_PREVIOUS is True. If files absent or unreadable, returns empty set.
    """
    processed = set()
    for path in [METRICS_CSV, COEF_CSV]:
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
    """Append a single row (dict) to CSV, writing header if file absent/empty."""
    df = pd.DataFrame([row])
    write_header = (not path.exists()) or path.stat().st_size == 0
    try:
        df.to_csv(path, mode='a', header=write_header, index=False)
    except Exception as e:
        print(f"[io warn] failed to append to {path.name}: {e}")

def _artifact_path(session_id, unit_id) -> Path:
    return MODEL_DIR / f"{session_id}__{unit_id}.json"

def _save_model_artifact(session_id, unit_id, region, names, params, x_mean, x_std, extra: dict = None):
    art = {
        "version": 1,
        "session_id": session_id,
        "unit_id": unit_id,
        "region": region,
        "names": list(names),
        "params": [float(v) for v in params],   # includes intercept at index 0
        "x_mean": [float(v) for v in x_mean],
        "x_std": [float(v) for v in x_std],
        "alpha": ALPHA,
        "l1_wt": L1_WT,
        "dt": DT,
        "hazard_tau": HAZARD_TAU,
        "include_spike_history": INCLUDE_SPIKE_HISTORY,
        "n_basis_bg": N_BASIS_BG,
        "n_basis_wait": N_BASIS_WAIT,
        "n_basis_bgwait": N_BASIS_BGWAIT,
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

def fit_models():
    # Load vetted units & sessions
    units_vetted = pd.read_csv(os.path.join(p.LOGS_DIR, "units_vetted.csv"), index_col=0).sort_values("unit_id")
    session_ids = sorted(units_vetted["session_id"].unique().tolist())
    # For fitting stage, consider artifacts as the resume source
    processed_units = set()

    for session_id in session_ids:
        print(f"\n=== Session {session_id} ===")
        session_units = units_vetted[units_vetted["session_id"] == session_id]
        events, trials, units = utils.get_session_data(session_id)
        # Filter out missed trials if column present (keep only trials where missed is False)
        if "missed" in trials.columns:
            try:
                missed_bool = coerce_bool_series(trials["missed"])
                trials = trials.loc[~missed_bool].copy()
            except Exception as e:
                print(f"[warn] could not filter missed trials: {e}")
        trials_idx = trials.set_index("trial_id", drop=False)

        # Prefer a single region label per session when available
        session_region = session_units["region"].iloc[0] if "region" in session_units.columns and len(session_units) > 0 else ""

        n_units = len(session_units)
        for ui, (_, unit_info) in enumerate(session_units.iterrows(), start=1):
            unit_id = unit_info["unit_id"]
            region = unit_info.get("region", session_region)
            unit_key = unit_info["id"]
            spikes_df = units[unit_key]
            spikes_by_trial = spikes_df_to_trial_map(spikes_df)

            # Skip if artifact exists and resuming
            art_path = _artifact_path(session_id, unit_id)
            if RESUME_PREVIOUS and art_path.exists():
                print(f"[skip] session {session_id} • unit {unit_id} artifact exists; skipping fit.")
                continue

            X_blocks, y_blocks = [], []
            names = None

            for tid, spikes_trial in spikes_by_trial.items():
                if tid not in trials_idx.index:
                    continue
                tr = trials_idx.loc[tid]
                try:
                    X, y, nm = build_Xy_for_trial(tr, events, spikes_trial)
                except Exception as e:
                    print(f"[warn] skipping trial {tid}: {e}")
                    continue
                X_blocks.append(X); y_blocks.append(y)
                if names is None: names = nm

            if not X_blocks:
                continue

            X_all = np.vstack(X_blocks); y_all = np.concatenate(y_blocks)


            # ---- production fit (save artifact only) ----
            if USE_STABLE_PRODUCTION_FIT:
                Xs, mean, std = standardize_columns(X_all)
                X_design = sm.add_constant(Xs, has_constant='add')
                res = sm.GLM(y_all, X_design, family=sm.families.Poisson()).fit_regularized(alpha=ALPHA, L1_wt=L1_WT, maxiter=1000)
                params = res.params
                # Save artifact with means/stds to allow later prediction
                _save_model_artifact(
                    session_id, unit_id, region, names, params,
                    x_mean=mean.ravel().tolist(), x_std=std.ravel().tolist(),
                    extra={
                        "n_timepoints": int(y_all.shape[0]),
                        "mean_rate_hz": float(y_all.sum() / (y_all.shape[0] * DT)),
                    }
                )
            else:
                X_design = sm.add_constant(X_all, has_constant='add')
                res = sm.GLM(y_all, X_design, family=sm.families.Poisson()).fit(maxiter=200)
                params = res.params
                # For unstandardized fit, means/stds are identity
                _save_model_artifact(
                    session_id, unit_id, region, names, params,
                    x_mean=(np.zeros(X_all.shape[1]).tolist()),
                    x_std=(np.ones(X_all.shape[1]).tolist()),
                    extra={
                        "n_timepoints": int(y_all.shape[0]),
                        "mean_rate_hz": float(y_all.sum() / (y_all.shape[0] * DT)),
                    }
                )

            print(f"[fit] saved model artifact for unit {unit_id}  {ui}/{n_units}")

    print("\nDone fitting models.")

def evaluate_models():
    # Load vetted units & sessions
    units_vetted = pd.read_csv(os.path.join(p.LOGS_DIR, "units_vetted.csv"), index_col=0).sort_values("unit_id")
    session_ids = sorted(units_vetted["session_id"].unique().tolist())

    processed_units = _load_processed_units() if (INCREMENTAL_SAVE and RESUME_PREVIOUS) else set()
    if processed_units:
        print(f"[resume] loaded {len(processed_units)} processed unit entries; will skip duplicates.")

    if not INCREMENTAL_SAVE:
        all_coef_rows, all_metrics_rows = [], []

    for session_id in session_ids:
        print(f"\n=== Evaluate session {session_id} ===")
        session_units = units_vetted[units_vetted["session_id"] == session_id]
        events, trials, units = utils.get_session_data(session_id)
        if "missed" in trials.columns:
            try:
                trials = trials.loc[~coerce_bool_series(trials["missed"])].copy()
            except Exception as e:
                print(f"[warn] could not filter missed trials: {e}")
        trials_idx = trials.set_index("trial_id", drop=False)

        session_region = session_units["region"].iloc[0] if "region" in session_units.columns and len(session_units) > 0 else ""

        n_units = len(session_units)
        for ui, (_, unit_info) in enumerate(session_units.iterrows(), start=1):
            unit_id = unit_info["unit_id"]
            region = unit_info.get("region", session_region)
            unit_key = unit_info["id"]

            # Resume check using existing CSVs
            unit_key_tuple = (session_id, str(unit_id))
            if unit_key_tuple in processed_units:
                print(f"[skip] session {session_id} • unit {unit_id} already evaluated; skipping.")
                continue

            # Load artifact
            art = _load_model_artifact(session_id, unit_id)
            if art is None:
                print(f"[warn] no model artifact for unit {unit_id}; skipping.")
                continue

            spikes_df = units[unit_info["id"]]
            spikes_by_trial = spikes_df_to_trial_map(spikes_df)

            X_blocks, y_blocks = [], []
            names = None
            for tid, sp in spikes_by_trial.items():
                if tid not in trials_idx.index:
                    continue
                tr = trials_idx.loc[tid]
                try:
                    X, y, nm = build_Xy_for_trial(tr, events, sp)
                except Exception as e:
                    print(f"[warn] skipping trial {tid}: {e}")
                    continue
                X_blocks.append(X); y_blocks.append(y)
                if names is None: names = nm

            if not X_blocks:
                continue

            X_all = np.vstack(X_blocks); y_all = np.concatenate(y_blocks)

            # Align names with artifact
            art_names = art.get("names", [])
            if names is None or len(names) != len(art_names) or any(a != b for a, b in zip(names, art_names)):
                print(f"[warn] design names mismatch for unit {unit_id}; proceeding with recomputed names.")

            # Recreate standardized design using saved mean/std
            mean = np.array(art.get("x_mean", [0.0]*X_all.shape[1])).reshape(1, -1)
            std  = np.array(art.get("x_std",  [1.0]*X_all.shape[1])).reshape(1, -1)
            std[std == 0] = 1.0
            Xs = (X_all - mean) / std
            X_design = sm.add_constant(Xs, has_constant='add')

            params = np.array(art["params"], dtype=float)
            # Poisson log link → mu = exp(Xβ)
            mu = np.exp(X_design @ params)
            pr2 = pseudo_R2_mcfadden(y_all, mu, y_all.mean())

            # ---- family pseudoR² (unregularized, raw X) vs null (family-only models) ----
            fam = family_indices(names)
            delta_pr2 = {f"delta_pr2_{k}": float(fit_and_pr2(X_all[:, idx], y_all))
                         for k, idx in fam.items() if idx.size > 0}

            # ---- CV-based pseudoR² vs null (family-only models) ----
            delta_pr2_cvnull = {}
            for fam_name, idx in fam.items():
                if idx.size == 0:
                    continue
                delta_cv = cv_delta_pr2_family_vs_null(X_all, y_all, idx, n_splits=5)
                delta_pr2_cvnull[f"delta_pr2_cvnull_{fam_name}"] = float(delta_cv)

            # ---- classification ----
            coef_names = ["const"] + names
            label = classify_unit(delta_pr2, names, params)

            # ---- save rows ----
            coef_row = {"session_id": session_id, "unit_id": unit_id, "region": region}
            coef_row.update({f"beta_{n}": float(v) for n, v in zip(coef_names, params)})

            metrics_row = {
                "session_id": session_id,
                "unit_id": unit_id,
                "region": region,
                "n_timepoints": int(y_all.shape[0]),
                "mean_rate_hz": float(y_all.sum() / (y_all.shape[0] * DT)),
                "pseudoR2_mcfadden": float(pr2),
                "alpha": float(art.get("alpha", ALPHA)),
                "l1_wt": float(art.get("l1_wt", L1_WT)),
                "label": label,
                **delta_pr2,
                **delta_pr2_cvnull,
            }

            if INCREMENTAL_SAVE:
                _append_row(COEF_CSV, coef_row)
                _append_row(METRICS_CSV, metrics_row)
                processed_units.add(unit_key_tuple)
            else:
                all_coef_rows.append(coef_row)
                all_metrics_rows.append(metrics_row)

            print(f"[eval] unit {unit_id}  {ui}/{n_units}  PR2={pr2:.3f}  label={label}")

    if not INCREMENTAL_SAVE:
        if all_coef_rows:
            pd.DataFrame(all_coef_rows).to_csv(COEF_CSV, index=False)
            print(f"\nSaved coefficients → {COEF_CSV}")
        if all_metrics_rows:
            pd.DataFrame(all_metrics_rows).to_csv(METRICS_CSV, index=False)
            print(f"Saved metrics → {METRICS_CSV}")
    print("\nDone evaluating models.")

def debug_example():
    """
    End-to-end smoke test on ONE example unit.
    Expects: events, trials, spikes = utils.get_data_for_debugging(units_vetted)
    Uses helpers already defined in this file.
    """
    # 1) Load a tiny debug bundle
    units_vetted = pd.read_csv(os.path.join(p.LOGS_DIR, "units_vetted.csv"), index_col=0).sort_values("unit_id")
    events, trials, spikes = utils.get_data_for_debugging(units_vetted)

    # 3) Prep lookups
    if "missed" in trials.columns:
        trials = trials.loc[~coerce_bool_series(trials["missed"])].copy()
    trials_idx = trials.set_index("trial_id", drop=False)
    spikes_by_trial = spikes_df_to_trial_map(spikes)

    # 4) Build design/response across overlapping trials
    X_blocks, y_blocks = [], []
    names = None
    used_trials = 0

    for tid, sp in spikes_by_trial.items():
        if tid not in trials_idx.index:
            continue
        tr = trials_idx.loc[tid]
        X, y, nm = build_Xy_for_trial(tr, events, sp)
        X_blocks.append(X); y_blocks.append(y)
        if names is None: names = nm
        used_trials += 1

    if not X_blocks:
        print("[debug] no overlapping trials between spikes and trials; nothing to fit.")
        return

    # 5) Stack & sanity
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks).astype(float)
    assert X.shape[0] == y.shape[0], f"Row mismatch: X={X.shape}, y={y.shape}"

    nz_per_col = (X != 0).sum(axis=0)
    print(f"[debug] trials={used_trials}  X={X.shape}  y={y.shape}")
    print(f"[debug] nonzero entries per regressor (first 12): {nz_per_col[:12].tolist()}")

    # 5b) Matrix diagnostics: rank and condition numbers
    try:
        rank_full = np.linalg.matrix_rank(X)
        cond_full = np.linalg.cond(X)
    except Exception as e:
        rank_full, cond_full = np.nan, np.nan
        print(f"[diag] full matrix diagnostics failed: {e}")
    print(f"[diag] rank(X) = {rank_full} / {X.shape[1]} cols | cond(X) = {cond_full:.3e}")

    # Subset condition numbers for suspect collinearity
    cols = np.array(names)
    idx_bg   = np.where(np.char.startswith(cols, "bg_time_"))[0]
    idx_wait = np.where(np.char.startswith(cols, "wait_time_"))[0]
    # cue_box removed, keep indices for current families only
    idx_hazard  = np.where(cols == "hazard_ortho")[0]

    def diag_subset(label: str, idx: np.ndarray):
        if idx.size == 0:
            print(f"[diag] {label}: no columns")
            return
        Xsub = X[:, np.unique(idx)]
        try:
            r = np.linalg.matrix_rank(Xsub)
            c = np.linalg.cond(Xsub)
            print(f"[diag] {label}: n={Xsub.shape[1]}  rank={r}  cond={c:.3e}")
        except Exception as e:
            print(f"[diag] {label}: diagnostics failed: {e}")

    diag_subset("BG_time basis", idx_bg)
    # BG_time + cue_box diagnostic removed (cue_box dropped)
    diag_subset("WAIT_time basis", idx_wait)
    if idx_wait.size > 0 and idx_hazard.size > 0:
        diag_subset("WAIT_time + hazard", np.r_[idx_wait, idx_hazard])

    # 6) Single debug fit: unregularized Poisson (IRLS)
    X_ = sm.add_constant(X, has_constant='add')
    res = sm.GLM(y, X_, family=sm.families.Poisson()).fit(maxiter=200)
    mu  = res.predict(X_)
    pr2 = pseudo_R2_mcfadden(y, mu, y.mean())
    mean_rate = y.sum() / (len(y) * DT)
    print(f"[debug] pseudoR2={pr2:.3f}  mean_rate={mean_rate:.2f} Hz")

    # 7) Top 10 coefficients (by |beta|), skipping intercept for clarity
    coef_names = ["const"] + names
    betas = list(res.params)
    top = sorted(zip(coef_names[1:], betas[1:]), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    print("[debug] top |beta| (name: value):")
    for n, v in top:
        print(f"    {n:>18s}: {v:+.4f}")

    cols = np.array(names)
    families = {
        "BG_time": np.where(np.char.startswith(cols, "bg_time_"))[0],
        "WAIT_time": np.where(np.char.startswith(cols, "wait_time_"))[0],
        "hazard": np.where(cols == "hazard_ortho")[0],
        "cue": np.where(np.isin(cols, ["cue_on","cue_off"]))[0],
        "licks": np.where(np.isin(cols, ["lick_bg","lick_decision","lick_cons"]))[0],
        "outcome": np.where(cols == "outcome_rewarded")[0],
    }

    for k, idx in families.items():
        if idx.size == 0:
            print(f"[debug] +{k:10s} ΔpseudoR2 = (no columns)")
            continue
        print(f"[debug] +{k:10s} ΔpseudoR2 = {fit_and_pr2(X[:, idx], y):+.4f}")
    
    try:
        tid_candidates = [tid for tid, sp in spikes_by_trial.items() if len(sp) > 0]
        tid_plot = tid_candidates[0] if len(tid_candidates) else next(iter(spikes_by_trial.keys()))
        if tid_plot in trials_idx.index:
            plot_one_trial(
                trials_idx.loc[tid_plot],
                events,
                spikes_by_trial[tid_plot],
                title=f"Debug plot • trial {tid_plot}",
                save_path=PLOT_SAVE_DIR / f"debug_trial_{tid_plot}.png"
            )
        else:
            print("[debug] chosen trial not in trials_idx; skip plotting.")
    except StopIteration:
        print("[debug] no trials available to plot.")

# Run:  python 4c_encoding_GLM_w_convolution_take2.py debug
if __name__ == "__main__":
    # fit_models()
    evaluate_models()
