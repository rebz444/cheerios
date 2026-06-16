#!/usr/bin/env python3
"""
4c_simple_glm.py

Descriptive per-unit Poisson GLM — the "what does this neuron respond to?" map.

Motivation
----------
The full reward-history GLM (4c_encoding_GLM_w_history.py) bolts a trial-history
block, reward × time interactions, a session-drift spline, a current-wait
confound family, and a wait-band trial filter on top of a within-trial encoding
model. Those layers exist to make the M1_vs_M2 *interaction* LRT well-calibrated.
They are dead weight — and a source of confounds — if the question is simply
"which event/epoch families does this unit encode?".

This script keeps ONLY the within-trial M0 design and turns it into a clean
descriptive tool:

    Family          What it captures
    --------------  -------------------------------------------------------
    BG_time         Slow ramping / decay across the background period
    WAIT_time       Slow ramping during the wait
    cue             cue_on / cue_off transients
    licks           Per-period lick-locked activity (bg / decision / cons)
    outcome         Reward delivery (consumption epoch)
    hazard          Reward-hazard kernel during wait
    spike_history   Autoregressive (refractoriness + bursts)

NO history features, NO drift basis, NO interactions, NO current-wait control,
NO wait-band filter. Just the per-trial within-trial design. This is the same
M0 design 4c fits as its baseline — see glm_design_helpers.assemble_design.

What it produces, per unit
--------------------------
  - Full-model pseudo-R² (in-sample and cross-validated) — overall fit quality.
  - Per-family CV ΔpseudoR² — the load-bearing output:
      * delta_pr2_marg_{family}   : family-vs-null (marginal; "how much can
                                    this family explain ON ITS OWN").
      * delta_pr2_unique_{family} : full-vs-(full−family) (unique contribution
                                    after every other family is in the model).
    Marginal answers "is this unit lick-driven at all"; unique answers "does
    licks explain variance no other family already accounts for". Collinear
    families (e.g. hazard vs wait_time) split their unique credit, so report
    both.
  - Rule-based classification (descriptive label, e.g. "wait-only ramping",
    "decision-locked") — reuses 4c.classify_unit.
  - Fitted coefficients (kernel betas) for downstream kernel-shape inspection.

Population output
-----------------
  - region_summary.csv  : per region_group, distribution (median / IQR) of each
                          family's ΔpR², plus the fraction of units "dominated"
                          by each family (argmax marginal ΔpR²). This is the map:
                          which families dominate VAL vs MOp vs V1.
  - cell_type_summary.csv : same breakdown by cell_type (MSN / FSI / RS / …).

Relationship to the other tests
-------------------------------
This is descriptive, not inferential. It is meant to be run FIRST, to learn what
each population encodes, so the targeted interaction tests (4d Test 2, 4f Test 1)
can be aimed at the right kernels. Classification labels are descriptive; the
formal tests live in 4d / 4f.

Design reuse
------------
The within-trial design and the fitting / CV / classification machinery are
imported (via importlib, since the filename starts with a digit) from
4c_encoding_GLM_w_history.py so there is a single source of truth. The basis
parameters (DT, N_BASIS_*, hazard_tau, absolute-time basis, t_max_*) are pulled
straight from that module — this script fits the identical M0 design.

CLI
---
    python 4c_simple_glm.py run [session_id ...]   # full run (+ summary + plot)
    python 4c_simple_glm.py summary                # region/cell_type summaries
    python 4c_simple_glm.py plot                   # family ΔpR² distribution plot
    python 4c_simple_glm.py debug <session_id> <unit_id>   # single-unit smoke test

External dependencies:
  - utils.get_session_data(session_id)
  - paths (LOGS_DIR, DATA_DIR)
  - glm_design_helpers.assemble_design, make_time_bins
  - 4c_encoding_GLM_w_history.py (machinery reuse, loaded via importlib)
"""

import os
import sys
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import utils
import paths as p
from glm_design_helpers import assemble_design, make_time_bins


# =====================================================================
# Reuse 4c machinery (single source of truth for the M0 design + fitting)
# =====================================================================
def _load_glm4c():
    """Load 4c_encoding_GLM_w_history.py as a module (filename starts with a
    digit, so it can't be imported by name)."""
    path = Path(__file__).resolve().parent / "4c_encoding_GLM_w_history.py"
    spec = importlib.util.spec_from_file_location("glm_w_history", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


glm4c = _load_glm4c()

# Within-trial design params — pulled from 4c (DT, hazard, spike history) so the
# event/lick/outcome design is identical. The BG/WAIT time bases differ: see
# per-group grids below.
DT = glm4c.DT
N_BASIS_BGWAIT = glm4c.N_BASIS_BGWAIT
HAZARD_TAU = glm4c.HAZARD_TAU
ABSOLUTE_TIME_BASIS = glm4c.ABSOLUTE_TIME_BASIS
INCLUDE_SPIKE_HISTORY = glm4c.INCLUDE_SPIKE_HISTORY

# --- Per-BG-group absolute-time grids for the BG and WAIT raised-cosine bases ---
# The basis is ABSOLUTE time (real seconds since epoch onset), not normalized;
# bins past t_max clamp to the last grid point. A single global BG grid of
# [0,5s] clamped 57% of Long-BG trials (median Long BG ≈ 5.2s) and wasted most
# columns on Short-BG (median ≈ 1.1s). So t_max is set PER BG GROUP to cover
# ~the 90-95th pct of that group's durations (from the duration audit), with
# n_basis scaled to hold the bump spacing t_max/(n_basis-1) ≈ 1.3-1.6s — so
# resolution is comparable across groups even though the grids differ.
#
# Values are (t_max_seconds, n_basis):
#   BG   Short ≈1.1s (95th 2.3s) → [0,2.5s]   |  Long ≈5.2s (95th 10.8s) → [0,11s]
#   WAIT Short med 2.9s (95th 10s) → [0,11s]   |  Long med 3.0s (90th 15.6s) → [0,16s]
BG_BASIS_BY_GROUP   = {"Short BG": (2.5, 4),  "Long BG": (11.0, 9)}
WAIT_BASIS_BY_GROUP = {"Short BG": (11.0, 8), "Long BG": (16.0, 11)}
# Fallback when a session's BG group is unknown (matches the old 4c global grid).
BG_BASIS_DEFAULT, WAIT_BASIS_DEFAULT = (5.0, 8), (10.0, 8)


def basis_params_for_group(bg_group: Optional[str]):
    """Return ((t_max_bg, n_basis_bg), (t_max_wait, n_basis_wait)) for a group."""
    return (BG_BASIS_BY_GROUP.get(bg_group, BG_BASIS_DEFAULT),
            WAIT_BASIS_BY_GROUP.get(bg_group, WAIT_BASIS_DEFAULT))


# Reused helpers
coerce_bool_series = glm4c.coerce_bool_series
spikes_df_to_trial_map = glm4c.spikes_df_to_trial_map
get_lick_times_for_trial = glm4c.get_lick_times_for_trial
load_bg_group_lookup = glm4c.load_bg_group_lookup
family_indices = glm4c.family_indices
pseudo_R2_mcfadden = glm4c.pseudo_R2_mcfadden
fit_and_pr2_unreg = glm4c.fit_and_pr2_unreg
glm_fit_predict_unreg = glm4c.glm_fit_predict_unreg
get_cv_splits = glm4c.get_cv_splits
drop_zero_variance_columns = glm4c.drop_zero_variance_columns
standardize_columns = glm4c.standardize_columns
classify_unit = glm4c.classify_unit
bh_fdr = glm4c.bh_fdr


# =====================================================================
# Config
# =====================================================================
# Families present in the M0 design (assemble_design output, no history).
SIMPLE_FAMILIES = [
    "BG_time", "WAIT_time", "hazard", "cue", "licks", "outcome", "spike_history",
]

# The "what does this neuron respond to?" map is about task/event encoding.
# spike_history is intrinsic dynamics (refractoriness/bursting) — a nuisance
# regressor that otherwise wins the dominant-family argmax on nearly every unit
# and washes out cross-region task contrasts. It stays a reported ΔpR² column,
# but is excluded from the dominant_family classification.
TASK_FAMILIES = [f for f in SIMPLE_FAMILIES if f != "spike_history"]

# A unit whose best TASK family explains less than this (CV marginal ΔpR²) is
# labeled 'untuned/weak' rather than handed a noise-driven dominant family.
DOMINANT_MIN_DPR2 = 0.002

# Sub-families: licks and cue are lumped in SIMPLE_FAMILIES, but the individual
# kernels are biologically distinct (background vs choice vs consummatory licking;
# cue onset vs offset). We additionally save per-sub-family ΔpR² so a later
# recategorization can split them WITHOUT a refit. These are single-column
# families drawn from the assemble_design output names. They are NOT used for
# the dominant_family argmax (that stays over the lumped TASK_FAMILIES).
SUBFAMILIES = ["lick_bg", "lick_decision", "lick_cons", "cue_on", "cue_off"]

# Every family we compute/store ΔpR² for (headline lumped families + sub-families).
EVAL_FAMILIES = SIMPLE_FAMILIES + SUBFAMILIES

# Unit inclusion (no history-based criteria — those were for the interaction test)
MIN_SPIKE_COUNT = 50
MAX_MEAN_FR_HZ = 50.0
MIN_TRIALS = 30          # min valid (non-miss) trials with a finite design

# CV unique-ΔpR² is the most expensive piece (refits full − each family per
# fold; ~60 near-full Poisson fits/unit at full scale). Marginal ΔpR², in-sample,
# sub-family splits, and per-fold CV are all cheap and always computed. Off for
# the full-dataset sweep — re-enable for a targeted subset via a resume-aware
# re-run when unique variance attribution is needed. With this off, the
# delta_pr2_unique_* columns are written as NaN and dominant_family (which uses
# marginal ΔpR²) is unaffected.
COMPUTE_UNIQUE_DPR2 = False

# Save the per-fold CV ΔpR² (one row per unit × family × kind × fold) so CV
# error bars / significance can be derived later without a refit.
SAVE_CV_FOLDS = True

# Region grouping: drop anatomically-unplaceable units from headline tables.
EXCLUDE_REGION_GROUPS = ("Excluded", "Other")

# Paths
OUT_DIR = Path(p.DATA_DIR) / "glm_simple"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_UNIT_CSV = OUT_DIR / "per_unit.csv"
COEF_CSV = OUT_DIR / "coefficients_per_unit.csv"
EXCLUDED_CSV = OUT_DIR / "excluded_units.csv"
CV_FOLDS_CSV = OUT_DIR / "per_unit_cv_folds.csv"
REGION_SUMMARY_CSV = OUT_DIR / "region_summary.csv"
CELLTYPE_SUMMARY_CSV = OUT_DIR / "cell_type_summary.csv"
DPR2_PLOT_PATH = OUT_DIR / "family_dpr2_by_region.png"

INCREMENTAL_SAVE = True
INCREMENTAL_EVERY = 25   # write partial CSVs every N units within a session
PROGRESS_EVERY = 50      # print a running progress line every N units processed

# Resume: on `run`, if per_unit.csv already exists, skip units already present
# (fit OR previously excluded) and append to the existing CSVs instead of
# wiping them. Lets a crashed/interrupted overnight run be restarted cheaply.
# To force a clean run, delete the output CSVs first (or set RESUME = False).
RESUME = True


# =====================================================================
# Anatomical labels (same join as 4d/4f)
# =====================================================================
def load_anatomical_labels() -> pd.DataFrame:
    """Load unit-level anatomical labels keyed by (session_id, id).

    unit_properties_final.csv keys by (mouse, date_only, insertion_number, id)
    and carries no session_id; join to units_vetted.csv to attach it.
    """
    ana_path = Path(p.LOGS_DIR) / "unit_properties_final.csv"
    uv_path = Path(p.LOGS_DIR) / "units_vetted.csv"
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
    ana["date"] = pd.to_datetime(ana["date_only"]).dt.strftime("%Y-%m-%d")
    ana_keep = ana[["mouse", "date", "insertion_number", "id",
                    "corrected_region", "region_group", "cell_type"]]
    uv_keep = uv[["mouse", "date", "insertion_number", "id", "session_id"]]
    merged = uv_keep.merge(ana_keep,
                           on=["mouse", "date", "insertion_number", "id"],
                           how="inner")
    return merged[["session_id", "id",
                   "corrected_region", "region_group", "cell_type"]]


# =====================================================================
# Per-trial within-trial design (M0 only — no history augmentation)
# =====================================================================
def build_trial_Xy(tr: pd.Series, events: pd.DataFrame,
                   spikes_trial: np.ndarray,
                   bg_basis: Tuple[float, int] = BG_BASIS_DEFAULT,
                   wait_basis: Tuple[float, int] = WAIT_BASIS_DEFAULT):
    """Build (X, y, names) for one trial using the within-trial M0 design.

    This is build_Xy_for_trial from 4c with the trial-history augmentation
    removed. `bg_basis`/`wait_basis` are (t_max_seconds, n_basis) for the BG and
    WAIT raised-cosine time bases — passed per BG group (see basis_params_for_group).
    Raises ValueError if the trial can't be processed.
    """
    t_max_bg, n_basis_bg = bg_basis
    t_max_wait, n_basis_wait = wait_basis
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
        n_basis_bg=n_basis_bg, n_basis_wait=n_basis_wait,
        n_basis_bgwait=N_BASIS_BGWAIT,
        absolute_time_basis=ABSOLUTE_TIME_BASIS,
        t_max_bg=t_max_bg, t_max_wait=t_max_wait,
        hazard_tau=HAZARD_TAU,
        include_spike_history=INCLUDE_SPIKE_HISTORY,
        drop_cue_box=True,
    )
    return X, spike_counts, names


def build_unit_design(spikes_by_trial: Dict[int, np.ndarray],
                      trials_idx: pd.DataFrame, events: pd.DataFrame,
                      bg_basis: Tuple[float, int] = BG_BASIS_DEFAULT,
                      wait_basis: Tuple[float, int] = WAIT_BASIS_DEFAULT):
    """Concatenate per-trial designs for one unit across all valid trials.

    `bg_basis`/`wait_basis` are the session's per-group time-basis grids.
    Returns (X_all, y_all, names, trial_id_per_row, used, skipped).
    """
    X_blocks, y_blocks, tid_blocks = [], [], []
    names: Optional[List[str]] = None
    used: List[int] = []
    skipped: List[Tuple[int, str]] = []

    for tid, sp in spikes_by_trial.items():
        if tid not in trials_idx.index:
            skipped.append((tid, "not in trials_idx"))
            continue
        tr = trials_idx.loc[tid]
        try:
            X, y, nm = build_trial_Xy(tr, events, sp,
                                      bg_basis=bg_basis, wait_basis=wait_basis)
        except Exception as e:
            skipped.append((tid, str(e)))
            continue
        if names is None:
            names = nm
        elif nm != names:
            skipped.append((tid, "design names mismatch"))
            continue
        X_blocks.append(X)
        y_blocks.append(y)
        tid_blocks.append(np.full(X.shape[0], tid, dtype=int))
        used.append(tid)

    if not X_blocks:
        return None, None, names, None, used, skipped

    X_all = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks).astype(float)
    trial_id_per_row = np.concatenate(tid_blocks)
    return X_all, y_all, names, trial_id_per_row, used, skipped


# =====================================================================
# Unit inclusion (no history-based criteria)
# =====================================================================
def check_unit_inclusion(spikes_by_trial: Dict[int, np.ndarray],
                         trials_idx: pd.DataFrame) -> Tuple[bool, str]:
    """Min spikes / max FR / min valid (non-miss) trials. Returns (ok, reason)."""
    valid_tids = [tid for tid in spikes_by_trial.keys() if tid in trials_idx.index]
    if len(valid_tids) < MIN_TRIALS:
        return False, f"only {len(valid_tids)} valid trials (need {MIN_TRIALS})"

    total_spikes = sum(len(spikes_by_trial[tid]) for tid in valid_tids)
    if total_spikes < MIN_SPIKE_COUNT:
        return False, f"only {total_spikes} spikes (need {MIN_SPIKE_COUNT})"

    total_dur = 0.0
    for tid in valid_tids:
        tr = trials_idx.loc[tid]
        total_dur += (float(tr.get("decision_time", 0.0)) +
                      float(tr.get("consumption_length", 3.0)))
    if total_dur > 0:
        mean_fr = total_spikes / total_dur
        if mean_fr > MAX_MEAN_FR_HZ:
            return False, f"mean FR {mean_fr:.1f} Hz > {MAX_MEAN_FR_HZ}"
    return True, "ok"


# =====================================================================
# Family index map (lumped families + sub-families)
# =====================================================================
def build_family_map(names: List[str]) -> Dict[str, np.ndarray]:
    """Column indices for every EVAL_FAMILIES entry present in `names`.

    Lumped families come from 4c.family_indices; sub-families are the matching
    single columns (lick_bg / lick_decision / lick_cons / cue_on / cue_off).
    """
    fam = family_indices(names)
    cols = np.array(names)
    out: Dict[str, np.ndarray] = {}
    for f in SIMPLE_FAMILIES:
        idx = fam.get(f)
        out[f] = idx if idx is not None else np.array([], dtype=int)
    for sf in SUBFAMILIES:
        out[sf] = np.where(cols == sf)[0]
    return out


# =====================================================================
# Single cross-validation pass: full pR² + per-family marginal & unique ΔpR²
# =====================================================================
def cv_all_families(X_all: np.ndarray, y_all: np.ndarray,
                    fam_map: Dict[str, np.ndarray],
                    trial_id_per_row: Optional[np.ndarray]):
    """One CV pass over all families (lumped + sub). Returns:
        full_cv      : mean full-model CV pR²
        marg_mean    : {family: mean marginal CV ΔpR² (family vs null)}
        uniq_mean    : {family: mean unique CV ΔpR² (full − family)}
        fold_records : list of {family, kind, fold, dpr2} per-fold scores
                       (kind ∈ {full, marg, unique}; family '__full__' for full)

    Marginal and unique share the SAME folds (get_cv_splits / CV_SEED) so they're
    directly comparable. Unique is skipped when COMPUTE_UNIQUE_DPR2 is False.
    """
    splits = get_cv_splits(X_all, y_all, trial_id_per_row=trial_id_per_row)
    all_cols = np.arange(X_all.shape[1])
    full_scores: List[float] = []
    marg: Dict[str, List[float]] = {f: [] for f in fam_map}
    uniq: Dict[str, List[float]] = {f: [] for f in fam_map}
    fold_records: List[dict] = []

    for k, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) < 5 or len(test_idx) < 5:
            continue
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        null_mu = float(y_train.mean())
        try:
            _, mu_full = glm_fit_predict_unreg(
                X_all[train_idx], y_train, X_all[test_idx])
            pr2_full = pseudo_R2_mcfadden(y_test, mu_full, null_mu)
        except Exception:
            continue
        if not np.isfinite(pr2_full):
            continue
        full_scores.append(pr2_full)
        fold_records.append({"family": "__full__", "kind": "full",
                             "fold": k, "dpr2": float(pr2_full)})

        for f, idx in fam_map.items():
            if idx.size == 0:
                continue
            # marginal: family-only vs null
            try:
                _, mu_m = glm_fit_predict_unreg(
                    X_all[train_idx][:, idx], y_train, X_all[test_idx][:, idx])
                pr2_m = pseudo_R2_mcfadden(y_test, mu_m, null_mu)
                if np.isfinite(pr2_m):
                    marg[f].append(pr2_m)
                    fold_records.append({"family": f, "kind": "marg",
                                        "fold": k, "dpr2": float(pr2_m)})
            except Exception:
                pass
            # unique: full minus this family
            if COMPUTE_UNIQUE_DPR2:
                red_cols = np.setdiff1d(all_cols, idx)
                try:
                    if red_cols.size == 0:
                        pr2_r = 0.0  # reduced model is intercept-only
                    else:
                        _, mu_r = glm_fit_predict_unreg(
                            X_all[train_idx][:, red_cols], y_train,
                            X_all[test_idx][:, red_cols])
                        pr2_r = pseudo_R2_mcfadden(y_test, mu_r, null_mu)
                    if np.isfinite(pr2_r):
                        marg_unique = pr2_full - pr2_r
                        uniq[f].append(marg_unique)
                        fold_records.append({"family": f, "kind": "unique",
                                            "fold": k, "dpr2": float(marg_unique)})
                except Exception:
                    pass

    full_cv = float(np.mean(full_scores)) if full_scores else 0.0
    marg_mean = {f: (float(np.mean(v)) if v else np.nan) for f, v in marg.items()}
    uniq_mean = {f: (float(np.mean(v)) if v else np.nan) for f, v in uniq.items()}
    return full_cv, marg_mean, uniq_mean, fold_records


# =====================================================================
# Per-unit evaluation
# =====================================================================
def evaluate_unit(X_all: np.ndarray, y_all: np.ndarray, names: List[str],
                  trial_id_per_row: np.ndarray) -> Tuple[dict, dict, List[dict]]:
    """Fit + score one unit.

    Returns (metrics_row_partial, coef_row_partial, fold_records). fold_records
    carry the per-fold CV ΔpR² (caller stamps session_id/unit_id and writes them
    to CV_FOLDS_CSV).
    """
    fam_map = build_family_map(names)

    # Full-model fit for coefficients + classification.
    # Prefer unregularized IRLS (interpretable, raw-space kernels); fall back to
    # standardized ridge if IRLS diverges.
    Xc = sm.add_constant(X_all, has_constant="add")
    fit_status = "ok"
    try:
        res = sm.GLM(y_all, Xc, family=sm.families.Poisson()).fit(maxiter=200)
        params = np.asarray(res.params, dtype=float)
        mu = res.predict(Xc)
        pr2_insample = pseudo_R2_mcfadden(y_all, mu, y_all.mean())
        if not np.isfinite(pr2_insample):
            raise RuntimeError("non-finite in-sample pR2")
    except Exception:
        fit_status = "regularized"
        Xs, _, _ = standardize_columns(X_all)
        Xsc = sm.add_constant(Xs, has_constant="add")
        alpha_vec = np.full(Xsc.shape[1], 0.1)
        alpha_vec[0] = 0.0
        res = sm.GLM(y_all, Xsc, family=sm.families.Poisson()).fit_regularized(
            alpha=alpha_vec, L1_wt=0.0, maxiter=1000)
        params = np.asarray(res.params, dtype=float)
        mu = res.predict(Xsc)
        pr2_insample = pseudo_R2_mcfadden(y_all, mu, y_all.mean())

    # In-sample per-family ΔpR² (family vs null; base feeds classify_unit).
    delta_insample = {
        f: float(fit_and_pr2_unreg(X_all[:, idx], y_all))
        for f, idx in fam_map.items() if idx.size > 0
    }

    # Single CV pass: full pR² + marginal & unique per-family ΔpR² + per-fold.
    full_cv, delta_marg, delta_unique, fold_records = cv_all_families(
        X_all, y_all, fam_map, trial_id_per_row)

    classify_in = {f"delta_pr2_{f}": delta_insample.get(f, 0.0)
                   for f in SIMPLE_FAMILIES}
    label = classify_unit(classify_in, names, params)

    # Dominant TASK family = argmax of marginal CV ΔpR² over TASK_FAMILIES
    # (excludes spike_history). Units whose best task family is below
    # DOMINANT_MIN_DPR2 are 'untuned/weak' rather than noise-labeled.
    task_marg = {f: delta_marg[f] for f in TASK_FAMILIES
                 if f in delta_marg and np.isfinite(delta_marg[f])}
    if task_marg:
        dom_family = max(task_marg, key=task_marg.get)
        dom_value = task_marg[dom_family]
        if dom_value < DOMINANT_MIN_DPR2:
            dom_family = "untuned/weak"
    else:
        dom_family, dom_value = "untuned/weak", np.nan

    # Efficiency (basis-count-normalized) dominant family — same tuned gate,
    # argmax of ΔpR²-per-column so multi-basis families don't win by df alone.
    if dom_family == "untuned/weak":
        dom_family_eff = "untuned/weak"
    else:
        ncol = task_family_ncol(fam_map)
        eff = {f: delta_marg[f] / ncol[f]
               for f in TASK_FAMILIES
               if f in delta_marg and np.isfinite(delta_marg[f])
               and ncol.get(f, 0) > 0}
        dom_family_eff = max(eff, key=eff.get) if eff else "untuned/weak"

    metrics = {
        "n_timepoints": int(y_all.shape[0]),
        "mean_rate_hz": float(y_all.sum() / (y_all.shape[0] * DT)),
        "pseudoR2_insample": float(pr2_insample),
        "pseudoR2_cv": float(full_cv),
        "label": label,
        "dominant_family": dom_family,
        "dominant_family_eff": dom_family_eff,
        "dominant_dpr2_marg": float(dom_value) if np.isfinite(dom_value) else np.nan,
        "fit_status": fit_status,
    }
    for f in EVAL_FAMILIES:
        metrics[f"delta_pr2_marg_{f}"] = float(delta_marg.get(f, np.nan))
        metrics[f"delta_pr2_unique_{f}"] = float(delta_unique.get(f, np.nan))
        metrics[f"delta_pr2_insample_{f}"] = float(delta_insample.get(f, np.nan))

    coef_row = {f"beta_{n}": float(b)
                for n, b in zip(["const"] + list(names), params)}
    coef_row["fit_status"] = fit_status
    return metrics, coef_row, fold_records


# =====================================================================
# Main run loop
# =====================================================================
def run(session_ids: Optional[List[str]] = None):
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
            print("[abort] no sessions to run after filtering.")
            return

    print(f"\nWill run {len(session_ids)} session(s). "
          f"unique ΔpR²={'on' if COMPUTE_UNIQUE_DPR2 else 'off'}")

    anatomy = load_anatomical_labels()
    if not anatomy.empty:
        anatomy_idx = anatomy.set_index(["session_id", "id"], drop=False)
        print(f"[anat] loaded {len(anatomy)} unit labels")
    else:
        anatomy_idx = None
        print("[anat] no labels; region columns will be probe-target only")

    bg_group_lookup = load_bg_group_lookup()
    print(f"[bg_group] loaded {len(bg_group_lookup)} session groups "
          f"(unknown groups use default grid {BG_BASIS_DEFAULT}/{WAIT_BASIS_DEFAULT})")

    # Resume vs clean start. On resume we load prior per_unit/coef/excluded rows
    # back into memory (so the full-rewrite saves stay correct) and build a
    # `seen` set of (session_id, unit_id) — both fit and previously-excluded —
    # to skip. cv_folds is append-only (too big to reload), so skipped units
    # never re-append their fold rows.
    seen = set()
    all_unit_rows, all_coef_rows = [], []
    all_excluded_rows, all_fold_rows = [], []
    resuming = RESUME and PER_UNIT_CSV.exists()
    if resuming:
        prev = pd.read_csv(PER_UNIT_CSV)
        all_unit_rows = prev.to_dict("records")
        seen |= set(zip(prev["session_id"].astype(str), prev["unit_id"].astype(str)))
        if COEF_CSV.exists():
            all_coef_rows = pd.read_csv(COEF_CSV).to_dict("records")
        if EXCLUDED_CSV.exists():
            exc_prev = pd.read_csv(EXCLUDED_CSV)
            all_excluded_rows = exc_prev.to_dict("records")
            if len(exc_prev):
                seen |= set(zip(exc_prev["session_id"].astype(str),
                                exc_prev["unit_id"].astype(str)))
        print(f"[resume] {len(seen)} units already processed; will skip them.")
    else:
        for f in (PER_UNIT_CSV, COEF_CSV, EXCLUDED_CSV, CV_FOLDS_CSV):
            if f.exists():
                f.unlink()

    # Units to process this run = all selected units not already in `seen`.
    n_target = sum(
        (str(sid), str(uid)) not in seen
        for sid in session_ids
        for uid in units_vetted.loc[units_vetted["session_id"] == sid, "unit_id"]
    )
    t_start = time.time()
    prog = {"n": 0, "fit": 0, "excl": 0}
    print(f"[progress] {n_target} units to process across {len(session_ids)} "
          f"session(s); progress every {PROGRESS_EVERY}.", flush=True)

    def _progress():
        """Print a running progress line every PROGRESS_EVERY units."""
        if prog["n"] % PROGRESS_EVERY and prog["n"] != n_target:
            return
        elapsed = time.time() - t_start
        rate = prog["n"] / elapsed * 60 if elapsed > 0 else 0.0   # units/min
        eta = (n_target - prog["n"]) / rate if rate > 0 else float("nan")
        print(f"[progress] {prog['n']}/{n_target} units "
              f"({100 * prog['n'] / max(n_target, 1):.0f}%) | "
              f"fit={prog['fit']} excl={prog['excl']} | {rate:.1f} u/min | "
              f"elapsed {elapsed / 60:.1f}m | ETA {eta:.0f}m", flush=True)

    def _record_excluded(sid, unit_id, unit_key, reason):
        all_excluded_rows.append({
            "session_id": sid, "unit_id": unit_id,
            "unit_key": unit_key, "reason": reason,
        })
        prog["n"] += 1
        prog["excl"] += 1
        _progress()

    def _flush():
        # per_unit / coef / excluded: full rewrite (small, ragged coef columns).
        pd.DataFrame(all_unit_rows).to_csv(PER_UNIT_CSV, index=False)
        pd.DataFrame(all_coef_rows).to_csv(COEF_CSV, index=False)
        pd.DataFrame(all_excluded_rows,
                     columns=["session_id", "unit_id", "unit_key", "reason"]
                     ).to_csv(EXCLUDED_CSV, index=False)
        # cv_folds: append-only (can be ~340k rows at full scale).
        if SAVE_CV_FOLDS and all_fold_rows:
            header = (not CV_FOLDS_CSV.exists()) or CV_FOLDS_CSV.stat().st_size == 0
            pd.DataFrame(all_fold_rows).to_csv(
                CV_FOLDS_CSV, mode="a", header=header, index=False)
            all_fold_rows.clear()

    for sid in session_ids:
        print(f"\n=== Session {sid} ===")
        try:
            events, trials, units = utils.get_session_data(sid)
        except Exception as e:
            print(f"[skip session] {sid}: {e}")
            continue

        if "missed" in trials.columns:
            try:
                trials = trials.loc[~coerce_bool_series(trials["missed"])].copy()
            except Exception as e:
                print(f"[warn] could not filter missed trials: {e}")
        trials_idx = trials.set_index("trial_id", drop=False)

        session_units = units_vetted[units_vetted["session_id"] == sid]
        probe_region = (
            session_units["region"].iloc[0]
            if "region" in session_units.columns and len(session_units) > 0
            else ""
        )

        bg_group = bg_group_lookup.get(sid)
        bg_basis, wait_basis = basis_params_for_group(bg_group)
        print(f"[bg_group] {sid}: {bg_group or 'UNKNOWN'} → "
              f"BG grid {bg_basis}, WAIT grid {wait_basis}")

        n_units = len(session_units)
        n_ok = 0
        for ui, (_, unit_info) in enumerate(session_units.iterrows(), start=1):
            unit_id = unit_info["unit_id"]
            unit_key = unit_info["id"]
            if (str(sid), str(unit_id)) in seen:
                continue
            try:
                spikes_df = units[unit_key]
            except KeyError:
                _record_excluded(sid, unit_id, unit_key, "not in session spikes dict")
                continue
            spikes_by_trial = spikes_df_to_trial_map(spikes_df)

            ok, reason = check_unit_inclusion(spikes_by_trial, trials_idx)
            if not ok:
                _record_excluded(sid, unit_id, unit_key, reason)
                continue

            X_raw, y_all, names_raw, tid_row, used, _ = build_unit_design(
                spikes_by_trial, trials_idx, events,
                bg_basis=bg_basis, wait_basis=wait_basis)
            if X_raw is None or names_raw is None:
                _record_excluded(sid, unit_id, unit_key, "no valid design")
                continue
            X_all, names, _ = drop_zero_variance_columns(X_raw, names_raw)

            try:
                metrics, coef_row, fold_records = evaluate_unit(
                    X_all, y_all, names, tid_row)
            except Exception as e:
                _record_excluded(sid, unit_id, unit_key, f"eval failed: {e}")
                continue

            # Anatomy lookup
            corrected_region = region_group = cell_type = ""
            if anatomy_idx is not None:
                try:
                    arow = anatomy_idx.loc[(sid, unit_key)]
                    if isinstance(arow, pd.DataFrame):
                        arow = arow.iloc[0]
                    corrected_region = str(arow.get("corrected_region", ""))
                    region_group = str(arow.get("region_group", ""))
                    cell_type = str(arow.get("cell_type", ""))
                except KeyError:
                    pass

            base = {
                "session_id": sid,
                "unit_id": unit_id,
                "unit_key": unit_key,
                "probe_region": probe_region,
                "corrected_region": corrected_region,
                "region_group": region_group,
                "cell_type": cell_type,
                "bg_group": bg_group or "",
                "n_basis_wait": int(wait_basis[1]),
                "n_trials_used": int(len(used)),
            }
            all_unit_rows.append({**base, **metrics})
            all_coef_rows.append({"session_id": sid, "unit_id": unit_id,
                                  "region_group": region_group, **coef_row})
            if SAVE_CV_FOLDS:
                for rec in fold_records:
                    all_fold_rows.append({"session_id": sid, "unit_id": unit_id,
                                          **rec})
            n_ok += 1
            prog["n"] += 1
            prog["fit"] += 1
            _progress()

            if INCREMENTAL_SAVE and (ui % INCREMENTAL_EVERY == 0 or ui == n_units):
                _flush()

        print(f"[session done] {sid}: {n_ok}/{n_units} units fit, "
              f"{len(all_excluded_rows)} excluded so far")

    _flush()
    print(f"\nSaved → {PER_UNIT_CSV}  ({len(all_unit_rows)} rows)")
    print(f"Saved → {COEF_CSV}  ({len(all_coef_rows)} rows)")
    print(f"Saved → {EXCLUDED_CSV}  ({len(all_excluded_rows)} excluded)")
    if SAVE_CV_FOLDS:
        print(f"Saved → {CV_FOLDS_CSV} (append-only; per-fold CV records)")


# =====================================================================
# Fair (basis-count-normalized) dominant family
# =====================================================================
# Raw marginal ΔpR² over-credits flexible multi-column families: WAIT_time has
# 8-11 basis columns and BG_time 4-9, vs licks=3 / outcome=1, so they win the
# per-unit argmax disproportionately. The "efficiency" dominant divides each
# family's ΔpR² by its column count (degrees of freedom) before the argmax, so
# the label reflects ΔpR²-per-parameter rather than raw flexibility. Both views
# share the same 'tuned' gate (best RAW task ΔpR² >= DOMINANT_MIN_DPR2) so the
# untuned/weak set is identical between them.
FAMILY_NCOL_FIXED = {
    "licks": 3, "outcome": 1, "cue": 2,
    "lick_bg": 1, "lick_decision": 1, "lick_cons": 1, "cue_on": 1, "cue_off": 1,
}


def task_family_ncol(fam_map: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Actual (post zero-variance drop) column count per TASK family."""
    return {f: int(fam_map[f].size) for f in TASK_FAMILIES if f in fam_map}


def compute_dominant_views(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return (dom_raw, dom_eff) Series for a per_unit DataFrame.

    dom_raw uses raw marginal ΔpR²; dom_eff uses ΔpR²-per-column (nominal
    counts from bg_group / n_basis_wait). Same tuned gate for both.
    """
    marg = df[[f"delta_pr2_marg_{f}" for f in TASK_FAMILIES]].copy()
    marg.columns = TASK_FAMILIES
    tuned = marg.max(axis=1, skipna=True) >= DOMINANT_MIN_DPR2

    ncol = pd.DataFrame(index=df.index)
    bg_series = df.get("bg_group", pd.Series("", index=df.index)).fillna("")
    wait_series = df.get("n_basis_wait",
                         pd.Series(WAIT_BASIS_DEFAULT[1], index=df.index)
                         ).fillna(WAIT_BASIS_DEFAULT[1])
    for f in TASK_FAMILIES:
        if f == "WAIT_time":
            ncol[f] = wait_series.astype(float)
        elif f == "BG_time":
            ncol[f] = bg_series.map(
                lambda g: BG_BASIS_BY_GROUP.get(g, BG_BASIS_DEFAULT)[1])
        else:
            ncol[f] = FAMILY_NCOL_FIXED.get(f, 1)
    eff = marg / ncol

    dom_raw = marg.fillna(-np.inf).idxmax(axis=1).where(tuned, "untuned/weak")
    dom_eff = eff.fillna(-np.inf).idxmax(axis=1).where(tuned, "untuned/weak")
    return dom_raw, dom_eff


# =====================================================================
# Population summary (by region_group or cell_type)
# =====================================================================
def summarize(group_col: str = "region_group",
              out_csv: Optional[Path] = None,
              per_unit_csv: Path = PER_UNIT_CSV,
              exclude_groups: Tuple[str, ...] = EXCLUDE_REGION_GROUPS,
              dpr2_kind: str = "marg") -> pd.DataFrame:
    """Aggregate per-unit family ΔpR² by group_col.

    Reports, per group: n, median full pR², per-family median/IQR ΔpR² (of the
    chosen kind), and the fraction of units dominated by each family.
    """
    if not per_unit_csv.exists():
        print(f"[summary warn] {per_unit_csv} not found.")
        return pd.DataFrame()
    df = pd.read_csv(per_unit_csv)
    if group_col not in df.columns or df[group_col].isna().all():
        print(f"[summary warn] {group_col} missing/empty; falling back to probe_region")
        group_col = "probe_region"
    df = df.loc[df[group_col].notna() & (df[group_col].astype(str) != "")]
    if exclude_groups:
        df = df.loc[~df[group_col].isin(exclude_groups)]
    if "fit_status" in df.columns:
        df = df.loc[df["fit_status"] == "ok"].copy()
    if df.empty:
        print("[summary] no usable rows after filters.")
        return pd.DataFrame()

    # Raw and efficiency (basis-normalized) dominant family. Prefer a stored
    # dominant_family_eff column (from a native run); else compute on the fly.
    dom_raw, dom_eff = compute_dominant_views(df)
    df = df.copy()
    df["__dom_raw"] = (df["dominant_family"] if "dominant_family" in df.columns
                       else dom_raw)
    df["__dom_eff"] = (df["dominant_family_eff"]
                       if "dominant_family_eff" in df.columns else dom_eff)

    rows = []
    for grp, sub in df.groupby(group_col):
        row = {
            group_col: grp,
            "n_units": int(len(sub)),
            "median_pr2_cv": float(np.nanmedian(sub["pseudoR2_cv"])),
            "median_pr2_insample": float(np.nanmedian(sub["pseudoR2_insample"])),
        }
        # Median/IQR over ALL families (spike_history included — informative).
        for f in SIMPLE_FAMILIES:
            vals = sub[f"delta_pr2_{dpr2_kind}_{f}"].dropna().values
            row[f"med_{f}"] = float(np.median(vals)) if vals.size else np.nan
            row[f"iqr_{f}"] = float(np.percentile(vals, 75) -
                                    np.percentile(vals, 25)) if vals.size else np.nan
        # Dominant-family fractions: raw AND efficiency-normalized, + untuned.
        for f in TASK_FAMILIES:
            row[f"fracdom_raw_{f}"] = float((sub["__dom_raw"] == f).mean())
            row[f"fracdom_eff_{f}"] = float((sub["__dom_eff"] == f).mean())
        row["fracdom_untuned"] = float((sub["__dom_eff"] == "untuned/weak").mean())
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("n_units", ascending=False)
    if out_csv is not None:
        summary.to_csv(out_csv, index=False)
        print(f"\nSaved → {out_csv}")

    # Population fair view: rank TASK families by median ΔpR² per region.
    print(f"\n[{group_col}] TASK families ranked by median {dpr2_kind} ΔpR² "
          f"(the fair cross-family view):")
    for _, r in summary.iterrows():
        ranked = sorted(TASK_FAMILIES, key=lambda f: -(r[f"med_{f}"]
                        if np.isfinite(r[f"med_{f}"]) else -np.inf))
        order = "  >  ".join(f"{f} {r[f'med_{f}']:.4f}" for f in ranked[:4])
        print(f"  {str(r[group_col]):14s} n={int(r['n_units']):4d} | {order}")

    # Efficiency-normalized dominant fractions (replaces the raw headline).
    show_cols = [group_col, "n_units", "median_pr2_cv"] + \
                [f"fracdom_eff_{f}" for f in TASK_FAMILIES] + ["fracdom_untuned"]
    print(f"\n[{group_col}] dominant-family fractions — EFFICIENCY "
          f"(ΔpR²-per-column) normalized:")
    print(summary[show_cols].to_string(index=False))
    return summary


# =====================================================================
# Plot: per-region family ΔpR² distributions
# =====================================================================
def plot_family_dpr2(per_unit_csv: Path = PER_UNIT_CSV,
                     group_col: str = "region_group",
                     exclude_groups: Tuple[str, ...] = EXCLUDE_REGION_GROUPS,
                     dpr2_kind: str = "marg"):
    """Boxplots of per-family ΔpR² for each region group."""
    if not per_unit_csv.exists():
        print(f"[plot warn] {per_unit_csv} not found.")
        return
    df = pd.read_csv(per_unit_csv)
    if group_col not in df.columns or df[group_col].isna().all():
        group_col = "probe_region"
    df = df.loc[df[group_col].notna() & (df[group_col].astype(str) != "")]
    if exclude_groups:
        df = df.loc[~df[group_col].isin(exclude_groups)]
    if "fit_status" in df.columns:
        df = df.loc[df["fit_status"] == "ok"].copy()
    if df.empty:
        print("[plot] no usable rows.")
        return

    regions = sorted(df[group_col].unique())
    n = len(regions)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, reg in zip(axes, regions):
        sub = df.loc[df[group_col] == reg]
        data = [sub[f"delta_pr2_{dpr2_kind}_{f}"].dropna().values
                for f in SIMPLE_FAMILIES]
        ax.boxplot(data, tick_labels=SIMPLE_FAMILIES, showfliers=False)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_title(f"{reg}\nn={len(sub)}", fontsize=10)
        ax.tick_params(axis="x", rotation=90)
    axes[0].set_ylabel(f"ΔpR² ({dpr2_kind})")
    fig.suptitle("Per-family CV ΔpR² by region", fontsize=11)
    fig.tight_layout()
    fig.savefig(DPR2_PLOT_PATH, dpi=150)
    print(f"Saved → {DPR2_PLOT_PATH}")
    plt.close(fig)


# =====================================================================
# Single-unit smoke test
# =====================================================================
def debug_one(session_id: str, unit_id: int):
    """Run the full pipeline on one (session, id) and print verbose output."""
    units_vetted = pd.read_csv(
        os.path.join(p.LOGS_DIR, "units_vetted.csv"), index_col=0)
    sub = units_vetted[(units_vetted["session_id"] == session_id) &
                       (units_vetted["id"] == unit_id)]
    if sub.empty:
        print(f"[abort] unit not found: {session_id} / id={unit_id}")
        return
    unit_info = sub.iloc[0]
    unit_key = unit_info["id"]
    print(f"[debug] {session_id} / id={unit_key} / unit_id={unit_info['unit_id']}")

    events, trials, units = utils.get_session_data(session_id)
    if "missed" in trials.columns:
        trials = trials.loc[~coerce_bool_series(trials["missed"])].copy()
    trials_idx = trials.set_index("trial_id", drop=False)

    bg_group = load_bg_group_lookup().get(session_id)
    bg_basis, wait_basis = basis_params_for_group(bg_group)
    print(f"[debug] bg_group={bg_group or 'UNKNOWN'} → BG grid {bg_basis}, "
          f"WAIT grid {wait_basis}")

    spikes_by_trial = spikes_df_to_trial_map(units[unit_key])
    ok, reason = check_unit_inclusion(spikes_by_trial, trials_idx)
    print(f"[debug] inclusion: {ok} ({reason})")
    if not ok:
        return

    X_raw, y_all, names_raw, tid_row, used, skipped = build_unit_design(
        spikes_by_trial, trials_idx, events,
        bg_basis=bg_basis, wait_basis=wait_basis)
    if X_raw is None:
        print("[debug] no design built.")
        return
    X_all, names, kept_mask = drop_zero_variance_columns(X_raw, names_raw)
    dropped = [n for n, k in zip(names_raw, kept_mask) if not k]
    print(f"[debug] trials_used={len(used)}  X={X_all.shape}  "
          f"y={y_all.shape}  dropped_zero_var={dropped}")

    try:
        rank = np.linalg.matrix_rank(X_all)
        print(f"[diag] rank(X)={rank}/{X_all.shape[1]}  cond={np.linalg.cond(X_all):.2e}")
    except Exception as e:
        print(f"[diag] rank/cond failed: {e}")

    metrics, coef_row, fold_records = evaluate_unit(X_all, y_all, names, tid_row)
    print(f"\n[debug] full pR²: in-sample={metrics['pseudoR2_insample']:.3f}  "
          f"CV={metrics['pseudoR2_cv']:.3f}  mean_rate={metrics['mean_rate_hz']:.2f} Hz")
    print(f"[debug] label={metrics['label']}  "
          f"dominant_family={metrics['dominant_family']} "
          f"({metrics['dominant_dpr2_marg']:.4f})  fit_status={metrics['fit_status']}")
    print(f"\n[debug] per-family CV ΔpR² (lumped families, then sub-families):")
    print(f"   {'family':>15s} {'marginal':>10s} {'unique':>10s} {'insample':>10s}")
    for f in EVAL_FAMILIES:
        sep = "  --" if f == SUBFAMILIES[0] else ""
        print(f"   {f:>15s} {metrics[f'delta_pr2_marg_{f}']:>10.4f} "
              f"{metrics[f'delta_pr2_unique_{f}']:>10.4f} "
              f"{metrics[f'delta_pr2_insample_{f}']:>10.4f}{sep}")
    print(f"[debug] {len(fold_records)} per-fold CV records "
          f"(would be saved to {CV_FOLDS_CSV.name})")


# =====================================================================
# CLI
# =====================================================================
def _summary_and_plot():
    summarize(group_col="region_group", out_csv=REGION_SUMMARY_CSV)
    summarize(group_col="cell_type", out_csv=CELLTYPE_SUMMARY_CSV)
    plot_family_dpr2()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  python {sys.argv[0]} run [session_id ...]")
        print(f"  python {sys.argv[0]} summary")
        print(f"  python {sys.argv[0]} plot")
        print(f"  python {sys.argv[0]} debug <session_id> <unit_id>")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "run":
        run(session_ids=sys.argv[2:] if len(sys.argv) > 2 else None)
        _summary_and_plot()
    elif mode == "summary":
        summarize(group_col="region_group", out_csv=REGION_SUMMARY_CSV)
        summarize(group_col="cell_type", out_csv=CELLTYPE_SUMMARY_CSV)
    elif mode == "plot":
        plot_family_dpr2()
    elif mode == "debug":
        if len(sys.argv) < 4:
            print(f"Usage: python {sys.argv[0]} debug <session_id> <unit_id>")
            sys.exit(1)
        debug_one(sys.argv[2], int(sys.argv[3]))
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
