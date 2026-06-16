#!/usr/bin/env python3
"""
4c_diversity_validation.py

Sketch: run the debug pipeline (rank/cond, family ΔpR², nested LRTs,
example-trial plot) on a small panel of canonical units spanning regions
and firing-rate regimes. Print a side-by-side comparison.

Predictions before looking at the numbers:
  - VAL strong rescaler:    M1_vs_M2 highly significant; large WAIT_time pR²
  - MOp/MOs strong rescaler: M1_vs_M2 significant; WAIT_time + outcome pR² both nonzero
  - V1 (predicted negative): M1_vs_M2 should be non-significant or weak;
                              sensory cortex shouldn't carry reward-history × time
  - Low-FR sanity:           fit should converge without numerical blow-up;
                              LRT may be non-significant just from low power

If V1 fires as strongly as VAL on M1_vs_M2, something is wrong with the
shuffle null or the design (e.g., interaction columns aliasing across
trial-history-poor units). Diagnose before scaling.

Run:
    MPLBACKEND=Agg python -u 4c_diversity_validation.py
or override the panel:
    python -u 4c_diversity_validation.py --panel custom_panel.csv
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import paths as p
import utils

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "glm_hist", _HERE / "4c_encoding_GLM_w_history.py"
)
glm_hist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glm_hist)

OUT_DIR = Path(p.DATA_DIR) / "glm_encoding_w_history" / "diversity_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical panel — keep this short. Each row picks a unit chosen for an
# explicit prediction; the comparison only earns trust if the predictions
# line up. Replace these with your favorites before running.
CANONICAL_PANEL = [
    # session_id,                unit_id, label,                            prediction
    ("RZ063_2025-03-05_str",     180,    "VAL strong rescaler",            "M1_vs_M2 significant"),
    ("RZ047_2024-11-21_str",     104,    "MOp/MOs strong rescaler (9 Hz)","M1_vs_M2 significant"),
    ("RZ063_2025-03-06_v1",      128,    "V1 (predicted negative)",        "M1_vs_M2 non-sig"),
    ("RZ034_2024-07-13_str",     7,      "Low-FR sanity (~3.5 Hz str)",    "fits cleanly, weak signal"),
]


def run_one_unit(session_id: str, unit_id: int, label: str, prediction: str):
    """Run the debug pipeline on a single unit; return summary dict + trial plot path."""
    print(f"\n=== {label}: {session_id} unit {unit_id} ===")
    print(f"    prediction: {prediction}")
    out = {
        "session_id": session_id,
        "unit_id": unit_id,
        "label": label,
        "prediction": prediction,
        "ok": False,
        "reason": "",
    }
    try:
        events, trials, units = utils.get_session_data(session_id)
    except Exception as e:
        out["reason"] = f"session load failed: {e}"
        print(f"  [skip] {out['reason']}")
        return out, None
    if unit_id not in units:
        out["reason"] = f"unit {unit_id} not in session"
        print(f"  [skip] {out['reason']}")
        return out, None

    spikes_df = units[unit_id]
    hist_idx = glm_hist.build_trial_history_features(trials)
    if "missed" in trials.columns:
        trials = trials.loc[~glm_hist.coerce_bool_series(trials["missed"])].copy()
    trials_idx = trials.set_index("trial_id", drop=False)
    spikes_by_trial = glm_hist.spikes_df_to_trial_map(spikes_df)

    ok, reason = glm_hist.check_unit_inclusion(spikes_by_trial, trials_idx, hist_idx)
    out["inclusion_ok"] = bool(ok)
    out["inclusion_reason"] = reason
    print(f"  inclusion: {ok} ({reason})")
    if not ok:
        out["reason"] = f"failed inclusion: {reason}"
        return out, None

    X_raw, y_all, names_raw, trial_id_per_row, used, skipped = (
        glm_hist.build_session_design_for_unit(
            spikes_by_trial, trials_idx, hist_idx, events
        )
    )
    if X_raw is None:
        out["reason"] = "no design"
        return out, None

    X_all, names, kept_mask = glm_hist.drop_zero_variance_columns(X_raw, names_raw)
    dropped = [n for n, k in zip(names_raw, kept_mask) if not k]
    print(f"  trials_used={len(used)}  X={X_all.shape}  dropped_zero_var={dropped}")

    rank = int(np.linalg.matrix_rank(X_all))
    cond = float(np.linalg.cond(X_all))
    Xs, _, _ = glm_hist.standardize_columns(X_all)
    sv = np.linalg.svd(Xs, compute_uv=False)
    sigma_ratio = float(sv.min() / sv.max()) if sv.max() > 0 else np.nan
    print(f"  rank={rank}/{X_all.shape[1]}  cond={cond:.3e}  min(σ)/max(σ)={sigma_ratio:.3e}")

    # Full-model unregularized fit (in-sample pR² + mean rate)
    Xc = sm.add_constant(X_all, has_constant="add")
    try:
        res = sm.GLM(y_all, Xc, family=sm.families.Poisson()).fit(maxiter=200)
        mu = res.predict(Xc)
        pr2 = glm_hist.pseudo_R2_mcfadden(y_all, mu, y_all.mean())
    except Exception as e:
        out["reason"] = f"GLM fit failed: {e}"
        print(f"  [warn] {out['reason']}")
        return out, None
    mean_rate = float(y_all.sum() / (len(y_all) * glm_hist.DT))
    print(f"  pR²={pr2:.3f}  mean_rate={mean_rate:.2f} Hz")

    # Per-family ΔpR² + nested LRTs
    fam = glm_hist.family_indices(names)
    delta_pr2 = {
        k: float(glm_hist.fit_and_pr2_unreg(X_all[:, idx], y_all))
        for k, idx in fam.items() if idx.size > 0
    }
    lrt = glm_hist.compute_nested_lrts(X_all, y_all, fam)
    p_m0m1 = lrt.get("lrts", {}).get("M0_vs_M1", {}).get("p_value", np.nan)
    p_m1m2 = lrt.get("lrts", {}).get("M1_vs_M2", {}).get("p_value", np.nan)
    print(f"  M0_vs_M1 p={p_m0m1:.3g}    M1_vs_M2 p={p_m1m2:.3g}")

    # Trial plot
    plot_path = None
    try:
        tid_plot = next(
            (tid for tid in used if len(spikes_by_trial.get(tid, [])) > 0),
            None,
        )
        if tid_plot is not None:
            plot_path = OUT_DIR / f"trial_plot__{session_id}__u{unit_id}.png"
            glm_hist.plot_one_trial(
                trials_idx.loc[tid_plot], events, spikes_by_trial[tid_plot],
                trial_history_idx=hist_idx,
                title=f"{label}\n{session_id}  unit {unit_id}  trial {tid_plot}",
                save_path=plot_path,
            )
    except Exception as e:
        print(f"  [plot warn] {e}")

    out.update({
        "ok": True,
        "n_trials_used": int(len(used)),
        "n_timepoints": int(y_all.shape[0]),
        "n_cols_raw": int(X_raw.shape[1]),
        "n_cols_kept": int(X_all.shape[1]),
        "dropped_zero_var_cols": ";".join(dropped),
        "rank": rank,
        "cond_num": cond,
        "sigma_ratio_standardized": sigma_ratio,
        "pseudoR2_full": float(pr2),
        "mean_rate_hz": mean_rate,
        "p_M0_vs_M1": float(p_m0m1) if np.isfinite(p_m0m1) else np.nan,
        "p_M1_vs_M2": float(p_m1m2) if np.isfinite(p_m1m2) else np.nan,
        **{f"delta_pr2_{k}": v for k, v in delta_pr2.items()},
    })
    return out, plot_path


def _load_panel(path: Path | None):
    if path is None:
        return CANONICAL_PANEL
    df = pd.read_csv(path)
    return list(df[["session_id", "unit_id", "label", "prediction"]].itertuples(
        index=False, name=None
    ))


def run_diversity_validation(panel_path: Path | None = None):
    panel = _load_panel(panel_path)
    rows = []
    for session_id, unit_id, label, prediction in panel:
        row, _ = run_one_unit(session_id, int(unit_id), label, prediction)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "diversity_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    # Compact comparison table to stdout
    print("\n=== Summary ===")
    cols = ["label", "n_trials_used", "mean_rate_hz", "pseudoR2_full",
            "p_M0_vs_M1", "p_M1_vs_M2", "rank", "dropped_zero_var_cols"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=None,
                    help="optional CSV with cols session_id,unit_id,label,prediction")
    args = ap.parse_args()
    run_diversity_validation(args.panel)
