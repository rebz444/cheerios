#!/usr/bin/env python3
"""
4c_wait_quartile_stratified.py

Stratified-analysis diagnostic for the M1_vs_M2 headline test.

Splits each session's valid trials into N quartile bins of CURRENT-trial
wait duration, then re-runs the nested-model LRT WITHIN each bin per unit.

Logic:
  - If the M1_vs_M2 signal reflects a real "reward history × wait-period
    time code" effect, it should appear within each quartile bin
    (within a bin, wait durations are similar so the duration-warped basis
    shapes don't differ much; the prev_rewarded × wait_basis interaction
    can't piggyback on duration variation).
  - If the signal is primarily the duration-warping artifact (where
    prev_rewarded → current wait duration → different basis shapes),
    the within-quartile M1_vs_M2 should be much weaker — because we've
    controlled for the duration variation that drives the confound.

Compares fraction-significant and median χ² across quartile bins,
region by region. Reuses compute_nested_lrts on row-subsetted X/y;
no refit of production artifacts needed.

Outputs into glm_encoding_w_history/wait_quartile_stratified/:
  - per_unit_per_quartile.csv  (one row per (unit, quartile))
  - region_quartile_summary.csv  (region_group × quartile)
  - region_quartile_chi2.png    (median χ² per region across quartiles)
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import paths as p
import utils

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "glm_hist", _HERE / "4c_encoding_GLM_w_history.py"
)
glm_hist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glm_hist)

DEFAULT_SESSIONS = ["RZ063_2025-03-05_str", "RZ063_2025-03-05_v1"]
N_QUARTILES = 4
MIN_TRIALS_PER_QUARTILE = 20   # below this, skip the bin (too few trials for LRT)

OUT_DIR = Path(p.DATA_DIR) / "glm_encoding_w_history" / "wait_quartile_stratified"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _chi2_from_p(p_val: float, dof: float) -> float:
    if not (p_val > 0 and np.isfinite(dof) and dof > 0):
        return np.nan
    return float(stats.chi2.isf(p_val, dof))


def _load_anat_lookup() -> dict:
    """(session_id, id) → {corrected_region, region_group, cell_type}."""
    upf = pd.read_csv(os.path.join(p.LOGS_DIR, "unit_properties_final.csv"))
    upf["session_id"] = (
        upf["mouse"].astype(str) + "_"
        + upf["date_only"].astype(str) + "_"
        + upf["probe_region"].astype(str)
    )
    keep = [c for c in ["session_id", "id", "corrected_region",
                        "region_group", "cell_type"] if c in upf.columns]
    upf_sub = upf[keep].drop_duplicates(subset=["session_id", "id"])
    return {
        (r.session_id, int(r.id)): {
            "corrected_region": getattr(r, "corrected_region", None),
            "region_group": getattr(r, "region_group", None),
            "cell_type": getattr(r, "cell_type", None),
        }
        for r in upf_sub.itertuples(index=False)
    }


def stratify_one_unit(
    session_id: str, unit_int_id: int, unit_label: str,
    region_probe: str, anat: dict,
    spikes_df, trials_idx, trial_history_idx, events,
    n_quartiles: int = N_QUARTILES,
) -> list:
    """Return one dict per quartile bin for this unit (or skipped reason)."""
    spikes_by_trial = glm_hist.spikes_df_to_trial_map(spikes_df)
    out_base = {
        "session_id": session_id,
        "unit_id": unit_label,
        "region_probe": region_probe,
        "corrected_region": anat.get("corrected_region"),
        "region_group": anat.get("region_group"),
        "cell_type": anat.get("cell_type"),
    }

    X_raw, y_all, names_raw, trial_id_per_row, used, _ = (
        glm_hist.build_session_design_for_unit(
            spikes_by_trial, trials_idx, trial_history_idx, events
        )
    )
    if X_raw is None:
        return [{**out_base, "q": q, "skipped": "no design"}
                for q in range(n_quartiles)]

    X_all, names, _ = glm_hist.drop_zero_variance_columns(X_raw, names_raw)

    # Current wait duration per used trial
    wait_dur = {
        tid: float(trials_idx.loc[tid, "decision_time"] -
                   trials_idx.loc[tid, "cue_off_time"])
        for tid in used
    }
    durs = np.array([wait_dur[tid] for tid in used])
    edges = np.quantile(durs, np.linspace(0, 1, n_quartiles + 1))

    rows = []
    for q in range(n_quartiles):
        lo, hi = float(edges[q]), float(edges[q + 1])
        if q == n_quartiles - 1:
            q_tids = [tid for tid in used if lo <= wait_dur[tid] <= hi]
        else:
            q_tids = [tid for tid in used if lo <= wait_dur[tid] < hi]

        row = {**out_base, "q": q, "wait_lo": lo, "wait_hi": hi,
               "n_trials": len(q_tids)}
        if len(q_tids) < MIN_TRIALS_PER_QUARTILE:
            row["skipped"] = f"only {len(q_tids)} trials"
            rows.append(row)
            continue

        mask = np.isin(trial_id_per_row, q_tids)
        X_q = X_all[mask, :]
        y_q = y_all[mask]
        # Re-drop zero-var: small bins may make some history columns constant
        X_q2, names_q, _ = glm_hist.drop_zero_variance_columns(X_q, names)
        fam = glm_hist.family_indices(names_q)
        lrt = glm_hist.compute_nested_lrts(X_q2, y_q, fam)
        info = lrt.get("lrts", {}).get("M1_vs_M2", {})
        models = lrt.get("models", {})
        row.update({
            "n_bin_rows": int(mask.sum()),
            "n_cols_used": int(X_q2.shape[1]),
            "p_M1_vs_M2": float(info.get("p_value", np.nan)),
            "df_M1_vs_M2": float(info.get("df", np.nan)),
            "chi2_M1_vs_M2": _chi2_from_p(
                info.get("p_value", np.nan), info.get("df", np.nan)),
            "ll_M0": float(models.get("M0", {}).get("ll", np.nan)),
            "ll_M1": float(models.get("M1", {}).get("ll", np.nan)),
            "ll_M2": float(models.get("M2", {}).get("ll", np.nan)),
        })
        rows.append(row)
    return rows


def run(sessions=None, n_quartiles=N_QUARTILES):
    if sessions is None:
        sessions = DEFAULT_SESSIONS
    units_vetted = pd.read_csv(
        os.path.join(p.LOGS_DIR, "units_vetted.csv"), index_col=0
    ).sort_values("unit_id")
    anat = _load_anat_lookup()

    all_rows = []
    for session_id in sessions:
        print(f"\n=== {session_id} ===")
        session_units = units_vetted[units_vetted.session_id == session_id]
        if len(session_units) == 0:
            print(f"  [skip] no vetted units")
            continue
        try:
            events, trials, units = utils.get_session_data(session_id)
        except Exception as e:
            print(f"  [skip] could not load session: {e}")
            continue
        trial_history_idx = glm_hist.build_trial_history_features(trials)
        if "missed" in trials.columns:
            trials = trials.loc[~glm_hist.coerce_bool_series(trials["missed"])].copy()
        trials_idx = trials.set_index("trial_id", drop=False)

        # Apply the same per-bg_group wait-band filter as the main pipeline
        # (so quartiles match what production sees). Set
        # glm_hist.APPLY_WAIT_BAND_FILTER=False before running to disable.
        bg_group_lookup = glm_hist.load_bg_group_lookup() \
            if glm_hist.APPLY_WAIT_BAND_FILTER else {}
        n_pre = len(trials_idx)
        trials_idx, band = glm_hist.filter_trials_by_wait_band(
            trials_idx, session_id, bg_group_lookup
        )
        if band is not None:
            print(f"  wait-band {bg_group_lookup.get(session_id)} {band}: "
                  f"{len(trials_idx)}/{n_pre} trials retained")

        # Quartile edges per session (using band-filtered wait times)
        wt = (trials_idx["decision_time"] - trials_idx["cue_off_time"]).values
        edges = np.quantile(wt, np.linspace(0, 1, n_quartiles + 1))
        print(f"  wait quartile edges: {np.round(edges, 2)}")

        for i, (_, info) in enumerate(session_units.iterrows(), start=1):
            unit_label = info["unit_id"]
            unit_int = int(info["id"])
            region_probe = info.get("region", "")
            spikes_df = units[unit_int]
            try:
                rows = stratify_one_unit(
                    session_id, unit_int, unit_label, region_probe,
                    anat.get((session_id, unit_int), {}),
                    spikes_df, trials_idx, trial_history_idx, events,
                    n_quartiles=n_quartiles,
                )
                all_rows.extend(rows)
            except Exception as e:
                print(f"  [warn] unit {unit_label}: {e}")
                continue
            if i % 10 == 0 or i == len(session_units):
                print(f"  [{i}/{len(session_units)}] units processed")

    df = pd.DataFrame(all_rows)
    per_unit_path = OUT_DIR / "per_unit_per_quartile.csv"
    df.to_csv(per_unit_path, index=False)
    print(f"\n[save] {per_unit_path}")

    # --- Region × quartile aggregation ---
    ok = df.dropna(subset=["p_M1_vs_M2", "region_group"]).copy()
    ok["neglog10p"] = -np.log10(ok.p_M1_vs_M2.clip(lower=1e-300))

    summary = []
    for (rg, q), sub in ok.groupby(["region_group", "q"]):
        if rg in ("Excluded", "Other", None):
            continue
        if len(sub) == 0: continue
        summary.append({
            "region_group": rg, "q": int(q),
            "n_units": int(len(sub)),
            "wait_lo_med": float(sub.wait_lo.median()),
            "wait_hi_med": float(sub.wait_hi.median()),
            "frac_sig_uncorr": float((sub.p_M1_vs_M2 < 0.05).mean()),
            "frac_sig_p001":   float((sub.p_M1_vs_M2 < 0.001).mean()),
            "median_p":        float(sub.p_M1_vs_M2.median()),
            "median_neglog10p": float(sub.neglog10p.median()),
            "median_chi2":     float(sub.chi2_M1_vs_M2.median()),
            "frac_chi2_gt_50": float((sub.chi2_M1_vs_M2 > 50).mean()),
        })
    sdf = pd.DataFrame(summary).sort_values(["region_group", "q"])
    summary_path = OUT_DIR / "region_quartile_summary.csv"
    sdf.to_csv(summary_path, index=False)
    print(f"[save] {summary_path}")
    print()
    print(sdf.to_string(index=False))

    # --- Plot: median χ² per region across quartiles ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    regions_sorted = sorted(
        sdf.region_group.unique(),
        key=lambda r: -sdf[sdf.region_group == r].median_chi2.median()
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(regions_sorted), 4)))
    ax = axes[0]
    for r, c in zip(regions_sorted, colors):
        sub = sdf[sdf.region_group == r].sort_values("q")
        ax.plot(sub.q, sub.median_chi2, "o-", color=c,
                label=f"{r} (n≈{int(sub.n_units.median())})", lw=2, ms=7)
    ax.axhline(stats.chi2.isf(0.05, 16), color="k", linestyle="--", lw=1, alpha=0.5,
               label="χ²₁₆ p=0.05")
    ax.set_xlabel("wait-duration quartile (0=shortest)")
    ax.set_ylabel("median χ²(M1_vs_M2) within quartile")
    ax.set_xticks(range(N_QUARTILES))
    ax.set_title("Stratified by current-wait quartile")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    ax = axes[1]
    for r, c in zip(regions_sorted, colors):
        sub = sdf[sdf.region_group == r].sort_values("q")
        ax.plot(sub.q, sub.frac_sig_p001, "o-", color=c, lw=2, ms=7,
                label=f"{r}")
    ax.axhline(0.05, color="k", linestyle="--", lw=1, alpha=0.5, label="chance @ p<0.001")
    ax.set_xlabel("wait-duration quartile (0=shortest)")
    ax.set_ylabel("fraction units with p<0.001")
    ax.set_xticks(range(N_QUARTILES))
    ax.set_title("Stratified fraction-significant (p<0.001)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    plot_path = OUT_DIR / "region_quartile_chi2.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {plot_path}")
    return df, sdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="session_ids to process (default: RZ063_2025-03-05 str+v1)")
    ap.add_argument("--n", type=int, default=N_QUARTILES,
                    help="number of wait-duration quartile bins (default 4)")
    args = ap.parse_args()
    run(sessions=args.sessions, n_quartiles=args.n)
