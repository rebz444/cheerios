#!/usr/bin/env python3
"""
4c_shuffle_control.py

Calibration / null-distribution check for the M1_vs_M2 LRT in
4c_encoding_GLM_w_history.

For ONE canonical unit, permute the trial-history feature block N times
(breaking any genuine prev-trial → current-trial relationship while
preserving within-trial spike-time structure, spike-history kernels,
and the marginal distribution of history features) and refit M0/M1/M2.

Outputs into glm_encoding_w_history/shuffle_control/:
  - CSV with per-seed (ll_M0, ll_M1, ll_M2, p_M0_vs_M1, p_M1_vs_M2)
  - Histogram of shuffled M1_vs_M2 p-values + real p as a red line
  - Q-Q plot vs uniform (calibration check)

Reads the strong-signal real p once as a reference; the shuffled p-values
are the null distribution. Under correctly-specified LRT they should look
roughly uniform on [0, 1]; ridge shrinkage typically pulls them toward
conservative (skewed to 1) — that bias is exactly what we want to measure
before relying on nominal p-values across regions.
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

import paths as p
import utils

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "glm_hist", _HERE / "4c_encoding_GLM_w_history.py"
)
glm_hist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glm_hist)

DEFAULT_SESSION = "RZ063_2025-03-05_str"   # VAL strong rescaler validated in debug
DEFAULT_UNIT = 180
DEFAULT_N_SHUFFLES = 200

OUT_DIR = Path(p.DATA_DIR) / "glm_encoding_w_history" / "shuffle_control"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns to permute jointly (preserves the within-history-block covariance,
# e.g. prev_rewarded ↔ prev_wait_z correlation, but breaks the link to
# current-trial spiking).
HISTORY_FEATURE_COLS_BASE = [
    "prev_rewarded", "prev_missed", "prev_wait_z",
    "num_bg_repeats_z", "trial_number_z",
]


def shuffle_history_features(hist_idx: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    out = hist_idx.copy()
    cols = [c for c in HISTORY_FEATURE_COLS_BASE if c in out.columns]
    cols += [c for c in out.columns if c.endswith("back")]
    cols += [c for c in out.columns if c.startswith("ew_reward_rate_tau")]
    cols = sorted(set(cols))
    if not cols:
        return out
    perm = rng.permutation(len(out))
    block = out[cols].to_numpy()
    out.loc[:, cols] = block[perm]
    return out


def run_one(session_id: str, unit_id: int, seed: int, real: bool,
            cached_session=None):
    """Build design (optionally shuffled) and return LRT row dict."""
    if cached_session is None:
        events, trials, units = utils.get_session_data(session_id)
    else:
        events, trials, units = cached_session
    if unit_id not in units:
        raise KeyError(f"unit {unit_id} not in {session_id}")
    spikes_df = units[unit_id]

    hist_idx = glm_hist.build_trial_history_features(trials)
    if not real:
        hist_idx = shuffle_history_features(hist_idx, seed)

    trials_local = trials
    if "missed" in trials_local.columns:
        trials_local = trials_local.loc[
            ~glm_hist.coerce_bool_series(trials_local["missed"])
        ].copy()
    trials_idx = trials_local.set_index("trial_id", drop=False)

    # Apply the same per-bg_group wait-band filter as the main pipeline.
    bg_group_lookup = getattr(run_one, "_bg_group_lookup", None)
    if bg_group_lookup is None:
        bg_group_lookup = glm_hist.load_bg_group_lookup() \
            if glm_hist.APPLY_WAIT_BAND_FILTER else {}
        run_one._bg_group_lookup = bg_group_lookup
    trials_idx, _band = glm_hist.filter_trials_by_wait_band(
        trials_idx, session_id, bg_group_lookup
    )

    spikes_by_trial = glm_hist.spikes_df_to_trial_map(spikes_df)

    X_raw, y_all, names_raw, _, used, _ = glm_hist.build_session_design_for_unit(
        spikes_by_trial, trials_idx, hist_idx, events
    )
    if X_raw is None:
        return None
    X_all, names, _ = glm_hist.drop_zero_variance_columns(X_raw, names_raw)
    fam = glm_hist.family_indices(names)
    lrt = glm_hist.compute_nested_lrts(X_all, y_all, fam)
    models = lrt.get("models", {})
    lrts = lrt.get("lrts", {})
    return {
        "seed": seed,
        "real": bool(real),
        "n_trials_used": int(len(used)),
        "n_cols": int(X_all.shape[1]),
        "ll_M0": models.get("M0", {}).get("ll", np.nan),
        "ll_M1": models.get("M1", {}).get("ll", np.nan),
        "ll_M2": models.get("M2", {}).get("ll", np.nan),
        "p_M0_vs_M1": lrts.get("M0_vs_M1", {}).get("p_value", np.nan),
        "p_M1_vs_M2": lrts.get("M1_vs_M2", {}).get("p_value", np.nan),
    }


def run_shuffle_control(session_id=DEFAULT_SESSION, unit_id=DEFAULT_UNIT,
                        n_shuffles=DEFAULT_N_SHUFFLES, seed_start=0):
    # Cache session data once — every shuffle reuses it
    cached = utils.get_session_data(session_id)

    print(f"[real] {session_id} unit {unit_id}")
    real_row = run_one(session_id, unit_id, seed=-1, real=True, cached_session=cached)
    if real_row is None:
        print("[abort] could not build real design")
        return None, None
    print(f"  real M1_vs_M2 p = {real_row['p_M1_vs_M2']:.3g}  "
          f"M0_vs_M1 p = {real_row['p_M0_vs_M1']:.3g}")

    rows = []
    for i in range(n_shuffles):
        seed = seed_start + i
        try:
            r = run_one(session_id, unit_id, seed=seed, real=False,
                        cached_session=cached)
        except Exception as e:
            print(f"  [seed {seed}] error: {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{n_shuffles}] shuffled M1_vs_M2 p={r['p_M1_vs_M2']:.3g}")

    df = pd.DataFrame(rows)
    real_df = pd.DataFrame([real_row])
    tag = f"{session_id}__u{unit_id}"
    csv_path = OUT_DIR / f"shuffle_lrt__{tag}.csv"
    pd.concat([real_df, df], ignore_index=True).to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    pvals = df["p_M1_vs_M2"].dropna().values
    ax.hist(pvals, bins=20, edgecolor="k", alpha=0.7)
    ax.axvline(real_row["p_M1_vs_M2"], color="r", lw=2,
               label=f"real p={real_row['p_M1_vs_M2']:.1e}")
    n_below = int((pvals < 0.05).sum())
    ax.set_xlabel("M1_vs_M2 p-value (shuffled)")
    ax.set_ylabel("count")
    ax.set_title(f"{tag}  n={len(pvals)}  frac<0.05={n_below/max(len(pvals),1):.3f}")
    ax.legend()

    ax = axes[1]
    if pvals.size:
        pvals_sorted = np.sort(pvals)
        unif = (np.arange(1, len(pvals_sorted) + 1) - 0.5) / len(pvals_sorted)
        ax.plot(-np.log10(unif), -np.log10(pvals_sorted + 1e-300), "o", ms=4)
        lim = float(max(-np.log10(unif).max(),
                        -np.log10(pvals_sorted + 1e-300).max()))
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5)
        ax.set_xlabel(r"$-\log_{10}$(uniform quantile)")
        ax.set_ylabel(r"$-\log_{10}$(shuffled p)")
        ax.set_title("Q-Q vs uniform (calibration)")

    fig.tight_layout()
    plot_path = OUT_DIR / f"shuffle_lrt__{tag}.png"
    fig.savefig(plot_path, dpi=150)
    print(f"[plot] {plot_path}")
    plt.close(fig)
    return df, real_row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--unit", type=int, default=DEFAULT_UNIT)
    ap.add_argument("--n", type=int, default=DEFAULT_N_SHUFFLES,
                    help="number of shuffles")
    ap.add_argument("--seed", type=int, default=0, help="starting seed")
    args = ap.parse_args()
    run_shuffle_control(args.session, args.unit, args.n, args.seed)
