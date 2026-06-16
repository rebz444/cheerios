"""
1_plot_raster_histo.py
======================
Per-unit raster + PSTH plots, forward-aligned from one of three anchors
(cue_on, cue_off, last_lick) to the decision lick. Trials split by either
outcome (rewarded vs unrewarded) or by quantile of the anchor's natural
"time waited" sort column. Missed trials are excluded — there is no
anchor → decision interval without a decision.

Outputs:
  p.RASTER_PLOTS_DIR / {corrected_region} / {cell_type} / {unit_id} /
      {anchor}_{split_mode}.png

  corrected_region and cell_type come from unit_properties_final.csv (output
  of 0h_cell_type_relabeling.py).

Replaces:
  1_plot_raster_histo.ipynb, 1a_plot_raster_histo_by_quantile.ipynb,
  1aa_plot_raster_histo_by_quantile_bgtw.ipynb,
  1b_plot_raster_histo_by_outcome.ipynb
"""

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import constants as k
import paths as p
import utils
from calculate_instantaneous_firing_rate import calculate_firing_rates
from raster_plot import plot_raster

# ── Parameters ────────────────────────────────────────────────────────────────
TIME_STEP = 0.1
SIGMA = 2  # bins; 0.2 s with TIME_STEP=0.1
TRIAL_COUNT_MASK = 5
N_QUANTILES = 4
REGENERATE = False

ANCHORS = [k.TO_CUE_ON, k.TO_CUE_OFF, k.TO_LAST_LICK]
SPLIT_MODES = ["outcome", "quantile"]

# Which "time waited" column orders the raster and defines quantile bins per anchor.
ANCHOR_SORT_COL = {
    k.TO_CUE_ON:    k.TIME_WAITED_SINCE_CUE_ON,    # = decision_time
    k.TO_CUE_OFF:   k.WAIT_LENGTH,                  # = decision_time - cue_off_time
    k.TO_LAST_LICK: k.TIME_WAITED_SINCE_LAST_LICK,  # = decision_time - last_lick_time
}

# Forward-only spike-period filters (anchor → decision). Distinct from
# constants.ANCHORED_PERIODS, which encodes the legacy back-looking model still
# used by other scripts.
FORWARD_ANCHOR_PERIODS = {
    k.TO_CUE_ON:    [k.BACKGROUND, k.WAIT],
    k.TO_CUE_OFF:   [k.WAIT],
    k.TO_LAST_LICK: [k.LICK_TO_CUE, k.WAIT],
}

# ── Colors ────────────────────────────────────────────────────────────────────
viridis = ListedColormap(plt.cm.viridis(np.linspace(0, 1, N_QUANTILES)))
QUANTILE_COLORS = [viridis(i) for i in range(N_QUANTILES)]
OUTCOME_COLORS = {"unrewarded": "tab:red", "rewarded": "tab:blue"}


# ── Trial / spike preparation ────────────────────────────────────────────────
def add_derived_trial_columns(trials):
    """Add anchor-aligned 'time waited' columns. Cheap aliases over existing data,
    so computed inline rather than baked into 0b's pickles."""
    trials = trials.copy()
    trials[k.TIME_WAITED_SINCE_CUE_ON] = trials["decision_time"]
    trials[k.TIME_WAITED_SINCE_LAST_LICK] = trials["decision_time"] - trials["last_lick_time"]
    return trials


def aligned_window(trials, anchor):
    """Forward window: anchor at t=0, decision at t=aligned_end_time."""
    trials = trials.copy()
    trials["aligned_start_time"] = 0.0
    trials["aligned_end_time"] = trials[ANCHOR_SORT_COL[anchor]]
    return trials


def drop_trials_for_anchor(trials, anchor):
    """Drop trials lacking a valid anchor → decision interval."""
    trials = trials.loc[~trials["missed"]].copy()
    if anchor == k.TO_LAST_LICK:
        trials = trials.loc[trials["last_lick_time"].notna()]
    return trials


def filter_spikes_to_forward_periods(spikes, anchor):
    period_col = k.ANCHOR_PERIOD_COL[anchor]
    return spikes.loc[spikes[period_col].isin(FORWARD_ANCHOR_PERIODS[anchor])].copy()


# ── Group builders ────────────────────────────────────────────────────────────
def build_outcome_groups():
    return [
        ("unrewarded", OUTCOME_COLORS["unrewarded"], lambda t: ~t["rewarded"]),
        ("rewarded",   OUTCOME_COLORS["rewarded"],   lambda t: t["rewarded"]),
    ]


def build_quantile_groups(trials, sort_col, n=N_QUANTILES):
    """Adds a 'quantile' column to `trials` in place; returns groups list.
    On qcut failure (e.g. duplicate edges) all trials get NaN quantile and
    each group mask is empty — caller still gets the all-trials baseline."""
    try:
        trials["quantile"] = pd.qcut(
            trials[sort_col], q=n, labels=[f"Q{i+1}" for i in range(n)]
        )
    except ValueError:
        trials["quantile"] = pd.NA
    return [
        (f"Q{i+1}", QUANTILE_COLORS[i],
         lambda t, label=f"Q{i+1}": t["quantile"].eq(label))
        for i in range(n)
    ]


# ── PSTH ──────────────────────────────────────────────────────────────────────
def plot_firing_rates(ax, trials, spikes, anchor, groups,
                      time_step=TIME_STEP, sigma=SIGMA,
                      trial_count_mask=TRIAL_COUNT_MASK, show_legend=True):
    ax.axvline(0, color="tab:gray", linestyle="--", alpha=0.5, label=anchor)

    bc, mf, sf = calculate_firing_rates(
        trials, spikes, anchor, time_step, trial_count_mask, sigma
    )
    if len(bc):
        ax.plot(bc, mf, color="gray", lw=1, alpha=0.6, label="all")
        ax.fill_between(bc, mf - sf, mf + sf, color="gray", alpha=0.1)

    for label, color, mask_fn in groups:
        mask = mask_fn(trials)
        t_grp = trials.loc[mask]
        s_grp = spikes.loc[spikes["trial_id"].isin(t_grp["trial_id"])]
        if t_grp.empty or s_grp.empty:
            continue
        bc, mf, sf = calculate_firing_rates(
            t_grp, s_grp, anchor, time_step, trial_count_mask, sigma
        )
        if len(bc):
            ax.plot(bc, mf, color=color, lw=1, label=str(label))
            ax.fill_between(bc, mf - sf, mf + sf, color=color, alpha=0.15)

    if show_legend:
        ax.legend(bbox_to_anchor=(1, 1.05), loc="upper left")


# ── Combined raster + PSTH figure ─────────────────────────────────────────────
def plot_unit(unit_id, events, trials, spikes, anchor, split_mode,
              corrected_region, cell_type):
    trials_a = drop_trials_for_anchor(trials, anchor)
    trials_a = aligned_window(trials_a, anchor)
    spikes_a = filter_spikes_to_forward_periods(spikes, anchor)
    spikes_a = spikes_a.loc[spikes_a["trial_id"].isin(trials_a["trial_id"])]

    sort_col = ANCHOR_SORT_COL[anchor]

    if split_mode == "outcome":
        groups = build_outcome_groups()
    elif split_mode == "quantile":
        groups = build_quantile_groups(trials_a, sort_col, n=N_QUANTILES)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    fig, (ax_r, ax_p) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    plot_raster(ax_r, events, trials_a, spikes_a, anchor, sort_col, show_legend=False)
    plot_firing_rates(ax_p, trials_a, spikes_a, anchor, groups)

    p99 = float(trials_a["aligned_end_time"].quantile(0.99))
    for ax in (ax_r, ax_p):
        ax.set_xlim(-0.2, p99 + 0.2)

    ax_r.set_ylabel("Trial (sorted)")
    ax_p.set_ylabel("Firing rate (Hz)")
    ax_p.set_xlabel(f"Time from {anchor} (s)")
    fig.suptitle("  |  ".join([
        unit_id, corrected_region, cell_type,
        f"anchor={anchor}", f"split={split_mode}",
    ]))
    fig.tight_layout()
    return fig


# ── Main loop ─────────────────────────────────────────────────────────────────
def load_unit_labels():
    """Merge units_vetted with corrected_region + cell_type from 0h's
    unit_properties_final.csv. Filter to QC-pass units."""
    units_vetted = (
        pd.read_csv(p.LOGS_DIR / "units_vetted.csv", index_col=0)
        .sort_values("unit_id")
    )
    units_vetted = units_vetted[units_vetted["qc_pass_all"] == True].copy()

    upf = pd.read_csv(p.LOGS_DIR / "unit_properties_final.csv")
    upf = upf[["mouse", "date_only", "insertion_number", "id",
               "corrected_region", "cell_type"]].drop_duplicates(
        subset=["mouse", "date_only", "insertion_number", "id"]
    )
    merged = units_vetted.merge(
        upf,
        left_on=["mouse", "date", "insertion_number", "id"],
        right_on=["mouse", "date_only", "insertion_number", "id"],
        how="left",
    ).drop(columns=["date_only"])
    n_unmatched = merged["corrected_region"].isna().sum()
    if n_unmatched:
        print(f"Warning: {n_unmatched} units missing from unit_properties_final.csv "
              "— labelled 'unknown'")
    merged["corrected_region"] = merged["corrected_region"].fillna("unknown")
    merged["cell_type"] = merged["cell_type"].fillna("unknown")
    return merged


def main():
    units_vetted = load_unit_labels()

    p.RASTER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    grouped = list(units_vetted.groupby("session_id"))
    n_sessions = len(grouped)

    for s_idx, (session_id, session_units) in enumerate(grouped, start=1):
        print(f"[{s_idx}/{n_sessions}] {session_id}: {len(session_units)} units",
              flush=True)
        events, trials, units = utils.get_session_data(session_id)
        trials = add_derived_trial_columns(trials)

        for _, ui in session_units.iterrows():
            unit_id = ui["unit_id"]
            corrected_region = ui["corrected_region"]
            cell_type = ui["cell_type"]
            spikes = units[ui["id"]]
            unit_dir = (p.RASTER_PLOTS_DIR / corrected_region /
                        cell_type / str(unit_id))
            unit_dir.mkdir(parents=True, exist_ok=True)

            for anchor in ANCHORS:
                for split_mode in SPLIT_MODES:
                    fig_path = unit_dir / f"{anchor}_{split_mode}.png"
                    if fig_path.exists() and not REGENERATE:
                        continue
                    try:
                        fig = plot_unit(
                            unit_id, events, trials, spikes, anchor, split_mode,
                            corrected_region=corrected_region,
                            cell_type=cell_type,
                        )
                        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
                        plt.close(fig)
                    except Exception as e:
                        failed.append({
                            "unit_id": unit_id, "anchor": anchor,
                            "split_mode": split_mode, "error": str(e),
                        })
                    finally:
                        plt.close("all")

    if failed:
        out = p.LOGS_DIR / "raster_histo_failed.csv"
        pd.DataFrame(failed).to_csv(out, index=False)
        print(f"{len(failed)} failures → {out}")
    else:
        print("All units processed.")


if __name__ == "__main__":
    main()
