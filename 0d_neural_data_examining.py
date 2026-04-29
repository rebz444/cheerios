"""
0d_neural_data_examining.py
============================
Adds group/cohort labels to the session log, computes per-session performance
metrics, and produces filtered session and unit lists. Also merges in the
spike-sorting QC flags from 0c so units_vetted reflects BOTH:

  - session/recording-level filters (num_units, length, percent_trials_w_spikes)
  - spike-sorting QC (qc_pass_all from RZ_unit_properties_with_qc.csv)

Inputs:
  p.LOGS_DIR / 'sessions_official_raw.csv'                 - from 0b
  per-session pickles in p.PICKLE_DIR                      - from 0b
  p.LOGS_DIR / 'RZ_unit_properties_with_qc.csv' (optional) - from 0c
    if absent, units_vetted falls back to recording-level filtering only.

Outputs (p.LOGS_DIR):
  sessions_all.csv         - all sessions w/ performance metrics
  sessions_vetted.csv      - sessions passing num_units / length thresholds
  units_all.csv            - all units with recording-level metrics + qc_pass_all
  units_vetted.csv         - units passing recording filter AND spike-sorting QC
  units_qc_missing.csv     - units in units_all but not in 0c output (if any)

Outputs (p.DATA_DIR / 'dataset_overview'):
  Histograms of session length, percent_missed, percent_bg_penalty,
  num_units, percent_trials_w_spikes, session_fr.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import constants as k
import paths as p
import utils


OVERVIEW_DIR = p.DATA_DIR / "dataset_overview"
OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)

QC_FILE = p.LOGS_DIR / "RZ_unit_properties_with_qc.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_session_length(trials):
    return trials["event_end_time"].iloc[-1] - trials["event_start_time"].iloc[0]


def add_session_performance(sessions_all_raw):
    rows = []
    for _, session_info in sessions_all_raw.iterrows():
        session_id = session_info["id"]
        _, trials, _ = utils.get_session_data(session_id)

        num_trials = len(trials)
        num_missed = trials.missed.sum()
        num_bg_penalty = (trials["num_bg_repeat"] > 0).sum()

        rows.append({
            "id": session_id,
            "length": get_session_length(trials),
            "num_trials": num_trials,
            "num_missed": num_missed,
            "percent_missed": num_missed / num_trials * 100,
            "num_bg_penalty": num_bg_penalty,
            "percent_bg_penalty": num_bg_penalty / num_trials * 100,
            "wait_length_mean": trials.wait_length.mean(),
        })
    perf = pd.DataFrame(rows)
    # sessions_official_raw.csv has a stale `num_trials` column (mostly NaN) that
    # would otherwise be picked up by pd.merge as an implicit join key and silently
    # drop every row. Drop it so we merge on `id` only.
    raw = sessions_all_raw.drop(columns=["num_trials"], errors="ignore")
    return pd.merge(raw, perf, on="id")


def print_dataset_overview(sessions_df):
    bar = "=" * 64
    print(bar)
    print("DATASET OVERVIEW")
    print(bar)

    for label, grp in [("Short (s)", "s"), ("Long  (l)", "l")]:
        rows = sessions_df[sessions_df["group"] == grp]
        print(
            f"{label}: {rows['mouse'].nunique():>3} mice | "
            f"{len(rows):>3} sessions | "
            f"{int(rows['num_trials'].sum()):>6} trials | "
            f"{int(rows['num_units'].sum()):>5} units"
        )
    print("-" * 64)
    print(
        f"Total    : {sessions_df['mouse'].nunique():>3} mice | "
        f"{len(sessions_df):>3} sessions | "
        f"{int(sessions_df['num_trials'].sum()):>6} trials | "
        f"{int(sessions_df['num_units'].sum()):>5} units"
    )

    print("\nUnits by region:")
    print(sessions_df.groupby("region")["num_units"].sum().to_string())

    histo_done = set(k.ALL_ANIMALS)
    print(
        f"\nHistology + clockspeed/rescaling complete "
        f"(constants.ALL_ANIMALS, n={len(histo_done)}):"
    )
    print(f"  early cohort ({len(k.EARLY_COHORT)}): {k.EARLY_COHORT}")
    print(f"  later cohort ({len(k.LATER_COHORT)}): {k.LATER_COHORT}")
    for label, grp in [("  short", "s"), ("  long ", "l")]:
        grp_mice = set(sessions_df.loc[sessions_df["group"] == grp, "mouse"].unique())
        done = sorted(grp_mice & histo_done)
        print(f"{label}: {len(done)}/{len(grp_mice)} mice done -> {done}")
    print(bar)


def hist(series, *, title, xlabel, filename, bins="auto", kde=True):
    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=series,
        kde=kde,
        stat="density" if kde else "count",
        bins=bins,
        color="skyblue",
        edgecolor="white",
        line_kws={"color": "red", "lw": 2},
    )
    plt.xlabel(xlabel)
    plt.ylabel("Density" if kde else "Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OVERVIEW_DIR / filename, bbox_inches="tight")
    plt.close()


def kde_by_group(df, col, *, xlabel, title, filename):
    plt.figure(figsize=(7, 4))
    for label, color in [("All", "gray"), ("l", "blue"), ("s", "orange")]:
        data = df if label == "All" else df[df["group"] == label]
        sns.kdeplot(data=data, x=col, label=label, color=color, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OVERVIEW_DIR / filename, bbox_inches="tight")
    plt.close()


def generate_units_all(sessions_vetted):
    rows = []
    for _, session_info in sessions_vetted.iterrows():
        session_id = session_info["id"]
        _, _, units = utils.get_session_data(session_id)
        for id_, spikes in units.items():
            spiked_trials = spikes["trial_id"].nunique()
            rows.append({
                "session_id": session_id,
                "mouse": session_info["mouse"],
                "date": session_info["date"],
                "insertion_number": session_info["insertion_number"],
                "id": id_,
                "unit_id": f"{session_id}_unit-{id_}",
                "region": session_info["region"],
                "group": session_info["group"],
                "percent_trials_w_spikes": spiked_trials / session_info["num_trials"],
                "session_fr": len(spikes) / session_info["length"],
            })
    return pd.DataFrame(rows)


def merge_qc_flags(units_all, qc_file):
    """Left-merge qc_pass_all from 0c's output into units_all.
    Returns (units_all_with_qc, units_missing_qc) where the second df is units
    that could not be matched to a QC row."""
    if not qc_file.exists():
        print(f"  Warning: {qc_file.name} not found.")
        print("  units_vetted will fall back to recording-level filter only.")
        units_all = units_all.copy()
        units_all["qc_pass_all"] = pd.NA
        return units_all, units_all.iloc[0:0].copy()

    qc = pd.read_csv(qc_file)
    qc_keys = ["mouse", "date_only", "insertion_number", "id"]
    qc_subset = qc[qc_keys + ["qc_pass_all"]].copy()
    qc_subset["mouse"] = qc_subset["mouse"].astype(str).str.strip()
    qc_subset["insertion_number"] = qc_subset["insertion_number"].astype(int)
    qc_subset["id"] = qc_subset["id"].astype(int)
    qc_subset = qc_subset.rename(columns={"date_only": "date"})

    units_all = units_all.copy()
    units_all["mouse"] = units_all["mouse"].astype(str).str.strip()
    units_all["insertion_number"] = units_all["insertion_number"].astype(int)
    units_all["id"] = units_all["id"].astype(int)

    merged = units_all.merge(
        qc_subset, on=["mouse", "date", "insertion_number", "id"], how="left"
    )
    missing = merged[merged["qc_pass_all"].isna()].copy()
    if len(missing):
        print(f"  Warning: {len(missing):,} / {len(merged):,} units have no matching QC row")
        print("  (saved to units_qc_missing.csv; treated as QC-fail in units_vetted)")
    return merged, missing


# ── 1. Load + add group/cohort ────────────────────────────────────────────────

print("Loading sessions_official_raw.csv...")
sessions_all_raw = pd.read_csv(p.LOGS_DIR / "sessions_official_raw.csv", index_col=0)

mouse_to_group = {m: g for g, mice in k.GROUP_DICT.items() for m in mice}
mouse_to_cohort = {m: c for c, mice in k.COHORT_DICT.items() for m in mice}
sessions_all_raw["group"] = sessions_all_raw["mouse"].map(mouse_to_group)
sessions_all_raw["cohort"] = sessions_all_raw["mouse"].map(mouse_to_cohort)

# ── 2. Per-session performance ────────────────────────────────────────────────

print("Computing session performance...")
sessions_all = add_session_performance(sessions_all_raw)
sessions_all.to_csv(p.LOGS_DIR / "sessions_all.csv")

# ── 3. Dataset overview + plots ───────────────────────────────────────────────

print()
print_dataset_overview(sessions_all)

print("\nPlotting session-level histograms...")
hist(
    sessions_all["length"],
    title="Session Length Distribution",
    xlabel="Time (s)",
    filename="hist_session_length.png",
)
kde_by_group(
    sessions_all, "percent_missed",
    xlabel="Percent Missed", title="Distribution of Percent Missed by Group",
    filename="kde_percent_missed_by_group.png",
)
kde_by_group(
    sessions_all, "percent_bg_penalty",
    xlabel="Percent BG Repeat", title="Distribution of Percent BG Repeat by Group",
    filename="kde_percent_bg_penalty_by_group.png",
)

# ── 4. Filter sessions ────────────────────────────────────────────────────────

sessions_vetted = sessions_all.loc[
    (sessions_all["num_units"] > k.MIN_UNITS)
    & (sessions_all["length"] >= k.MIN_SESSION_LENGTH)
].sort_values("id")
sessions_vetted.to_csv(p.LOGS_DIR / "sessions_vetted.csv")

print(f"\nVetted sessions: {len(sessions_vetted)} / {len(sessions_all)}")
print("Group wait_length_mean:")
print(sessions_vetted.groupby("group")["wait_length_mean"].mean().to_string())

hist(
    sessions_vetted["num_units"],
    title="Session Unit Count",
    xlabel="Number of units per insertion",
    filename="hist_num_units.png",
    bins=15,
    kde=False,
)

# ── 5. Generate units log ─────────────────────────────────────────────────────

print("\nGenerating units log...")
units_all = generate_units_all(sessions_vetted)

print("Merging spike-sorting QC flags from 0c...")
units_all, units_missing_qc = merge_qc_flags(units_all, QC_FILE)
units_all.to_csv(p.LOGS_DIR / "units_all.csv")
units_missing_qc.to_csv(p.LOGS_DIR / "units_qc_missing.csv")
print(f"  units_all: {len(units_all):,}")
print(f"  v1: {(units_all['region'] == 'v1').sum()} | str: {(units_all['region'] == 'str').sum()}")

hist(
    units_all["percent_trials_w_spikes"],
    title="Percent Trials with Spikes Distribution",
    xlabel="Percent Trials with Spikes",
    filename="hist_percent_trials_w_spikes.png",
)
hist(
    units_all["session_fr"],
    title="Session Firing Rate Distribution",
    xlabel="Firing Rate (Hz)",
    filename="hist_session_fr.png",
)

# ── 6. Units vetting (recording filter AND spike-sorting QC) ──────────────────

recording_pass = units_all["percent_trials_w_spikes"] >= k.MIN_PERCENT_TRIALS_WITH_SPIKES

if units_all["qc_pass_all"].isna().all():
    # 0c output not available — recording-level only
    units_vetted = units_all.loc[recording_pass]
    print("\nWARNING: no QC data merged — units_vetted is recording-filter only.")
else:
    qc_pass = units_all["qc_pass_all"].fillna(False).astype(bool)
    units_vetted = units_all.loc[recording_pass & qc_pass]

units_vetted.to_csv(p.LOGS_DIR / "units_vetted.csv")
print(f"\nVetted units: {len(units_vetted):,} / {len(units_all):,}")
print(
    f"  passing recording filter: {int(recording_pass.sum()):,}"
    + (
        f" | passing QC: {int(units_all['qc_pass_all'].fillna(False).astype(bool).sum()):,}"
        if not units_all["qc_pass_all"].isna().all()
        else ""
    )
)

print("\nDone.")
