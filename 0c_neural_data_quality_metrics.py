"""
0c_neural_data_quality_metrics.py
================================
Applies spike-sorting QC thresholds to unit properties and exports pass/fail flags.

Inputs:
  RZ_unit_properties.csv  — unit properties (firing rate, SNR, ISI, drift, waveform, etc.)
  Google Sheet recording log — session metadata (region, simultaneous recording flag)

Outputs (p.DATA_DIR / 'qc_metrics/'):
  hist_{metric}.png               — histograms for each primary and secondary QC metric
  region_qc_metrics_counts.png    — stacked bar chart of QC pass/fail counts by region
  scat_fr_vs_isi.png              — scatter: ISI violation vs. firing rate
  scat_fr_vs_iso.png              — scatter: isolation distance vs. firing rate
  qc_summary_overall.csv          — overall pass rate
  qc_summary_by_session.csv       — pass rate per session
  qc_summary_by_region.csv        — pass rate per region

Outputs (p.LOGS_DIR/):
  RZ_unit_properties_with_qc.csv  — per-unit QC pass/fail flags
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import constants as c
import paths as p

# ── Paths ──────────────────────────────────────────────────────────────────────
QC_DIR = p.DATA_DIR / "qc_metrics"
QC_DIR.mkdir(parents=True, exist_ok=True)

# ── Google Sheet — recording log ───────────────────────────────────────────────
_RECORDING_LOG_SHEET_ID = "1xpb52EO7BII-_5zIcB2HaOpZqepFcYTvBYgxA4zLyNo"
RECORDING_LOG_URL = (
    f"https://docs.google.com/spreadsheets/d/{_RECORDING_LOG_SHEET_ID}"
    "/export?format=csv&gid=0"
)

# ── QC thresholds ──────────────────────────────────────────────────────────────
qc_thresholds           = c.QC_THRESHOLDS
qc_thresholds_secondary = c.QC_THRESHOLDS_SECONDARY

# STR-specific firing rate floor (MSNs in behaving mice can fire below the default 0.1 Hz)
FR_THRESH_STR = 0.05

# ISI values above this are flagged as likely spike-sorting artifacts
ISI_OUTLIER_THRESH = 20.0


# ── Helper functions ───────────────────────────────────────────────────────────

def apply_rule(series: pd.Series, op: str, val: float) -> pd.Series:
    ops = {
        "<":  series.__lt__,
        "<=": series.__le__,
        ">":  series.__gt__,
        ">=": series.__ge__,
        "==": series.__eq__,
    }
    return ops.get(op, lambda v: pd.Series(False, index=series.index))(val)


def compute_qc_summary(df: pd.DataFrame, groupby_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(groupby_col, dropna=False)
          .agg(total_units=("id", "count"),
               qc_pass_units=("qc_pass_all", "sum"))
          .reset_index()
    )
    summary["qc_pass_pct"] = 100.0 * summary["qc_pass_units"] / summary["total_units"]
    return summary


def hist_plot(series, title, bins=50, filename=None, xlim=None, qc_threshold=None):
    plt.figure(figsize=(8, 5), dpi=140)
    sns.histplot(series.dropna(), bins=bins, color="cornflowerblue", edgecolor="black")
    plt.title(title)
    plt.ylabel("Count")
    if qc_threshold is not None:
        op, threshold = qc_threshold
        ax = plt.gca()
        if op == ">":
            plt.axvspan(threshold, ax.get_xlim()[1], color="skyblue", alpha=0.18, label="Kept region")
        elif op == "<":
            plt.axvspan(ax.get_xlim()[0], threshold, color="skyblue", alpha=0.18, label="Kept region")
        plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold: {threshold}")
        plt.legend()
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    if filename:
        plt.savefig(QC_DIR / filename, bbox_inches="tight")
    plt.close()


def scatter_plot(df, x_col, y_col, qc_pass_col, title, xlabel, ylabel,
                 alpha=0.6, filename=None, xlim=None, ylim=None):
    x      = df[x_col]
    y      = df[y_col]
    qc_pass = df[qc_pass_col]
    mask   = ~(np.isnan(x.values) | np.isnan(y.values) | pd.isnull(qc_pass.values))
    xc, yc, qcc = x.values[mask], y.values[mask], qc_pass.values[mask]

    plt.figure(figsize=(7, 5), dpi=140)
    for val, color, label in zip(
        [True, False], ["lightblue", "lightcoral"], ["QC Pass", "QC Fail"]
    ):
        idx = qcc == val
        plt.scatter(xc[idx], yc[idx], alpha=alpha, s=6, color=color, label=label, edgecolor="none")
    plt.title(title)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=11)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    if filename:
        plt.savefig(QC_DIR / filename, bbox_inches="tight")
    plt.close()


# ── 1. Load unit properties ────────────────────────────────────────────────────
print("Loading unit properties...")
up_df = pd.read_csv(p.LOGS_DIR / "RZ_unit_properties.csv")
up_df = up_df.rename(columns={"subject": "mouse", "unit": "id", "session_datetime": "datetime"})
up_df["date_only"] = pd.to_datetime(up_df["datetime"]).dt.date.astype(str)

print(f"  {len(up_df):,} units loaded")

# Keep only manually labelled "good" units
up_df = up_df[up_df["manual_label"].str.lower().eq("good")].reset_index(drop=True)
up_df.to_csv(p.LOGS_DIR / "RZ_unit_properties_good.csv", index=False)
print(f"  {len(up_df):,} 'good' units retained → saved RZ_unit_properties_good.csv")

# ── 2. Load recording log and merge session metadata ──────────────────────────
print("Loading recording log from Google Sheets...")
log = pd.read_csv(RECORDING_LOG_URL)
# Normalize column names: lowercase, strip whitespace, replace spaces with underscores
log.columns = log.columns.str.lower().str.strip().str.replace(" ", "_")
log["date_only"] = pd.to_datetime(log["date"]).dt.date.astype(str)
log["mouse"] = log["mouse"].astype(str).str.strip()

# Build list of columns to pull from the log (simultaneous may or may not exist)
log_cols = ["mouse", "date_only", "insertion_number", "region"]
if "simultaneous" in log.columns:
    log_cols.append("simultaneous")

_pre_merge_len = len(up_df)
up_df = up_df.merge(
    log[log_cols].drop_duplicates(),
    on=["mouse", "date_only", "insertion_number"],
    how="left",
)
assert len(up_df) == _pre_merge_len, (
    f"Recording log merge duplicated rows: {len(up_df):,} vs {_pre_merge_len:,}. "
    "Check for duplicate entries in the log."
)

if "simultaneous" not in up_df.columns:
    up_df["simultaneous"] = np.nan

# Construct a session key for grouping
up_df["session_key"] = (
    up_df["mouse"]
    + "|"
    + up_df["date_only"]
    + "|"
    + up_df["insertion_number"].astype(str)
)

n_no_region = up_df["region"].isna().sum()
if n_no_region:
    print(f"  Warning: {n_no_region} units could not be matched to the recording log")

print(f"  {len(up_df):,} units after recording log merge")

# Working copy
df = up_df.copy()

# ── 3. Histograms — primary QC metrics ────────────────────────────────────────
print("Plotting primary QC metric histograms...")

METRIC_PLOT_CONFIG = {
    "firing_rate":        {"title": "Firing Rate",          "bins": 200,  "xlim": (-2, 100)},
    "isi_violation":      {"title": "ISI Violations",       "bins": 2000, "xlim": (0, 2)},
    "amplitude_cutoff":   {"title": "Amplitude Cutoff",     "bins": 500,  "xlim": (0, 0.012)},
    "presence_ratio":     {"title": "Presence Ratio",       "bins": 60,   "xlim": (0.77, 1.0)},
    "contamination_rate": {"title": "Contamination Rate",   "bins": 200,  "xlim": (0, 1.02)},
    "number_violation":   {"title": "Number of Violations", "bins": 2000, "xlim": (-5, 400)},
}

for metric, cfg in METRIC_PLOT_CONFIG.items():
    thresh = qc_thresholds.get(metric)
    hist_plot(df[metric], cfg["title"], bins=cfg["bins"],
              filename=f"hist_{metric}.png", xlim=cfg["xlim"],
              qc_threshold=thresh)

# ── 4. Histograms — secondary QC metrics ──────────────────────────────────────
print("Plotting secondary QC metric histograms...")

METRIC_PLOT_CONFIG_SECONDARY = {
    "isolation_distance": {"title": "Isolation Distance",   "bins": 5000, "xlim": (0, 1000)},
    "snr":                {"title": "SNR Distribution",     "bins": 200,  "xlim": (0, 30)},
    "l_ratio":            {"title": "L Ratio",              "bins": 200,  "xlim": (0, 2.5)},
    "d_prime":            {"title": "D Prime",              "bins": 200,  "xlim": None},
    "nn_hit_rate":        {"title": "NN Hit Rate",          "bins": 200,  "xlim": None},
    "nn_miss_rate":       {"title": "NN Miss Rate",         "bins": 200,  "xlim": (0, 0.3)},
    "silhouette_score":   {"title": "Silhouette Score",     "bins": 200,  "xlim": (0, 0.5)},
    "max_drift":          {"title": "Max Drift",            "bins": 200,  "xlim": (0, 25)},
    "cumulative_drift":   {"title": "Cumulative Drift",     "bins": 200,  "xlim": (0, 7)},
}

for metric, cfg in METRIC_PLOT_CONFIG_SECONDARY.items():
    thresh = qc_thresholds_secondary.get(metric)
    hist_plot(df[metric], cfg["title"], bins=cfg["bins"],
              filename=f"hist_{metric}.png", xlim=cfg["xlim"],
              qc_threshold=thresh)

# Compute qc_pass columns for secondary metrics (informational, not in qc_pass_all)
for metric, (op, val) in qc_thresholds_secondary.items():
    if metric in df.columns:
        df[f"qc_pass_{metric}"] = apply_rule(df[metric], op, val)
        df[f"qc_not_computable_{metric}"] = df[metric].isna()

# ── 4b. ISI outlier detection ─────────────────────────────────────────────────
df["qc_flag_isi_outlier"] = df["isi_violation"] > ISI_OUTLIER_THRESH
isi_outliers = df[df["qc_flag_isi_outlier"]][
    ["mouse", "date_only", "insertion_number", "id", "isi_violation", "manual_label"]
]
if not isi_outliers.empty:
    print(f"\nISI outliers (isi_violation > {ISI_OUTLIER_THRESH}):")
    print(isi_outliers.to_string(index=False))
else:
    print(f"  No ISI outliers above {ISI_OUTLIER_THRESH}")

# ── 5. Apply QC thresholds ────────────────────────────────────────────────────
print("Applying QC thresholds...")

# NaN policy: units with uncomputable metrics explicitly fail that metric and are tracked
# via qc_not_computable_*. qc_pass_* columns are always clean booleans (no NaN).
qc_mask_parts = []
for metric, (op, val) in qc_thresholds.items():
    if metric in df.columns:
        has_value = df[metric].notna()
        m = apply_rule(df[metric], op, val)
        df[f"qc_not_computable_{metric}"] = ~has_value
        df[f"qc_pass_{metric}"] = m.fillna(False)   # NaN → False (uncomputable = fail)
        qc_mask_parts.append(df[f"qc_pass_{metric}"])
    else:
        print(f"  Warning: metric '{metric}' not found in dataframe")
        df[f"qc_not_computable_{metric}"] = True
        df[f"qc_pass_{metric}"] = True   # metric absent → don't penalise
        qc_mask_parts.append(pd.Series(True, index=df.index))

# STR override: lower firing rate floor for MSNs
str_mask = df["region"].str.lower().eq("str")
df.loc[str_mask, "qc_pass_firing_rate"] = (
    df.loc[str_mask, "firing_rate"] > FR_THRESH_STR
).fillna(False)
df.loc[str_mask, "qc_not_computable_firing_rate"] = df.loc[str_mask, "firing_rate"].isna()
fr_idx = list(qc_thresholds.keys()).index("firing_rate")
qc_mask_parts[fr_idx] = df["qc_pass_firing_rate"]

df["qc_pass_all"] = np.logical_and.reduce(qc_mask_parts) if qc_mask_parts else False

# ── 6. QC summary statistics ──────────────────────────────────────────────────
print("Computing QC summaries...")

summary_overall = pd.DataFrame({
    "total_units":    [len(df)],
    "qc_pass_units":  [int(df["qc_pass_all"].sum())],
    "qc_pass_pct":    [float(100.0 * df["qc_pass_all"].mean())],
})
summary_by_session = compute_qc_summary(df, "session_key")
summary_by_region  = compute_qc_summary(df, "region")

summary_overall.to_csv(QC_DIR / "qc_summary_overall.csv", index=False)
summary_by_session.to_csv(QC_DIR / "qc_summary_by_session.csv", index=False)
summary_by_region.to_csv(QC_DIR / "qc_summary_by_region.csv", index=False)

print(summary_overall.to_string(index=False))
print()
print(summary_by_region.sort_values("qc_pass_pct", ascending=False).to_string(index=False))

# V1 session audit — identify bad recordings pulling the average down
v1_session_keys = df.loc[df["region"].str.lower().eq("v1"), "session_key"].unique()
v1_sessions = summary_by_session[
    summary_by_session["session_key"].isin(v1_session_keys)
].sort_values("qc_pass_pct")
print("\nV1 session audit (sorted by pass rate):")
print(v1_sessions.to_string(index=False))
v1_sessions.to_csv(QC_DIR / "qc_summary_v1_sessions.csv", index=False)

# ── 7. Region × QC metric stacked bar chart ───────────────────────────────────
print("Plotting QC counts by region...")

qc_metrics = [f"qc_pass_{m}" for m in qc_thresholds] + ["qc_pass_all"]

fig, axs = plt.subplots(1, len(qc_metrics), figsize=(2 * len(qc_metrics), 6), sharey=True)

for i, metric in enumerate(qc_metrics):
    region_qc_counts = df.groupby(["region", metric]).size().reset_index(name="count")
    pivot_qc = region_qc_counts.pivot(index="region", columns=metric, values="count").fillna(0)
    pivot_qc = pivot_qc.reindex(columns=[True, False], fill_value=0)
    x = np.arange(len(pivot_qc.index))

    bars_pass = axs[i].bar(x, pivot_qc[True],  width=0.6, label="QC-Pass", color="lightblue",  edgecolor="black")
    bars_fail = axs[i].bar(x, pivot_qc[False], width=0.6, bottom=pivot_qc[True], label="QC-Fail", color="lightcoral", edgecolor="black")
    axs[i].set_xticks(x)
    axs[i].set_xticklabels(pivot_qc.index, rotation=45, ha="right", fontsize=10)
    axs[i].set_title(metric.replace("qc_pass_", "").replace("_", " ").title())
    axs[i].set_xlabel("Region", fontsize=12)
    for bar in bars_pass + bars_fail:
        height = bar.get_height()
        if height > 0:
            axs[i].text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2,
                        f"{int(height)}", ha="center", va="center", fontsize=9, color="black")
    if i == 0:
        axs[i].set_ylabel("Count", fontsize=12)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)
plt.tight_layout()
plt.savefig(QC_DIR / "region_qc_metrics_counts.png", dpi=300, bbox_inches="tight")
plt.close()

# ── 8. Scatter plots ──────────────────────────────────────────────────────────
print("Plotting scatter plots...")

scatter_plot(df, "firing_rate", "isi_violation", "qc_pass_isi_violation",
             "ISI Violation vs Firing Rate", "Firing rate (Hz)", "ISI violation",
             filename="scat_fr_vs_isi.png", xlim=(-1, 40), ylim=(-1, 60))

scatter_plot(df, "firing_rate", "isolation_distance", "qc_pass_isolation_distance",
             "Isolation Distance vs Firing Rate", "Firing rate (Hz)", "Isolation distance",
             filename="scat_fr_vs_iso.png", xlim=(0, 80), ylim=(0, 4000))

# ── 9. Export per-unit QC pass/fail flags ─────────────────────────────────────
print("Exporting per-unit QC flags...")

# Propagate qc_pass columns back to up_df
qc_cols = [col for col in df.columns if col.startswith("qc_pass_")]
up_df[qc_cols] = df[qc_cols]

# Save full up_df (all unit properties + session metadata + qc_pass columns)
out_path = p.LOGS_DIR / "RZ_unit_properties_with_qc.csv"
up_df.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")

print("Done.")
