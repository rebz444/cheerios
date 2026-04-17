"""
qc_metrics.py
=============
Generates presentation-ready QC figures for thesis committee meeting.

Inputs:
  RZ_unit_properties_with_qc.csv  — per-unit QC flags (output of 0d_neural_data_quality_metrics.py)

Outputs (p.DATA_DIR / 'qc_metrics/committee/'):
  fig1_qc_distributions.png   — 2×3 panel: distribution of each primary QC metric with cutoffs
  fig2a_str_pass_rates.png    — STR probe pass rate per QC metric (bar chart)
  fig2b_v1_pass_rates.png     — V1 probe pass rate per QC metric (bar chart)
  fig3_histology_status.png   — per-mouse QC-pass unit count, colored by histology status
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import constants as c
import paths as p

# ── Output directory ───────────────────────────────────────────────────────────
OUT_DIR = p.DATA_DIR / "qc_metrics" / "committee"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── STR-specific firing rate threshold ────────────────────────────────────────
FR_THRESH_STR = 0.05

# ── Presentation style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def apply_op(series, op, val):
    ops = {">": series.__gt__, "<": series.__lt__,
           ">=": series.__ge__, "<=": series.__le__, "==": series.__eq__}
    return ops[op](val)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading unit properties with QC flags...")
df = pd.read_csv(p.LOGS_DIR / "RZ_unit_properties_with_qc.csv")
print(f"  {len(df):,} units loaded")
print(f"  Regions present: {sorted(df['region'].dropna().unique())}")


# ── Figure 1: QC metric distributions ────────────────────────────────────────
print("Plotting Fig 1 — QC metric distributions...")

METRIC_CONFIG = {
    "firing_rate":        {"title": "Firing Rate (Hz)",          "xlim": (-2,   100),   "bins": 200},
    "isi_violation":      {"title": "ISI Violation",             "xlim": (0,    2),     "bins": 500},
    "amplitude_cutoff":   {"title": "Amplitude Cutoff",          "xlim": (0,    0.012), "bins": 200},
    "presence_ratio":     {"title": "Presence Ratio",            "xlim": (0.77, 1.0),   "bins": 60},
    "contamination_rate": {"title": "Contamination Rate",        "xlim": (0,    1.02),  "bins": 200},
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (metric, cfg) in enumerate(METRIC_CONFIG.items()):
    ax = axes[i]
    data = df[metric].dropna()

    # Clip to xlim for histogram binning so outliers don't squash the main distribution
    xlim = cfg["xlim"]
    data_clipped = data.clip(lower=xlim[0], upper=xlim[1])

    ax.hist(data_clipped, bins=cfg["bins"], color="cornflowerblue", edgecolor="none", alpha=0.85)
    ax.set_title(cfg["title"])
    ax.set_xlabel(cfg["title"])
    ax.set_ylabel("Unit Count")
    ax.set_xlim(*xlim)

    op, thresh = c.QC_THRESHOLDS[metric]

    # Shade the "pass" region
    if op == ">":
        ax.axvspan(thresh, xlim[1], color="skyblue", alpha=0.22, zorder=0, label="Pass region")
    else:
        ax.axvspan(xlim[0], thresh, color="skyblue", alpha=0.22, zorder=0, label="Pass region")

    # Threshold lines
    if metric == "firing_rate":
        ax.axvline(FR_THRESH_STR, color="darkorange", linestyle="--", linewidth=2,
                   label=f"STR threshold ({FR_THRESH_STR} Hz)")
        ax.axvline(thresh, color="crimson", linestyle="--", linewidth=2,
                   label=f"Default threshold ({thresh} Hz)")
    else:
        ax.axvline(thresh, color="crimson", linestyle="--", linewidth=2,
                   label=f"Threshold: {thresh}")

    ax.legend(fontsize=10, framealpha=0.7)
    n_pass = int((apply_op(df[metric].dropna(), op, thresh)).sum())
    n_total = int(df[metric].notna().sum())
    ax.text(0.98, 0.97, f"Pass: {n_pass}/{n_total}\n({100*n_pass/n_total:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Hide the unused 6th panel
axes[5].set_visible(False)

fig.suptitle("Metric Distributions — All Units")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_qc_distributions.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved fig1_qc_distributions.png")


# ── Figure 2: STR + V1 pass rates (side-by-side) ─────────────────────────────
print("Plotting Fig 2 — STR and V1 pass rates...")

str_df = df[df["region"].str.upper().eq("STR")]
v1_df  = df[df["region"].str.upper().eq("V1")]
print(f"  STR units: {len(str_df):,}  |  V1 units: {len(v1_df):,}")

METRICS = [
    ("firing_rate",        "Firing\nRate"),
    ("isi_violation",      "ISI\nViolation"),
    ("amplitude_cutoff",   "Amplitude\nCutoff"),
    ("presence_ratio",     "Presence\nRatio"),
    ("contamination_rate", "Contamination\nRate"),
    ("all",                "All\nMetrics"),
]
pass_cols = [f"qc_pass_{m}" if m != "all" else "qc_pass_all" for m, _ in METRICS]
xlabels   = [lbl for _, lbl in METRICS]

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

for ax, region_df, region_label in [
    (axes[0], str_df, "STR Probe"),
    (axes[1], v1_df,  "V1 Probe"),
]:
    n_total = len(region_df)
    pass_rates  = []
    n_pass_list = []
    for col in pass_cols:
        n = int(region_df[col].sum()) if col in region_df.columns else 0
        pass_rates.append(100.0 * n / n_total if n_total > 0 else 0.0)
        n_pass_list.append(n)

    x      = np.arange(len(xlabels))
    colors = ["steelblue"] * (len(xlabels) - 1) + ["#1a4f8a"]
    bars   = ax.bar(x, pass_rates, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

    for bar, n_p in zip(bars, n_pass_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{n_p}/{n_total}", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=13)
    ax.set_ylim(0, 115)
    ax.set_ylabel("% Units Passing", fontsize=14)
    ax.set_title(f"{region_label} — QC Pass Rate per Metric\n(n = {n_total} units total)")
    ax.axhline(100, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_pass_rates.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved fig2_pass_rates.png")


# ── Figure 3: Histology status (bar) + pie combined ──────────────────────────
print("Plotting Fig 3 — histology status + pie...")

# Build per-mouse summary
mouse_summary = (
    df.groupby("mouse")
      .agg(
          n_total   =("qc_pass_all", "count"),
          n_qc_pass =("qc_pass_all", "sum"),
      )
      .reset_index()
)
mouse_summary["histology_done"] = mouse_summary["mouse"].isin(c.ALL_ANIMALS)

def get_cohort(mouse):
    for cohort, members in c.COHORT_DICT.items():
        if mouse in members:
            return int(cohort)
    return 99

mouse_summary["cohort"] = mouse_summary["mouse"].apply(get_cohort)
mouse_summary = mouse_summary.sort_values(["cohort", "mouse"]).reset_index(drop=True)

COLOR_DONE    = "#2ca02c"
COLOR_PENDING = "#aec7e8"

n_mice  = len(mouse_summary)
fig_h   = max(7, n_mice * 0.45 + 2)
fig, (ax_bar, ax_pie) = plt.subplots(
    1, 2,
    figsize=(16, fig_h),
    gridspec_kw={"width_ratios": [2, 1]},
)

# ── Left: horizontal bar chart ────────────────────────────────────────────────
y_positions = np.arange(n_mice)
x_max = float(mouse_summary["n_total"].max())

for idx, row in mouse_summary.iterrows():
    y     = y_positions[idx]
    color = COLOR_DONE if row["histology_done"] else COLOR_PENDING

    ax_bar.barh(y, row["n_total"],   color=color, alpha=0.3, edgecolor="none",  height=0.6)
    ax_bar.barh(y, row["n_qc_pass"], color=color, alpha=0.9, edgecolor="black",
                linewidth=0.6, height=0.6)

    ax_bar.text(row["n_qc_pass"] + x_max * 0.01, y,
                f"{int(row['n_qc_pass'])} / {int(row['n_total'])}",
                va="center", fontsize=11)

ax_bar.set_yticks(y_positions)
ax_bar.set_yticklabels(mouse_summary["mouse"], fontsize=12)
ax_bar.set_xlabel("Number of Units", fontsize=14)
ax_bar.set_title("Per-Mouse QC-Pass Units & Histology Status\n(solid bar = QC-pass, faded = total)")
ax_bar.set_xlim(0, x_max * 1.2)
ax_bar.invert_yaxis()

patch_done    = mpatches.Patch(color=COLOR_DONE,    label="Histology Complete")
patch_pending = mpatches.Patch(color=COLOR_PENDING, label="Histology Pending")
ax_bar.legend(handles=[patch_done, patch_pending], fontsize=11, loc="lower right")

# ── Right: pie chart ──────────────────────────────────────────────────────────
n_pass_with    = int(mouse_summary.loc[mouse_summary["histology_done"],  "n_qc_pass"].sum())
n_pass_without = int(mouse_summary.loc[~mouse_summary["histology_done"], "n_qc_pass"].sum())
total_pass     = n_pass_with + n_pass_without

wedges, texts, autotexts = ax_pie.pie(
    [n_pass_with, n_pass_without],
    labels=["Histology\nComplete", "Histology\nPending"],
    colors=[COLOR_DONE, COLOR_PENDING],
    autopct=lambda p: f"{p:.1f}%\n({int(round(p * total_pass / 100))})",
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=12),
)
for at in autotexts:
    at.set_fontsize(11)

ax_pie.set_title(f"QC-Pass Units\nHistology Coverage\n(total: {total_pass})")

fig.tight_layout()
fig.subplots_adjust(wspace=0.01)
fig.savefig(OUT_DIR / "fig3_histology_status.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved fig3_histology_status.png")

print(f"\nDone! All figures saved to: {OUT_DIR}")
