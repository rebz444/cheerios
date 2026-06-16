"""
0g_waveform_diagnostic.py
=========================
Combined waveform visualisation and FR/PT biological-prior validation.
Replaces 0f_waveform_by_region.py and 0g_waveform_fr_validation.py.

Requires:
  LOGS_DIR / unit_properties_with_qc_and_regions.csv  (output of 0f)
  RAW_DATA_DIR / RZ_unit_templates.npz

Section 1 — Waveform by region
  [Plot 1] Mean ± SD waveforms per region
  [Plot 2] PT duration distributions + PT vs amplitude scatter
  [Plot 3] Waveform heatmap sorted by PT duration

Section 2 — FR / PT biological-prior validation
  [Plot 4] FR × PT scatter (overview + faceted by circuit group)
  [Plot 5] Per-animal bio-consistency heatmap
  [Plot 6] Prior range strips: observed vs expected PT and FR
  [Plot 7] Pass-rate summary bar per region
  [Plot 8] Flagged unit waveforms vs all units
  flagged_units.csv

Outputs → DATA_DIR / location_matching / diagnostic/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

import paths as p
from constants import REGION_COLORS
from utils import load_waveform_metrics

# ── Output directory ───────────────────────────────────────────────────────────
PLOT_DIR = p.DATA_DIR / "location_matching" / "diagnostic"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Region lists ───────────────────────────────────────────────────────────────
# Waveform plots use a tighter set of well-populated regions
WAVEFORM_REGIONS = [
    "CP", "GPe",
    "MOp5", "MOp2/3",
    "VISp5", "VISp2/3",
    "VAL", "RT",
]

# Validation plots include all regions with defined priors
VALIDATION_REGIONS = [
    "CP", "GPe",
    "MOp5", "MOp2/3", "MOp1", "MOp6a",
    "VISp5", "VISp2/3", "VISp4", "VISp6a", "VISp6b",
    "VAL", "AV", "CL", "PO", "VPM", "LD",
    "RT",
    "CA1", "SUB",
    "PAL",          # pallidum — borders CP, recorded on STR probes, priors defined
]

# Probe-context expectations: which probe types should we expect units from for
# each region? "_str" probes target striatum and pass through cortex/thalamus on
# the way; "_v1" probes target visual cortex. Units appearing on the wrong probe
# are not necessarily wrong — but they're worth flagging in the cross-check.
EXPECTED_PROBE_CONTEXT = {
    "CP":      {"str"},
    "GPe":     {"str"},
    "GPi":     {"str"},
    "PAL":     {"str"},
    "MOp5":    {"str"},   # STR probe enters via MOp
    "MOp2/3":  {"str"},
    "MOp1":    {"str"},
    "MOp6a":   {"str"},
    "VISp5":   {"v1"},
    "VISp2/3": {"v1"},
    "VISp4":   {"v1"},
    "VISp1":   {"v1"},
    "VISp6a":  {"v1"},
    "VISp6b":  {"v1"},
    "VISl1":   {"v1"},
    "VISl2/3": {"v1"},
    # Thalamic relays sit between cortex and CP — STR probes pass through them.
    "VAL":     {"str"},
    "AV":      {"str"},
    "CL":      {"str"},
    "LD":      {"str"},
    "LP":      {"str", "v1"},
    "PO":      {"str", "v1"},
    "VPM":     {"str"},
    "RT":      {"str"},
    "CA1":     {"v1"},   # V1 probe overshoots into HPC
    "SUB":     {"v1"},
}

# ── Biological priors ──────────────────────────────────────────────────────────
# Format: (pt_min_ms, pt_max_ms, fr_min_hz, fr_max_hz)
# None = no constraint on that bound.
#
# Sources:
#   STR MSNs  : Mallet 2006, Berke 2004 — very low FR, broad waveform
#   GPe       : Mallet 2008, Gage 2010  — high FR (30–100 Hz)
#   Cortex    : McCormick 1985, Barthó 2004 — pyramidal broad, FS narrow
#   Thalamus  : Steriade 1993 — relay broad; RT narrow fast-spiking
#   Hippocampus: O'Keefe 1978, Ylinen 1995 — place cells low FR

REGION_PRIORS = {
    "CP":      (0.40, None, 0.05,  15.0),
    "STR":     (0.40, None, 0.05,  15.0),
    "GPe":     (0.35, None, 20.0, 120.0),
    "GPi":     (0.35, None, 40.0, 150.0),
    "MOp5":    (0.40, None,  0.5,  40.0),
    "MOp2/3":  (0.30, None,  0.5,  40.0),
    "MOp1":    (0.25, None,  0.2,  30.0),
    "MOp6a":   (0.35, None,  0.5,  30.0),
    "MOs5":    (0.40, None,  0.5,  40.0),
    "MOs2/3":  (0.30, None,  0.5,  40.0),
    "MOs1":    (0.25, None,  0.2,  30.0),
    "VISp5":   (0.40, None,  0.5,  30.0),
    "VISp2/3": (0.25, None,  0.5,  30.0),
    "VISp4":   (0.30, None,  1.0,  40.0),
    "VISp6a":  (0.35, None,  0.5,  25.0),
    "VISp6b":  (0.35, None,  0.5,  25.0),
    "VISp1":   (0.25, None,  0.2,  20.0),
    "VISl1":   (0.25, None,  0.2,  30.0),
    "VISl2/3": (0.25, None,  0.5,  30.0),
    "VAL":     (0.35, None,  2.0,  40.0),
    "AV":      (0.35, None,  2.0,  30.0),
    "CL":      (0.30, None,  2.0,  40.0),
    "LD":      (0.35, None,  2.0,  30.0),
    "LP":      (0.35, None,  2.0,  30.0),
    "PO":      (0.35, None,  2.0,  35.0),   # posterior thalamic relay (Sherman 2007)
    "VPM":     (0.35, None,  2.0,  35.0),   # ventral posteromedial relay (Steriade 1993)
    "RT":      (None, 0.35,  5.0,  80.0),   # NARROW mandatory
    "CA1":     (0.35, None,  0.5,  10.0),
    "SUB":     (0.35, None,  0.5,  15.0),
    "PAL":     (0.35, None,  0.5,  20.0),
}


# ══════════════════════════════════════════════════════════════════════════════
# Load data (once)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("0g_waveform_diagnostic.py")
print("=" * 60)

_csv_candidates = [
    p.LOGS_DIR / "unit_properties_with_qc_and_regions.csv",
    p.LOGS_DIR / "unit_properties_with_regions.csv",
]
_csv_path = next((c for c in _csv_candidates if c.exists()), None)
if _csv_path is None:
    raise FileNotFoundError(
        "Could not find region CSV. Tried:\n" +
        "\n".join(str(c) for c in _csv_candidates)
    )
print(f"\nLoading: {_csv_path.name}")
regions_df = pd.read_csv(_csv_path)
regions_df["datetime"]         = pd.to_datetime(regions_df["datetime"])
regions_df["insertion_number"] = regions_df["insertion_number"].astype(int)
regions_df["id"]               = regions_df["id"].astype(int)
regions_df["paramset_idx"]     = regions_df["paramset_idx"].astype(int)

# Strip any waveform columns left behind by 0f's boundary-recovery step;
# load_waveform_metrics re-computes them and a duplicate template_idx would
# collide on its merge.
_STALE_WF_COLS = ["template_idx", "pt_duration_ms", "trough_amp_uv",
                  "peak_amp_uv", "pt_ratio"]
regions_df = regions_df.drop(
    columns=[c for c in _STALE_WF_COLS if c in regions_df.columns]
)
print(f"  {len(regions_df):,} units loaded")

# ── Detect firing rate column ──────────────────────────────────────────────────
_FR_CANDIDATES = ["firing_rate", "mean_firing_rate", "fr", "spike_rate",
                  "mean_fr", "avg_firing_rate"]
fr_col = next((c for c in _FR_CANDIDATES if c in regions_df.columns), None)

if fr_col is None:
    print(f"\n  WARNING: No firing rate column found. Searched: {_FR_CANDIDATES}")
    print("  FR-based validation will be skipped. Only PT duration will be used.")
    HAS_FR = False
else:
    print(f"  Firing rate column: '{fr_col}'  "
          f"(median={regions_df[fr_col].median():.2f} Hz, "
          f"n_valid={regions_df[fr_col].notna().sum():,})")
    HAS_FR = True

# ── Load waveform templates and compute metrics ────────────────────────────────
print("\nLoading waveform templates and computing metrics...")
merged, waveforms, waveforms_norm, t_ms = load_waveform_metrics(regions_df)
merged["waveform_row"] = np.arange(len(merged))
print(f"  {len(merged):,} units matched to templates")

# ── Available regions ──────────────────────────────────────────────────────────
avail_wf  = [r for r in WAVEFORM_REGIONS   if r in merged["region_acronym"].values]
avail_val = [r for r in VALIDATION_REGIONS if r in merged["region_acronym"].values]

# ── Apply biological priors ────────────────────────────────────────────────────
def check_prior(row):
    region = row["region_acronym"]
    prior  = REGION_PRIORS.get(region)
    if prior is None:
        return {"pt_ok": None, "fr_ok": None, "bio_consistent": None}

    pt_min, pt_max, fr_min, fr_max = prior

    pt = row["pt_duration_ms"]
    if not np.isnan(pt):
        pt_ok = True
        if pt_min is not None and pt < pt_min:
            pt_ok = False
        if pt_max is not None and pt > pt_max:
            pt_ok = False
    else:
        pt_ok = None

    fr_ok = None
    if HAS_FR:
        fr = row[fr_col]
        if pd.notna(fr):
            fr_ok = True
            if fr_min is not None and fr < fr_min:
                fr_ok = False
            if fr_max is not None and fr > fr_max:
                fr_ok = False

    criteria = [c for c in [pt_ok, fr_ok] if c is not None]
    return {"pt_ok": pt_ok, "fr_ok": fr_ok,
            "bio_consistent": all(criteria) if criteria else None}

print("\nApplying biological priors...")
flags = merged.apply(check_prior, axis=1, result_type="expand")
merged = pd.concat([merged, flags], axis=1)

n_evaluated = merged["bio_consistent"].notna().sum()
n_pass      = (merged["bio_consistent"] == True).sum()
n_fail      = (merged["bio_consistent"] == False).sum()
n_no_prior  = merged["bio_consistent"].isna().sum()
print(f"  Evaluated  : {n_evaluated:,}")
print(f"  Consistent : {n_pass:,}  ({n_pass/n_evaluated*100:.1f}%)")
print(f"  Flagged    : {n_fail:,}  ({n_fail/n_evaluated*100:.1f}%)")
print(f"  No prior   : {n_no_prior:,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Waveform by region
# ══════════════════════════════════════════════════════════════════════════════

# ── [Plot 1] Mean ± SD waveforms by region ────────────────────────────────────
print("\n[Plot 1] Mean ± SD waveforms by region...")

n_regions = len(avail_wf)
ncols = 4
nrows = int(np.ceil(n_regions / ncols))

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 3.5, nrows * 3),
                         sharey=False)
axes = np.array(axes).flatten()

for ax_idx, region in enumerate(avail_wf):
    ax    = axes[ax_idx]
    idx   = merged["region_acronym"] == region
    wv    = waveforms_norm[idx.values]
    n     = wv.shape[0]
    color = REGION_COLORS.get(region, "#555555")

    mean_wv = wv.mean(axis=0)
    std_wv  = wv.std(axis=0)

    ax.fill_between(t_ms, mean_wv - std_wv, mean_wv + std_wv,
                    alpha=0.25, color=color)
    ax.plot(t_ms, mean_wv, color=color, linewidth=1.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(t_ms[int(np.round(np.argmin(mean_wv)))],
               color="gray", linewidth=0.5, linestyle="--")

    pt_med = merged.loc[idx, "pt_duration_ms"].median()
    ax.set_title(f"{region}  (n={n})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Norm. amplitude", fontsize=8)
    ax.text(0.97, 0.05, f"PT={pt_med:.2f} ms",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=color)
    ax.tick_params(labelsize=7)

for ax_idx in range(len(avail_wf), len(axes)):
    axes[ax_idx].set_visible(False)

plt.suptitle("Mean ± SD waveforms by brain region (normalised to trough)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "waveform_by_region.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved → waveform_by_region.png")


# ── [Plot 2] PT duration distributions + PT vs amplitude scatter ──────────────
print("[Plot 2] Waveform metrics scatter...")

plot_wf = merged[merged["region_acronym"].isin(avail_wf) &
                 merged["pt_duration_ms"].notna()].copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

region_order = (plot_wf.groupby("region_acronym")["pt_duration_ms"]
                .median().sort_values().index.tolist())

for i, region in enumerate(region_order):
    d     = plot_wf.loc[plot_wf["region_acronym"] == region, "pt_duration_ms"]
    color = REGION_COLORS.get(region, "#555555")
    axes[0].scatter(d, np.full(len(d), i) + np.random.uniform(-0.3, 0.3, len(d)),
                    color=color, alpha=0.5, s=18, linewidths=0)
    axes[0].plot([d.median()], [i], marker="|", color="black",
                 markersize=12, markeredgewidth=2)

axes[0].set_yticks(range(len(region_order)))
axes[0].set_yticklabels(region_order, fontsize=9)
axes[0].axvline(0.35, color="tomato",  linestyle="--", linewidth=1,
                label="FS threshold (0.35 ms)")
axes[0].axvline(0.45, color="orange", linestyle="--", linewidth=1,
                label="RS threshold (0.45 ms)")
axes[0].set_xlabel("Peak-to-trough duration (ms)", fontsize=10)
axes[0].set_title("Waveform width by region", fontsize=11)
axes[0].legend(fontsize=8)

for region in avail_wf:
    d     = plot_wf[plot_wf["region_acronym"] == region]
    color = REGION_COLORS.get(region, "#555555")
    axes[1].scatter(d["pt_duration_ms"], np.abs(d["trough_amp_uv"]),
                    color=color, alpha=0.5, s=18, linewidths=0, label=region)

axes[1].set_xlabel("Peak-to-trough duration (ms)", fontsize=10)
axes[1].set_ylabel("|Trough amplitude| (µV)", fontsize=10)
axes[1].set_title("Waveform width vs amplitude", fontsize=11)
axes[1].axvline(0.35, color="tomato", linestyle="--", linewidth=0.8)
axes[1].axvline(0.45, color="orange", linestyle="--", linewidth=0.8)
axes[1].legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig(PLOT_DIR / "waveform_metrics_scatter.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved → waveform_metrics_scatter.png")


# ── [Plot 3] Waveform heatmap ─────────────────────────────────────────────────
print("[Plot 3] Waveform heatmap...")

heatmap_df  = merged[merged["region_acronym"].isin(avail_wf) &
                     merged["pt_duration_ms"].notna()].copy()
sort_order  = heatmap_df["pt_duration_ms"].argsort().values
wv_sorted   = waveforms_norm[heatmap_df.index.values][sort_order]
regions_sorted = heatmap_df["region_acronym"].values[sort_order]
pt_sorted      = heatmap_df["pt_duration_ms"].values[sort_order]

fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                         gridspec_kw={"width_ratios": [6, 1]})

im = axes[0].imshow(wv_sorted, aspect="auto", cmap="RdBu_r",
                    vmin=-1.5, vmax=1.5,
                    extent=[t_ms[0], t_ms[-1], len(wv_sorted), 0])
axes[0].set_xlabel("Time (ms)", fontsize=10)
axes[0].set_ylabel("Units (sorted by PT duration)", fontsize=10)
axes[0].set_title("All waveforms — sorted by peak-to-trough duration", fontsize=11)
plt.colorbar(im, ax=axes[0], fraction=0.02, pad=0.01).set_label(
    "Norm. amplitude", fontsize=8)

colors_sorted = [REGION_COLORS.get(r, "#555555") for r in regions_sorted]
axes[1].barh(range(len(pt_sorted)), pt_sorted,
             color=colors_sorted, height=1.0, linewidth=0)
axes[1].axvline(0.35, color="tomato", linestyle="--", linewidth=0.8)
axes[1].axvline(0.45, color="orange", linestyle="--", linewidth=0.8)
axes[1].set_xlabel("PT (ms)", fontsize=8)
axes[1].set_yticks([])
axes[1].invert_yaxis()

handles = [Patch(facecolor=REGION_COLORS.get(r, "#555555"), label=r)
           for r in avail_wf]
axes[1].legend(handles=handles, fontsize=7, loc="lower right",
               bbox_to_anchor=(2.2, 0))

plt.tight_layout()
plt.savefig(PLOT_DIR / "waveform_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved → waveform_heatmap.png")

# ── Waveform summary stats ─────────────────────────────────────────────────────
print("\nWaveform summary by region:")
summary_wf = (plot_wf.groupby("region_acronym")
              .agg(
                  n             = ("pt_duration_ms", "count"),
                  pt_median_ms  = ("pt_duration_ms", "median"),
                  pt_iqr_ms     = ("pt_duration_ms", lambda x: x.quantile(0.75) - x.quantile(0.25)),
                  pct_narrow    = ("pt_duration_ms", lambda x: (x < 0.35).mean() * 100),
                  amp_median_uv = ("trough_amp_uv",  lambda x: np.abs(x).median()),
              )
              .loc[avail_wf]
              .round(3))
print(summary_wf.to_string())
print("\nExpected:")
print("  Narrow (<0.35 ms): RT, some CP interneurons")
print("  Broad  (>0.45 ms): MOp5, MOp2/3, VISp5")
print("  High amplitude   : GPe")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FR / PT biological-prior validation
# ══════════════════════════════════════════════════════════════════════════════

# ── Per-region validation summary ─────────────────────────────────────────────
print("\nPer-region validation summary:")
summary_rows = []
for region in avail_val:
    sub    = merged[merged["region_acronym"] == region]
    n      = len(sub)
    pt_med = sub["pt_duration_ms"].median()
    pt_iqr = sub["pt_duration_ms"].quantile(0.75) - sub["pt_duration_ms"].quantile(0.25)
    pct_nw = (sub["pt_duration_ms"] < 0.35).mean() * 100
    fr_med = sub[fr_col].median() if HAS_FR else np.nan
    n_eval = sub["bio_consistent"].notna().sum()
    pct_ok = (sub["bio_consistent"] == True).sum() / n_eval * 100 if n_eval else np.nan
    prior  = REGION_PRIORS.get(region, (None,) * 4)
    summary_rows.append({
        "region": region, "n": n,
        "pt_median": pt_med, "pt_iqr": pt_iqr,
        "pct_narrow": pct_nw, "fr_median": fr_med,
        "pct_bio_ok": pct_ok,
        "prior_pt_min": prior[0], "prior_pt_max": prior[1],
        "prior_fr_min": prior[2], "prior_fr_max": prior[3],
    })

summary_df = pd.DataFrame(summary_rows).set_index("region")
with pd.option_context("display.float_format", "{:.2f}".format,
                       "display.max_columns", 20):
    print(summary_df[["n", "pt_median", "pt_iqr", "pct_narrow",
                       "fr_median", "pct_bio_ok"]].to_string())

print("\nNotes:")
print("  GPe: check fr_median is >20 Hz — if not, probe is still in STR/cortex")
print("  CP: fr_median should be <5 Hz for MSN-dominated recordings")

# ── CP failure mode breakdown ──────────────────────────────────────────────────
# SPNs are ~90–95% of striatal neurons anatomically, but in recordings ~75–85%
# of units are classified as putative MSNs because FSIs (~1–2% anatomically)
# have high FR and large amplitude and are overrepresented. A rate below ~65%
# suggests edge/boundary recording. Failure modes:
#   Fail PT only → narrow waveform → FSI or fast-spiking contaminant
#   Fail FR only → fast but broad   → TAN (cholinergic), ChAT+, or misassignment
#   Fail both    → narrow + fast    → genuine interneuron or noise
#   Pass both    → broad + low FR   → putative MSN candidate for 0h
if "CP" in merged["region_acronym"].values and HAS_FR:
    cp = merged[merged["region_acronym"] == "CP"].copy()
    n_cp   = len(cp)
    n_fail_pt_only = int(((cp["pt_ok"] == False) & (cp["fr_ok"] == True)).sum())
    n_fail_fr_only = int(((cp["pt_ok"] == True)  & (cp["fr_ok"] == False)).sum())
    n_fail_both    = int(((cp["pt_ok"] == False)  & (cp["fr_ok"] == False)).sum())
    n_pass_both    = int(((cp["pt_ok"] == True)   & (cp["fr_ok"] == True)).sum())
    # Strict MSN gate: Reinhold 2025 (halfwidth → PT proxy) + low FR
    n_msn_strict = int(((cp["pt_duration_ms"] >= 0.40) & (cp[fr_col] < 4.0)).sum())
    print(f"\n── CP failure breakdown (n={n_cp}) ──")
    print(f"  Fail PT only  (narrow → FSI-like)     : {n_fail_pt_only:3d}  ({n_fail_pt_only/n_cp*100:.1f}%)")
    print(f"  Fail FR only  (fast+broad → TAN/ChAT?): {n_fail_fr_only:3d}  ({n_fail_fr_only/n_cp*100:.1f}%)")
    print(f"  Fail both     (narrow+fast → noise?)  : {n_fail_both:3d}  ({n_fail_both/n_cp*100:.1f}%)")
    print(f"  Pass both     (broad+low FR → MSN-like): {n_pass_both:3d}  ({n_pass_both/n_cp*100:.1f}%)")
    print(f"  Strict MSN gate (PT≥0.40ms, FR<4Hz)   : {n_msn_strict:3d}  ({n_msn_strict/n_cp*100:.1f}%)")
    print(f"  Expected ~75–85% putative MSNs in a well-targeted STR recording.")


# ── [Plot 4] FR × PT scatter ──────────────────────────────────────────────────
print("\n[Plot 4] FR × PT scatter...")

if not HAS_FR:
    print("  Skipped — no firing rate column.")
else:
    plot_val = merged[
        merged["region_acronym"].isin(avail_val) &
        merged["pt_duration_ms"].notna() &
        merged[fr_col].notna()
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    for region in avail_val:
        d     = plot_val[plot_val["region_acronym"] == region]
        color = REGION_COLORS.get(region, "#888888")
        ax.scatter(d["pt_duration_ms"], d[fr_col],
                   color=color, alpha=0.55, s=22, linewidths=0, label=region)
        prior = REGION_PRIORS.get(region)
        if prior:
            pt_min, pt_max, fr_min, fr_max = prior
            x0 = pt_min if pt_min is not None else 0.0
            x1 = pt_max if pt_max is not None else 1.2
            y0 = fr_min if fr_min is not None else 0.0
            y1 = fr_max if fr_max is not None else 150.0
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                    color=color, linewidth=0.6, alpha=0.35, linestyle="--")

    ax.axvline(0.35, color="tomato", linestyle="--", linewidth=1, label="FS<0.35ms")
    ax.axvline(0.45, color="orange", linestyle="--", linewidth=1, label="RS>0.45ms")
    ax.set_xlabel("Peak-to-trough duration (ms)", fontsize=11)
    ax.set_ylabel(f"Firing rate (Hz)  [{fr_col}]", fontsize=11)
    ax.set_title("FR × PT duration — all regions", fontsize=12)
    ax.set_yscale("log"); ax.set_ylim(0.02, 200)
    ax.legend(fontsize=7, markerscale=1.8, ncol=2,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.2)

    colors_flag = merged[
        merged["region_acronym"].isin(avail_val) &
        merged["pt_duration_ms"].notna() &
        merged[fr_col].notna()
    ]["bio_consistent"].map({True: "#2196F3", False: "#F44336", None: "#BBBBBB"})

    ax = axes[1]
    ax.scatter(plot_val["pt_duration_ms"], plot_val[fr_col],
               c=colors_flag.values, alpha=0.55, s=22, linewidths=0)
    ax.axvline(0.35, color="tomato", linestyle="--", linewidth=1)
    ax.set_xlabel("Peak-to-trough duration (ms)", fontsize=11)
    ax.set_ylabel(f"Firing rate (Hz)  [{fr_col}]", fontsize=11)
    ax.set_title("Bio-consistent flag (blue=pass, red=fail)", fontsize=12)
    ax.set_yscale("log"); ax.set_ylim(0.02, 200)
    ax.grid(True, alpha=0.2)
    ax.legend(handles=[
        mpatches.Patch(color="#2196F3", label="Bio-consistent"),
        mpatches.Patch(color="#F44336", label="Flagged"),
        mpatches.Patch(color="#BBBBBB", label="No prior"),
    ], fontsize=9)

    fig.suptitle("FR × peak-to-trough duration — waveform validation",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fr_pt_scatter_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → fr_pt_scatter_overview.png")

    # Faceted by circuit group
    facet_groups = {
        "STR circuit\n(CP, GPe)":
            [r for r in avail_val if r in {"CP", "STR", "GPe", "GPi", "PAL"}],
        "Cortex\n(MOp, MOs, VISp)":
            [r for r in avail_val if r.startswith(("MOp", "MOs", "VISp", "VISl"))],
        "Thalamus\n(VAL, AV, CL, RT)":
            [r for r in avail_val if r in {"VAL", "AV", "CL", "LD", "LP", "RT"}],
        "Hippocampus\n(CA1, SUB)":
            [r for r in avail_val if r in {"CA1", "SUB"}],
    }
    facet_groups = {k: v for k, v in facet_groups.items() if v}

    n_facets = len(facet_groups)
    fig2, faxes = plt.subplots(1, n_facets, figsize=(4.5 * n_facets, 5), sharey=False)
    if n_facets == 1:
        faxes = [faxes]

    for fax, (group_name, group_regions) in zip(faxes, facet_groups.items()):
        for region in group_regions:
            d     = plot_val[plot_val["region_acronym"] == region]
            color = REGION_COLORS.get(region, "#888888")
            fax.scatter(d["pt_duration_ms"], d[fr_col],
                        color=color, alpha=0.6, s=28, linewidths=0, label=region)
            prior = REGION_PRIORS.get(region)
            if prior:
                pt_min, pt_max, fr_min, fr_max = prior
                x0 = pt_min if pt_min is not None else 0.0
                x1 = pt_max if pt_max is not None else 1.2
                y0 = fr_min if fr_min is not None else 0.02
                y1 = fr_max if fr_max is not None else 200.0
                fax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                         color=color, linewidth=1.2, alpha=0.5, linestyle="--")
        fax.axvline(0.35, color="tomato", linestyle="--", linewidth=0.8, alpha=0.7)
        fax.set_xlabel("PT duration (ms)", fontsize=10)
        fax.set_ylabel("Firing rate (Hz)", fontsize=10)
        fax.set_title(group_name, fontsize=11)
        fax.set_yscale("log"); fax.set_ylim(0.02, 200)
        fax.legend(fontsize=8, markerscale=1.5)
        fax.grid(True, alpha=0.2)

    fig2.suptitle("FR × PT duration by circuit group\n(dashed boxes = biological priors)",
                  fontsize=12, y=1.02)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "fr_pt_scatter_faceted.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved → fr_pt_scatter_faceted.png")


# ── [Plot 5] Per-animal bio-consistency heatmap ───────────────────────────────
print("\n[Plot 5] Per-animal heatmap...")

heatmap_animals = sorted(merged["mouse"].unique())
heatmap_regions = [r for r in avail_val if r in REGION_PRIORS]

heatmap_pct = np.full((len(heatmap_animals), len(heatmap_regions)), np.nan)
heatmap_n   = np.zeros((len(heatmap_animals), len(heatmap_regions)), dtype=int)

for i, mouse in enumerate(heatmap_animals):
    for j, region in enumerate(heatmap_regions):
        sub    = merged[(merged["mouse"] == mouse) &
                        (merged["region_acronym"] == region)]
        n_eval = sub["bio_consistent"].notna().sum()
        n_ok   = (sub["bio_consistent"] == True).sum()
        heatmap_n[i, j] = len(sub)
        if n_eval > 0:
            heatmap_pct[i, j] = n_ok / n_eval * 100

rg_cmap = LinearSegmentedColormap.from_list(
    "rg", ["#d62728", "#ffdd57", "#2ca02c"], N=256)

fig, ax = plt.subplots(figsize=(max(10, len(heatmap_regions) * 0.8),
                                max(5,  len(heatmap_animals) * 0.6)))
im = ax.imshow(heatmap_pct, cmap=rg_cmap, vmin=0, vmax=100,
               aspect="auto", interpolation="nearest")

for i in range(len(heatmap_animals)):
    for j in range(len(heatmap_regions)):
        n   = heatmap_n[i, j]
        pct = heatmap_pct[i, j]
        if n == 0:
            ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#aaaaaa")
        elif np.isnan(pct):
            ax.text(j, i, f"n={n}\n?", ha="center", va="center", fontsize=7)
        else:
            txt_color = "white" if pct / 100.0 < 0.55 else "black"
            ax.text(j, i, f"{pct:.0f}%\n(n={n})",
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color=txt_color)

ax.set_xticks(range(len(heatmap_regions)))
ax.set_xticklabels(heatmap_regions, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(heatmap_animals)))
ax.set_yticklabels(heatmap_animals, fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label(
    "% bio-consistent units", fontsize=9)
ax.set_title(
    "Bio-consistency by animal × region\n"
    "(% units passing FR + PT priors)",
    fontsize=12)
fig.tight_layout()
fig.savefig(PLOT_DIR / "per_animal_consistency_heatmap.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved → per_animal_consistency_heatmap.png")


# ── [Plot 6] Prior range strips ────────────────────────────────────────────────
print("\n[Plot 6] Prior range strips...")

n_regions = len(avail_val)
ncols = 2 if HAS_FR else 1
fig, axes = plt.subplots(n_regions, ncols,
                         figsize=(ncols * 5, n_regions * 1.4),
                         squeeze=False)

for row_i, region in enumerate(avail_val):
    sub   = merged[merged["region_acronym"] == region]
    prior = REGION_PRIORS.get(region)
    color = REGION_COLORS.get(region, "#888888")

    ax  = axes[row_i][0]
    pts = sub["pt_duration_ms"].dropna()
    if len(pts):
        ax.scatter(pts, np.random.uniform(-0.3, 0.3, len(pts)),
                   color=color, alpha=0.45, s=10, linewidths=0)
        ax.plot([pts.median()], [0], "|", color="black",
                markersize=14, markeredgewidth=2)
    if prior:
        pt_min = prior[0] if prior[0] is not None else 0.0
        pt_max = prior[1] if prior[1] is not None else 1.4
        ax.axvspan(pt_min, pt_max, color=color, alpha=0.15)
    ax.axvline(0.35, color="tomato", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.0, 1.4); ax.set_yticks([])
    ax.set_ylabel(region, fontsize=9, rotation=0, labelpad=55, va="center")
    if row_i == 0:
        ax.set_title("PT duration (ms)", fontsize=10)
    if row_i == n_regions - 1:
        ax.set_xlabel("ms", fontsize=9)

    if HAS_FR:
        ax2 = axes[row_i][1]
        frs = sub[fr_col].dropna()
        if len(frs):
            ax2.scatter(frs, np.random.uniform(-0.3, 0.3, len(frs)),
                        color=color, alpha=0.45, s=10, linewidths=0)
            ax2.plot([frs.median()], [0], "|", color="black",
                     markersize=14, markeredgewidth=2)
        if prior:
            fr_min = prior[2] if prior[2] is not None else 0.02
            fr_max = prior[3] if prior[3] is not None else 150.0
            ax2.axvspan(fr_min, fr_max, color=color, alpha=0.15)
        ax2.set_xscale("log"); ax2.set_xlim(0.02, 200); ax2.set_yticks([])
        if row_i == 0:
            ax2.set_title("Firing rate (Hz, log)", fontsize=10)
        if row_i == n_regions - 1:
            ax2.set_xlabel("Hz", fontsize=9)

fig.suptitle(
    "Observed PT duration and FR vs biological priors\n"
    "(shaded band = prior range, | = median, red dashes = FS threshold)",
    fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "prior_range_strips.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved → prior_range_strips.png")


# ── [Plot 7] Pass-rate summary bar ────────────────────────────────────────────
print("\n[Plot 7] Pass rate summary bar...")

pass_rates = []
for region in avail_val:
    sub    = merged[merged["region_acronym"] == region]
    n_eval = sub["bio_consistent"].notna().sum()
    n_ok   = (sub["bio_consistent"] == True).sum()
    pass_rates.append({
        "region":   region,
        "pct_ok":   n_ok / n_eval * 100 if n_eval else np.nan,
        "n_total":  len(sub),
    })
pass_df = pd.DataFrame(pass_rates).sort_values("pct_ok", ascending=True)

fig, ax = plt.subplots(figsize=(7, max(4, len(pass_df) * 0.45)))
colors_bar = [REGION_COLORS.get(r, "#888888") for r in pass_df["region"]]
bars = ax.barh(pass_df["region"], pass_df["pct_ok"],
               color=colors_bar, edgecolor="white", linewidth=0.5)
ax.axvline(50, color="gray",  linestyle="--", linewidth=1, label="50%")
ax.axvline(80, color="green", linestyle="--", linewidth=1, label="80%")
for bar, row in zip(bars, pass_df.itertuples()):
    if not np.isnan(row.pct_ok):
        ax.text(row.pct_ok + 1, bar.get_y() + bar.get_height() / 2,
                f"{row.pct_ok:.0f}%  (n={row.n_total})",
                va="center", fontsize=8)
ax.set_xlim(0, 115)
ax.set_xlabel("% units passing biological priors", fontsize=11)
ax.set_title("Bio-consistency pass rate by region\n(PT duration + firing rate vs priors)",
             fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(PLOT_DIR / "pass_rate_summary.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved → pass_rate_summary.png")


# ── [Plot 8] Flagged unit waveforms ───────────────────────────────────────────
print("\n[Plot 8] Flagged unit waveforms...")

flagged = merged[merged["bio_consistent"] == False].copy()
print(f"  {len(flagged):,} bio-inconsistent units")

if len(flagged) > 0:
    flag_regions = [r for r in avail_val if r in flagged["region_acronym"].values]
    ncols_f = 4
    nrows_f = int(np.ceil(len(flag_regions) / ncols_f))

    fig, axes = plt.subplots(nrows_f, ncols_f,
                             figsize=(ncols_f * 3.5, nrows_f * 3),
                             sharey=False)
    axes = np.array(axes).flatten()

    for ax_i, region in enumerate(flag_regions):
        ax    = axes[ax_i]
        idx_f = flagged["region_acronym"] == region
        idx_a = merged["region_acronym"] == region
        wv_f  = waveforms_norm[flagged[idx_f]["waveform_row"].values]
        wv_a  = waveforms_norm[merged[idx_a]["waveform_row"].values]
        color = REGION_COLORS.get(region, "#888888")

        if len(wv_a):
            mean_all = wv_a.mean(axis=0)
            std_all  = wv_a.std(axis=0)
            ax.fill_between(t_ms, mean_all - std_all, mean_all + std_all,
                            alpha=0.10, color="gray")
            ax.plot(t_ms, mean_all, color="gray", linewidth=1, alpha=0.4,
                    label=f"All (n={len(wv_a)})")
        if len(wv_f):
            mean_f = wv_f.mean(axis=0)
            std_f  = wv_f.std(axis=0)
            ax.fill_between(t_ms, mean_f - std_f, mean_f + std_f,
                            alpha=0.25, color=color)
            ax.plot(t_ms, mean_f, color=color, linewidth=1.8,
                    label=f"Flagged (n={len(wv_f)})")

        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        fr_str = (f", FR={flagged[idx_f][fr_col].median():.1f}Hz" if HAS_FR else "")
        pt_str = f"PT={flagged[idx_f]['pt_duration_ms'].median():.2f}ms"
        ax.set_title(f"{region}\n{pt_str}{fr_str}", fontsize=9,
                     fontweight="bold", color=color)
        ax.set_xlabel("Time (ms)", fontsize=7)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for ax_i in range(len(flag_regions), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("Waveforms of bio-inconsistent units (coloured) vs all units (gray)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "flagged_unit_waveforms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → flagged_unit_waveforms.png")


# ── Probe-context cross-check ──────────────────────────────────────────────────
# For each region with units, report the mix of probe types those units came
# from. Units appearing on a probe outside EXPECTED_PROBE_CONTEXT are flagged.
print("\n── Probe-context cross-check ──")
if "probe_region" in merged.columns:
    ctx_rows = []
    for region, expected in EXPECTED_PROBE_CONTEXT.items():
        sub = merged[merged["region_acronym"] == region]
        if not len(sub):
            continue
        counts = sub["probe_region"].value_counts().to_dict()
        unexpected = {p: n for p, n in counts.items() if p not in expected}
        ctx_rows.append({
            "region": region,
            "expected_probes": "/".join(sorted(expected)),
            "n_total": len(sub),
            "by_probe": ", ".join(f"{p}={n}" for p, n in counts.items()),
            "n_unexpected": int(sum(unexpected.values())),
        })
    ctx_df = pd.DataFrame(ctx_rows).sort_values("n_unexpected", ascending=False)
    print(ctx_df.to_string(index=False))
    ctx_df.to_csv(PLOT_DIR / "probe_context_check.csv", index=False)
    print(f"  Saved → probe_context_check.csv")
else:
    print("  probe_region column missing — skipping cross-check.")


# ── ccb boundary units ────────────────────────────────────────────────────────
# ccb (corpus callosum body) has no neuronal cell bodies, but units labelled
# ccb here are genuine recordings. They arise from two sources:
#   1. Daily CSF-mediated position variability — brain is suspended in spinal
#      fluid, so small day-to-day shifts in probe landing position occur even
#      when targeting the same depth. A single blended histological track cannot
#      capture this. Units near the CP/ccb or cortex/ccb border may be assigned
#      to ccb on any given session.
#   2. Atlas registration error — the Allen CCF ccb boundary doesn't perfectly
#      match every individual brain. A ≤200 µm atlas offset shifts real CP or
#      cortical neurons across the ccb label.
# 0f does NO boundary recovery — all ccb units from the atlas assignment are
# present here. 0h will use the nearest-region information saved below to
# recover units near CP or cortex using waveform + location criteria.
print(f"\n── ccb boundary units ──")
ccb_units = merged[merged["region_acronym"] == "ccb"].copy()
print(f"  ccb units (raw atlas assignment): {len(ccb_units)}")
if len(ccb_units):
    n_narrow_ccb = int((ccb_units["pt_duration_ms"] < 0.35).sum())
    n_broad_ccb  = int((ccb_units["pt_duration_ms"] >= 0.40).sum())
    n_mid_ccb    = len(ccb_units) - n_narrow_ccb - n_broad_ccb
    n_msn_like_ccb = (
        int(((ccb_units["pt_duration_ms"] >= 0.40) & (ccb_units[fr_col] < 4.0)).sum())
        if HAS_FR else "n/a"
    )
    print(f"  Waveform profile:")
    print(f"    Narrow (PT < 0.35ms, FSI/cortical-FS-like): {n_narrow_ccb}")
    print(f"    Mid    (PT 0.35–0.40ms, ambiguous)         : {n_mid_ccb}")
    print(f"    Broad  (PT ≥ 0.40ms, MSN/pyramidal-like)   : {n_broad_ccb}")
    if HAS_FR:
        print(f"    Strict MSN-like (PT≥0.40ms, FR<4Hz)        : {n_msn_like_ccb}")
    print(f"  By mouse:")
    print(ccb_units.groupby("mouse").size().rename("n_ccb").to_string())

    # Find nearest non-ccb neighbour on the same track to infer likely origin.
    # The nearest_region and gap_um columns are consumed by 0h's recovery rules.
    if "track_file" in ccb_units.columns and "dist_along_track_um" in ccb_units.columns:
        neighbor_rows = []
        for _, u in ccb_units.iterrows():
            track = u.get("track_file")
            dist  = u.get("dist_along_track_um", np.nan)
            same_track = merged[
                (merged["track_file"] == track) &
                (~merged["region_acronym"].isin({"ccb"})) &
                merged["dist_along_track_um"].notna()
            ] if not pd.isna(dist) else pd.DataFrame()
            if len(same_track):
                gaps        = (same_track["dist_along_track_um"] - dist).abs()
                nearest     = same_track.loc[gaps.idxmin()]
                gap_um      = round(float(gaps.min()), 0)
                near_region = nearest["region_acronym"]
            else:
                near_region, gap_um = "none", np.nan
            row = {
                # Unit keys needed by 0h for merge
                "mouse":            u["mouse"],
                "datetime":         u.get("datetime"),
                "insertion_number": u.get("insertion_number"),
                "paramset_idx":     u.get("paramset_idx"),
                "id":               u.get("id"),
                # Location
                "probe_region":     u.get("probe_region"),
                "dist_along_track_um": round(dist, 0) if not np.isnan(dist) else np.nan,
                "pt_duration_ms":   round(u["pt_duration_ms"], 3),
                # Nearest-region info for recovery rules
                "nearest_region":   near_region,
                "gap_um":           gap_um,
            }
            if HAS_FR:
                row[fr_col] = round(u[fr_col], 2)
            neighbor_rows.append(row)
        nb_df = pd.DataFrame(neighbor_rows)
        print(f"\n  Nearest non-ccb region on same track:")
        print(nb_df.to_string(index=False))
        print(f"\n  Nearest-region breakdown:")
        print(nb_df["nearest_region"].value_counts().to_string())
        nb_df.to_csv(PLOT_DIR / "ccb_boundary_units.csv", index=False)
    else:
        # Fallback: save unit keys only (no neighbor info)
        ccb_export_cols = ["mouse", "datetime", "insertion_number", "paramset_idx", "id",
                           "region_acronym", "depth_confidence", "probe_region",
                           "dist_along_track_um", "pt_duration_ms"]
        if HAS_FR:
            ccb_export_cols.append(fr_col)
        ccb_units[
            [c for c in ccb_export_cols if c in ccb_units.columns]
        ].sort_values(["mouse", "datetime"]).to_csv(PLOT_DIR / "ccb_boundary_units.csv", index=False)
    print(f"  Saved → ccb_boundary_units.csv")


# ── Export flagged unit list (split: GPe boundary candidates vs others) ───────
print("\nExporting flagged unit list...")
JOIN_KEYS = ["mouse", "datetime", "insertion_number", "paramset_idx", "id"]
export_cols = (
    JOIN_KEYS +
    ["region_acronym", "region_name", "depth_confidence", "probe_region",
     "pt_duration_ms", "trough_amp_uv",
     "pt_ok", "fr_ok", "bio_consistent"]
)
if HAS_FR:
    export_cols.insert(export_cols.index("pt_duration_ms"), fr_col)

flagged_all = merged[merged["bio_consistent"] == False][
    [c for c in export_cols if c in merged.columns]
].copy()

# GPe units flagged here are typically broad+slow → MSN-like → boundary
# candidates. They warrant their own file because the action is different
# (re-examine the CP/GPe boundary, not "drop the unit").
GPE_REGIONS = {"GPe", "GPi", "GP"}
gpe_boundary_mask = (
    flagged_all["region_acronym"].isin(GPE_REGIONS)
    & (flagged_all["pt_duration_ms"] >= 0.40)
)
if HAS_FR:
    gpe_boundary_mask &= (flagged_all[fr_col] <= 15.0)

gpe_boundary_export = flagged_all[gpe_boundary_mask].sort_values(["mouse", "datetime"])
flagged_other       = flagged_all[~gpe_boundary_mask].sort_values(["region_acronym", "mouse"])

flagged_other.to_csv(PLOT_DIR / "flagged_units.csv", index=False)
print(f"  Saved {len(flagged_other):,} flagged units (excl. GPe boundary) → flagged_units.csv")

gpe_boundary_export.to_csv(PLOT_DIR / "flagged_units_gpe_boundary.csv", index=False)
print(f"  Saved {len(gpe_boundary_export):,} GPe → CP boundary candidates "
      f"→ flagged_units_gpe_boundary.csv")


# ── Final summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)
print(f"Total units evaluated : {n_evaluated:,}")
print(f"Bio-consistent        : {n_pass:,}  ({n_pass/n_evaluated*100:.1f}%)")
print(f"Flagged               : {n_fail:,}  ({n_fail/n_evaluated*100:.1f}%)")
print()
print("Regions to action on:")
for _, row in pass_df.iterrows():
    pct = row["pct_ok"]
    if np.isnan(pct):
        continue
    if pct < 30:
        flag = "DROP / re-examine track"
    elif pct < 60:
        flag = "Suspicious — verify manually"
    else:
        flag = "Acceptable"
    print(f"  {row['region']:12s}  {pct:5.1f}%  {flag}")


# ── RT broad-waveform boundary check ──────────────────────────────────────────
# RT (reticular thalamus) requires narrow waveforms (PT < 0.35 ms) — it is
# entirely GABAergic fast-spiking. Any RT unit with PT ≥ 0.35 ms fails the
# prior and is a candidate misassignment: atlas boundary error can place a
# CP-edge or VAL-edge neuron into RT. We check whether these units sit within
# 300 µm of the CP depth range on their track, making reclassification plausible.
print(f"\n── RT broad-waveform boundary check ──")
rt_broad = merged[
    (merged["region_acronym"] == "RT") &
    (merged["pt_duration_ms"] >= 0.35)
].copy()
print(f"  RT units with PT ≥ 0.35 ms (violates narrow prior): {len(rt_broad)}")

if len(rt_broad) and "dist_along_track_um" in merged.columns and "track_file" in merged.columns:
    # Per-(mouse, track) CP depth range (from CP-assigned units on the same probe).
    # Keying on track_file alone collapses across mice when track names repeat.
    cp_depth_by_track = (
        merged[merged["region_acronym"] == "CP"]
        .groupby(["mouse", "track_file"])["dist_along_track_um"]
        .agg(cp_min="min", cp_max="max")
    )

    rows = []
    for _, u in rt_broad.iterrows():
        track = u.get("track_file")
        dist  = u.get("dist_along_track_um", np.nan)
        key = (u["mouse"], track)
        if key in cp_depth_by_track.index and not np.isnan(dist):
            cp_min = cp_depth_by_track.loc[key, "cp_min"]
            cp_max = cp_depth_by_track.loc[key, "cp_max"]
            # Gap = 0 if unit is within the CP range, else distance to nearest edge
            if dist < cp_min:
                gap = cp_min - dist
            elif dist > cp_max:
                gap = dist - cp_max
            else:
                gap = 0.0
            near_cp = gap <= 300
            cp_range_str = f"{cp_min:.0f}–{cp_max:.0f}"
        else:
            cp_min = cp_max = gap = np.nan
            near_cp = False
            cp_range_str = "no CP on track"
        row = {
            # Unit keys needed by 0h for merge
            "mouse":            u["mouse"],
            "datetime":         u.get("datetime"),
            "insertion_number": u.get("insertion_number"),
            "paramset_idx":     u.get("paramset_idx"),
            "id":               u.get("id"),
            # Location
            "track_file":       track,
            "dist_along_track_um": round(dist, 0) if not np.isnan(dist) else np.nan,
            "pt_duration_ms":   round(u["pt_duration_ms"], 3),
            "cp_range_um":      cp_range_str,
            "gap_to_cp_um":     round(gap, 0) if not np.isnan(gap) else np.nan,
            "near_cp_300um":    near_cp,
        }
        if HAS_FR:
            row[fr_col] = round(u[fr_col], 2)
        rows.append(row)

    rt_check_df = pd.DataFrame(rows)
    n_near = int(rt_check_df["near_cp_300um"].sum())
    n_no_cp = (rt_check_df["cp_range_um"] == "no CP on track").sum()
    print(f"  Within 300 µm of CP on same track : {n_near}")
    print(f"  No CP units on same track          : {n_no_cp}")
    print(f"  Far from CP (>300 µm, likely thal.): {len(rt_broad) - n_near - n_no_cp}")

    if n_near:
        print(f"\n  Candidate CP-boundary misassignments (within 300 µm of CP):")
        print(rt_check_df[rt_check_df["near_cp_300um"]].to_string(index=False))
    not_near = rt_check_df[~rt_check_df["near_cp_300um"]]
    if len(not_near):
        print(f"\n  Far from CP or no CP on track (likely RT boundary / atlas edge):")
        print(not_near.to_string(index=False))

    rt_check_df.to_csv(PLOT_DIR / "rt_broad_boundary_check.csv", index=False)
    print(f"  Saved → rt_broad_boundary_check.csv")
elif len(rt_broad):
    print("  dist_along_track_um or track_file column missing — skipping distance check.")

print(f"\nOutputs saved to: {PLOT_DIR}")
for fname in [
    "waveform_by_region.png", "waveform_metrics_scatter.png", "waveform_heatmap.png",
    "fr_pt_scatter_overview.png", "fr_pt_scatter_faceted.png",
    "per_animal_consistency_heatmap.png", "prior_range_strips.png",
    "pass_rate_summary.png", "flagged_unit_waveforms.png",
    "flagged_units.csv", "flagged_units_gpe_boundary.csv",
    "ccb_boundary_units.csv", "rt_broad_boundary_check.csv",
    "probe_context_check.csv",
]:
    print(f"  {fname}")
