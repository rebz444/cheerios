"""
0h_msn_waveforms.py
====================
Plot MSN waveforms from depth-based cell type assignment (0g_cell_type_relabeling_v3.py).

Panels:
  1. MSN individual traces + mean ± SD (trough-normalised)
  2. Mean waveforms by cell type (MSN, FSI, RS, TAN overlaid)
  3. PT duration distributions by cell type
  4. FR distribution for MSN vs other cell types

Inputs:
  LOGS_DIR / RZ_unit_properties_final.csv  (output of 0g_cell_type_relabeling_v3.py)
  LOGS_DIR / RZ_unit_templates.npz

Outputs → DATA_DIR / location_matching / cell_type_v3 / msn_waveforms.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import paths as p
from utils import load_waveform_metrics

# ── Output directory ───────────────────────────────────────────────────────────
PLOT_DIR = p.DATA_DIR / "location_matching" / "cell_type_v3"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CELL_TYPE_COLORS = {
    "MSN":      "#2166AC",
    "FSI":      "#D6604D",
    "RS":       "#4393C3",
    "TAN":      "#8C564B",
    "high_FR":  "#762A83",
    "ambiguous":"#AAAAAA",
}

MAX_INDIVIDUAL_TRACES = 50   # cap individual traces per cell type

print("=" * 70)
print("0h_msn_waveforms.py")
print("  MSN waveform plots (depth-based labeling)")
print("=" * 70)

# ── Load unit properties ───────────────────────────────────────────────────────
csv_path = p.LOGS_DIR / "RZ_unit_properties_final.csv"
if not csv_path.exists():
    raise FileNotFoundError(
        f"Run 0g_cell_type_relabeling_v3.py first. Missing: {csv_path}"
    )

print(f"\nLoading: {csv_path.name}")
df = pd.read_csv(csv_path)

# Ensure join-key types match what load_waveform_metrics expects
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"])
if "insertion_number" in df.columns:
    df["insertion_number"] = df["insertion_number"].astype(int)
if "id" in df.columns:
    df["id"] = df["id"].astype(int)
if "paramset_idx" in df.columns:
    df["paramset_idx"] = df["paramset_idx"].astype(int)

if "cell_type" not in df.columns:
    raise ValueError("Missing 'cell_type' column — run 0g_cell_type_relabeling_v3.py first.")

print(f"  {len(df):,} units loaded")
print(f"  MSN count: {(df['cell_type'] == 'MSN').sum()}")

# ── Detect firing rate column ──────────────────────────────────────────────────
fr_col = next(
    (c for c in ["firing_rate", "mean_firing_rate", "fr"] if c in df.columns), None
)

# ── Load waveforms ─────────────────────────────────────────────────────────────
print("\nLoading waveform templates...")
merged, waveforms_raw, waveforms_norm, t_ms = load_waveform_metrics(df)
merged["_wf_row"] = np.arange(len(merged))

print(f"  Waveform time axis: {t_ms[0]:.2f} – {t_ms[-1]:.2f} ms  ({len(t_ms)} samples)")

n_msn = (merged["cell_type"] == "MSN").sum()
print(f"  MSN units with waveforms: {n_msn}")
if n_msn == 0:
    raise RuntimeError("No MSN units found in the merged waveform dataset.")


# ── Helper: get waveforms for a cell type ─────────────────────────────────────
def get_waveforms(ct):
    mask = merged["cell_type"] == ct
    rows = merged.loc[mask, "_wf_row"].values
    return waveforms_norm[rows], mask.sum()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — MSN traces + mean by cell type (side by side)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Plot 1] MSN individual traces + mean waveforms by cell type...")

wv_msn, n_msn_wf = get_waveforms("MSN")
mean_msn = wv_msn.mean(axis=0)
sd_msn   = wv_msn.std(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Left: MSN individual traces + mean ± SD ───────────────────────────────────
ax = axes[0]
n_plot = min(MAX_INDIVIDUAL_TRACES, len(wv_msn))
rng = np.random.default_rng(seed=0)
idx = rng.choice(len(wv_msn), size=n_plot, replace=False)
for i in idx:
    ax.plot(t_ms, wv_msn[i], color=CELL_TYPE_COLORS["MSN"], alpha=0.08, linewidth=0.7)

ax.fill_between(t_ms, mean_msn - sd_msn, mean_msn + sd_msn,
                color=CELL_TYPE_COLORS["MSN"], alpha=0.25, label="± 1 SD")
ax.plot(t_ms, mean_msn, color=CELL_TYPE_COLORS["MSN"], linewidth=2.5,
        label=f"Mean (n={n_msn_wf})")

ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Normalised amplitude (trough = −1)")
ax.set_title("MSN Waveforms — Individual Traces + Mean ± SD")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

# ── Right: Mean waveforms by cell type (overlaid) ─────────────────────────────
ax = axes[1]
cell_types_to_plot = ["MSN", "FSI", "RS", "TAN"]

for ct in cell_types_to_plot:
    wv, n = get_waveforms(ct)
    if n == 0:
        continue
    mean_wv = wv.mean(axis=0)
    sd_wv   = wv.std(axis=0)
    color   = CELL_TYPE_COLORS[ct]
    ax.fill_between(t_ms, mean_wv - sd_wv, mean_wv + sd_wv, color=color, alpha=0.15)
    ax.plot(t_ms, mean_wv, color=color, linewidth=2, label=f"{ct} (n={n})")

ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax.axvline(0.35, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="PT=0.35 ms (FSI)")
ax.axvline(0.40, color="gray", linestyle="-.", linewidth=1, alpha=0.6, label="PT=0.40 ms (MSN)")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Normalised amplitude (trough = −1)")
ax.set_title("Mean Waveforms by Cell Type (± 1 SD)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.15)

fig.tight_layout()
fig.savefig(PLOT_DIR / "msn_waveforms.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> msn_waveforms.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — PT duration + FR distributions (MSN highlighted)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Plot 3] PT duration and FR distributions...")

has_fr = fr_col is not None and fr_col in merged.columns
n_cols = 2 if has_fr else 1
fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
if n_cols == 1:
    axes = [axes]

# Panel A: PT duration histogram
ax = axes[0]
for ct in ["ambiguous", "high_FR", "TAN", "RS", "FSI", "MSN"]:
    mask = merged["cell_type"] == ct
    if mask.sum() < 5:
        continue
    pt_vals = merged.loc[mask, "pt_duration_ms"].dropna()
    ax.hist(pt_vals, bins=40, range=(0.1, 0.9),
            color=CELL_TYPE_COLORS.get(ct, "#888"),
            alpha=0.6 if ct != "MSN" else 0.85,
            label=f"{ct} (n={len(pt_vals)})",
            density=True,
            linewidth=0.5, edgecolor="white")

ax.axvline(0.35, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="0.35 ms (FSI)")
ax.axvline(0.40, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="0.40 ms (MSN)")
ax.set_xlabel("PT duration (ms)")
ax.set_ylabel("Density")
ax.set_title("PT Duration Distribution by Cell Type")
ax.legend(fontsize=7, loc="upper left")
ax.grid(True, alpha=0.15)

# Panel B: Firing rate (log-scale)
if has_fr:
    ax = axes[1]
    fr_bins = np.logspace(np.log10(0.05), np.log10(200), 45)
    for ct in ["ambiguous", "high_FR", "TAN", "RS", "FSI", "MSN"]:
        mask = merged["cell_type"] == ct
        if mask.sum() < 5:
            continue
        fr_vals = merged.loc[mask, fr_col].dropna()
        fr_vals = fr_vals[fr_vals > 0]
        ax.hist(fr_vals, bins=fr_bins,
                color=CELL_TYPE_COLORS.get(ct, "#888"),
                alpha=0.6 if ct != "MSN" else 0.85,
                label=f"{ct} (n={len(fr_vals)})",
                density=True,
                linewidth=0.5, edgecolor="white")

    ax.axvline(15, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="15 Hz (MSN)")
    ax.set_xscale("log")
    ax.set_xlabel("Firing rate (Hz)")
    ax.set_ylabel("Density")
    ax.set_title("Firing Rate Distribution by Cell Type")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.15)

fig.tight_layout()
fig.savefig(PLOT_DIR / "msn_feature_distributions.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> msn_feature_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"\nOutput directory: {PLOT_DIR}")
print("\nGenerated plots:")
for fname in [
    "msn_waveforms.png",
    "msn_feature_distributions.png",
]:
    status = "ok" if (PLOT_DIR / fname).exists() else "MISSING"
    print(f"  [{status}] {fname}")

# Print MSN waveform stats
msn_pt = merged.loc[merged["cell_type"] == "MSN", "pt_duration_ms"].dropna()
print(f"\nMSN PT duration: {msn_pt.mean():.3f} ± {msn_pt.std():.3f} ms  "
      f"(median {msn_pt.median():.3f})")
if has_fr:
    msn_fr = merged.loc[merged["cell_type"] == "MSN", fr_col].dropna()
    print(f"MSN firing rate: {msn_fr.mean():.2f} ± {msn_fr.std():.2f} Hz  "
          f"(median {msn_fr.median():.2f})")
