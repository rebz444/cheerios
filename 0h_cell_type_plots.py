"""
0h_cell_type_plots.py
======================
Generate diagnostic plots for committee meeting after running 0g_cell_type_relabeling_v2.py

Plots:
  1. Units per region BEFORE correction (aggregated + detailed)
  2. Units per region AFTER correction (aggregated + detailed)
  3. Cell type counts by probe type
  4. Cell type × region cross-tabulation heatmap
  5. Waveforms by cell type (mean ± SD)
  6. Cortex-STR boundary validation (depth distributions)
  7. FR × PT scatter colored by cell type
  8. Per-animal breakdown

Inputs:
  RZ_unit_properties_final.csv (output of 0g_v2)
  RZ_unit_templates.npz (for waveforms)

Outputs → DATA_DIR/location_matching/cell_type_v2/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
class Paths:
    DATA_DIR = Path("./data")
    LOGS_DIR = Path("./logs")

try:
    import paths as p
except ImportError:
    p = Paths()

PLOT_DIR = p.DATA_DIR / "location_matching" / "cell_type_v2"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Region sets (same as 0g_v2) ───────────────────────────────────────────────
STR_REGIONS = {
    "CP", "STR", "STRd", "STRv", "ACB", "OT",
    "PAL", "SI", "LSX", "CEA", "LA", "BLA", "EP", "MEA"
}

CORTEX_PREFIXES = ["MOp", "MOs", "VIS", "SSp", "SSs", "ACA", "PL", "ILA", "ORB", "AI", "RSP"]

# ── Colors ────────────────────────────────────────────────────────────────────
CELL_TYPE_COLORS = {
    "MSN": "#2166AC",
    "FSI": "#D6604D",
    "RS": "#4393C3",
    "TAN": "#8C564B",
    "high_FR": "#762A83",
    "ambiguous": "#AAAAAA",
}

REGION_GROUP_COLORS = {
    "Striatum": "#2166AC",
    "Motor cortex": "#D6604D",
    "Visual cortex": "#9467BD",
    "Thalamus": "#FF7F0E",
    "Pallidum": "#4393C3",
    "Hippocampus": "#8C564B",
    "Other": "#AAAAAA",
    "Excluded": "#CCCCCC",
}


def get_region_group(region):
    """Map detailed region to broad category."""
    r = str(region)
    if r in STR_REGIONS:
        return "Striatum"
    if r in {"GPe", "GPi", "GP"}:
        return "Pallidum"
    if any(r.startswith(p) for p in ["MOp", "MOs"]):
        return "Motor cortex"
    if r.startswith("VIS"):
        return "Visual cortex"
    if r in {"VAL", "AV", "CL", "LD", "LP", "VPL", "VPM", "PO", "MD", "RT", "TH"}:
        return "Thalamus"
    if r in {"CA1", "CA2", "CA3", "SUB", "DG", "HIP", "HPF"}:
        return "Hippocampus"
    return "Other"


def is_in_cortex(region):
    r = str(region)
    return any(r.startswith(p) for p in CORTEX_PREFIXES)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("0h_cell_type_plots.py")
print("=" * 70)

csv_path = p.LOGS_DIR / "RZ_unit_properties_final.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Run 0g_cell_type_relabeling_v2.py first. Missing: {csv_path}")

print(f"\nLoading: {csv_path.name}")
df = pd.read_csv(csv_path)
n_total = len(df)
print(f"  {n_total:,} units")

# Ensure columns exist
required_cols = ["corrected_region", "cell_type", "region_acronym", "probe_region"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}. Run 0g_cell_type_relabeling_v2.py first.")

# Add region groups if not present
if "region_group" not in df.columns:
    df["region_group"] = df["corrected_region"].apply(get_region_group)
df["original_region_group"] = df["region_acronym"].apply(get_region_group)

str_probe = df["probe_region"] == "str"
v1_probe = df["probe_region"] == "v1"

# Detect FR column
fr_col = next((c for c in ["firing_rate", "mean_firing_rate", "fr"] if c in df.columns), None)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Aggregated region counts BEFORE and AFTER correction
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Plot 1] Aggregated region counts before/after correction...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row, (probe, probe_name) in enumerate([("str", "STR Probe"), ("v1", "V1 Probe")]):
    probe_mask = df["probe_region"] == probe
    subset = df[probe_mask]
    
    # Before (original)
    ax = axes[row, 0]
    counts_before = subset["original_region_group"].value_counts()
    order = ["Striatum", "Motor cortex", "Visual cortex", "Thalamus", "Pallidum", "Hippocampus", "Other"]
    counts_before = counts_before.reindex([r for r in order if r in counts_before.index])
    colors = [REGION_GROUP_COLORS.get(r, "#888") for r in counts_before.index]
    bars = ax.barh(counts_before.index, counts_before.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts_before.values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=9)
    ax.set_xlabel("Unit count")
    ax.set_title(f"{probe_name} — BEFORE correction\n(n={len(subset):,})")
    ax.invert_yaxis()
    
    # After (corrected)
    ax = axes[row, 1]
    counts_after = subset["region_group"].value_counts()
    counts_after = counts_after.reindex([r for r in order if r in counts_after.index])
    colors = [REGION_GROUP_COLORS.get(r, "#888") for r in counts_after.index]
    bars = ax.barh(counts_after.index, counts_after.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts_after.values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=9)
    ax.set_xlabel("Unit count")
    ax.set_title(f"{probe_name} — AFTER correction\n(n={len(subset):,})")
    ax.invert_yaxis()

fig.suptitle("Units per Region Group — Before vs After Correction", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "1_region_counts_aggregated.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> 1_region_counts_aggregated.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Detailed region counts (top 15 per probe)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Plot 2] Detailed region counts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 10))

for ax, (probe, probe_name) in zip(axes, [("str", "STR Probe"), ("v1", "V1 Probe")]):
    probe_mask = df["probe_region"] == probe
    subset = df[probe_mask]
    
    counts = subset["corrected_region"].value_counts().head(20).sort_values()
    colors = [REGION_GROUP_COLORS.get(get_region_group(r), "#888") for r in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.3)
    
    for bar, val in zip(bars, counts.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=8)
    
    ax.set_xlabel("Unit count")
    ax.set_title(f"{probe_name} — Detailed Regions\n(top 20, n={len(subset):,})")

# Legend
patches = [mpatches.Patch(facecolor=c, label=r) for r, c in REGION_GROUP_COLORS.items() if r != "Excluded"]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Units per Brain Region (Corrected)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "2_region_counts_detailed.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> 2_region_counts_detailed.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Cell type counts by probe
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Plot 3] Cell type counts by probe...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# STR probe
ax = axes[0]
str_df = df[str_probe]
str_ct = str_df["cell_type"].value_counts()
ct_order = ["MSN", "FSI", "TAN", "RS", "high_FR", "ambiguous"]
str_ct = str_ct.reindex([c for c in ct_order if c in str_ct.index])
colors = [CELL_TYPE_COLORS.get(ct, "#888") for ct in str_ct.index]
bars = ax.bar(str_ct.index, str_ct.values, color=colors, edgecolor="white")
for bar, val in zip(bars, str_ct.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, str(val), ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Unit count")
ax.set_title(f"STR Probe Cell Types\n(n={len(str_df):,})")

# V1 probe
ax = axes[1]
v1_df = df[v1_probe]
v1_ct = v1_df["cell_type"].value_counts()
v1_ct = v1_ct.reindex([c for c in ct_order if c in v1_ct.index])
colors = [CELL_TYPE_COLORS.get(ct, "#888") for ct in v1_ct.index]
bars = ax.bar(v1_ct.index, v1_ct.values, color=colors, edgecolor="white")
for bar, val in zip(bars, v1_ct.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, str(val), ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Unit count")
ax.set_title(f"V1 Probe Cell Types\n(n={len(v1_df):,})")

# Proportional comparison
ax = axes[2]
str_total = len(str_df)
v1_total = len(v1_df)

all_cts = ["MSN", "RS", "FSI", "TAN", "high_FR", "ambiguous"]
str_props = [str_ct.get(ct, 0) / str_total * 100 if str_total > 0 else 0 for ct in all_cts]
v1_props = [v1_ct.get(ct, 0) / v1_total * 100 if v1_total > 0 else 0 for ct in all_cts]

x = np.arange(2)
bottom = np.zeros(2)
for i, ct in enumerate(all_cts):
    color = CELL_TYPE_COLORS.get(ct, "#888")
    vals = [str_props[i], v1_props[i]]
    ax.bar(x, vals, bottom=bottom, color=color, width=0.5, edgecolor="white", label=ct)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 4:
            ax.text(j, b + v/2, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    bottom = [b + v for b, v in zip(bottom, vals)]

ax.set_xticks(x)
ax.set_xticklabels(["STR Probe", "V1 Probe"])
ax.set_ylabel("Percentage")
ax.set_ylim(0, 100)
ax.set_title("Cell Type Proportions")
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

fig.suptitle("Cell Type Distribution by Probe", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "3_cell_type_counts.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> 3_cell_type_counts.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Cell type × Region heatmap (STR probe)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Plot 4] Cell type × Region heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, (probe, title) in zip(axes, [("str", "STR Probe"), ("v1", "V1 Probe")]):
    probe_mask = df["probe_region"] == probe
    subset = df[probe_mask]
    
    # Get top regions
    top_regions = subset["corrected_region"].value_counts().head(12).index.tolist()
    subset_top = subset[subset["corrected_region"].isin(top_regions)]
    
    # Create crosstab
    ct = pd.crosstab(subset_top["corrected_region"], subset_top["cell_type"])
    ct = ct.reindex(columns=[c for c in ct_order if c in ct.columns], fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    
    # Plot heatmap
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title(f"{title}\nRegion × Cell Type (top 12 regions)")
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Region")

fig.suptitle("Region × Cell Type Cross-tabulation", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "4_region_celltype_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> 4_region_celltype_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5: FR × PT scatter by cell type
# ══════════════════════════════════════════════════════════════════════════════
if fr_col and "pt_duration_ms" in df.columns:
    print("\n[Plot 5] FR × PT scatter by cell type...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (probe, title) in zip(axes, [("str", "STR Probe"), ("v1", "V1 Probe")]):
        probe_mask = df["probe_region"] == probe
        subset = df[probe_mask].dropna(subset=[fr_col, "pt_duration_ms"])
        
        for ct in ["MSN", "RS", "FSI", "TAN", "high_FR", "ambiguous"]:
            mask = subset["cell_type"] == ct
            if mask.sum() > 0:
                ax.scatter(subset.loc[mask, "pt_duration_ms"], 
                          subset.loc[mask, fr_col],
                          c=CELL_TYPE_COLORS.get(ct, "#888"),
                          label=f"{ct} (n={mask.sum()})",
                          alpha=0.5, s=15, edgecolors="none")
        
        # Reference lines
        ax.axvline(0.35, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="FSI threshold")
        ax.axvline(0.40, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Broad threshold")
        ax.axhline(15, color="gray", linestyle="-.", linewidth=1, alpha=0.5, label="Low FR threshold")
        
        ax.set_xlabel("PT duration (ms)")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title(f"{title}")
        ax.set_xlim(0.1, 0.9)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 200)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)
    
    fig.suptitle("Firing Rate × PT Duration by Cell Type", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "5_fr_pt_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> 5_fr_pt_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Depth distribution by region (cortex vs STR boundary check)
# ══════════════════════════════════════════════════════════════════════════════
if "peak_channel_depth" in df.columns:
    print("\n[Plot 6] Depth distribution by region...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # STR probe only
    str_df = df[str_probe].copy()
    
    # Left: depth distribution by region group
    ax = axes[0]
    for rg in ["Motor cortex", "Striatum", "Thalamus"]:
        mask = str_df["region_group"] == rg
        if mask.sum() > 5:
            depths = str_df.loc[mask, "peak_channel_depth"].dropna()
            ax.hist(depths, bins=30, alpha=0.5, 
                   label=f"{rg} (n={len(depths)})",
                   color=REGION_GROUP_COLORS.get(rg, "#888"))
    
    ax.axvline(1500, color="red", linestyle="--", linewidth=2, label="1500 µm (old cutoff)")
    ax.set_xlabel("Peak channel depth (µm from tip)")
    ax.set_ylabel("Unit count")
    ax.set_title("STR Probe: Depth by Region Group")
    ax.legend()
    
    # Right: depth by cell type (to verify MSN vs RS separation)
    ax = axes[1]
    for ct in ["MSN", "RS", "FSI"]:
        mask = (str_df["cell_type"] == ct)
        if mask.sum() > 5:
            depths = str_df.loc[mask, "peak_channel_depth"].dropna()
            ax.hist(depths, bins=30, alpha=0.5,
                   label=f"{ct} (n={len(depths)})",
                   color=CELL_TYPE_COLORS.get(ct, "#888"))
    
    ax.axvline(1500, color="red", linestyle="--", linewidth=2, label="1500 µm")
    ax.set_xlabel("Peak channel depth (µm from tip)")
    ax.set_ylabel("Unit count")
    ax.set_title("STR Probe: Depth by Cell Type")
    ax.legend()
    
    fig.suptitle("Depth Distributions — Cortex-STR Boundary Validation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "6_depth_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> 6_depth_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7: Per-animal breakdown
# ══════════════════════════════════════════════════════════════════════════════
if "mouse" in df.columns:
    print("\n[Plot 7] Per-animal breakdown...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Get animals sorted
    animals = sorted(df["mouse"].unique())
    
    # Top: STR units by animal (stacked by cell type)
    ax = axes[0]
    str_df = df[str_probe & df["corrected_region"].isin(STR_REGIONS)]
    
    x = np.arange(len(animals))
    width = 0.7
    bottom = np.zeros(len(animals))
    
    for ct in ["MSN", "FSI", "TAN", "high_FR", "ambiguous"]:
        counts = [str_df[(str_df["mouse"] == m) & (str_df["cell_type"] == ct)].shape[0] for m in animals]
        ax.bar(x, counts, width, bottom=bottom, label=ct, color=CELL_TYPE_COLORS.get(ct, "#888"), edgecolor="white", linewidth=0.3)
        bottom = bottom + np.array(counts)
    
    ax.set_xticks(x)
    ax.set_xticklabels(animals, rotation=45, ha="right")
    ax.set_ylabel("Unit count")
    ax.set_title("STR Units per Animal (by cell type)")
    ax.legend(loc="upper right")
    
    # Bottom: Total STR vs cortex on STR probe
    ax = axes[1]
    
    str_counts = [df[str_probe & df["corrected_region"].isin(STR_REGIONS) & (df["mouse"] == m)].shape[0] for m in animals]
    ctx_counts = [df[str_probe & df["corrected_region"].apply(is_in_cortex) & (df["mouse"] == m)].shape[0] for m in animals]
    
    ax.bar(x - 0.2, str_counts, 0.35, label="In STR", color="#2166AC", edgecolor="white")
    ax.bar(x + 0.2, ctx_counts, 0.35, label="In Cortex", color="#D6604D", edgecolor="white")
    
    ax.set_xticks(x)
    ax.set_xticklabels(animals, rotation=45, ha="right")
    ax.set_ylabel("Unit count")
    ax.set_title("STR Probe: Units in STR vs Cortex per Animal")
    ax.legend()
    
    fig.suptitle("Per-Animal Unit Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "7_per_animal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> 7_per_animal.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8: Waveforms by cell type (placeholder - needs actual waveform data)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Plot 8] Waveforms by cell type...")

# Check if waveform template file exists
template_path = p.LOGS_DIR / "RZ_unit_templates.npz"
if template_path.exists():
    print(f"  Loading waveforms from {template_path}")
    templates = np.load(template_path)
    waveforms = templates.get("templates", templates.get("waveforms", None))
    
    if waveforms is not None and "template_idx" in df.columns:
        n_samples = waveforms.shape[1] if waveforms.ndim > 1 else 82
        t_ms = np.linspace(-0.5, 1.5, n_samples)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for ax, ct in zip(axes, ["MSN", "FSI", "RS", "TAN", "high_FR", "ambiguous"]):
            mask = (df["cell_type"] == ct) & df["template_idx"].notna()
            if mask.sum() > 0:
                indices = df.loc[mask, "template_idx"].astype(int).values
                valid_indices = indices[indices < len(waveforms)]
                
                if len(valid_indices) > 0:
                    wv = waveforms[valid_indices]
                    
                    # Normalize each waveform
                    wv_norm = wv / np.abs(wv.min(axis=1, keepdims=True) + 1e-10)
                    
                    mean_wv = wv_norm.mean(axis=0)
                    std_wv = wv_norm.std(axis=0)
                    
                    ax.fill_between(t_ms, mean_wv - std_wv, mean_wv + std_wv,
                                   alpha=0.3, color=CELL_TYPE_COLORS.get(ct, "#888"))
                    ax.plot(t_ms, mean_wv, linewidth=2, color=CELL_TYPE_COLORS.get(ct, "#888"))
                    
                    # Plot some individual traces
                    for i in range(min(30, len(wv_norm))):
                        ax.plot(t_ms, wv_norm[i], alpha=0.1, linewidth=0.5,
                               color=CELL_TYPE_COLORS.get(ct, "#888"))
            
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
            ax.set_title(f"{ct} (n={mask.sum()})", fontsize=11, 
                        color=CELL_TYPE_COLORS.get(ct, "#888"), fontweight="bold")
            ax.set_xlabel("Time (ms)")
            ax.set_xlim(-0.3, 1.2)
        
        axes[0].set_ylabel("Normalized amplitude")
        axes[3].set_ylabel("Normalized amplitude")
        
        fig.suptitle("Mean Waveforms by Cell Type (± 1 SD)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "8_waveforms_by_celltype.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> 8_waveforms_by_celltype.png")
    else:
        print(f"  Waveform data not in expected format, skipping plot 8")
else:
    print(f"  No waveform file found at {template_path}, skipping plot 8")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PLOTS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {PLOT_DIR}")
print(f"\nGenerated plots:")
for i, name in enumerate([
    "region_counts_aggregated",
    "region_counts_detailed", 
    "cell_type_counts",
    "region_celltype_heatmap",
    "fr_pt_scatter",
    "depth_distributions",
    "per_animal",
    "waveforms_by_celltype",
], start=1):
    fpath = PLOT_DIR / f"{i}_{name}.png"
    status = "✓" if fpath.exists() else "✗"
    print(f"  {status} {i}_{name}.png")

print("\n" + "=" * 70)
