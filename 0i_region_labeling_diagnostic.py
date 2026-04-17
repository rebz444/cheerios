"""
0i_region_labeling_diagnostic.py
=================================
Diagnose whether histology-based region labeling makes sense.

Key questions:
1. BEFORE any correction, what regions does the track say units are in?
2. Does firing rate vs depth show a clear cortex→STR transition?
3. Is the histology-estimated boundary consistent with physiology?

The cortex→STR transition should show:
  - Cortex (upper): Higher FR (5-20 Hz average), mix of RS and FSI
  - STR (lower): Very low FR (0.5-5 Hz), mostly MSN-like waveforms
  - Transition should be sharp, not gradual

Analysis is done PER INSERTION (not per session) since each insertion has:
  - Its own track file (l_str.csv, r_str.csv, etc.)
  - Its own recorded insertion depth
  - Its own shrinkage-corrected mapping

Outputs:
  diagnostic_original_regions.png     : Region distribution BEFORE any correction
  diagnostic_fr_vs_depth.png          : Firing rate vs depth (all STR probe units)
  diagnostic_fr_vs_depth_insertions.png : Per-insertion FR vs depth profiles
  diagnostic_boundary_comparison.png  : Physiological vs histological boundary
  diagnostic_insertion_summary.csv    : Per-insertion statistics
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
class Paths:
    DATA_DIR = Path("./data")
    LOGS_DIR = Path("./logs")

try:
    import paths as p
except ImportError:
    p = Paths()

PLOT_DIR = p.DATA_DIR / "location_matching" / "diagnostic"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Region sets ───────────────────────────────────────────────────────────────
STR_REGIONS = {
    "CP", "STR", "STRd", "STRv", "ACB", "OT",
    "PAL", "SI", "LSX", "CEA", "LA", "BLA", "EP", "MEA"
}

CORTEX_PREFIXES = ["MOp", "MOs", "VIS", "SSp", "SSs", "ACA", "PL", "ILA", "ORB", "AI", "RSP"]

THAL_REGIONS = {
    "VAL", "AV", "CL", "LD", "LP", "VPL", "VPM", "PO", "MD", "AM",
    "IAD", "RE", "VM", "CM", "ATN", "ILM", "PIL", "SPF", "MG",
    "LGd", "LGv", "PCN", "VL", "SGN", "TH", "RT"
}

GPE_REGIONS = {"GPe", "GPi", "GP"}

def get_region_group(region):
    """Map detailed region to broad category (NO corrections applied)."""
    r = str(region)
    if r in STR_REGIONS:
        return "Striatum"
    if r in GPE_REGIONS:
        return "Pallidum"
    if any(r.startswith(px) for px in ["MOp", "MOs"]):
        return "Motor cortex"
    if r.startswith("VIS"):
        return "Visual cortex"
    if r in THAL_REGIONS:
        return "Thalamus"
    if r in {"CA1", "CA2", "CA3", "SUB", "DG", "HIP", "HPF"}:
        return "Hippocampus"
    return "Other/Excluded"


def is_in_cortex(region):
    r = str(region)
    return any(r.startswith(px) for px in CORTEX_PREFIXES)


def is_in_str(region):
    return str(region) in STR_REGIONS


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("0i_region_labeling_diagnostic.py")
print("  Validating histology-based region labels")
print("=" * 70)

# Load the ORIGINAL data (before cell type corrections)
csv_candidates = [
    p.LOGS_DIR / "RZ_unit_properties_with_qc_and_regions.csv",
    p.LOGS_DIR / "RZ_unit_properties_final.csv",
]
csv_path = next((c for c in csv_candidates if c.exists()), None)
if csv_path is None:
    raise FileNotFoundError(f"Could not find data. Tried: {csv_candidates}")

print(f"\nLoading: {csv_path.name}")
df = pd.read_csv(csv_path)
print(f"  {len(df):,} units")

# Key columns
depth_col = "peak_channel_depth"
fr_col = next((c for c in ["firing_rate", "mean_firing_rate", "fr"] if c in df.columns), None)
region_col = "region_acronym"  # ORIGINAL histology labels

if depth_col not in df.columns:
    raise ValueError(f"Missing depth column: {depth_col}")
if fr_col is None:
    raise ValueError("No firing rate column found")

# Add region group (based on ORIGINAL labels, no correction)
df["region_group_original"] = df[region_col].apply(get_region_group)

# Identify probe type
if "probe_region" not in df.columns:
    if "track_file" in df.columns:
        df["probe_region"] = df["track_file"].apply(
            lambda t: "str" if "_str" in str(t) else ("v1" if "_v1" in str(t) else "other")
        )
    else:
        raise ValueError("Cannot determine probe_region")

str_probe = df["probe_region"] == "str"
print(f"  STR probe units: {str_probe.sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: Original region distribution (BEFORE any correction)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("DIAGNOSTIC 1: Original region labels (BEFORE any correction)")
print("-" * 70)

str_df = df[str_probe].copy()

# Region counts from ORIGINAL histology labels
original_counts = str_df["region_group_original"].value_counts()
print("\nSTR probe — region distribution from ORIGINAL histology:")
for rg, count in original_counts.items():
    pct = count / len(str_df) * 100
    print(f"  {rg:20s}: {count:4d}  ({pct:5.1f}%)")

# Detailed region counts
detailed_counts = str_df[region_col].value_counts().head(20)
print("\nTop 20 detailed regions (ORIGINAL labels):")
for region, count in detailed_counts.items():
    rg = get_region_group(region)
    print(f"  {region:15s} ({rg:15s}): {count:4d}")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: Firing rate vs depth — find cortex-STR boundary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("DIAGNOSTIC 2: Firing rate vs depth (finding cortex-STR boundary)")
print("-" * 70)

# For each INSERTION (not session), plot FR vs depth
# Each insertion has its own track file and depth calibration
# Unique insertion = mouse + datetime + insertion_number
insertion_cols = ["mouse", "datetime", "insertion_number"]

# Ensure datetime is string for grouping
str_df["datetime_str"] = str_df["datetime"].astype(str)

insertions = str_df.groupby(["mouse", "datetime_str", "insertion_number"]).size().reset_index(name="n_units")
insertions = insertions[insertions["n_units"] >= 10]  # Only insertions with enough units
print(f"\n  Insertions with ≥10 units on STR probe: {len(insertions)}")

# Create insertion key
str_df["insertion_key"] = (
    str_df["mouse"] + "_" + 
    str_df["datetime_str"] + "_" +
    str_df["insertion_number"].astype(str)
)

# For each insertion, estimate the cortex-STR boundary from physiology
boundary_estimates = []

for _, row in insertions.iterrows():
    mouse = row["mouse"]
    dt_str = row["datetime_str"]
    ins_num = row["insertion_number"]
    ins_key = f"{mouse}_{dt_str}_{ins_num}"
    
    ins_df = str_df[str_df["insertion_key"] == ins_key].copy()
    
    if len(ins_df) < 10:
        continue
    
    # Get track file if available
    track_file = ins_df["track_file"].iloc[0] if "track_file" in ins_df.columns else "unknown"
    
    # Sort by depth
    ins_df = ins_df.sort_values(depth_col)
    depths = ins_df[depth_col].values
    frs = ins_df[fr_col].values
    
    # Skip if too many NaNs
    valid = ~np.isnan(depths) & ~np.isnan(frs)
    if valid.sum() < 10:
        continue
    
    depths_v = depths[valid]
    frs_v = frs[valid]
    
    # Smooth FR to find transition
    # Use sliding window median
    window = max(3, len(frs_v) // 10)
    
    # Find where FR drops significantly (cortex→STR transition)
    # Cortex: higher FR, STR: lower FR
    # Look for the depth where FR drops below a threshold
    
    # Method: find depth where cumulative low-FR fraction increases sharply
    fr_threshold = 5.0  # Hz — units below this are likely MSNs
    is_low_fr = frs_v < fr_threshold
    
    # Sliding window: fraction of low-FR units
    low_fr_frac = np.zeros(len(depths_v))
    for i in range(len(depths_v)):
        start = max(0, i - window)
        end = min(len(depths_v), i + window)
        low_fr_frac[i] = is_low_fr[start:end].mean()
    
    # Find depth where low_fr_frac crosses 0.5 (50% low-FR units)
    # This approximates the cortex-STR boundary
    cross_idx = np.where(low_fr_frac > 0.5)[0]
    if len(cross_idx) > 0:
        boundary_depth_phys = depths_v[cross_idx[0]]
    else:
        boundary_depth_phys = np.nan
    
    # What does histology say? Find where units transition from cortex to STR labels
    is_ctx = ins_df[region_col].apply(is_in_cortex).values[valid]
    is_str = ins_df[region_col].apply(is_in_str).values[valid]
    
    # Find depth where majority changes from cortex to STR
    ctx_mask = is_ctx.astype(float)
    ctx_frac = np.zeros(len(depths_v))
    for i in range(len(depths_v)):
        start = max(0, i - window)
        end = min(len(depths_v), i + window)
        ctx_frac[i] = ctx_mask[start:end].mean()
    
    cross_ctx = np.where(ctx_frac < 0.5)[0]
    if len(cross_ctx) > 0:
        boundary_depth_hist = depths_v[cross_ctx[0]]
    else:
        boundary_depth_hist = np.nan
    
    # Statistics
    n_ctx = is_ctx.sum()
    n_str = is_str.sum()
    n_other = len(is_ctx) - n_ctx - n_str
    
    boundary_estimates.append({
        "mouse": mouse,
        "datetime": dt_str,
        "insertion_number": ins_num,
        "track_file": track_file,
        "insertion_key": ins_key,
        "n_units": len(ins_df),
        "n_cortex": n_ctx,
        "n_striatum": n_str,
        "n_other": n_other,
        "pct_cortex": n_ctx / len(ins_df) * 100,
        "pct_striatum": n_str / len(ins_df) * 100,
        "boundary_physiology_um": boundary_depth_phys,
        "boundary_histology_um": boundary_depth_hist,
        "depth_min": depths_v.min(),
        "depth_max": depths_v.max(),
        "median_fr_shallow": np.nanmedian(frs_v[depths_v < np.median(depths_v)]),
        "median_fr_deep": np.nanmedian(frs_v[depths_v >= np.median(depths_v)]),
    })

boundary_df = pd.DataFrame(boundary_estimates)
print(f"\n  Insertions analyzed: {len(boundary_df)}")

if len(boundary_df) > 0:
    print("\n  Per-insertion summary (STR probe):")
    print(boundary_df[["mouse", "datetime", "insertion_number", "track_file", "n_units", "n_cortex", 
                       "n_striatum", "pct_striatum", "boundary_physiology_um", 
                       "boundary_histology_um"]].to_string(index=False))
    
    # Overall statistics
    print(f"\n  Overall statistics:")
    print(f"    Mean % cortex on STR probe: {boundary_df['pct_cortex'].mean():.1f}%")
    print(f"    Mean % striatum on STR probe: {boundary_df['pct_striatum'].mean():.1f}%")
    
    valid_phys = boundary_df["boundary_physiology_um"].dropna()
    valid_hist = boundary_df["boundary_histology_um"].dropna()
    if len(valid_phys) > 0:
        print(f"    Physiological boundary (FR-based): {valid_phys.mean():.0f} ± {valid_phys.std():.0f} µm")
    if len(valid_hist) > 0:
        print(f"    Histological boundary (label-based): {valid_hist.mean():.0f} ± {valid_hist.std():.0f} µm")
    
    if len(valid_phys) > 0 and len(valid_hist) > 0:
        offset = valid_hist.mean() - valid_phys.mean()
        print(f"    Offset (histology - physiology): {offset:.0f} µm")
        if abs(offset) > 300:
            print(f"    *** WARNING: Large offset suggests systematic depth error! ***")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("Generating diagnostic plots")
print("-" * 70)

REGION_COLORS = {
    "Striatum": "#2166AC",
    "Motor cortex": "#D6604D",
    "Visual cortex": "#9467BD",
    "Thalamus": "#FF7F0E",
    "Pallidum": "#4393C3",
    "Hippocampus": "#8C564B",
    "Other/Excluded": "#AAAAAA",
}


# ── Plot 1: Original region distribution ──────────────────────────────────────
print("\n[Plot 1] Original region distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Pie chart of region groups
ax = axes[0]
colors = [REGION_COLORS.get(rg, "#888") for rg in original_counts.index]
wedges, texts, autotexts = ax.pie(original_counts.values, labels=original_counts.index,
                                   autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                                   colors=colors, startangle=90)
ax.set_title(f"STR Probe — ORIGINAL Histology Labels\n(n={len(str_df):,} units)")

# Right: Bar chart of detailed regions
ax = axes[1]
top_regions = str_df[region_col].value_counts().head(15)
colors = [REGION_COLORS.get(get_region_group(r), "#888") for r in top_regions.index]
bars = ax.barh(range(len(top_regions)), top_regions.values, color=colors, edgecolor="white")
ax.set_yticks(range(len(top_regions)))
ax.set_yticklabels(top_regions.index)
ax.invert_yaxis()
ax.set_xlabel("Unit count")
ax.set_title("Top 15 regions (ORIGINAL labels)")

for bar, val in zip(bars, top_regions.values):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=9)

fig.suptitle("DIAGNOSTIC: Region Labels BEFORE Any Correction", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diagnostic_original_regions.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> diagnostic_original_regions.png")


# ── Plot 2: FR vs depth scatter (all STR probe sessions) ──────────────────────
print("\n[Plot 2] FR vs depth scatter...")

fig, ax = plt.subplots(figsize=(10, 8))

# Color by original region group
for rg in ["Motor cortex", "Striatum", "Thalamus", "Pallidum", "Other/Excluded"]:
    mask = (str_df["region_group_original"] == rg) & str_df[depth_col].notna() & str_df[fr_col].notna()
    if mask.sum() > 0:
        ax.scatter(str_df.loc[mask, depth_col], str_df.loc[mask, fr_col],
                  c=REGION_COLORS.get(rg, "#888"), label=f"{rg} (n={mask.sum()})",
                  alpha=0.5, s=20, edgecolors="none")

ax.axhline(5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="FR=5Hz (MSN threshold)")
ax.axhline(15, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="FR=15Hz")

ax.set_xlabel("Peak channel depth (µm from tip)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_yscale("log")
ax.set_ylim(0.05, 200)
ax.set_title("STR Probe: Firing Rate vs Depth\n(colored by ORIGINAL histology label)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(PLOT_DIR / "diagnostic_fr_vs_depth.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> diagnostic_fr_vs_depth.png")


# ── Plot 3: Per-insertion FR profile ────────────────────────────────────────────
print("\n[Plot 3] Per-insertion FR vs depth profiles...")

# Plot up to 12 insertions
n_insertions = min(12, len(insertions))
if n_insertions > 0:
    ncols = 4
    nrows = int(np.ceil(n_insertions / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
    axes = np.array(axes).flatten()
    
    for i, (_, row) in enumerate(insertions.head(n_insertions).iterrows()):
        ax = axes[i]
        ins_key = f"{row['mouse']}_{row['datetime_str']}_{row['insertion_number']}"
        ins_df = str_df[str_df["insertion_key"] == ins_key].copy()
        
        # Plot all units
        for rg in ["Motor cortex", "Striatum", "Thalamus"]:
            mask = (ins_df["region_group_original"] == rg)
            if mask.sum() > 0:
                ax.scatter(ins_df.loc[mask, depth_col], ins_df.loc[mask, fr_col],
                          c=REGION_COLORS.get(rg, "#888"), s=30, alpha=0.7,
                          label=f"{rg[:3]} ({mask.sum()})")
        
        # Add smoothed FR line
        valid = ins_df[depth_col].notna() & ins_df[fr_col].notna()
        if valid.sum() > 5:
            sorted_df = ins_df[valid].sort_values(depth_col)
            depths = sorted_df[depth_col].values
            frs = sorted_df[fr_col].values
            
            # Binned median
            n_bins = min(10, len(depths) // 3)
            if n_bins >= 3:
                bins = np.linspace(depths.min(), depths.max(), n_bins + 1)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_medians = []
                for j in range(n_bins):
                    mask = (depths >= bins[j]) & (depths < bins[j+1])
                    if mask.sum() > 0:
                        bin_medians.append(np.median(frs[mask]))
                    else:
                        bin_medians.append(np.nan)
                ax.plot(bin_centers, bin_medians, 'k-', linewidth=2, alpha=0.7, label="Median")
        
        ax.axhline(5, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 100)
        
        # Title with track file if available
        track_info = ""
        if "track_file" in ins_df.columns:
            track_info = f"\n{ins_df['track_file'].iloc[0]}"
        # Extract date from datetime for cleaner title
        date_short = row['datetime_str'].split()[0] if ' ' in row['datetime_str'] else row['datetime_str'][:10]
        ax.set_title(f"{row['mouse']} {date_short} ins{row['insertion_number']}{track_info}\n(n={row['n_units']})", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("Depth (µm)")
        ax.set_ylabel("FR (Hz)")
    
    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle("Per-Insertion: Firing Rate vs Depth (STR probe)\nRed line = 5Hz (MSN threshold)", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "diagnostic_fr_vs_depth_insertions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> diagnostic_fr_vs_depth_insertions.png")


# ── Plot 4: Boundary comparison ───────────────────────────────────────────────
if len(boundary_df) > 0 and boundary_df["boundary_physiology_um"].notna().sum() > 0:
    print("\n[Plot 4] Physiological vs histological boundary...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Scatter of physiological vs histological boundary
    ax = axes[0]
    valid = boundary_df["boundary_physiology_um"].notna() & boundary_df["boundary_histology_um"].notna()
    if valid.sum() > 0:
        ax.scatter(boundary_df.loc[valid, "boundary_physiology_um"],
                  boundary_df.loc[valid, "boundary_histology_um"],
                  s=60, alpha=0.7, c="#2166AC", edgecolors="white")
        
        # Unity line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'k--', linewidth=1, label="Unity")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        # Label points with mouse and insertion number
        for _, row in boundary_df[valid].iterrows():
            label = f"{row['mouse']}_ins{row['insertion_number']}"
            ax.annotate(label, 
                       (row["boundary_physiology_um"], row["boundary_histology_um"]),
                       fontsize=7, alpha=0.7)
    
    ax.set_xlabel("Physiological boundary (µm)\n(FR-based: where low-FR fraction > 50%)")
    ax.set_ylabel("Histological boundary (µm)\n(label-based: where cortex fraction < 50%)")
    ax.set_title("Cortex-STR Boundary: Physiology vs Histology")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # Right: Distribution of offsets
    ax = axes[1]
    offsets = boundary_df["boundary_histology_um"] - boundary_df["boundary_physiology_um"]
    offsets = offsets.dropna()
    if len(offsets) > 0:
        ax.hist(offsets, bins=15, color="#2166AC", edgecolor="white", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No offset")
        ax.axvline(offsets.median(), color="orange", linestyle="-", linewidth=2, 
                  label=f"Median: {offsets.median():.0f}µm")
        ax.set_xlabel("Offset (histology - physiology) (µm)")
        ax.set_ylabel("Insertion count")
        ax.set_title("Distribution of Boundary Offsets\n(positive = histology deeper than physiology)")
        ax.legend()
    
    fig.suptitle("DIAGNOSTIC: Is Histology-Based Boundary Consistent with Physiology?", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "diagnostic_boundary_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> diagnostic_boundary_comparison.png")


# ── Save insertion summary ──────────────────────────────────────────────────────
if len(boundary_df) > 0:
    boundary_df.to_csv(PLOT_DIR / "diagnostic_insertion_summary.csv", index=False)
    print(f"  Saved -> diagnostic_insertion_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

pct_str = (str_df["region_group_original"] == "Striatum").sum() / len(str_df) * 100
pct_ctx = (str_df["region_group_original"] == "Motor cortex").sum() / len(str_df) * 100
pct_thal = (str_df["region_group_original"] == "Thalamus").sum() / len(str_df) * 100

print(f"""
STR probe region breakdown (ORIGINAL histology labels):
  Striatum:     {pct_str:.1f}%
  Motor cortex: {pct_ctx:.1f}%
  Thalamus:     {pct_thal:.1f}%
""")

if pct_str < 40:
    print("⚠️  WARNING: Less than 40% of STR probe units are labeled as striatum!")
    print("   This suggests a systematic issue with:")
    print("   1. Track reconstruction depth")
    print("   2. Shrinkage correction factor")
    print("   3. Or probe placement")

if pct_ctx > pct_str:
    print("\n⚠️  WARNING: More cortical than striatal units on STR probe!")
    print("   Expected: STR probe should have MAJORITY of units in striatum")

if len(boundary_df) > 0:
    offsets = (boundary_df["boundary_histology_um"] - boundary_df["boundary_physiology_um"]).dropna()
    if len(offsets) > 0 and abs(offsets.median()) > 300:
        print(f"\n⚠️  WARNING: Median boundary offset = {offsets.median():.0f}µm")
        print("   Histology boundary is significantly different from physiology!")
        if offsets.median() > 0:
            print("   → Histology thinks cortex-STR boundary is DEEPER than FR suggests")
            print("   → This would cause cortical units to be mislabeled as STR")
            print("   → Consider: shrinkage correction may be OVER-correcting")
        else:
            print("   → Histology thinks boundary is SHALLOWER than FR suggests")
            print("   → This would cause STR units to be mislabeled as cortex")
            print("   → Consider: shrinkage correction may be UNDER-correcting")

print(f"""
NEXT STEPS:
1. Check the per-session FR vs depth plots to see if there's a clear
   cortex→STR transition in the physiology
2. Compare the physiological boundary estimate to histological labels
3. If offset is systematic, consider adjusting the shrinkage correction
4. Review track reconstructions for sessions with largest discrepancies

Output files in: {PLOT_DIR}
""")
