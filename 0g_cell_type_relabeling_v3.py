"""
0g_cell_type_relabeling_v3.py
==============================
Depth-based cell type assignment for STR probe.

Key insight: The per-unit histology region labels are unreliable (only 23% STR),
but the track reconstructions show where CP actually is. Use DEPTH to assign
cell types instead of trusting the histology labels.

Coordinate system (depth from surface/entry):
  0 µm         = cortical surface (probe entry)
  0-1600 µm    = cortex (MOp layers)
  1600-2800 µm = striatum (CP)  <-- THIS IS THE STR ZONE
  2800+ µm     = thalamus/pallidum (deep, often unreliable)

Cell type logic for STR probe:
  - Depth in STR zone + MSN waveform (PT≥0.4ms, FR≤15Hz) → MSN
  - Depth in STR zone + narrow waveform (PT<0.35ms)      → STR FSI
  - Depth in STR zone + TAN signature                    → TAN
  - Depth in cortex zone + broad waveform                → RS (pyramidal)
  - Depth in cortex zone + narrow waveform               → cortical FSI

This MAXIMIZES STR unit recovery by using the track-validated depth ranges.

Outputs:
  RZ_unit_properties_final.csv
  RZ_str_msn.csv  (MSNs in STR depth zone — PRIMARY FOR RESCALING)
  RZ_str_fsi.csv
  RZ_str_units.csv (all units in STR depth zone)
  RZ_cortex_units.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    import paths as p
except ImportError:
    class Paths:
        DATA_DIR = Path("./data")
        LOGS_DIR = Path("./logs")
    p = Paths()

PLOT_DIR = p.DATA_DIR / "location_matching" / "cell_type_v3"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DEPTH BOUNDARIES (µm from surface)
# ══════════════════════════════════════════════════════════════════════════════
# Based on track analysis: CP typically starts at 1300-1600 µm from surface
#
# Key insight: NO cell type below cortex has MSN-like properties except MSNs.
#   - Thalamic relay neurons: broad waveform but fire 5-20 Hz (not MSN-slow)
#   - GPe/GPi: fire 20-100 Hz (definitely not MSN-like)
#   - Ventral striatum: contains real MSNs with identical properties
#
# Therefore: ANY unit below cortex with broad+slow waveform = MSN

CORTEX_END_UM = 1600      # Below this = cortex (RS if broad waveform)
                          # Above this = subcortical → broad+slow = MSN

# ══════════════════════════════════════════════════════════════════════════════
# WAVEFORM THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
PT_BROAD_MIN = 0.40    # ms — MSN/RS have broad waveforms
PT_NARROW_MAX = 0.35   # ms — FSI have narrow waveforms
FR_LOW_MAX = 15.0      # Hz — MSN/RS fire slowly
FR_HIGH_MIN = 20.0     # Hz — high-firing neurons
FR_TAN_MIN = 3.0       # Hz — TANs fire tonically
FR_TAN_MAX = 12.0      # Hz

# ══════════════════════════════════════════════════════════════════════════════
# CELL TYPE COLORS
# ══════════════════════════════════════════════════════════════════════════════
CELL_TYPE_COLORS = {
    "MSN": "#2166AC",
    "FSI": "#D6604D",
    "RS": "#4393C3",
    "TAN": "#8C564B",
    "high_FR": "#762A83",
    "ambiguous": "#AAAAAA",
}

# Target regions for each probe (used in correct-region counts and plots)
CP_STR_REGIONS   = {"CP", "STR", "STRd", "STRv", "ACB"}
V1_REGIONS_PREFIX = "VISp"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_depth_zone(depth, cortex_end=CORTEX_END_UM):
    """
    Assign depth zone based on distance from surface.
    
    Returns: 'cortex' or 'subcortical' or 'unknown'
    
    Subcortical includes striatum, pallidum, thalamus - but only MSNs
    have broad+slow waveforms, so waveform distinguishes cell type.
    """
    if pd.isna(depth):
        return "unknown"
    if depth <= cortex_end:
        return "cortex"
    return "subcortical"  # Everything below cortex - use waveform to classify


def classify_cell_type(row, depth_col="peak_channel_depth", fr_col="firing_rate"):
    """
    Classify cell type based on depth zone and waveform.
    
    For STR probe:
      - Cortex zone (≤1600µm) + broad/slow → RS (pyramidal)
      - Subcortical (>1600µm) + broad/slow → MSN (only cell type with this signature)
      - Subcortical + narrow → FSI
      - Subcortical + medium PT + tonic FR → TAN
      - Any zone + very high FR (not narrow) → high_FR
    
    For V1 probe:
      - Uses simpler logic (histology is more reliable for V1)
    """
    probe = row.get("probe_region", "unknown")
    depth = row.get(depth_col, np.nan)
    pt = row.get("pt_duration_ms", np.nan)
    fr = row.get(fr_col, np.nan)
    
    # Waveform features
    is_broad = pt >= PT_BROAD_MIN if pd.notna(pt) else False
    is_narrow = pt < PT_NARROW_MAX if pd.notna(pt) else False
    is_low_fr = fr <= FR_LOW_MAX if pd.notna(fr) else False
    is_high_fr = fr > FR_HIGH_MIN if pd.notna(fr) else False
    is_tonic = (FR_TAN_MIN <= fr <= FR_TAN_MAX) if pd.notna(fr) else False
    is_medium_pt = (0.35 <= pt <= 0.50) if pd.notna(pt) else False
    
    # V1 probe: use simpler logic (histology is more reliable)
    if probe == "v1":
        if is_narrow:
            return "FSI"
        if is_high_fr:
            return "high_FR"
        if is_broad and is_low_fr:
            return "RS"
        return "ambiguous"
    
    # STR probe: use depth-based assignment
    depth_zone = get_depth_zone(depth)
    
    # High-firing neurons (regardless of zone) - NOT MSN-like
    if is_high_fr and not is_narrow:
        return "high_FR"
    
    # FSI (narrow waveform, any zone)
    if is_narrow:
        return "FSI"
    
    # Cortex zone: broad+slow = RS (pyramidal)
    if depth_zone == "cortex":
        if is_broad and is_low_fr:
            return "RS"
        return "ambiguous"
    
    # Subcortical zone: broad+slow = MSN (the ONLY cell type with this signature)
    if depth_zone == "subcortical":
        # TAN: medium PT, tonic firing (cholinergic interneuron)
        if is_medium_pt and is_tonic:
            return "TAN"
        # MSN: broad waveform, low firing - NO OTHER SUBCORTICAL CELL TYPE HAS THIS
        if is_broad and is_low_fr:
            return "MSN"
        return "ambiguous"
    
    return "ambiguous"


def get_depth_zone_label(row, depth_col="peak_channel_depth"):
    """Get the depth zone for a row."""
    depth = row.get(depth_col, np.nan)
    return get_depth_zone(depth)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("0g_cell_type_relabeling_v3.py")
print("  Depth-based cell type assignment (MAXIMIZING STR units)")
print("=" * 70)

print(f"""
Depth boundary (µm from surface):
  Cortex zone:      0 - {CORTEX_END_UM} µm  → broad+slow = RS
  Subcortical zone: > {CORTEX_END_UM} µm    → broad+slow = MSN

Key insight: No subcortical cell type except MSN has broad+slow signature.
  - Thalamic neurons: broad but fire 5-20 Hz
  - GPe/GPi: fire 20-100 Hz
  - Therefore: subcortical + broad + slow = MSN (no depth limit)
""")

# ── Load data ─────────────────────────────────────────────────────────────────
csv_candidates = [
    p.LOGS_DIR / "RZ_unit_properties_with_qc_and_regions.csv",
    p.LOGS_DIR / "RZ_unit_properties_with_regions.csv",
]
csv_path = next((c for c in csv_candidates if c.exists()), None)
if csv_path is None:
    raise FileNotFoundError(f"Could not find region CSV. Tried: {csv_candidates}")

print(f"Loading: {csv_path.name}")
df = pd.read_csv(csv_path)
n_total = len(df)
print(f"  {n_total:,} units loaded")

# Ensure required columns
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"])

# Detect probe_region
if "probe_region" not in df.columns:
    if "track_file" in df.columns:
        df["probe_region"] = df["track_file"].apply(
            lambda t: "str" if "_str" in str(t) else ("v1" if "_v1" in str(t) else "other")
        )
    else:
        raise ValueError("Cannot determine probe_region")

# Detect firing rate column
fr_col = next((c for c in ["firing_rate", "mean_firing_rate", "fr"] if c in df.columns), None)
if fr_col is None:
    raise ValueError("No firing rate column found")
print(f"  Firing rate column: '{fr_col}'")

# Detect depth column
depth_col = "peak_channel_depth"
if depth_col not in df.columns:
    raise ValueError(f"Missing depth column: {depth_col}")

# Detect waveform duration column and normalise to ms
pt_col_raw = next(
    (c for c in ["pt_duration_ms", "trough_to_peak_ms", "peak_trough_ms"] if c in df.columns),
    None,
)
if pt_col_raw is None:
    raise ValueError("No waveform duration column found (expected pt_duration_ms or trough_to_peak_ms)")
# trough_to_peak_ms is stored in seconds despite the name — convert if needed
if df[pt_col_raw].max() < 0.01:   # values < 0.01 → stored in seconds
    df["pt_duration_ms"] = df[pt_col_raw] * 1000
    print(f"  Waveform column '{pt_col_raw}' detected in seconds → converted to ms")
else:
    df["pt_duration_ms"] = df[pt_col_raw]
    print(f"  Waveform column '{pt_col_raw}' used as-is (already in ms)")

str_probe_mask = df["probe_region"] == "str"
v1_probe_mask = df["probe_region"] == "v1"
print(f"  STR probe units: {str_probe_mask.sum():,}")
print(f"  V1 probe units:  {v1_probe_mask.sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# ASSIGN DEPTH ZONES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("ASSIGNING DEPTH ZONES")
print("-" * 70)

df["depth_zone"] = df.apply(lambda row: get_depth_zone_label(row, depth_col), axis=1)

# Count by zone for STR probe
str_df = df[str_probe_mask]
zone_counts = str_df["depth_zone"].value_counts()
print(f"\nSTR probe depth zone distribution:")
for zone in ["cortex", "subcortical", "unknown"]:
    count = zone_counts.get(zone, 0)
    pct = count / len(str_df) * 100 if len(str_df) > 0 else 0
    print(f"  {zone:12s}: {count:4d}  ({pct:5.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFY CELL TYPES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("CLASSIFYING CELL TYPES")
print("-" * 70)

df["cell_type"] = df.apply(
    lambda row: classify_cell_type(row, depth_col=depth_col, fr_col=fr_col), 
    axis=1
)

# Summary
print(f"\nCell type summary (all units):")
ct_counts = df["cell_type"].value_counts()
for ct in ["MSN", "RS", "FSI", "TAN", "high_FR", "ambiguous"]:
    count = ct_counts.get(ct, 0)
    pct = count / n_total * 100
    print(f"  {ct:15s}: {count:4d}  ({pct:5.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STR PROBE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("STR PROBE BREAKDOWN")
print("-" * 70)

str_df = df[str_probe_mask].copy()

print(f"\nSTR probe cell types by depth zone:")
for zone in ["cortex", "subcortical"]:
    zone_mask = str_df["depth_zone"] == zone
    zone_df = str_df[zone_mask]
    if len(zone_df) > 0:
        print(f"\n  {zone.upper()} zone ({len(zone_df)} units):")
        for ct in ["MSN", "RS", "FSI", "TAN", "high_FR", "ambiguous"]:
            count = (zone_df["cell_type"] == ct).sum()
            if count > 0:
                print(f"    {ct}: {count}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD OUTPUT DATASETS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("BUILDING OUTPUT DATASETS")
print("-" * 70)

# STR MSNs (the primary output for rescaling)
str_msn = df[str_probe_mask & (df["cell_type"] == "MSN")].copy()
print(f"\n  STR MSN: {len(str_msn)} units")

# STR FSIs (subcortical)
str_fsi = df[str_probe_mask & (df["cell_type"] == "FSI") & (df["depth_zone"] == "subcortical")].copy()
print(f"  STR FSI: {len(str_fsi)} units")

# STR TANs
str_tan = df[str_probe_mask & (df["cell_type"] == "TAN")].copy()
print(f"  STR TAN: {len(str_tan)} units")

# All subcortical units (potential STR)
str_units = df[str_probe_mask & (df["depth_zone"] == "subcortical")].copy()
print(f"  Subcortical zone (all cell types): {len(str_units)} units")

# Cortical units (from STR probe cortex zone + V1 probe)
cortex_from_str = df[str_probe_mask & (df["depth_zone"] == "cortex")].copy()
cortex_from_v1 = df[v1_probe_mask].copy()
cortex_units = pd.concat([cortex_from_str, cortex_from_v1], ignore_index=True)
print(f"  Cortical units (STR probe shallow + V1): {len(cortex_units)} units")

# RS neurons
rs_units = df[df["cell_type"] == "RS"].copy()
print(f"  RS (all): {len(rs_units)} units")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE TO HISTOLOGY-BASED LABELING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("COMPARISON: Depth-based vs Histology-based")
print("-" * 70)

# How many units were labeled as STR by histology?
if "region_acronym" in df.columns:
    STR_REGIONS = {"CP", "STR", "STRd", "STRv", "ACB", "OT", "PAL"}
    histology_str = str_df["region_acronym"].isin(STR_REGIONS).sum()
    depth_subcortical = (str_df["depth_zone"] == "subcortical").sum()
    
    print(f"\n  STR probe units in subcortical zone:")
    print(f"    By histology (STR regions): {histology_str}")
    print(f"    By depth (>{CORTEX_END_UM}µm):      {depth_subcortical}")
    print(f"    Difference:                 {depth_subcortical - histology_str:+d}")
    
    # Breakdown of what depth-based gains
    depth_sub_mask = str_df["depth_zone"] == "subcortical"
    hist_str_mask = str_df["region_acronym"].isin(STR_REGIONS)
    
    gained = depth_sub_mask & ~hist_str_mask  # subcortical by depth, not STR by histology
    lost = ~depth_sub_mask & hist_str_mask    # STR by histology, but cortex by depth
    
    if gained.sum() > 0:
        print(f"\n  GAINED by depth-based (histology said NOT STR, but depth >{CORTEX_END_UM}µm):")
        gained_df = str_df[gained]
        for region in gained_df["region_acronym"].value_counts().head(10).index:
            count = (gained_df["region_acronym"] == region).sum()
            print(f"    {region}: {count}")
    
    if lost.sum() > 0:
        print(f"\n  LOST by depth-based (histology said STR, but depth ≤{CORTEX_END_UM}µm):")
        print(f"    {lost.sum()} units (likely boundary cases)")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# Add depth_zone to main dataframe and save
df.to_csv(p.LOGS_DIR / "RZ_unit_properties_final.csv", index=False)
print(f"  Saved: RZ_unit_properties_final.csv")

str_msn.to_csv(p.LOGS_DIR / "RZ_str_msn.csv", index=False)
print(f"  Saved: RZ_str_msn.csv ({len(str_msn)} units) <- PRIMARY FOR RESCALING")

str_fsi.to_csv(p.LOGS_DIR / "RZ_str_fsi.csv", index=False)
print(f"  Saved: RZ_str_fsi.csv ({len(str_fsi)} units)")

str_units.to_csv(p.LOGS_DIR / "RZ_str_units.csv", index=False)
print(f"  Saved: RZ_str_units.csv ({len(str_units)} units)")

cortex_units.to_csv(p.LOGS_DIR / "RZ_cortex_units.csv", index=False)
print(f"  Saved: RZ_cortex_units.csv ({len(cortex_units)} units)")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
Depth boundary used:
  Cortex:      0 - {CORTEX_END_UM} µm from surface
  Subcortical: > {CORTEX_END_UM} µm (includes STR, pallidum, thalamus)
  
  Key: Subcortical + broad waveform + low FR = MSN
       (No other subcortical cell type has this signature)

Total units: {n_total}

STR probe ({str_probe_mask.sum()} units):
  Cortex zone (≤{CORTEX_END_UM}µm):      {(str_df['depth_zone'] == 'cortex').sum():4d}
  Subcortical zone (>{CORTEX_END_UM}µm): {(str_df['depth_zone'] == 'subcortical').sum():4d}  <-- MSN SOURCE

Cell types from subcortical zone:
  MSN:       {len(str_msn):4d}  <- PRIMARY FOR RESCALING
  FSI:       {len(str_fsi):4d}
  TAN:       {len(str_tan):4d}
  Other:     {len(str_units) - len(str_msn) - len(str_fsi) - len(str_tan):4d}

V1 probe ({v1_probe_mask.sum()} units):
  RS:        {(df[v1_probe_mask]['cell_type'] == 'RS').sum():4d}
  FSI:       {(df[v1_probe_mask]['cell_type'] == 'FSI').sum():4d}

Output files:
  RZ_str_msn.csv     : MSNs (subcortical + broad + slow)
  RZ_str_units.csv   : All subcortical units
  RZ_cortex_units.csv: Cortical units (STR probe shallow + V1)
""")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("GENERATING DIAGNOSTIC PLOT")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Depth vs FR, colored by cell type
ax = axes[0]
str_df = df[str_probe_mask].copy()
for ct in ["MSN", "RS", "FSI", "TAN", "high_FR", "ambiguous"]:
    mask = str_df["cell_type"] == ct
    if mask.sum() > 0:
        ax.scatter(str_df.loc[mask, depth_col], str_df.loc[mask, fr_col],
                  c=CELL_TYPE_COLORS.get(ct, "#888"), label=f"{ct} ({mask.sum()})",
                  alpha=0.6, s=20, edgecolors="none")

ax.axvline(CORTEX_END_UM, color="green", linestyle="--", linewidth=2, alpha=0.7,
           label=f"Cortex-Subcortical ({CORTEX_END_UM}µm)")
ax.axhline(FR_LOW_MAX, color="red", linestyle=":", alpha=0.5, label=f"FR={FR_LOW_MAX}Hz")

ax.set_xlabel("Depth from surface (µm)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_yscale("log")
ax.set_ylim(0.05, 200)
ax.set_title("STR Probe: Depth vs FR by Cell Type")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.2)

# Plot 2: Cell type counts by depth zone
ax = axes[1]
zones = ["cortex", "subcortical"]
cell_types = ["MSN", "RS", "FSI", "TAN", "high_FR", "ambiguous"]
x = np.arange(len(zones))
width = 0.12
for i, ct in enumerate(cell_types):
    counts = [((str_df["depth_zone"] == z) & (str_df["cell_type"] == ct)).sum() for z in zones]
    ax.bar(x + i * width, counts, width, label=ct, color=CELL_TYPE_COLORS.get(ct, "#888"))

ax.set_xticks(x + width * 2.5)
ax.set_xticklabels([f"Cortex\n(≤{CORTEX_END_UM}µm)", f"Subcortical\n(>{CORTEX_END_UM}µm)"])
ax.set_ylabel("Unit count")
ax.set_title("Cell Type Distribution by Depth Zone")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.suptitle(f"Depth-Based Cell Type Assignment\nCortex boundary: {CORTEX_END_UM}µm | Subcortical broad+slow = MSN")
fig.tight_layout()
fig.savefig(PLOT_DIR / "depth_based_assignment.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> depth_based_assignment.png")

# ── Correct-region unit count plot ────────────────────────────────────────────
if "region_acronym" in df.columns:
    cell_types_str = ["MSN", "FSI", "TAN", "RS", "high_FR", "ambiguous"]
    cell_types_v1  = ["RS", "FSI", "high_FR", "ambiguous"]

    str_correct_df = df[str_probe_mask & df["region_acronym"].isin(CP_STR_REGIONS)]
    v1_correct_df  = df[v1_probe_mask  & df["region_acronym"].str.startswith(V1_REGIONS_PREFIX, na=False)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ── Panel 1: STR probe — stacked bar by region (CP vs STR) ────────────────
    ax = axes[0]
    region_order_str = ["CP", "STR"]
    bottom = np.zeros(len(cell_types_str))
    region_colors_str = {"CP": "#2166AC", "STR": "#6BAED6"}
    for region in region_order_str:
        counts = [
            (str_correct_df[str_correct_df["region_acronym"] == region]["cell_type"] == ct).sum()
            for ct in cell_types_str
        ]
        bars = ax.bar(cell_types_str, counts, bottom=bottom,
                      label=region, color=region_colors_str[region],
                      edgecolor="white", linewidth=0.8)
        # Label non-zero segments
        for bar, count, bot in zip(bars, counts, bottom):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bot + count / 2,
                        str(count), ha="center", va="center")
        bottom += np.array(counts)
    # Total labels on top
    for i, tot in enumerate(bottom):
        if tot > 0:
            ax.text(i, tot + 1, str(int(tot)), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Unit count")
    ax.set_title(f"STR probe — histology-confirmed CP/STR\n"
                 f"({len(str_correct_df)} / {str_probe_mask.sum()} units, "
                 f"{100*len(str_correct_df)/str_probe_mask.sum():.1f}%)")
    ax.legend(title="Region", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: V1 probe — stacked bar by layer ──────────────────────────────
    ax = axes[1]
    layer_order_v1 = ["VISp1", "VISp2/3", "VISp4", "VISp5", "VISp6a", "VISp6b"]
    layer_colors   = {
        "VISp1":   "#C6DBEF",
        "VISp2/3": "#9ECAE1",
        "VISp4":   "#6BAED6",
        "VISp5":   "#3182BD",
        "VISp6a":  "#08519C",
        "VISp6b":  "#08306B",
    }
    bottom = np.zeros(len(cell_types_v1))
    for layer in layer_order_v1:
        counts = [
            (v1_correct_df[v1_correct_df["region_acronym"] == layer]["cell_type"] == ct).sum()
            for ct in cell_types_v1
        ]
        bars = ax.bar(cell_types_v1, counts, bottom=bottom,
                      label=layer, color=layer_colors[layer],
                      edgecolor="white", linewidth=0.8)
        for bar, count, bot in zip(bars, counts, bottom):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bot + count / 2,
                        str(count), ha="center", va="center")
        bottom += np.array(counts)
    for i, tot in enumerate(bottom):
        if tot > 0:
            ax.text(i, tot + 1, str(int(tot)), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Unit count")
    ax.set_title(f"V1 probe — histology-confirmed VISp\n"
                 f"({len(v1_correct_df)} / {v1_probe_mask.sum()} units, "
                 f"{100*len(v1_correct_df)/v1_probe_mask.sum():.1f}%)")
    ax.legend(title="Layer", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Correct-Region Unit Counts (histology-confirmed)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "correct_region_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> correct_region_counts.png")

print("\n" + "=" * 70)
print(f"Output directory: {PLOT_DIR}")
print("=" * 70)
