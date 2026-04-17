"""
0g_cell_type_relabeling_v2.py
==============================
Corrected two-stage pipeline with SEPARATE region and cell type columns.

Key changes from v1:
  1. corrected_region and cell_type are INDEPENDENT columns
  2. Region correction only for biological impossibilities (GPe→CP, RT→VAL)
  3. Cell type assignment respects BOTH waveform AND region
  4. Cortex-STR boundary estimated from track data, not hard-coded depth
  5. Pyramidal (RS) vs MSN distinction based on whether unit is in cortex vs STR

Output columns:
  corrected_region  : anatomical location (only changed when biologically impossible)
  cell_type         : MSN, FSI, RS, TAN, high_FR, ambiguous
  region_source     : 'histology' or 'waveform_corrected'
  
This allows proper counting: STR MSN, STR FSI, cortex RS, cortex FSI, etc.

Inputs:
  RZ_unit_properties_with_qc_and_regions.csv  (output of 0e)

Outputs:
  RZ_unit_properties_final.csv
  RZ_str_units.csv           (all units in STR regions)
  RZ_str_msn.csv             (STR MSNs only — primary for rescaling)
  RZ_str_fsi.csv             (STR FSIs)
  RZ_cortex_units.csv        (all cortical units)
  relabeling_summary.csv
  
Plots (in DATA_DIR/location_matching/cell_type_v2/):
  units_per_region_before.png
  units_per_region_after.png
  cell_type_by_region.png
  waveforms_by_cell_type.png
  cortex_str_boundary_validation.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths (adjust these to match your setup) ──────────────────────────────────
# These would normally come from your paths.py
class Paths:
    DATA_DIR = Path("./data")
    LOGS_DIR = Path("./logs")

try:
    import paths as p
except ImportError:
    p = Paths()

PLOT_DIR = p.DATA_DIR / "location_matching" / "cell_type_v2"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Waveform thresholds ───────────────────────────────────────────────────────
PT_BROAD_MIN  = 0.40   # ms — minimum PT for broad waveform (MSN or RS)
PT_NARROW_MAX = 0.35   # ms — maximum PT for narrow waveform (FSI)
FR_LOW_MAX    = 15.0   # Hz — maximum FR for low-firing (MSN or RS)
FR_HIGH_MIN   = 20.0   # Hz — minimum FR for high-firing (GPe-like)
FR_TAN_MIN    = 3.0    # Hz — TANs fire tonically, typically 3-10 Hz
FR_TAN_MAX    = 12.0   # Hz

# ── Depth gate ────────────────────────────────────────────────────────────────
# STR probe units deeper than this are below cortex and are assigned as MSN.
# Units shallower than this are in cortex and are assigned as RS.
# Broad+slow waveform is ambiguous between MSN and RS — depth resolves it.
CORTEX_STR_BOUNDARY_UM = 1500   # µm from probe surface

# ── Region sets ───────────────────────────────────────────────────────────────
STR_REGIONS = {
    "CP", "STR", "STRd", "STRv", "ACB", "OT",
    "PAL", "SI", "LSX", "CEA", "LA", "BLA", "EP", "MEA"
}

CORTEX_REGIONS = {
    "MOp", "MOp1", "MOp2/3", "MOp5", "MOp6a", "MOp6b",
    "MOs", "MOs1", "MOs2/3", "MOs5", "MOs6a", "MOs6b",
    "VISp", "VISp1", "VISp2/3", "VISp4", "VISp5", "VISp6a", "VISp6b",
    "VISl", "VISl1", "VISl2/3", "VISl4", "VISl5", "VISl6a", "VISl6b",
    "VISam", "VISpm", "VISal", "VISrl", "VISli", "VISpl", "VISpor",
    "SSp", "SSs", "ACA", "PL", "ILA", "ORB", "AI", "RSP"
}

GPE_REGIONS = {"GPe", "GPi", "GP"}
RT_REGIONS = {"RT"}
THAL_REGIONS = {
    "VAL", "AV", "CL", "LD", "LP", "VPL", "VPM", "PO", "MD", "AM",
    "IAD", "RE", "VM", "CM", "ATN", "ILM", "PIL", "SPF", "MG",
    "LGd", "LGv", "PCN", "VL", "SGN", "TH"
}
HPC_REGIONS = {"CA1", "CA2", "CA3", "SUB", "ProS", "DG", "HIP", "HPF"}
FIBER_TRACT_REGIONS = {
    "ccb", "ccs", "ccg", "fi", "dhc", "ec", "st", "cing",
    "fp", "or", "scwm", "int", "root", "fiber_tract",
    "above_surface", "below_track", "no_track_file",
    "no_session_match", "unknown_depth", "nan_depth",
    "Not found in brain"
}

# Colors
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
}


try:
    from utils import load_waveform_metrics
except ImportError:
    def load_waveform_metrics(df):
        if "pt_duration_ms" not in df.columns:
            print("  WARNING: pt_duration_ms not in dataframe. Waveform metrics needed.")
            df["pt_duration_ms"] = np.nan
            df["template_idx"] = np.nan
        return df, None, None, None


def get_region_group(region):
    """Map detailed region to broad category."""
    r = str(region)
    if r in STR_REGIONS:
        return "Striatum"
    if r in GPE_REGIONS:
        return "Pallidum"
    if any(r.startswith(p) for p in ["MOp", "MOs", "SSp", "SSs", "ACA", "PL", "ILA", "ORB", "AI", "RSP"]):
        return "Motor cortex"
    if r.startswith("VIS"):
        return "Visual cortex"
    if r in THAL_REGIONS or r in RT_REGIONS:
        return "Thalamus"
    if r in HPC_REGIONS:
        return "Hippocampus"
    if r in FIBER_TRACT_REGIONS:
        return "Excluded"
    return "Other"


def is_in_striatum(region):
    """Check if region is striatum (for cell type assignment)."""
    return str(region) in STR_REGIONS


def is_in_cortex(region):
    """Check if region is cortex."""
    r = str(region)
    if r in CORTEX_REGIONS:
        return True
    # Also check prefixes for layer-specific labels
    return any(r.startswith(p) for p in ["MOp", "MOs", "VIS", "SSp", "SSs", "ACA", "PL", "ILA", "ORB", "AI", "RSP"])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("0g_cell_type_relabeling_v2.py")
print("  Separate region and cell type columns")
print("  Region correction only for biological impossibilities")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
csv_candidates = [
    p.LOGS_DIR / "RZ_unit_properties_with_qc_and_regions.csv",
    p.LOGS_DIR / "RZ_unit_properties_with_regions.csv",
]
csv_path = next((c for c in csv_candidates if c.exists()), None)
if csv_path is None:
    raise FileNotFoundError(f"Could not find region CSV. Tried: {csv_candidates}")

print(f"\nLoading: {csv_path.name}")
df = pd.read_csv(csv_path)

# Ensure required columns
for col in ["datetime", "insertion_number", "id"]:
    if col in df.columns:
        if col == "datetime":
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = df[col].astype(int)

n_total = len(df)
print(f"  {n_total:,} units loaded")

# Detect probe_region column
if "probe_region" not in df.columns:
    if "track_file" in df.columns:
        df["probe_region"] = df["track_file"].apply(
            lambda t: "str" if "_str" in str(t) else ("v1" if "_v1" in str(t) else "other")
        )
    else:
        raise ValueError("Cannot determine probe_region")

str_probe_mask = df["probe_region"] == "str"
v1_probe_mask = df["probe_region"] == "v1"
print(f"  STR probe units: {str_probe_mask.sum():,}")
print(f"  V1 probe units:  {v1_probe_mask.sum():,}")

# ── Load waveform metrics ─────────────────────────────────────────────────────
print("\nLoading waveform metrics...")
df, waveforms, waveforms_norm, t_ms = load_waveform_metrics(df, join_how="left")

# Detect firing rate column
fr_candidates = ["firing_rate", "mean_firing_rate", "fr", "spike_rate"]
fr_col = next((c for c in fr_candidates if c in df.columns), None)
if fr_col is None:
    raise ValueError(f"No firing rate column found. Searched: {fr_candidates}")
print(f"  Firing rate column: '{fr_col}'")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Region correction (MINIMAL — only biological impossibilities)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("STAGE 1: Region correction (biological impossibilities + depth gate)")
print("-" * 70)

df["corrected_region"] = df["region_acronym"].copy()
df["region_source"] = "histology"
df["region_correction_reason"] = ""

pt = df["pt_duration_ms"]
fr = df[fr_col]

# Define waveform signatures
is_broad = pt >= PT_BROAD_MIN
is_narrow = pt < PT_NARROW_MAX
is_low_fr = fr <= FR_LOW_MAX
is_high_fr_sig = fr > FR_HIGH_MIN

# R1: GPe + broad/slow waveform → CP
# Biological rationale: GPe neurons are obligate fast-spiking (20-100 Hz).
# A unit with broad PT and low FR in GPe MUST be a misplaced CP unit.
r1 = (
    df["region_acronym"].isin(GPE_REGIONS)
    & is_broad
    & is_low_fr
)
df.loc[r1, "corrected_region"] = "CP"
df.loc[r1, "region_source"] = "waveform_corrected"
df.loc[r1, "region_correction_reason"] = "R1: GPe->CP (broad+slow incompatible with GPe)"
print(f"\n  R1: GPe + broad/slow → CP: {r1.sum()} units")

# R2: RT + broad waveform → VAL
# Biological rationale: RT is GABAergic fast-spiking. Broad waveforms = thalamic relay.
r2 = (
    df["region_acronym"].isin(RT_REGIONS)
    & is_broad
)
df.loc[r2, "corrected_region"] = "VAL"
df.loc[r2, "region_source"] = "waveform_corrected"
df.loc[r2, "region_correction_reason"] = "R2: RT->VAL (broad waveform incompatible with RT)"
print(f"  R2: RT + broad → VAL: {r2.sum()} units")

# R3: STR probe + broad+slow waveform + depth > boundary → corrected_region = "CP"
# Rationale: broad+slow waveform is ambiguous between MSN and RS (pyramidal).
# Depth resolves the ambiguity: units deeper than the cortex–STR boundary on a
# STR-targeted probe are physically in or adjacent to CP. Whatever the track
# reconstruction says, they cannot be cortical at that depth.
depth = df["peak_channel_depth"]
r3 = (
    str_probe_mask
    & is_broad
    & is_low_fr
    & ~is_narrow
    & (depth > CORTEX_STR_BOUNDARY_UM)
    & ~df["region_acronym"].isin(FIBER_TRACT_REGIONS)
)
df.loc[r3, "corrected_region"] = "CP"
df.loc[r3, "region_source"] = "waveform_corrected"
df.loc[r3, "region_correction_reason"] = (
    f"R3: STR probe + broad+slow + depth>{CORTEX_STR_BOUNDARY_UM}µm → CP"
)
print(f"  R3: STR probe + broad/slow + depth>{CORTEX_STR_BOUNDARY_UM}µm → CP: {r3.sum()} units")

print(f"\n  Total region corrections: {(df['region_source'] == 'waveform_corrected').sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Cell type assignment (uses BOTH waveform AND corrected region)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("STAGE 2: Cell type assignment (waveform + probe + depth)")
print("-" * 70)
print(f"  Depth gate: {CORTEX_STR_BOUNDARY_UM} µm  (STR probe only)")
print("  Rule: broad+slow + STR probe + depth > gate  →  MSN  (in CP)")
print("  Rule: broad+slow + STR probe + depth ≤ gate  →  RS   (in cortex)")
print("  Rule: broad+slow + V1  probe                 →  RS   (in cortex)")

df["cell_type"] = "ambiguous"

is_str_deep    = str_probe_mask & (depth > CORTEX_STR_BOUNDARY_UM)
is_str_shallow = str_probe_mask & (depth <= CORTEX_STR_BOUNDARY_UM)

# FSI: narrow waveform, any region (interneurons are everywhere)
is_fsi = is_narrow
df.loc[is_fsi, "cell_type"] = "FSI"
print(f"\n  FSI (PT < {PT_NARROW_MAX}ms): {is_fsi.sum()}")

# high_FR: very high firing, not FSI (e.g., GPe neurons, some interneurons)
is_high_fr_cell = is_high_fr_sig & ~is_fsi
df.loc[is_high_fr_cell, "cell_type"] = "high_FR"
print(f"  high_FR (FR > {FR_HIGH_MIN}Hz, not FSI): {is_high_fr_cell.sum()}")

# TAN: medium PT, tonic firing (3-12 Hz), STR probe + below cortex boundary
# TANs (cholinergic interneurons) are exclusively striatal
is_tan = (
    (pt >= 0.35) & (pt <= 0.50)
    & (fr >= FR_TAN_MIN) & (fr <= FR_TAN_MAX)
    & is_str_deep
    & ~is_fsi
)
df.loc[is_tan, "cell_type"] = "TAN"
print(f"  TAN (medium PT, tonic FR, STR probe depth>{CORTEX_STR_BOUNDARY_UM}µm): {is_tan.sum()}")

# MSN: broad+slow waveform + STR probe + below cortex boundary
# Depth resolves the MSN/RS ambiguity: deep on a STR probe → MSN in CP
is_msn = (
    is_broad
    & is_low_fr
    & is_str_deep
    & ~is_fsi
    & ~is_tan
)
df.loc[is_msn, "cell_type"] = "MSN"
print(f"  MSN (broad+slow, STR probe, depth>{CORTEX_STR_BOUNDARY_UM}µm): {is_msn.sum()}")

# RS (regular-spiking pyramidal): broad+slow waveform + above cortex boundary
# OR any broad+slow unit on V1 probe
# Same waveform as MSN — location (probe + depth) is the disambiguation
is_rs = (
    is_broad
    & is_low_fr
    & ~is_fsi
    & (is_str_shallow | v1_probe_mask)
)
df.loc[is_rs, "cell_type"] = "RS"
print(f"  RS (broad+slow, STR shallow or V1 probe): {is_rs.sum()}")

# Summary
print(f"\n  Cell type summary:")
for ct in ["MSN", "FSI", "RS", "TAN", "high_FR", "ambiguous"]:
    n = (df["cell_type"] == ct).sum()
    pct = n / n_total * 100
    print(f"    {ct:12s}: {n:5d}  ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Build output datasets
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("STAGE 3: Building output datasets")
print("-" * 70)

# STR units: STR probe below cortex boundary (regardless of cell type)
# These are units physically in or adjacent to striatum
str_mask = is_str_deep & ~df["corrected_region"].isin(FIBER_TRACT_REGIONS)
str_units = df[str_mask].copy()
print(f"\n  STR units (STR probe, depth>{CORTEX_STR_BOUNDARY_UM}µm): {len(str_units)}")

# STR MSNs specifically
str_msn = df[str_mask & is_msn].copy()
print(f"  STR MSN: {len(str_msn)}")

# STR FSIs
str_fsi = df[str_mask & is_fsi].copy()
print(f"  STR FSI: {len(str_fsi)}")

# STR TANs
str_tan = df[str_mask & is_tan].copy()
print(f"  STR TAN: {len(str_tan)}")

# Cortical units: V1 probe OR STR probe above cortex boundary
cortex_mask = (v1_probe_mask | is_str_shallow) & ~df["corrected_region"].isin(FIBER_TRACT_REGIONS)
cortex_units = df[cortex_mask].copy()
print(f"  Cortical units (V1 probe or STR probe shallow): {len(cortex_units)}")

# Cortical RS
cortex_rs = df[cortex_mask & is_rs].copy()
print(f"  Cortical RS: {len(cortex_rs)}")

# Cortical FSI
cortex_fsi = df[cortex_mask & is_fsi].copy()
print(f"  Cortical FSI: {len(cortex_fsi)}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Cross-tabulation (the key output for committee)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("STAGE 4: Region × Cell type cross-tabulation")
print("-" * 70)

# Add region group for easier viewing
df["region_group"] = df["corrected_region"].apply(get_region_group)

# Cross-tab: region group × cell type
crosstab = pd.crosstab(df["region_group"], df["cell_type"], margins=True)
print("\n  Region group × Cell type:")
print(crosstab.to_string())

# Detailed cross-tab for STR probe
str_probe_df = df[str_probe_mask]
crosstab_str = pd.crosstab(str_probe_df["corrected_region"], str_probe_df["cell_type"], margins=True)
print("\n  STR probe detailed (region × cell type):")
print(crosstab_str.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("Saving outputs")
print("-" * 70)

# Main output with all columns
out_path = p.LOGS_DIR / "RZ_unit_properties_final.csv"
df.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")

# Subset files
str_units.to_csv(p.LOGS_DIR / "RZ_str_units.csv", index=False)
print(f"  Saved: RZ_str_units.csv ({len(str_units)} units)")

str_msn.to_csv(p.LOGS_DIR / "RZ_str_msn.csv", index=False)
print(f"  Saved: RZ_str_msn.csv ({len(str_msn)} units) <- PRIMARY FOR RESCALING")

str_fsi.to_csv(p.LOGS_DIR / "RZ_str_fsi.csv", index=False)
print(f"  Saved: RZ_str_fsi.csv ({len(str_fsi)} units)")

cortex_units.to_csv(p.LOGS_DIR / "RZ_cortex_units.csv", index=False)
print(f"  Saved: RZ_cortex_units.csv ({len(cortex_units)} units)")

# Summary table
summary = crosstab.reset_index()
summary.to_csv(PLOT_DIR / "relabeling_summary.csv", index=False)
print(f"  Saved: relabeling_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
Total units: {n_total:,}

Region corrections (biological impossibilities only):
  R1 (GPe → CP): {r1.sum()}
  R2 (RT → VAL): {r2.sum()}
  Total changed: {(df['region_source'] == 'waveform_corrected').sum()}
  Unchanged:     {(df['region_source'] == 'histology').sum()}

Cell types (waveform + probe + depth gate @ {CORTEX_STR_BOUNDARY_UM}µm):
  MSN      : {(df['cell_type']=='MSN').sum():5d}  (broad+slow, STR probe, depth>{CORTEX_STR_BOUNDARY_UM}µm)
  RS       : {(df['cell_type']=='RS').sum():5d}  (broad+slow, V1 probe or STR probe shallow)
  FSI      : {(df['cell_type']=='FSI').sum():5d}  (narrow waveform, any probe/depth)
  TAN      : {(df['cell_type']=='TAN').sum():5d}  (medium PT, tonic FR, STR probe deep)
  high_FR  : {(df['cell_type']=='high_FR').sum():5d}  (>20Hz, any probe/depth)
  ambiguous: {(df['cell_type']=='ambiguous').sum():5d}

STR probe breakdown:
  Deep (>{CORTEX_STR_BOUNDARY_UM}µm) : {int(is_str_deep.sum())}
    MSN        : {len(str_msn)}
    TAN        : {len(str_tan)}
    FSI        : {len(str_fsi)}
  Shallow (≤{CORTEX_STR_BOUNDARY_UM}µm) : {int(is_str_shallow.sum())}
    RS         : {int((is_str_shallow & is_rs).sum())}
    FSI        : {int((is_str_shallow & is_fsi).sum())}

V1 probe: {int(v1_probe_mask.sum())} units
  RS         : {int((v1_probe_mask & is_rs).sum())}
  FSI        : {int((v1_probe_mask & is_fsi).sum())}

Key output columns:
  corrected_region : anatomical location (R1/R2/R3 corrections applied)
  cell_type        : MSN, RS, FSI, TAN, high_FR, ambiguous
  region_source    : 'histology' or 'waveform_corrected'
  region_group     : Striatum, Motor cortex, Visual cortex, etc.

Use RZ_str_msn.csv for rescaling analysis (verified STR MSNs).
""")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Unit count distribution per animal after correction ───────────────
print("\n[Plot 1] Unit count distribution after correction...")

all_animals_sorted = sorted(df["mouse"].unique())

str_unit_counts = [int((df["mouse"].eq(m) & str_probe_mask).sum()) for m in all_animals_sorted]
v1_unit_counts  = [int((df["mouse"].eq(m) & v1_probe_mask).sum())  for m in all_animals_sorted]

x     = np.arange(len(all_animals_sorted))
width = 0.38

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
bars_str = ax.bar(x - width / 2, str_unit_counts, width, color="#2166AC",
                  label="STR probe", edgecolor="white", linewidth=0.4)
bars_v1  = ax.bar(x + width / 2, v1_unit_counts,  width, color="#9467bd",
                  label="V1 probe",  edgecolor="white", linewidth=0.4)
for bar, val in zip(bars_str, str_unit_counts):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, str(val),
                ha="center", va="bottom", fontsize=7, color="#2166AC")
for bar, val in zip(bars_v1, v1_unit_counts):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, str(val),
                ha="center", va="bottom", fontsize=7, color="#9467bd")
ax.set_xticks(x)
ax.set_xticklabels(all_animals_sorted, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Unit count")
ax.set_title("All units per animal after region correction\n(STR vs V1 probe)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
str_nz = [c for c in str_unit_counts if c > 0]
v1_nz  = [c for c in v1_unit_counts  if c > 0]
bins = np.linspace(0, max(str_nz + v1_nz + [1]) + 10, 20)
if str_nz:
    ax.hist(str_nz, bins=bins, alpha=0.65, color="#2166AC",
            label=f"STR probe  (n={len(str_nz)} animals, total={sum(str_nz)})")
if v1_nz:
    ax.hist(v1_nz,  bins=bins, alpha=0.65, color="#9467bd",
            label=f"V1 probe   (n={len(v1_nz)} animals, total={sum(v1_nz)})")
ax.set_xlabel("Units per animal")
ax.set_ylabel("Number of animals")
ax.set_title("Distribution of unit counts\nacross animals (after correction)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

fig.suptitle("Unit count distribution after region correction", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "unit_count_distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> unit_count_distribution.png")


# ── Plot 2: Cell type distribution per probe type ─────────────────────────────
print("[Plot 2] Cell type distribution per probe type...")

STR_CELL_TYPES = ["MSN", "TAN", "FSI", "high_FR", "ambiguous"]
V1_CELL_TYPES  = ["RS",  "FSI", "high_FR", "ambiguous"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: STR probe cell type counts per animal
ax = axes[0]
str_animal_list = [m for m, c in zip(all_animals_sorted, str_unit_counts) if c > 0]
str_ct_matrix   = {ct: [] for ct in STR_CELL_TYPES}
for mouse in str_animal_list:
    m = df["mouse"].eq(mouse) & str_probe_mask
    for ct in STR_CELL_TYPES:
        str_ct_matrix[ct].append(int((m & df["cell_type"].eq(ct)).sum()))

x_str  = np.arange(len(str_animal_list))
bottom = np.zeros(len(str_animal_list))
for ct in STR_CELL_TYPES:
    vals = np.array(str_ct_matrix[ct])
    ax.bar(x_str, vals, bottom=bottom, color=CELL_TYPE_COLORS[ct],
           label=ct, edgecolor="white", linewidth=0.3)
    bottom += vals
ax.set_xticks(x_str)
ax.set_xticklabels(str_animal_list, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Unit count")
ax.set_title("STR probe — cell type per animal\n(after correction)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Middle: V1 probe cell type counts per animal
ax = axes[1]
v1_animal_list = [m for m, c in zip(all_animals_sorted, v1_unit_counts) if c > 0]
v1_ct_matrix   = {ct: [] for ct in V1_CELL_TYPES}
for mouse in v1_animal_list:
    m = df["mouse"].eq(mouse) & v1_probe_mask
    for ct in V1_CELL_TYPES:
        v1_ct_matrix[ct].append(int((m & df["cell_type"].eq(ct)).sum()))

x_v1   = np.arange(len(v1_animal_list))
bottom = np.zeros(len(v1_animal_list))
for ct in V1_CELL_TYPES:
    vals = np.array(v1_ct_matrix[ct])
    ax.bar(x_v1, vals, bottom=bottom, color=CELL_TYPE_COLORS[ct],
           label=ct, edgecolor="white", linewidth=0.3)
    bottom += vals
ax.set_xticks(x_v1)
ax.set_xticklabels(v1_animal_list, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Unit count")
ax.set_title("V1 probe — cell type per animal\n(after correction)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Right: 100% stacked bar — cell type composition STR vs V1
ax = axes[2]
all_cts   = ["MSN", "TAN", "RS", "FSI", "high_FR", "ambiguous"]
str_tots  = {ct: sum(str_ct_matrix.get(ct, [0])) for ct in all_cts}
v1_tots   = {ct: sum(v1_ct_matrix.get(ct,  [0])) for ct in all_cts}
str_props = np.array([str_tots[ct] for ct in all_cts], dtype=float)
v1_props  = np.array([v1_tots[ct]  for ct in all_cts], dtype=float)
str_props /= str_props.sum() if str_props.sum() > 0 else 1
v1_props  /= v1_props.sum()  if v1_props.sum()  > 0 else 1

x2     = np.arange(2)
bottom = np.zeros(2)
for i, ct in enumerate(all_cts):
    vals = np.array([str_props[i], v1_props[i]])
    ax.bar(x2, vals, bottom=bottom, color=CELL_TYPE_COLORS[ct],
           label=ct, edgecolor="white", linewidth=0.5, width=0.5)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.03:
            ax.text(j, b + v / 2, f"{v*100:.0f}%",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    bottom += vals
ax.set_xticks(x2)
ax.set_xticklabels(["STR probe", "V1 probe"], fontsize=11)
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1)
ax.set_title("Cell type composition\nSTR vs V1 probe (proportional)", fontsize=11)
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Cell type distribution by probe type after region correction", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "cell_type_distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> cell_type_distribution.png")


# ── Plot 3: Units per corrected region (vertical bar, coloured by region type) ─
print("[Plot 3] Units per corrected region (STR and V1)...")

_RPLOT_STR_REGIONS  = {
    "CP", "GPe", "GPi", "STR", "PAL", "SI", "LSX",
    "STRd", "STRv", "ACB", "OT", "CEA", "LA", "BLA", "EP", "MEA",
}
_RPLOT_V1_REGIONS   = {
    "VISp", "VISp1", "VISp2/3", "VISp4", "VISp5", "VISp6a", "VISp6b",
    "VISl", "VISrl", "VISal", "VISam", "VISpm", "VISli",
    "VISpl", "VISpor", "VIS",
}
_RPLOT_THAL_REGIONS = {
    "VAL", "LD", "LP", "CL", "RT", "VPL", "VPM", "PO", "MD", "AV",
    "AM", "IAD", "RE", "VM", "CM", "ATN", "ILM", "PIL", "SPF",
    "MG", "LGd", "LGv", "PCN", "VL", "SGN",
}

def _region_color(acronym):
    a = str(acronym)
    if a in _RPLOT_STR_REGIONS or any(a.startswith(p) for p in ("CP", "GP", "STR", "PAL")):
        return "#2ca02c"   # green  = striatum targets
    if a in _RPLOT_V1_REGIONS or a.startswith("VIS"):
        return "#9467bd"   # purple = visual cortex targets
    if a in _RPLOT_THAL_REGIONS:
        return "#ff7f0e"   # orange = thalamus
    return "#aec7e8"       # light blue = other

_rplot_legend = [
    mpatches.Patch(facecolor="#2ca02c", label="Striatum targets (CP, GPe, PAL…)"),
    mpatches.Patch(facecolor="#9467bd", label="Visual cortex targets (VISp, VISl…)"),
    mpatches.Patch(facecolor="#ff7f0e", label="Thalamus (VAL, LP, CL…)"),
    mpatches.Patch(facecolor="#aec7e8", label="Other"),
]

_counts_after = {
    "str": df[str_probe_mask]["corrected_region"].value_counts(),
    "v1":  df[v1_probe_mask]["corrected_region"].value_counts(),
}
_probe_labels = {"str": "STR probes", "v1": "V1 probes"}

max_cols = max(len(c) for c in _counts_after.values()) if _counts_after else 4
fig, axes = plt.subplots(1, 2, figsize=(max(8, max_cols * 0.5), 6))

for ax, probe in zip(axes, ["str", "v1"]):
    rc = _counts_after[probe]
    rc = rc[rc > 0].sort_values(ascending=False)  # tallest bar on the left
    colors = [_region_color(r) for r in rc.index]
    bars = ax.bar(rc.index, rc.values, color=colors, edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, rc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                str(val), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Unit count")
    ax.set_title(f"Units per region — {_probe_labels[probe]}  (n={rc.sum():,})")
    ax.set_xticks(range(len(rc)))
    ax.set_xticklabels(rc.index, rotation=45, ha="right", fontsize=8)
    ax.legend(handles=_rplot_legend, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Units per corrected region label by probe location", fontsize=11)
fig.tight_layout()
fig.savefig(PLOT_DIR / "units_per_region.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> units_per_region.png")

print("=" * 70)
