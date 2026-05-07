"""
0h_cell_type_relabeling.py
===========================
Two-stage pipeline that takes the atlas-only labels from 0f, the biological
flags from 0g, and produces final region + cell-type assignments.

Pipeline:
  0f → atlas labels (no waveforms)
  0g → biological flags (flagged_units.csv, flagged_units_gpe_boundary.csv,
       ccb_units.csv, probe_context_check.csv)
  0h → recovery rules + cell-type classification → final CSV → re-run 0g

Stage 1 — Region correction (location + waveform with explicit thresholds):
  R1: GPe → CP                    if unit appears in flagged_units_gpe_boundary
                                   (0g already gated PT≥0.40 ms, FR≤15 Hz)
  R2: RT  → VAL                   if PT ≥ PT_BROAD_MIN (broad ≠ RT FS)
  R3: STR probe + broad/slow + depth > 1500 µm → CP
  R4: ccb/VL/scwm/fi + STR probe + strict MSN (PT≥0.45, FR<4) + depth > 1500 µm → CP

Stage 2 — Cell type assignment (waveform + corrected_region):
  MSN       ⟸ in CP/STR + PT ≥ 0.40 ms + not FSI + not TAN
              (waveform-only; no FR bound — Berke 2004, Mello 2015)
  RS        ⟸ in cortex region + PT ≥ 0.40 ms + FR < 30 Hz
  FSI       ⟸ PT < 0.35 ms (any region)
  TAN       ⟸ in CP/STR + PT 0.35–0.55 ms + FR 3–12 Hz
  high_FR   ⟸ FR > 20 Hz + not FSI + NOT in CP/STR
  ambiguous ⟸ otherwise

Output columns:
  corrected_region   anatomical location (only changed by R1–R4)
  cell_type          MSN, FSI, RS, TAN, high_FR, ambiguous
  region_source      'histology' or 'waveform_corrected'
  relabel_reason     which rule fired (with explicit thresholds), if any
  region_group       broad bucket (Striatum, Motor cortex, Visual cortex, …)

Inputs:
  RZ_unit_properties_with_qc_and_regions.csv     (output of 0f, atlas-only)
  data/location_matching/diagnostic/
      flagged_units_gpe_boundary.csv             (0g — drives R1)
      ccb_units.csv                              (0g — cross-checked by R4)
      probe_context_check.csv                    (0g — informational)

Outputs:
  RZ_unit_properties_final.csv  (canonical post-correction)
  RZ_str_units.csv  RZ_str_msn.csv  RZ_str_fsi.csv
  RZ_cortex_units.csv  RZ_v1_cortical.csv
  relabeling_summary.csv

  Then 0g is re-run against the corrected labels to produce a "post-relabel"
  diagnostic. The pre-relabel 0g flag CSVs are first copied to
  data/location_matching/diagnostic/pre_relabel/ for comparison.

Plots (in DATA_DIR/location_matching/cell_type/):
  unit_count_distribution.png
  cell_type_distribution.png
  units_per_region.png
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import paths as p
import constants as c
from utils import load_waveform_metrics

PLOT_DIR = p.DATA_DIR / "location_matching" / "cell_type"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Waveform thresholds ───────────────────────────────────────────────────────
PT_BROAD_MIN  = 0.40   # ms — minimum PT for broad waveform (MSN or RS)
PT_NARROW_MAX = 0.35   # ms — maximum PT for narrow waveform (FSI)
FR_LOW_MAX    = 15.0   # Hz — appears in R1's relabel_reason string (0g gate doc)
FR_HIGH_MIN   = 20.0   # Hz — minimum FR for high-firing (GPe-like)

# ── Region sets ───────────────────────────────────────────────────────────────
STR_REGIONS = {
    "CP", "STR", "STRd", "STRv", "ACB", "OT",
    "SI", "LSX", "CEA", "LA", "BLA", "EP", "MEA"
    # PAL (pallidum) excluded: 66% narrow waveforms, 22.9% bio-ok — not MSN source
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
print("0h_cell_type_relabeling.py")
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

# Strip stale waveform columns from a previous 0h run — load_waveform_metrics
# re-introduces template_idx on its merge and a duplicate would collide.
_STALE_WF_COLS = ["template_idx", "pt_duration_ms", "trough_amp_uv",
                  "peak_amp_uv", "pt_ratio"]
df = df.drop(columns=[c for c in _STALE_WF_COLS if c in df.columns])

# Idempotency: if region_acronym_atlas is present, this CSV was already
# processed by 0h. Restore atlas labels to region_acronym (and discard the
# previous correction columns) so the rules below run against the same atlas
# state as the first invocation. Without this the canonical CSV ends up
# locked to whatever the previous run produced and re-runs are no-ops.
_PREV_RUN_COLS = ["corrected_region", "region_source", "relabel_reason",
                  "cell_type", "region_group"]
if "region_acronym_atlas" in df.columns:
    print("  Detected prior 0h output (region_acronym_atlas present); restoring atlas labels.")
    df["region_acronym"] = df["region_acronym_atlas"]
    df = df.drop(columns=["region_acronym_atlas"]
                 + [c for c in _PREV_RUN_COLS if c in df.columns])

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


# ── Load 0g diagnostic flag CSVs ──────────────────────────────────────────────
# Stage 1 rules load these directly inside the stage — G_DIAG_DIR is the path.
G_DIAG_DIR = p.DATA_DIR / "location_matching" / "diagnostic"

# Detect firing rate column
fr_candidates = ["firing_rate", "mean_firing_rate", "fr", "spike_rate"]
fr_col = next((c for c in fr_candidates if c in df.columns), None)
if fr_col is None:
    raise ValueError(f"No firing rate column found. Searched: {fr_candidates}")
print(f"  Firing rate column: '{fr_col}'")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Region correction (location + waveform, using 0g diagnostic CSVs)
# ══════════════════════════════════════════════════════════════════════════════
# Every rule requires BOTH a location criterion (from 0g's spatial analysis)
# AND a waveform criterion. Neither alone is sufficient.
#
#   R1: GPe → CP    near CP boundary + MSN-like waveform (from 0g GPe flag)
#   R2: RT  → CP    within CP depth range + broad waveform (from 0g RT check)
#   R3: ccb → CP    nearest atlas region = CP + broad waveform
#   R4: ccb → MOp6a nearest atlas region = MOp6a + gap ≤ 10 µm (cortex boundary)
#   R5: ccb → unplaceable  all remaining ccb (ventricle-adjacent or ambiguous)
#
# Rules are applied in order; a unit already relabeled by an earlier rule is
# not re-examined by subsequent rules.
print("\n" + "-" * 70)
print("STAGE 1: Region correction (0g location + waveform criteria)")
print("-" * 70)

df["corrected_region"] = df["region_acronym"].copy()
df["region_source"]    = "histology"
df["relabel_reason"]   = ""

pt = df["pt_duration_ms"]
fr = df[fr_col]
is_broad  = pt >= PT_BROAD_MIN
is_narrow = pt < PT_NARROW_MAX

# Build unit-key lookup for fast membership testing
UNIT_KEYS = ["mouse", "datetime", "insertion_number", "paramset_idx", "id"]
_INT_KEYS = ["insertion_number", "paramset_idx", "id"]
_unit_tuples = list(map(tuple, df[UNIT_KEYS].to_numpy()))


def _to_unit_set(rows):
    """Build a set of unit-key tuples from a DataFrame slice (rows)."""
    return set(map(tuple,
        rows[UNIT_KEYS].dropna()
            .astype({c: int for c in _INT_KEYS}).to_numpy()
    ))


def _unit_set(name):
    """Load a 0g flag CSV and return a set of unit-key tuples."""
    path = G_DIAG_DIR / name
    if not path.exists():
        print(f"  WARNING: {path} missing — run 0g first. Returning empty set.")
        return set()
    flag_df = pd.read_csv(path, parse_dates=["datetime"])
    flag_df["insertion_number"] = flag_df["insertion_number"].astype(int)
    flag_df["paramset_idx"]     = flag_df["paramset_idx"].astype(int)
    flag_df["id"]               = flag_df["id"].astype(int)
    return set(map(tuple, flag_df[UNIT_KEYS].to_numpy()))


def _unit_df(name):
    """Load a 0g flag CSV as a DataFrame (all columns)."""
    path = G_DIAG_DIR / name
    if not path.exists():
        print(f"  WARNING: {path} missing — run 0g first. Returning empty DataFrame.")
        return pd.DataFrame()
    df_out = pd.read_csv(path, parse_dates=["datetime"])
    if "insertion_number" in df_out.columns:
        df_out["insertion_number"] = df_out["insertion_number"].astype(int)
    if "paramset_idx" in df_out.columns:
        df_out["paramset_idx"] = df_out["paramset_idx"].astype(int)
    if "id" in df_out.columns:
        df_out["id"] = df_out["id"].astype(int)
    return df_out


print("\nLoading 0g diagnostic CSVs...")
gpe_boundary_set = _unit_set("flagged_units_gpe_boundary.csv")
rt_check_df      = _unit_df("rt_broad_boundary_check.csv")
ccb_df           = _unit_df("ccb_boundary_units.csv")
print(f"  GPe boundary candidates : {len(gpe_boundary_set)}")
print(f"  RT broad-waveform units : {len(rt_check_df)}")
print(f"  ccb boundary units      : {len(ccb_df)}")

in_gpe_flag = pd.Series([t in gpe_boundary_set for t in _unit_tuples], index=df.index)

# R1: GPe → CP
# GPe is fast-spiking (20–100 Hz). A GPe-labelled unit with broad waveform and
# low FR is biologically inconsistent with GPe and consistent with MSN.
# Gate already applied by 0g (PT ≥ 0.40 ms, FR ≤ 15 Hz).
r1 = in_gpe_flag & df["region_acronym"].isin(GPE_REGIONS)
df.loc[r1, "corrected_region"] = "CP"
df.loc[r1, "region_source"]    = "waveform_corrected"
df.loc[r1, "relabel_reason"]   = (
    f"R1: GPe→CP (0g gpe_boundary flag: PT≥{PT_BROAD_MIN}ms, FR≤{FR_LOW_MAX}Hz)"
)

# R2: RT + broad waveform + within CP depth range → CP
# RT (reticular thalamus) is exclusively GABAergic fast-spiking.
# Broad waveforms are biologically impossible for RT. Units near the CP depth
# range on their track are atlas boundary errors placing CP-edge neurons into RT.
# Units far from CP (>300 µm gap) stay as RT — they may be genuine thalamic
# boundary neurons that happen to be broad.
if len(rt_check_df):
    rt_cp_set = _to_unit_set(rt_check_df[rt_check_df["near_cp_300um"] == True])
    in_rt_cp  = pd.Series([t in rt_cp_set for t in _unit_tuples], index=df.index)
else:
    in_rt_cp  = pd.Series(False, index=df.index)

r2 = df["region_acronym"].isin(RT_REGIONS) & is_broad & in_rt_cp
df.loc[r2, "corrected_region"] = "CP"
df.loc[r2, "region_source"]    = "waveform_corrected"
df.loc[r2, "relabel_reason"]   = (
    f"R2: RT→CP (0g rt_check: within 300µm of CP depth range, PT≥{PT_BROAD_MIN}ms)"
)

# R3 / R4 / R5: ccb boundary units (nearest-region-driven)
# ccb has no neurons. Units labelled ccb are genuine recordings at the ccb
# boundary. The 0g neighbor analysis found the nearest non-ccb atlas region
# for each ccb unit on the same track. Rules:
#   R3: nearest = CP  + PT ≥ PT_BROAD_MIN → CP  (MSN/pyramidal at CP border)
#   R4: nearest = MOp6a + gap ≤ 10 µm    → MOp6a (cortex/ccb boundary)
#   R5: everything else                   → "unplaceable" (ventricle, scwm…)
already_relabeled = r1 | r2

if len(ccb_df) and "nearest_region" in ccb_df.columns:
    ccb_cp_set  = _to_unit_set(ccb_df[
        (ccb_df["nearest_region"] == "CP") &
        (ccb_df["pt_duration_ms"] >= PT_BROAD_MIN)
    ])
    ccb_mop_set = _to_unit_set(ccb_df[
        (ccb_df["nearest_region"] == "MOp6a") &
        (ccb_df["gap_um"] <= 10)
    ])
else:
    ccb_cp_set  = set()
    ccb_mop_set = set()

in_ccb_cp  = pd.Series([t in ccb_cp_set  for t in _unit_tuples], index=df.index)
in_ccb_mop = pd.Series([t in ccb_mop_set for t in _unit_tuples], index=df.index)

r3 = (df["region_acronym"] == "ccb") & in_ccb_cp & ~already_relabeled
df.loc[r3, "corrected_region"] = "CP"
df.loc[r3, "region_source"]    = "waveform_corrected"
df.loc[r3, "relabel_reason"]   = (
    f"R3: ccb→CP (nearest_region=CP, PT≥{PT_BROAD_MIN}ms)"
)

r4 = (df["region_acronym"] == "ccb") & in_ccb_mop & ~already_relabeled & ~r3
df.loc[r4, "corrected_region"] = "MOp6a"
df.loc[r4, "region_source"]    = "waveform_corrected"
df.loc[r4, "relabel_reason"]   = "R4: ccb→MOp6a (nearest_region=MOp6a, gap≤10µm)"

r5 = (df["region_acronym"] == "ccb") & ~already_relabeled & ~r3 & ~r4
df.loc[r5, "corrected_region"] = "unplaceable"
df.loc[r5, "region_source"]    = "waveform_corrected"
df.loc[r5, "relabel_reason"]   = "R5: ccb→unplaceable (ventricle-adjacent or no CP neighbour)"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Cell type assignment (waveform + corrected_region)
# ══════════════════════════════════════════════════════════════════════════════
# After Stage 1, corrected_region is the best available location estimate.
# Cell type is assigned using waveform criteria within each region class.
# peak_channel_depth is no longer used anywhere in this stage.
#
#   MSN : corrected_region ∈ STR_REGIONS + PT ≥ 0.40 ms + not FSI + not TAN
#         (waveform-only; no FR bound — Berke 2004, Mello 2015)
#   FSI : PT < 0.35 ms  (fast-spiking, any region)
#   TAN : corrected_region ∈ STR_REGIONS + PT 0.35–0.55 ms + FR 3–12 Hz
#   RS  : corrected_region ∈ CORTEX_REGIONS + PT ≥ 0.40 ms + FR < 30 Hz
#   high_FR : FR > 20 Hz + not FSI + NOT in STR_REGIONS
#             (in-STR high-FR broad units are MSNs by waveform, not high_FR)
#   ambiguous : all other
print("\n" + "-" * 70)
print("STAGE 2: Cell type assignment (waveform + corrected_region)")
print("-" * 70)

TAN_FR_MIN  = 3.0    # Hz — TANs fire tonically
TAN_FR_MAX  = 12.0   # Hz
TAN_PT_MAX  = 0.55   # ms — TANs are broad but not as wide as MSNs

df["cell_type"] = "ambiguous"

in_str     = df["corrected_region"].isin(STR_REGIONS)
in_cortex  = df["corrected_region"].apply(is_in_cortex)
is_unplace = df["corrected_region"] == "unplaceable"

# FSI: narrow waveform — applies across all regions
is_fsi = is_narrow
df.loc[is_fsi, "cell_type"] = "FSI"

# high_FR: very high firing, not FSI
is_high_fr_cell = (fr > FR_HIGH_MIN) & ~is_fsi
df.loc[is_high_fr_cell, "cell_type"] = "high_FR"

# TAN: cholinergic interneurons — exclusively striatal, medium PT, tonic FR
is_tan = (
    in_str
    & (pt >= PT_NARROW_MAX) & (pt <= TAN_PT_MAX)
    & (fr >= TAN_FR_MIN)    & (fr <= TAN_FR_MAX)
    & ~is_fsi
    & ~is_high_fr_cell
)
df.loc[is_tan, "cell_type"] = "TAN"

# MSN: broad waveform in CP/STR, not FSI, not TAN.
# No FR upper bound — Berke 2004 and Mello 2015 show MSNs span a wide FR
# range (resting <1 Hz to >20 Hz during movement). Waveform alone is the
# defining feature. This intentionally OVERWRITES the high_FR label for
# STR-broad units assigned by the earlier `is_high_fr_cell` block — those
# units are MSNs, not GPe-like high-FR neurons.
is_msn = (
    in_str
    & is_broad
    & ~is_fsi
    & ~is_tan
)
df.loc[is_msn, "cell_type"] = "MSN"

# RS: regular-spiking pyramidal — broad+low FR in cortical regions
is_rs = (
    in_cortex
    & is_broad
    & (fr < 30.0)
    & ~is_fsi
    & ~is_high_fr_cell
)
df.loc[is_rs, "cell_type"] = "RS"

# unplaceable units get their own label so they're easy to filter downstream
df.loc[is_unplace, "cell_type"] = "unplaceable"

print(f"\n  {'cell_type':<12} {'n':>5}  pct   criteria")
_ct_criteria = {
    "MSN":         f"STR region, PT≥{PT_BROAD_MIN}ms (waveform only, no FR bound)",
    "FSI":         f"PT<{PT_NARROW_MAX}ms, any region",
    "RS":          f"cortex region, PT≥{PT_BROAD_MIN}ms, FR<30Hz",
    "TAN":         f"STR region, PT {PT_NARROW_MAX}–{TAN_PT_MAX}ms, FR {TAN_FR_MIN}–{TAN_FR_MAX}Hz",
    "high_FR":     f"FR>{FR_HIGH_MIN}Hz, not FSI",
    "unplaceable": "ccb-derived, no neighbour",
    "ambiguous":   "no rule matched",
}
for ct, crit in _ct_criteria.items():
    n = (df["cell_type"] == ct).sum()
    print(f"  {ct:<12} {n:>5}  {n/n_total*100:>4.1f}%  {crit}")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT: region correction summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("AUDIT: Region correction summary")
print("-" * 70)

for rule, label in [("R1", "GPe→CP"), ("R2", "RT→CP"),
                    ("R3", "ccb→CP"), ("R4", "ccb→MOp6a"), ("R5", "ccb→unplaceable")]:
    n = df["relabel_reason"].str.startswith(rule).sum()
    print(f"  {label:20s}: {n}")

# Check: how many CP units came from corrections vs atlas
n_cp_atlas    = ((df["region_acronym"] == "CP") & (df["region_source"] == "histology")).sum()
n_cp_corrected = ((df["corrected_region"] == "CP") & (df["region_source"] == "waveform_corrected")).sum()
print(f"\n  CP from atlas (0f)        : {n_cp_atlas}")
print(f"  CP from corrections (0h)  : {n_cp_corrected}")
print(f"  CP total                  : {(df['corrected_region'] == 'CP').sum()}")

# Spot-check: any well-validated thalamic regions (VAL, PO) accidentally moved to CP?
for r in ["VAL", "PO", "VPM"]:
    n_moved = ((df["region_acronym"] == r) & (df["corrected_region"] == "CP")).sum()
    if n_moved:
        print(f"  WARNING: {n_moved} {r} units were moved to CP — review Stage 1 rules")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Build output datasets
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("STAGE 3: Building output datasets")
print("-" * 70)

df["region_group"] = df["corrected_region"].apply(get_region_group)
in_cortex_region = df["region_group"].isin({"Motor cortex", "Visual cortex"})
in_visual_cortex = df["region_group"] == "Visual cortex"

# STR units: corrected_region in STR_REGIONS (atlas + recovered)
str_mask  = df["corrected_region"].isin(STR_REGIONS)
str_units = df[str_mask].copy()
print(f"\n  STR units (corrected_region ∈ STR_REGIONS): {len(str_units)}")

str_msn = df[str_mask & is_msn].copy()
print(f"  STR MSN: {len(str_msn)}")

str_fsi = df[str_mask & is_fsi].copy()
print(f"  STR FSI: {len(str_fsi)}")

str_tan = df[str_mask & is_tan].copy()
print(f"  STR TAN: {len(str_tan)}")

# Cortical units: corrected_region in a motor/visual cortex label
cortex_mask  = in_cortex_region & (df["corrected_region"] != "unplaceable")
cortex_units = df[cortex_mask].copy()
print(f"  Cortical units (region_group ∈ {{Motor, Visual cortex}}): {len(cortex_units)}")

# V1 cortical: visual cortex on V1 probe
v1_cortical_mask = v1_probe_mask & in_visual_cortex
v1_cortical      = df[v1_cortical_mask].copy()
print(f"  V1 cortical (V1 probe + Visual cortex): {len(v1_cortical)}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Cross-tabulation (the key output for committee)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("STAGE 4: Region × Cell type cross-tabulation")
print("-" * 70)

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

# Preserve the atlas-only label from 0f under a new column, then promote
# corrected_region into region_acronym. This makes the saved CSV self-contained
# (downstream code can rely on region_acronym being the post-correction label)
# while keeping the pre-correction atlas label one column away.
df["region_acronym_atlas"] = df["region_acronym"]
df["region_acronym"]       = df["corrected_region"]

df.to_csv(p.LOGS_DIR / "RZ_unit_properties_final.csv", index=False)
str_units.to_csv(p.LOGS_DIR    / "RZ_str_units.csv",    index=False)
str_msn.to_csv(p.LOGS_DIR      / "RZ_str_msn.csv",      index=False)
str_fsi.to_csv(p.LOGS_DIR      / "RZ_str_fsi.csv",      index=False)
cortex_units.to_csv(p.LOGS_DIR / "RZ_cortex_units.csv", index=False)
v1_cortical.to_csv(p.LOGS_DIR  / "RZ_v1_cortical.csv",  index=False)
crosstab.reset_index().to_csv(PLOT_DIR / "relabeling_summary.csv", index=False)

print(f"  RZ_unit_properties_final.csv  ({len(df)} units, all columns)")
print(f"  RZ_str_units.csv              ({len(str_units)} units)  ← all CP/STR units (rescaling input)")
print(f"  RZ_str_msn.csv  RZ_str_fsi.csv  RZ_cortex_units.csv  RZ_v1_cortical.csv")
print(f"  relabeling_summary.csv        ({PLOT_DIR.name}/)")


# ══════════════════════════════════════════════════════════════════════════════
# POPULATION DECODER ELIGIBILITY (per population_decoder.py: MIN_UNITS=15)
# ══════════════════════════════════════════════════════════════════════════════
# population_decoder.py also requires >150 valid trials per session — that gate
# is applied at decode time (needs trial pickles), not here.
DECODER_MIN_UNITS = 15
SHORT_MICE = set(c.GROUP_DICT["s"])
LONG_MICE  = set(c.GROUP_DICT["l"])

print("\n" + "=" * 70)
print(f"POPULATION DECODER ELIGIBILITY  (≥{DECODER_MIN_UNITS} STR MSN units / session)")
print("=" * 70)

# Use date_only when present for readability; fall back to datetime.
_session_keys = (["mouse", "date_only", "insertion_number"]
                 if "date_only" in str_msn.columns
                 else ["mouse", "datetime", "insertion_number"])
msn_per_session = (
    str_msn.groupby(_session_keys)["id"].nunique().reset_index(name="n_msn")
)
msn_per_session["bg_group"] = msn_per_session["mouse"].apply(
    lambda m: "Short BG" if m in SHORT_MICE else ("Long BG" if m in LONG_MICE else "Unassigned")
)
msn_per_session["meets_threshold"] = msn_per_session["n_msn"] >= DECODER_MIN_UNITS

n_sessions     = len(msn_per_session)
n_qualify_total = int(msn_per_session["meets_threshold"].sum())
print(f"  Sessions with ≥1 STR MSN  : {n_sessions}")
print(f"  Sessions meeting threshold: {n_qualify_total}  "
      f"({100*n_qualify_total/n_sessions:.0f}%)" if n_sessions else
      "  No sessions with STR MSN units.")

if n_sessions:
    # ── Per-BG-group summary ──────────────────────────────────────────────────
    print(f"\n  {'BG group':<12} {'sessions':>8} {'qualify':>8} {'median':>7} {'max':>5}")
    for group in ("Short BG", "Long BG", "Unassigned"):
        sub = msn_per_session[msn_per_session["bg_group"] == group]
        if len(sub) == 0:
            continue
        n_q = int(sub["meets_threshold"].sum())
        print(f"  {group:<12} {len(sub):>8} {n_q:>8} "
              f"{sub['n_msn'].median():>7.0f} {sub['n_msn'].max():>5}")

    # ── Per-mouse summary ─────────────────────────────────────────────────────
    print("\n  Per-mouse summary:")
    per_mouse = (msn_per_session.groupby("mouse")
                 .agg(n_sessions=(_session_keys[1], "count"),
                      n_qualifying=("meets_threshold", "sum"),
                      median_msn=("n_msn", "median"),
                      max_msn=("n_msn", "max"))
                 .astype({"n_qualifying": int}))
    print(per_mouse.to_string())

    # ── Qualifying sessions (the downstream-facing answer) ────────────────────
    qualifying = msn_per_session[msn_per_session["meets_threshold"]].sort_values(
        ["mouse"] + _session_keys[1:]
    )
    print(f"\n  Sessions meeting the gate ({len(qualifying)}):")
    print(qualifying.to_string(index=False))

    # ── Below-gate sessions (top 20 by n_msn — closest to qualifying) ─────────
    below = msn_per_session[~msn_per_session["meets_threshold"]].sort_values(
        "n_msn", ascending=False
    )
    if len(below):
        print(f"\n  Sessions BELOW the gate (top 20 by n_msn):")
        print(below.head(20).to_string(index=False))

# ── Save the per-session table for downstream consumption ─────────────────────
session_decoder_path = PLOT_DIR / "session_msn_decoder_eligibility.csv"
msn_per_session.to_csv(session_decoder_path, index=False)
print(f"\n  Saved → {session_decoder_path}")
print(f"\n  Note: population_decoder.py also gates on >150 valid trials/session;")
print(f"  final session count after that gate may be lower than {n_qualify_total}.")


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

# Reuse the top-of-file region sets — no need to redefine them here.
def _region_color(acronym):
    a = str(acronym)
    if a in STR_REGIONS or a in GPE_REGIONS or a.startswith(("STR", "PAL")):
        return "#2ca02c"   # green  = striatum / pallidum
    if a.startswith("VIS"):
        return "#9467bd"   # purple = visual cortex
    if a in THAL_REGIONS or a in RT_REGIONS:
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


# ── Plot 4: Region × cell-type heatmap (top regions per probe) ────────────────
print("[Plot 4] Region × cell-type heatmap...")
_HEATMAP_CT_ORDER = ["MSN", "FSI", "TAN", "RS", "high_FR", "ambiguous", "unplaceable"]
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, (probe_key, label) in zip(axes, [("str", "STR Probe"), ("v1", "V1 Probe")]):
    sub = df[df["probe_region"] == probe_key]
    if not len(sub):
        ax.set_visible(False)
        continue
    top_regions = sub["corrected_region"].value_counts().head(12).index.tolist()
    sub_top = sub[sub["corrected_region"].isin(top_regions)]
    ct_tab  = pd.crosstab(sub_top["corrected_region"], sub_top["cell_type"])
    ct_tab  = ct_tab.reindex(columns=[c for c in _HEATMAP_CT_ORDER if c in ct_tab.columns],
                             fill_value=0)
    ct_tab  = ct_tab.loc[ct_tab.sum(axis=1).sort_values(ascending=False).index]
    im = ax.imshow(ct_tab.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(ct_tab.shape[1])); ax.set_xticklabels(ct_tab.columns, fontsize=9)
    ax.set_yticks(range(ct_tab.shape[0])); ax.set_yticklabels(ct_tab.index, fontsize=9)
    for i in range(ct_tab.shape[0]):
        for j in range(ct_tab.shape[1]):
            v = int(ct_tab.iat[i, j])
            if v == 0:
                continue
            txt_color = "white" if v > ct_tab.values.max() * 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color=txt_color)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(f"{label} — region × cell type (top 12)")
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Region")
fig.suptitle("Region × cell-type cross-tabulation", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "region_celltype_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> region_celltype_heatmap.png")


# ── Plot 5: Depth distributions on STR probes (sanity check on 1500 µm gate) ──
if "peak_channel_depth" in df.columns:
    print("[Plot 5] Depth distributions on STR probes...")
    str_df_plot = df[str_probe_mask].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for rg in ("Motor cortex", "Striatum", "Thalamus"):
        m = str_df_plot["region_group"] == rg
        if m.sum() < 5:
            continue
        d = str_df_plot.loc[m, "peak_channel_depth"].dropna()
        ax.hist(d, bins=30, alpha=0.5, label=f"{rg} (n={len(d)})",
                color=REGION_GROUP_COLORS.get(rg, "#888"))
    ax.axvline(1500, color="red", linestyle="--", linewidth=2, label="1500 µm gate")
    ax.set_xlabel("Peak channel depth (µm from probe tip)")
    ax.set_ylabel("Unit count")
    ax.set_title("STR probe — depth by region group")
    ax.legend(fontsize=8)

    ax = axes[1]
    for ct in ("MSN", "RS", "FSI"):
        m = str_df_plot["cell_type"] == ct
        if m.sum() < 5:
            continue
        d = str_df_plot.loc[m, "peak_channel_depth"].dropna()
        ax.hist(d, bins=30, alpha=0.5, label=f"{ct} (n={len(d)})",
                color=CELL_TYPE_COLORS.get(ct, "#888"))
    ax.axvline(1500, color="red", linestyle="--", linewidth=2, label="1500 µm gate")
    ax.set_xlabel("Peak channel depth (µm from probe tip)")
    ax.set_ylabel("Unit count")
    ax.set_title("STR probe — depth by cell type")
    ax.legend(fontsize=8)

    fig.suptitle("Depth distributions — cortex–STR boundary validation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "depth_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> depth_distributions.png")


# ── Plot 6: Mean waveforms by cell type (waveform-vs-classification sanity) ───
if "template_idx" in df.columns:
    print("[Plot 6] Mean waveforms by cell type...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax, ct in zip(axes, ["MSN", "FSI", "RS", "TAN", "high_FR", "ambiguous"]):
        m = (df["cell_type"] == ct) & df["template_idx"].notna()
        if m.sum() == 0:
            ax.set_title(f"{ct} (n=0)", fontsize=11,
                         color=CELL_TYPE_COLORS.get(ct, "#888"))
            ax.axis("off")
            continue
        idx = df.loc[m, "template_idx"].astype(int).to_numpy()
        idx = idx[idx < len(waveforms_norm)]
        if len(idx) == 0:
            ax.axis("off")
            continue
        wv = waveforms_norm[idx]
        mean_wv = wv.mean(axis=0)
        std_wv  = wv.std(axis=0)
        color   = CELL_TYPE_COLORS.get(ct, "#888")
        for i in range(min(30, len(wv))):
            ax.plot(t_ms, wv[i], alpha=0.12, linewidth=0.5, color=color)
        ax.fill_between(t_ms, mean_wv - std_wv, mean_wv + std_wv, alpha=0.3, color=color)
        ax.plot(t_ms, mean_wv, linewidth=2, color=color)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        ax.set_title(f"{ct} (n={int(m.sum())})", fontsize=11,
                     color=color, fontweight="bold")
        ax.set_xlabel("Time (ms)"); ax.set_xlim(t_ms[0], t_ms[-1])
        ax.set_ylabel("Normalised amplitude")
    fig.suptitle("Mean waveforms by cell type (± 1 SD)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "waveforms_by_celltype.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> waveforms_by_celltype.png")


# ── Plot 7: PT and FR 1D distributions by cell type (marginal sanity) ─────────
if "pt_duration_ms" in df.columns and fr_col in df.columns:
    print("[Plot 7] PT and FR distributions by cell type...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _ct_overlay = ["ambiguous", "high_FR", "TAN", "RS", "FSI", "MSN"]  # MSN drawn last

    ax = axes[0]
    for ct in _ct_overlay:
        m = df["cell_type"] == ct
        if m.sum() < 5:
            continue
        vals = df.loc[m, "pt_duration_ms"].dropna()
        ax.hist(vals, bins=40, range=(0.1, 0.9),
                color=CELL_TYPE_COLORS.get(ct, "#888"),
                alpha=0.85 if ct == "MSN" else 0.55,
                label=f"{ct} (n={len(vals)})",
                density=True, linewidth=0.4, edgecolor="white")
    ax.axvline(PT_NARROW_MAX, color="gray",  linestyle="--", linewidth=1.2,
               label=f"{PT_NARROW_MAX} ms (FSI)")
    ax.axvline(PT_BROAD_MIN,  color="black", linestyle="--", linewidth=1.2,
               label=f"{PT_BROAD_MIN} ms (broad)")
    ax.set_xlabel("PT duration (ms)"); ax.set_ylabel("Density")
    ax.set_title("PT duration by cell type")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    fr_bins = np.logspace(np.log10(0.05), np.log10(200), 45)
    for ct in _ct_overlay:
        m = df["cell_type"] == ct
        if m.sum() < 5:
            continue
        vals = df.loc[m, fr_col].dropna()
        vals = vals[vals > 0]
        ax.hist(vals, bins=fr_bins,
                color=CELL_TYPE_COLORS.get(ct, "#888"),
                alpha=0.85 if ct == "MSN" else 0.55,
                label=f"{ct} (n={len(vals)})",
                density=True, linewidth=0.4, edgecolor="white")
    ax.axvline(FR_HIGH_MIN, color="black", linestyle="--", linewidth=1.2,
               label=f"{FR_HIGH_MIN} Hz (high_FR)")
    ax.set_xscale("log")
    ax.set_xlabel("Firing rate (Hz, log)"); ax.set_ylabel("Density")
    ax.set_title("Firing rate by cell type")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.15)

    fig.suptitle("Marginal distributions by cell type",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pt_fr_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> pt_fr_distributions.png")


print(f"\nPlots saved to: {PLOT_DIR}")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION: re-run 0g against the corrected labels
# ══════════════════════════════════════════════════════════════════════════════
# The pre-relabel 0g output is the input we just consumed. Snapshot it under
# pre_relabel/, then promote the corrected labels into the canonical CSV that
# 0g reads, and re-invoke 0g. The post-relabel diagnostic shows whether the
# rules above actually moved units off the bio-inconsistent flag list.
print("\n" + "=" * 70)
print("VALIDATION — re-running 0g against corrected labels")
print("=" * 70)

# 1. Snapshot pre-relabel 0g flags so they aren't overwritten by the re-run.
PRE_RELABEL_DIR = G_DIAG_DIR / "pre_relabel"
PRE_RELABEL_DIR.mkdir(parents=True, exist_ok=True)
for fname in ("flagged_units.csv", "flagged_units_gpe_boundary.csv",
              "ccb_units.csv", "probe_context_check.csv",
              "pass_rate_summary.png"):
    src = G_DIAG_DIR / fname
    if src.exists():
        shutil.copy2(src, PRE_RELABEL_DIR / fname)
print(f"  Snapshotted pre-relabel flags → {PRE_RELABEL_DIR}")

# 2. Promote corrected labels into the canonical CSV that 0g reads. df was
#    already mutated above (region_acronym_atlas + region_acronym=corrected),
#    so we just save it back to the path 0g expects.
CANONICAL_CSV = p.LOGS_DIR / "RZ_unit_properties_with_qc_and_regions.csv"
df.to_csv(CANONICAL_CSV, index=False)
print(f"  Wrote corrected labels → {CANONICAL_CSV.name}")

# 3. Subprocess-run 0g. Streaming output to our stdout for visibility.
G_SCRIPT = Path(__file__).resolve().parent / "0g_waveform_diagnostic.py"
print(f"  Invoking: python {G_SCRIPT.name}")
print("-" * 70)
_result = subprocess.run(
    ["python", str(G_SCRIPT)],
    cwd=str(G_SCRIPT.parent),
    check=False,
)
print("-" * 70)
if _result.returncode != 0:
    print(f"  WARNING: 0g exited with code {_result.returncode}")
else:
    print("  0g validation pass complete.")

print(f"\n  Compare pre/post:")
print(f"    pre  → {PRE_RELABEL_DIR}")
print(f"    post → {G_DIAG_DIR}")
