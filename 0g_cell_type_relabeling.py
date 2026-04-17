"""
0g_cell_type_relabeling.py
===========================
Two-stage pipeline: cell type classification then region label correction.

Core principle
--------------
Waveform properties are a more reliable indicator of cell identity than
histological reconstruction. Reconstruction errors (probe too shallow/deep,
track tracing artefacts) are common. The MSN waveform signature — broad
peak-to-trough duration and very low firing rate — is highly distinctive
and not shared by any region the STR probe could plausibly mis-target:
  - Cortex (MOp/MOs)  : pyramidal cells are broad but fire faster on average
  - Thalamus (AV/VAL/CL): relay cells are broad but fire faster
  - GPe               : fires 20-100 Hz — completely different
  - White matter       : no neurons

Therefore, for units from STR-targeted probes:
    MSN waveform  =  MSN,  regardless of what the track reconstruction says

For V1-targeted probes the same logic does NOT apply — broad slow neurons
in VISp/MOp are genuine L5/L6 pyramidal cells, not MSNs.

Stage 1 — Cell type (waveform-based, location-agnostic)
  MSN   : PT >= 0.40 ms  AND  FR <= 15 Hz
  FSI   : PT < 0.35 ms
  high_FR: FR > 20 Hz  (and not FSI)
  ambiguous: everything else

Stage 2 — Region label correction (targeted rules)
  R1: GPe + MSN waveform            -> CP          (all probes)
  R2: RT  + non-FSI waveform        -> VAL         (all probes)
  R3: any region + MSN waveform     -> CP           IF from STR probe
      (the "waveform-primary" rule -- maximises MSN recovery from RZ034-039)
  R4: CA1/SUB + FSI waveform        -> HPC_border  IF from V1 probe
  R5: V1 probe + fiber tract label  -> fiber_tract (exclude)

  R3 is the key change from the previous version. On a STR probe, any MSN-
  waveform unit is an MSN. The histological region label tells us the track
  reconstruction got the depth wrong, not that the cell type is wrong.
  This is NOT applied to V1 probes: broad slow neurons there are genuine
  pyramidal cells and should stay as cortex labels.

Three output sets
-----------------
  Tier 1 -- strict     : CP/STR histology label  AND  MSN waveform
  Tier 2 -- waveform   : MSN waveform  AND  STR probe  (any region label)
  Tier 3 -- conservative: Tier 2 + ambiguous-waveform CP units

Use Tier 2 as primary for rescaling analysis.
Use Tier 1 as the highest-confidence reference.
Use Tier 3 as upper-bound sensitivity check.

Outputs -> DATA_DIR/location_matching/cell_type/
  RZ_unit_properties_final.csv
  RZ_msn_strict.csv           (Tier 1)
  RZ_msn_waveform.csv         (Tier 2 -- recommended primary)
  RZ_msn_conservative.csv     (Tier 3)
  RZ_v1_cortical.csv          (V1 cortical)
  RZ_v1_hippocampal.csv       (V1 hippocampal)
  relabeling_log.csv
  cell_type_overview.png
  msn_tiers_waterfall.png
  rescued_msn_breakdown.png
  msn_by_animal.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import paths as p
from constants import EARLY_COHORT
from utils import load_waveform_metrics

# ── Config ─────────────────────────────────────────────────────────────────────
PLOT_DIR = p.DATA_DIR / "location_matching" / "cell_type"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Waveform thresholds ────────────────────────────────────────────────────────
PT_MSN_MIN   = 0.40   # ms  -- minimum PT for MSN (broad)
PT_FSI_MAX   = 0.35   # ms  -- maximum PT for FSI (narrow)
FR_MSN_MAX   = 15.0   # Hz  -- maximum FR for MSN
FR_HIGH_MIN  = 20.0   # Hz  -- minimum FR for high_FR class (GPe etc.)
FR_TAN_MIN   = 3.0    # Hz  -- minimum FR for TAN (tonic cholinergic interneuron)
FR_TAN_MAX   = 12.0   # Hz  -- maximum FR for TAN
PT_TAN_MIN   = 0.35   # ms  -- TAN minimum PT (medium waveform; overlaps FSI boundary)
PT_TAN_MAX   = 0.50   # ms  -- TAN maximum PT (medium waveform)

# ── Depth gate for R3 ─────────────────────────────────────────────────────────
# STR probe units shallower than this are in cortex and should not be rescued
# as MSNs even if their waveform looks like one. Cortical L5/L6 pyramidal cells
# can overlap with the MSN waveform zone (broad PT, low FR). The depth gate
# restricts R3 to units deep enough to plausibly be in or near striatum.
# Adjust based on your cortex thickness estimate (typically 1000–1500 µm for mouse).
CORTEX_STR_BOUNDARY_UM = 1500   # µm from probe surface — units DEEPER than this qualify

# ── Region sets ───────────────────────────────────────────────────────────────
STR_HISTOLOGY_REGIONS = {
    "CP", "STR", "STRd", "STRv", "ACB", "OT",
    "PAL", "SI", "LSX", "CEA", "LA", "BLA", "EP", "MEA"
}
GPE_REGIONS  = {"GPe", "GPi", "GP"}
RT_REGIONS   = {"RT"}
THAL_REGIONS = {
    "VAL", "AV", "CL", "LD", "LP", "VPL", "VPM", "PO", "MD",
    "AM", "IAD", "RE", "VM", "CM", "ATN", "ILM", "PIL", "SPF",
    "MG", "LGd", "LGv", "PCN", "VL", "SGN"
}
# White matter / fiber tracts — no neuronal cell bodies expected here
FIBER_TRACT_REGIONS = {
    "ccb", "ccs", "ccg",          # corpus callosum (body, splenium, genu)
    "fi", "dhc", "ec",            # fimbria, dorsal hippocampal commissure, external capsule
    "st", "cing",                 # stria terminalis, cingulum bundle
    "fp", "or", "scwm", "int",    # forceps, optic radiation, subcortical WM, internal capsule
    "root", "Not found in brain", # outside atlas / off-brain
}
HPC_REGIONS = {"CA1", "CA2", "CA3", "SUB", "ProS", "DG", "DG-mo",
               "DG-sg", "DG-po", "HIP", "HPF"}

CELL_TYPE_COLORS = {
    "MSN":       "#2166AC",
    "RS":        "#4393C3",
    "FSI":       "#D6604D",
    "TAN":       "#8C564B",
    "high_FR":   "#762A83",
    "ambiguous": "#AAAAAA",
}

REGION_PALETTE = {
    "AV": "#FDB863", "VAL": "#E08214", "CL": "#B35806",
    "MOp5": "#D6604D", "MOp2/3": "#F4A582", "MOp1": "#FDDBC7",
    "MOp6a": "#e08070", "MOp6b": "#e08070",
    "MOs5": "#c45040", "MOs2/3": "#e8b090", "MOs1": "#f5c5b0",
    "CP": "#2166AC",
}


# ── Load data ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("0g_cell_type_relabeling.py  (waveform-primary)")
print("=" * 60)

_csv_candidates = [
    p.LOGS_DIR / "RZ_unit_properties_with_qc_and_regions.csv",
    p.LOGS_DIR / "RZ_unit_properties_with_regions.csv",
]
_csv_path = next((c for c in _csv_candidates if c.exists()), None)
if _csv_path is None:
    raise FileNotFoundError(
        "Could not find region CSV. Tried:\n" +
        "\n".join(str(c) for c in _csv_candidates)
    )
print(f"\nLoading: {_csv_path.name}")
df = pd.read_csv(_csv_path)
df["datetime"]         = pd.to_datetime(df["datetime"])
df["insertion_number"] = df["insertion_number"].astype(int)
df["id"]               = df["id"].astype(int)
n_total = len(df)
print(f"  {n_total:,} units loaded")

# Ensure probe_region column exists
if "probe_region" not in df.columns:
    if "track_file" in df.columns:
        df["probe_region"] = df["track_file"].apply(
            lambda t: "str" if "_str" in str(t) else ("v1" if "_v1" in str(t) else "other")
        )
    else:
        raise ValueError(
            "'probe_region' column not found and cannot be derived. "
            "Re-run 0e_neuron_location_matching.py first."
        )

str_probe_mask = df["probe_region"] == "str"
v1_probe_mask  = df["probe_region"] == "v1"
print(f"  STR probe units: {str_probe_mask.sum():,}")
print(f"  V1  probe units: {v1_probe_mask.sum():,}")


# ── Load waveform templates and compute metrics ────────────────────────────────
print("\nLoading waveform templates and computing metrics...")
df, waveforms, waveforms_norm, t_ms = load_waveform_metrics(df, join_how="left")

# Build a mapping from df row index → position in waveforms_norm (-1 = no waveform)
_matched_mask = df["template_idx"].notna()
_waveform_pos = pd.Series(-1, index=df.index, dtype=int)
_waveform_pos[_matched_mask] = np.arange(_matched_mask.sum())
matched = df["template_idx"].notna()

_FR_CANDIDATES = ["firing_rate", "mean_firing_rate", "fr",
                  "spike_rate", "mean_fr", "avg_firing_rate"]
fr_col = next((c for c in _FR_CANDIDATES if c in df.columns), None)
if fr_col is None:
    raise ValueError(f"No firing rate column. Searched: {_FR_CANDIDATES}")
print(f"  Firing rate column: '{fr_col}'")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 -- Cell type (waveform-based, location-agnostic)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("STAGE 1: Cell type classification")
print("-" * 60)

pt = df["pt_duration_ms"]
fr = df[fr_col]

is_fsi     = pt < PT_FSI_MAX
is_high_fr = (fr > FR_HIGH_MIN) & ~is_fsi
# TAN: medium PT + tonic FR + STR probe + subcortical depth (cholinergic interneuron).
# Identified before MSN because TAN PT range (0.35–0.50ms) overlaps the MSN zone (>=0.40ms).
is_tan     = (
    (pt >= PT_TAN_MIN) & (pt <= PT_TAN_MAX)
    & (fr >= FR_TAN_MIN) & (fr <= FR_TAN_MAX)
    & str_probe_mask
    & (df["peak_channel_depth"] > CORTEX_STR_BOUNDARY_UM)
)
is_msn     = (pt >= PT_MSN_MIN) & (fr <= FR_MSN_MAX) & ~is_fsi & ~is_high_fr & ~is_tan

df["putative_cell_type"] = "ambiguous"
df.loc[is_high_fr, "putative_cell_type"] = "high_FR"
df.loc[is_fsi,     "putative_cell_type"] = "FSI"
df.loc[is_tan,     "putative_cell_type"] = "TAN"
df.loc[is_msn,     "putative_cell_type"] = "MSN"

# For V1 probe units: broad+slow waveform = regular-spiking pyramidal (RS), not MSN.
df.loc[v1_probe_mask & (df["putative_cell_type"] == "MSN"),
       "putative_cell_type"] = "RS"

# For STR probe units shallower than the cortex boundary: broad+slow = RS (pyramidal),
# not MSN. Depth disambiguates MSN (deep) from cortical pyramidal (shallow).
df.loc[
    str_probe_mask
    & (df["peak_channel_depth"] <= CORTEX_STR_BOUNDARY_UM)
    & (df["putative_cell_type"] == "MSN"),
    "putative_cell_type"
] = "RS"

# STR probe MSN mask — used for all STR-specific logic below
msn_waveform = (df["putative_cell_type"] == "MSN") & str_probe_mask
rs_waveform  = (df["putative_cell_type"] == "RS")  & v1_probe_mask

ct_counts = df["putative_cell_type"].value_counts()
print(f"\n  Cell type counts (all {n_total:,} units):")
for ct, n in ct_counts.items():
    print(f"    {ct:12s}: {n:4d}  ({n/n_total*100:.1f}%)")

print(f"\n  STR probe cell types:")
for ct in ["MSN", "TAN", "RS", "FSI", "high_FR", "ambiguous"]:
    n = int((str_probe_mask & (df["putative_cell_type"] == ct)).sum())
    print(f"    {ct:12s}: {n:4d}")
print(f"\n  V1 probe cell types:")
for ct in ["RS", "FSI", "high_FR", "ambiguous"]:
    n = int((v1_probe_mask & (df["putative_cell_type"] == ct)).sum())
    print(f"    {ct:12s}: {n:4d}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 -- Region label correction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("STAGE 2: Region label correction")
print("-" * 60)

df["final_label"]    = df["region_acronym"].copy()
df["relabel_reason"] = ""
df["label_changed"]  = False

def apply_relabel(mask, new_label, reason):
    n = int(mask.sum())
    df.loc[mask, "final_label"]    = new_label
    df.loc[mask, "relabel_reason"] = reason
    df.loc[mask, "label_changed"]  = True
    print(f"\n  {reason}")
    print(f"    -> {n} units relabeled to '{new_label}'")
    if n > 0:
        for mouse, cnt in df.loc[mask].groupby("mouse").size().items():
            print(f"       {mouse}: {cnt}")

# R1: GPe + MSN waveform -> CP
# GPe is biologically impossible for an MSN (must fire 20-100 Hz).
# These are deep CP units where the track overestimates depth.
r1 = msn_waveform & df["region_acronym"].isin(GPE_REGIONS)
apply_relabel(r1, "CP",
    "R1: GPe->CP  [MSN waveform in GPe; track overestimates depth]")

# R2: RT + non-FSI waveform -> VAL
# Reticular thalamus is GABAergic fast-spiking. Broad slow units are
# VAL border spillover.
r2 = df["region_acronym"].isin(RT_REGIONS) & ~is_fsi
apply_relabel(r2, "VAL",
    "R2: RT->VAL  [broad/slow waveform in RT; not GABAergic FS]")

# R3: STR probe + MSN waveform + non-STR region label -> CP
# Core waveform-primary rule.
# The probe was aimed at dorsal striatum and physically passed through it.
# If a unit has an MSN waveform (broad + very low FR), it IS an MSN.
# The track reconstruction placed it in the wrong region due to depth error —
# it didn't get the cell type wrong.
# NOT applied to V1 probes: broad slow neurons there are genuine pyramidals.
r3 = (
    msn_waveform
    & str_probe_mask
    & (df["peak_channel_depth"] > CORTEX_STR_BOUNDARY_UM)
    & ~df["region_acronym"].isin(STR_HISTOLOGY_REGIONS)
    & ~df["region_acronym"].isin(FIBER_TRACT_REGIONS)
    & ~df["label_changed"]
)
r3_from_thal = r3 & df["region_acronym"].isin(THAL_REGIONS)
r3_from_ctx  = r3 & ~df["region_acronym"].isin(THAL_REGIONS | GPE_REGIONS | RT_REGIONS)

print(f"\n  R3: STR probe + MSN waveform + depth>{CORTEX_STR_BOUNDARY_UM}µm -> CP  [waveform-primary rescue]")
print(f"    From thalamic labels: {r3_from_thal.sum():3d} units")
print(f"    From cortical labels: {r3_from_ctx.sum():3d} units")
print(f"    From other labels   : {(r3 & ~r3_from_thal & ~r3_from_ctx).sum():3d} units")
apply_relabel(r3, "CP",
    "R3: STR probe->CP  [MSN waveform on STR probe; track depth error]")

r4 = (
    v1_probe_mask
    & is_fsi
    & df["region_acronym"].isin(HPC_REGIONS)
)
apply_relabel(r4, "HPC_border",
    "R4: CA1/SUB + FSI waveform -> HPC_border  [alveus border; not genuine CA1]")

r5 = (
    v1_probe_mask
    & df["region_acronym"].isin(FIBER_TRACT_REGIONS)
)
apply_relabel(r5, "fiber_tract",
    "R5: V1 probe + fiber tract label -> fiber_tract  [exclude: no cell bodies]")

n_changed    = int(df["label_changed"].sum())
n_cp_final   = int((df["final_label"] == "CP").sum())
n_hpc_border = int((df["final_label"] == "HPC_border").sum())
n_fiber      = int((df["final_label"] == "fiber_tract").sum())
print(f"\n  Total relabeled : {n_changed}")
print(f"  CP              : {n_cp_final}  (was {(df['region_acronym']=='CP').sum()}, rescued MSN waveform units included)")
print(f"  HPC_border      : {n_hpc_border}  (V1 probe alveus border, flagged)")
print(f"  fiber_tract     : {n_fiber}  (V1 probe white matter, excluded)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 -- Build MSN tiers
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("STAGE 3: MSN tier construction")
print("-" * 60)

# Tier 1: histology AND waveform agree (unchanged by any rule)
tier1 = (
    msn_waveform
    & df["final_label"].isin(STR_HISTOLOGY_REGIONS)
    & ~df["label_changed"]
)

# Tier 2 revised — depth-gated to exclude clear cortical zone
tier2 = (
    msn_waveform
    & str_probe_mask
    & (df["peak_channel_depth"] > CORTEX_STR_BOUNDARY_UM)  # 1500 µm
    & ~df["region_acronym"].isin(FIBER_TRACT_REGIONS)
)

# Tier 3: Tier 2 + ambiguous-waveform units in confirmed STR labels
tier3_extra = (
    (df["putative_cell_type"] == "ambiguous")
    & str_probe_mask
    & df["final_label"].isin(STR_HISTOLOGY_REGIONS)
)
tier3 = tier2 | tier3_extra

df["msn_tier1"] = tier1
df["msn_tier2"] = tier2
df["msn_tier3"] = tier3

# ── V1 cortical unit set ───────────────────────────────────────────────────────
EXCLUDE_V1 = {"HPC_border", "fiber_tract", "above_surface", "below_track",
              "no_track_file", "no_session_match", "unknown_depth", "nan_depth"}
VIS_REGIONS_SET = {r for r in df["region_acronym"].unique()
                   if str(r).startswith("VIS")}

v1_valid = (
    v1_probe_mask
    & ~df["final_label"].isin(EXCLUDE_V1)
    & ~df["region_acronym"].isin(FIBER_TRACT_REGIONS)
)
v1_cortical = v1_valid & df["region_acronym"].isin(VIS_REGIONS_SET)
v1_hippocampal = (
    v1_valid
    & df["region_acronym"].isin(HPC_REGIONS)
    & ~df["label_changed"]
)

df["v1_valid"]       = v1_valid
df["v1_cortical"]    = v1_cortical
df["v1_hippocampal"] = v1_hippocampal

n1 = int(tier1.sum())
n2 = int(tier2.sum())
n3 = int(tier3.sum())
n_v1c = int(v1_cortical.sum())
n_v1h = int(v1_hippocampal.sum())

print(f"""
  STR probe — MSN tiers:
  +-------------------------------------------------------+
  |  Tier 1 -- Strict            : {n1:4d} units            |
  |    Histology + waveform agree. Highest confidence.    |
  |                                                       |
  |  Tier 2 -- Waveform-primary  : {n2:4d} units  <- USE THIS|
  |    MSN waveform on STR probe, any region label.       |
  |    Recommended primary for rescaling analysis.        |
  |                                                       |
  |  Tier 3 -- Conservative      : {n3:4d} units            |
  |    Tier 2 + ambiguous PT/FR units in STR labels.      |
  |    Upper-bound sensitivity check only.                |
  +-------------------------------------------------------+

  V1 probe — cortical units:
  +-------------------------------------------------------+
  |  Cortical (VISp layers)      : {n_v1c:4d} units            |
  |  Hippocampal (CA1/SUB clean) : {n_v1h:4d} units            |
  +-------------------------------------------------------+
""")

# Per-animal breakdown
def animal_probes(mouse):
    m = df["mouse"] == mouse
    probes = df.loc[m, "probe_region"].unique()
    return "+".join(sorted(pr for pr in probes if pr in ("str", "v1")))

animals = sorted(df["mouse"].unique())
print(f"  {'mouse':8s} {'T1':>5s} {'T2':>5s} {'T3':>5s}  "
      f"{'Rescued':>7s}  {'FR_med':>7s}  {'PT_med':>7s}  "
      f"{'V1_ctx':>6s}  {'V1_hpc':>6s}  probes")
print("  " + "-" * 80)
for mouse in animals:
    m  = df["mouse"] == mouse
    t1 = int((tier1 & m).sum())
    t2 = int((tier2 & m).sum())
    t3 = int((tier3 & m).sum())
    vc = int((v1_cortical & m).sum())
    vh = int((v1_hippocampal & m).sum())
    sub = df[tier2 & m]
    fr_med = sub[fr_col].median() if len(sub) else float("nan")
    pt_med = sub["pt_duration_ms"].median() if len(sub) else float("nan")
    probes = animal_probes(mouse)
    print(f"  {mouse:8s} {t1:5d} {t2:5d} {t3:5d}  "
          f"{t2-t1:7d}  {fr_med:7.2f}  {pt_med:7.3f}  "
          f"{vc:6d}  {vh:6d}  {probes}")

print(f"\n  Early cohort Tier 2 MSNs — original region labels:")
for mouse in EARLY_COHORT:
    m = (df["mouse"] == mouse) & tier2
    if not m.any():
        print(f"    {mouse}: 0")
        continue
    region_counts = df.loc[m, "region_acronym"].value_counts()
    print(f"    {mouse} ({m.sum()} total):")
    for region, cnt in region_counts.items():
        print(f"      {region:12s}: {cnt}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("Saving outputs")
print("-" * 60)

df_out = df.drop(columns=["template_idx"], errors="ignore")

full_path = p.LOGS_DIR / "RZ_unit_properties_final.csv"
df_out.to_csv(full_path, index=False)
print(f"  Full table  -> RZ_unit_properties_final.csv  ({len(df_out):,} units)")

df_out[tier1].to_csv(p.LOGS_DIR / "RZ_msn_strict.csv", index=False)
print(f"  Tier 1      -> RZ_msn_strict.csv          ({n1:,} units)")

# Remove RZ038 high-FR LD units (thalamic relay, not MSN) before saving
rz038_bad = (
    df_out["mouse"].eq("RZ038") &
    df_out["region_acronym"].eq("LD") &
    (df_out[fr_col] >= 5.0)
)
n2_saved = int((tier2 & ~rz038_bad).sum())
df_out[tier2 & ~rz038_bad].to_csv(p.LOGS_DIR / "RZ_msn_waveform.csv", index=False)
print(f"  Tier 2      -> RZ_msn_waveform.csv        ({n2_saved:,} units, {n2 - n2_saved} RZ038/LD removed)  <- primary")

df_out[tier3].to_csv(p.LOGS_DIR / "RZ_msn_conservative.csv", index=False)
print(f"  Tier 3      -> RZ_msn_conservative.csv    ({n3:,} units)")

df_out[v1_cortical].to_csv(p.LOGS_DIR / "RZ_v1_cortical.csv", index=False)
print(f"  V1 cortical -> RZ_v1_cortical.csv         ({n_v1c:,} units)  <- V1 analysis")

df_out[v1_hippocampal].to_csv(p.LOGS_DIR / "RZ_v1_hippocampal.csv", index=False)
print(f"  V1 hippo    -> RZ_v1_hippocampal.csv      ({n_v1h:,} units)")

relabel_log = df_out[df_out["label_changed"]][[
    "mouse", "datetime", "insertion_number", "id",
    "region_acronym", "final_label", "relabel_reason",
    "putative_cell_type", "pt_duration_ms", fr_col,
]]
relabel_log.to_csv(PLOT_DIR / "relabeling_log.csv", index=False)
print(f"  Relabel log -> relabeling_log.csv  ({len(relabel_log):,} units changed)")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Plot 1: FR x PT — cell type, probe type, early cohort ─────────────────────
print("\n[Plot 1] FR x PT scatter...")

plot_df = df_out[df_out["pt_duration_ms"].notna() & df_out[fr_col].notna()]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Left: cell type classification
ax = axes[0]
for ct in ["FSI", "high_FR", "ambiguous", "MSN"]:
    d = plot_df[plot_df["putative_cell_type"] == ct]
    ax.scatter(d["pt_duration_ms"], d[fr_col],
               color=CELL_TYPE_COLORS[ct], alpha=0.5, s=16, linewidths=0,
               label=f"{ct} (n={len(d)})")
msn_box = plt.Rectangle((PT_MSN_MIN, 0.0), 1.0, FR_MSN_MAX,
                          lw=1.5, edgecolor="#2166AC",
                          facecolor="#2166AC", alpha=0.08)
ax.add_patch(msn_box)
ax.axvline(PT_FSI_MAX,  color="tomato",    linestyle="--", lw=1, label=f"FSI<{PT_FSI_MAX}ms")
ax.axvline(PT_MSN_MIN,  color="steelblue", linestyle="--", lw=1, label=f"MSN>{PT_MSN_MIN}ms")
ax.axhline(FR_MSN_MAX,  color="steelblue", linestyle=":",  lw=1, label=f"MSN FR<{FR_MSN_MAX}Hz")
ax.axhline(FR_HIGH_MIN, color="purple",    linestyle=":",  lw=1, label=f"high_FR>{FR_HIGH_MIN}Hz")
ax.set_yscale("log")
ax.set_ylim(0.02, 200)
ax.set_xlim(0, 1.2)
ax.set_xlabel("PT duration (ms)"); ax.set_ylabel("Firing rate (Hz, log)")
ax.set_title("Stage 1: Cell type\n(waveform + FR, location-agnostic)", fontsize=11)
ax.legend(fontsize=7, markerscale=2); ax.grid(True, alpha=0.2)

# Middle: STR vs V1 probe, Tier 2 highlighted
ax = axes[1]
probe_colors = {"str": "#2166AC", "v1": "#9467bd", "other": "#888888"}
for probe in ["v1", "str"]:
    d = plot_df[plot_df["probe_region"] == probe]
    ax.scatter(d["pt_duration_ms"], d[fr_col],
               color=probe_colors[probe], alpha=0.45, s=16, linewidths=0,
               label=f"{probe.upper()} probe (n={len(d)})")
d_t2 = plot_df[tier2[plot_df.index]]
ax.scatter(d_t2["pt_duration_ms"], d_t2[fr_col],
           facecolors="none", edgecolors="#2166AC", s=38, lw=0.9, alpha=0.65,
           label=f"Tier 2 MSN (n={len(d_t2)})")
ax.axvline(PT_MSN_MIN, color="steelblue", linestyle="--", lw=1)
ax.axhline(FR_MSN_MAX, color="steelblue", linestyle=":",  lw=1)
ax.set_yscale("log")
ax.set_ylim(0.02, 200)
ax.set_xlim(0, 1.2)
ax.set_xlabel("PT duration (ms)"); ax.set_ylabel("Firing rate (Hz, log)")
ax.set_title("STR vs V1 probe\n(circles = Tier 2 MSN selection)", fontsize=11)
ax.legend(fontsize=8, markerscale=1.5); ax.grid(True, alpha=0.2)

# Right: early cohort Tier 2 MSNs coloured by original region
ax = axes[2]
early_t2 = plot_df[tier2[plot_df.index] & plot_df["mouse"].isin(EARLY_COHORT)]
for region in early_t2["region_acronym"].unique():
    d = early_t2[early_t2["region_acronym"] == region]
    color = REGION_PALETTE.get(region, "#888888")
    ax.scatter(d["pt_duration_ms"], d[fr_col],
               color=color, alpha=0.7, s=24, linewidths=0,
               label=f"{region} (n={len(d)})")
ax.axvline(PT_MSN_MIN, color="steelblue", linestyle="--", lw=1)
ax.axhline(FR_MSN_MAX, color="steelblue", linestyle=":",  lw=1)
ax.set_yscale("log")
ax.set_ylim(0.02, 200)
ax.set_xlim(0, 1.2)
ax.set_xlabel("PT duration (ms)"); ax.set_ylabel("Firing rate (Hz, log)")
ax.set_title("Early cohort (RZ034-39) Tier 2 MSNs\ncoloured by original region label", fontsize=11)
ax.legend(fontsize=7, markerscale=1.5, ncol=2); ax.grid(True, alpha=0.2)

fig.suptitle("FR x PT duration -- cell type and MSN tier selection", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "cell_type_overview.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> cell_type_overview.png")


# ── Plot 2: Tier waterfall per animal ─────────────────────────────────────────
print("[Plot 2] MSN tier waterfall...")

str_animals = sorted([
    m for m in df_out["mouse"].unique()
    if (df_out.loc[df_out["mouse"] == m, "probe_region"] == "str").any()
])

t1_counts = [int((tier1 & (df["mouse"] == m)).sum()) for m in str_animals]
t2_counts = [int((tier2 & (df["mouse"] == m)).sum()) for m in str_animals]
rescued   = [t2 - t1 for t2, t1 in zip(t2_counts, t1_counts)]

x = np.arange(len(str_animals))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(x, t1_counts, color="#1a3a5c",
       label=f"Tier 1 strict (n={sum(t1_counts)})")
ax.bar(x, rescued, bottom=t1_counts, color="#4DAFEC",
       label=f"Waveform-rescued (n={sum(rescued)})")
for i, total in enumerate(t2_counts):
    if total > 0:
        ax.text(i, total + 0.5, str(total), ha="center", va="bottom",
                fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(str_animals, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("MSN unit count")
ax.set_title(f"Tier 2 MSNs per animal  (total = {n2})", fontsize=11)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
for i, mouse in enumerate(str_animals):
    sub = df_out[tier2 & (df_out["mouse"] == mouse)]
    if not len(sub): continue
    frs = sub[fr_col].dropna()
    ax.scatter(frs, np.full(len(frs), i) + np.random.uniform(-0.25, 0.25, len(frs)),
               color="#2166AC", alpha=0.35, s=10, linewidths=0)
    ax.plot([frs.median()], [i], "|", color="black", markersize=14, markeredgewidth=2.5)
ax.axvline(FR_MSN_MAX, color="steelblue", linestyle="--", lw=1)
ax.set_xscale("log")
ax.set_yticks(range(len(str_animals)))
ax.set_yticklabels(str_animals, fontsize=9)
ax.set_xlabel("Firing rate (Hz, log)")
ax.set_title("Tier 2 MSN firing rate per animal\n(| = median)", fontsize=11)
ax.grid(True, alpha=0.2)

fig.suptitle("MSN recovery by tier and animal (STR probes only)", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "msn_tiers_waterfall.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> msn_tiers_waterfall.png")


# ── Plot 3: Source region breakdown for rescued MSNs ──────────────────────────
print("[Plot 3] Rescued MSN breakdown...")

rescued_df = df_out[tier2 & df_out["label_changed"]]
source_counts = rescued_df["region_acronym"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
regions = source_counts.index.tolist()
counts  = source_counts.values
colors  = [REGION_PALETTE.get(r, "#888888") for r in regions]
bars = ax.barh(regions, counts, color=colors, edgecolor="white", linewidth=0.4)
for bar, val in zip(bars, counts):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=8)
ax.set_xlabel("Units rescued")
ax.set_title(f"Original region label of rescued MSNs\n(n={len(rescued_df)} total)", fontsize=11)
ax.grid(axis="x", alpha=0.3)

ax = axes[1]
t1_pts  = df_out.loc[tier1,  "pt_duration_ms"].dropna()
res_pts = df_out.loc[tier2 & df_out["label_changed"], "pt_duration_ms"].dropna()
bins = np.linspace(0.35, 1.2, 35)
ax.hist(t1_pts,  bins=bins, alpha=0.65, color="#1a3a5c",
        label=f"Tier 1 strict (n={len(t1_pts)})")
ax.hist(res_pts, bins=bins, alpha=0.65, color="#4DAFEC",
        label=f"Rescued (n={len(res_pts)})")
ax.axvline(PT_MSN_MIN, color="steelblue", linestyle="--", lw=1.5)
ax.set_xlabel("PT duration (ms)")
ax.set_ylabel("Unit count")
ax.set_title("PT distribution: strict vs rescued\n(similar shape = waveforms are consistent)", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

fig.suptitle("Rescued MSN characterisation", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "rescued_msn_breakdown.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> rescued_msn_breakdown.png")


# ── Plot 4: PT and FR per animal, Tier 2 ─────────────────────────────────────
print("[Plot 4] MSN properties by animal...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax_i, (col_name, xlabel, use_log) in enumerate([
    ("pt_duration_ms", "PT duration (ms)",      False),
    (fr_col,           "Firing rate (Hz, log)",  True),
]):
    ax = axes[ax_i]
    for i, mouse in enumerate(str_animals):
        m_strict  = tier1 & (df_out["mouse"] == mouse)
        m_rescued = tier2 & df_out["label_changed"] & (df_out["mouse"] == mouse)
        for sub_mask, color in [(m_strict, "#1a3a5c"), (m_rescued, "#4DAFEC")]:
            v = df_out.loc[sub_mask, col_name].dropna()
            if len(v):
                ax.scatter(v, np.full(len(v), i) + np.random.uniform(-0.25, 0.25, len(v)),
                           color=color, alpha=0.4, s=10, linewidths=0)
        # Overall median for this animal's Tier 2
        all_t2 = df_out.loc[tier2 & (df_out["mouse"] == mouse), col_name].dropna()
        if len(all_t2):
            ax.plot([all_t2.median()], [i], "|", color="black",
                    markersize=14, markeredgewidth=2.5)
    if use_log:
        ax.set_xscale("log")
        ax.axvline(FR_MSN_MAX, color="steelblue", linestyle="--", lw=1)
    else:
        ax.axvline(PT_MSN_MIN, color="steelblue", linestyle="--", lw=1)
    ax.set_yticks(range(len(str_animals)))
    ax.set_yticklabels(str_animals, fontsize=9)
    ax.set_xlabel(xlabel); ax.grid(True, alpha=0.2)
    ax.set_title(f"Tier 2 MSN {xlabel}\nper animal (| = median)", fontsize=11)

legend_els = [
    mpatches.Patch(color="#1a3a5c", label="Tier 1 strict"),
    mpatches.Patch(color="#4DAFEC", label="Waveform-rescued"),
]
axes[0].legend(handles=legend_els, fontsize=9)
fig.suptitle("Tier 2 MSN waveform properties by animal", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "msn_by_animal.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> msn_by_animal.png")


# ── Plot 5: Unit count distribution per animal after correction ───────────────
print("[Plot 5] Unit count distribution after correction...")

all_animals_sorted = sorted(df_out["mouse"].unique())

# Build per-animal counts after relabeling
str_unit_counts = []
v1_unit_counts  = []
for mouse in all_animals_sorted:
    m = df_out["mouse"] == mouse
    str_unit_counts.append(int((m & str_probe_mask).sum()))
    v1_unit_counts.append(int((m & v1_probe_mask).sum()))

x = np.arange(len(all_animals_sorted))
width = 0.38

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: stacked bar — STR and V1 unit counts per animal
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
ax.set_title("All units per animal after label correction\n(STR vs V1 probe)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Right: histogram of unit counts (distribution across animals)
ax = axes[1]
str_counts_nonzero = [c for c in str_unit_counts if c > 0]
v1_counts_nonzero  = [c for c in v1_unit_counts  if c > 0]
bins = np.linspace(0, max(str_counts_nonzero + v1_counts_nonzero + [1]) + 10, 20)
if str_counts_nonzero:
    ax.hist(str_counts_nonzero, bins=bins, alpha=0.65, color="#2166AC",
            label=f"STR probe (n={len(str_counts_nonzero)} animals, "
                  f"total={sum(str_counts_nonzero)})")
if v1_counts_nonzero:
    ax.hist(v1_counts_nonzero,  bins=bins, alpha=0.65, color="#9467bd",
            label=f"V1 probe (n={len(v1_counts_nonzero)} animals, "
                  f"total={sum(v1_counts_nonzero)})")
ax.set_xlabel("Units per animal")
ax.set_ylabel("Number of animals")
ax.set_title("Distribution of unit counts\nacross animals (after correction)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

fig.suptitle("Unit count distribution after relabeling correction", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "unit_count_distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> unit_count_distribution.png")


# ── Plot 6: Cell type distribution per probe type ─────────────────────────────
print("[Plot 6] Cell type distribution per probe type...")

# STR probe cell types: MSN, FSI, high_FR, ambiguous
# V1 probe cell types:  RS, FSI, high_FR, ambiguous
STR_CELL_TYPES = ["MSN", "TAN", "RS", "FSI", "high_FR", "ambiguous"]
V1_CELL_TYPES  = ["RS",  "FSI", "high_FR", "ambiguous"]

CT_COLORS_ALL = {
    "MSN":       "#2166AC",
    "RS":        "#4393C3",
    "TAN":       "#8C564B",
    "FSI":       "#D6604D",
    "high_FR":   "#762A83",
    "ambiguous": "#AAAAAA",
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: STR probe cell type counts per animal
ax = axes[0]
str_animal_list = [m for m, c in zip(all_animals_sorted, str_unit_counts) if c > 0]
str_ct_matrix   = {ct: [] for ct in STR_CELL_TYPES}
for mouse in str_animal_list:
    m = (df_out["mouse"] == mouse) & str_probe_mask
    for ct in STR_CELL_TYPES:
        str_ct_matrix[ct].append(int((m & (df_out["putative_cell_type"] == ct)).sum()))

x_str  = np.arange(len(str_animal_list))
bottom = np.zeros(len(str_animal_list))
for ct in STR_CELL_TYPES:
    vals = np.array(str_ct_matrix[ct])
    ax.bar(x_str, vals, bottom=bottom, color=CT_COLORS_ALL[ct],
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
    m = (df_out["mouse"] == mouse) & v1_probe_mask
    for ct in V1_CELL_TYPES:
        v1_ct_matrix[ct].append(int((m & (df_out["putative_cell_type"] == ct)).sum()))

x_v1   = np.arange(len(v1_animal_list))
bottom = np.zeros(len(v1_animal_list))
for ct in V1_CELL_TYPES:
    vals = np.array(v1_ct_matrix[ct])
    ax.bar(x_v1, vals, bottom=bottom, color=CT_COLORS_ALL[ct],
           label=ct, edgecolor="white", linewidth=0.3)
    bottom += vals
ax.set_xticks(x_v1)
ax.set_xticklabels(v1_animal_list, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Unit count")
ax.set_title("V1 probe — cell type per animal\n(after correction)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Right: proportional cell type breakdown, STR vs V1 (pie or 100% stacked bar)
ax = axes[2]
probe_labels = ["STR probe", "V1 probe"]
str_totals = {ct: sum(str_ct_matrix[ct]) for ct in STR_CELL_TYPES}
v1_totals  = {ct: sum(v1_ct_matrix[ct])  for ct in V1_CELL_TYPES}

all_cts = ["MSN", "TAN", "RS", "FSI", "high_FR", "ambiguous"]
str_props = np.array([str_totals.get(ct, 0) for ct in all_cts], dtype=float)
v1_props  = np.array([v1_totals.get(ct, 0)  for ct in all_cts], dtype=float)
str_props /= str_props.sum() if str_props.sum() > 0 else 1
v1_props  /= v1_props.sum()  if v1_props.sum()  > 0 else 1

x2     = np.arange(2)
bottom = np.zeros(2)
for i, ct in enumerate(all_cts):
    vals = np.array([str_props[i], v1_props[i]])
    ax.bar(x2, vals, bottom=bottom, color=CT_COLORS_ALL[ct],
           label=ct, edgecolor="white", linewidth=0.5, width=0.5)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.03:
            ax.text(j, b + v / 2, f"{v*100:.0f}%",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    bottom += vals
ax.set_xticks(x2)
ax.set_xticklabels(probe_labels, fontsize=11)
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1)
ax.set_title("Cell type composition\nSTR vs V1 probe (proportional)", fontsize=11)
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Cell type distribution by probe type after relabeling", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "cell_type_distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> cell_type_distribution.png")


# ── Plot 7: Units per corrected region (vertical bar, coloured by region type)
print("[Plot 7] Units per corrected region (STR and V1)...")

_RPLOT_STR_REGIONS  = {
    "CP", "GPe", "GPi", "STR", "PAL", "SI", "LSX",
    "STRd", "STRv", "ACB", "OT", "CEA", "LA", "BLA", "EP", "MEA",
}
_RPLOT_V1_REGIONS   = {
    "VISp", "VISp1", "VISp2/3", "VISp4", "VISp5", "VISp6a", "VISp6b",
    "VISl", "VISrl", "VISal", "VISam", "VISpm", "VISli",
    "VISpl", "VISpor", "VIS", "VISL1", "VISL4",
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
    "str": df_out[str_probe_mask]["final_label"].value_counts(),
    "v1":  df_out[v1_probe_mask]["final_label"].value_counts(),
}
_probe_labels = {"str": "STR probes", "v1": "V1 probes"}

max_cols = max(len(c) for c in _counts_after.values()) if _counts_after else 4
fig, axes = plt.subplots(1, 2, figsize=(max(8, max_cols * 0.5), 6))

for ax, probe in zip(axes, ["str", "v1"]):
    rc = _counts_after[probe]
    rc = rc[rc > 0].sort_values(ascending=False)  # descending so tallest bar is on the left
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


# ── Plot 8: PT vs FR scatter — final cell type assignment ─────────────────────
print("[Plot 8] PT vs FR scatter (final cell type)...")

_FINAL_CT_ORDER = ["MSN", "TAN", "RS", "FSI", "high_FR", "ambiguous"]
_CT_COLORS_FINAL = {**CT_COLORS_ALL, "RS": "#4393C3"}

plot_df_final = df_out[df_out["pt_duration_ms"].notna() & df_out[fr_col].notna()]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

for ax, (probe, probe_label) in zip(axes, [("str", "STR probe"), ("v1", "V1 probe")]):
    sub = plot_df_final[plot_df_final["probe_region"] == probe]
    for ct in _FINAL_CT_ORDER:
        d = sub[sub["putative_cell_type"] == ct]
        if len(d):
            ax.scatter(d["pt_duration_ms"], d[fr_col],
                       color=_CT_COLORS_FINAL.get(ct, "#888"), alpha=0.55, s=8,
                       linewidths=0, label=f"{ct} (n={len(d)})")

    # MSN zone box
    ax.add_patch(plt.Rectangle(
        (PT_MSN_MIN, 0.0), 1.2, FR_MSN_MAX,
        lw=1.5, edgecolor="#2166AC", facecolor="#2166AC", alpha=0.06,
    ))
    # TAN zone box
    ax.add_patch(plt.Rectangle(
        (PT_TAN_MIN, FR_TAN_MIN), PT_TAN_MAX - PT_TAN_MIN, FR_TAN_MAX - FR_TAN_MIN,
        lw=1.5, edgecolor="#8C564B", facecolor="#8C564B", alpha=0.08,
    ))

    ax.axvline(PT_FSI_MAX,  color="tomato",    linestyle="--", lw=1,
               label=f"FSI < {PT_FSI_MAX}ms")
    ax.axvline(PT_MSN_MIN,  color="steelblue", linestyle="--", lw=1,
               label=f"MSN ≥ {PT_MSN_MIN}ms")
    ax.axhline(FR_MSN_MAX,  color="steelblue", linestyle=":",  lw=1,
               label=f"MSN FR ≤ {FR_MSN_MAX}Hz")
    ax.axhline(FR_HIGH_MIN, color="purple",    linestyle=":",  lw=1,
               label=f"high_FR > {FR_HIGH_MIN}Hz")

    ax.set_yscale("log")
    ax.set_ylim(0.02, 200)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("PT duration (ms)", fontsize=7)
    ax.set_ylabel("Firing rate (Hz, log)", fontsize=7)
    ax.set_title(f"{probe_label} — final cell type assignment", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, markerscale=0.8, loc="upper right",
              handlelength=1, handletextpad=0.3, borderpad=0.3, labelspacing=0.2)
    ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(PLOT_DIR / "final_cell_type_scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> final_cell_type_scatter.png")


# ── Plot 9: Waveforms — mean by cell type (left) + MSN traces (right) ─────────
print("[Plot 9] Waveform plots...")

def _get_waveforms_for_ct(ct):
    """Return normalised waveforms for units with given final cell type."""
    rows = df_out.index[df_out["putative_cell_type"] == ct]
    pos  = _waveform_pos[rows]
    pos  = pos[pos >= 0].values
    return waveforms_norm[pos] if len(pos) else np.empty((0, waveforms_norm.shape[1]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: mean ± SD by cell type
ax = axes[0]
for ct in ["MSN", "FSI", "RS", "TAN"]:
    wv = _get_waveforms_for_ct(ct)
    if len(wv) == 0:
        continue
    mean  = wv.mean(axis=0)
    sd    = wv.std(axis=0)
    color = CELL_TYPE_COLORS.get(ct, "#888")
    ax.plot(t_ms, mean, color=color, lw=2, label=f"{ct} (n={len(wv)})")
    ax.fill_between(t_ms, mean - sd, mean + sd, color=color, alpha=0.18)
ax.axvline(PT_FSI_MAX, color="grey", linestyle="--", lw=1.2, label="PT=0.35 ms (FSI)")
ax.axvline(PT_MSN_MIN, color="grey", linestyle=":",  lw=1.2, label="PT=0.40 ms (MSN)")
ax.axhline(0, color="grey", linestyle=":", lw=0.8, alpha=0.5)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Normalised amplitude (trough = −1)")
ax.set_title("Mean Waveforms by Cell Type (± 1 SD)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Right: MSN individual traces + mean ± SD
ax = axes[1]
msn_wv = _get_waveforms_for_ct("MSN")
if len(msn_wv):
    mean = msn_wv.mean(axis=0)
    sd   = msn_wv.std(axis=0)
    n_traces = min(len(msn_wv), 200)
    idx = np.random.default_rng(42).choice(len(msn_wv), n_traces, replace=False)
    for i in idx:
        ax.plot(t_ms, msn_wv[i], color="#2166AC", alpha=0.08, lw=0.6)
    ax.fill_between(t_ms, mean - sd, mean + sd,
                    color="#2166AC", alpha=0.25, label="± 1 SD")
    ax.plot(t_ms, mean, color="#2166AC", lw=2.5, label=f"Mean (n={len(msn_wv)})")
ax.axvline(0, color="grey", linestyle="--", lw=1, alpha=0.6)
ax.axhline(0, color="grey", linestyle=":", lw=0.8, alpha=0.5)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Normalised amplitude (trough = −1)")
ax.set_title("MSN Waveforms — Individual Traces + Mean ± SD")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(PLOT_DIR / "waveforms.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved -> waveforms.png")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
Total units              : {n_total:,}

STR probe — MSN recovery:
  Tier 1 -- Strict           : {n1:4d}  (histology + waveform agree)
  Tier 2 -- Waveform-primary : {n2:4d}  <- PRIMARY for rescaling analysis
  Tier 3 -- Conservative     : {n3:4d}  (sensitivity check)
  Rescued (T2 - T1)          : {n2-n1:4d}
    R1 (GPe -> CP)           : {int(r1.sum()):4d}
    R2 (RT  -> VAL)          : {int(r2.sum()):4d}
    R3 (STR probe, non-STR)  : {int(r3.sum()):4d}
      from thalamic labels   : {int(r3_from_thal.sum()):4d}
      from cortical labels   : {int(r3_from_ctx.sum()):4d}

V1 probe — cortical units:
  Cortical (VISp layers)     : {n_v1c:4d}  <- V1 analysis
    RS (regular spiking)     : {int((v1_cortical & (df_out['putative_cell_type']=='RS')).sum()):4d}
    FSI                      : {int((v1_cortical & (df_out['putative_cell_type']=='FSI')).sum()):4d}
    high_FR                  : {int((v1_cortical & (df_out['putative_cell_type']=='high_FR')).sum()):4d}
    ambiguous                : {int((v1_cortical & (df_out['putative_cell_type']=='ambiguous')).sum()):4d}
  Hippocampal (CA1/SUB)      : {n_v1h:4d}
  HPC border flagged (R4)    : {n_hpc_border:4d}
  Fiber tract flagged (R5)   : {n_fiber:4d}

Key columns in RZ_unit_properties_final.csv:
  putative_cell_type : MSN (STR) / TAN (STR) / RS (V1 pyramidal) / FSI / high_FR / ambiguous
  final_label        : corrected region label
  relabel_reason     : rule applied (empty if unchanged)
  label_changed      : bool
  msn_tier1          : bool -- strict MSN set
  msn_tier2          : bool -- waveform-primary MSN  <- use for rescaling
  msn_tier3          : bool -- conservative MSN
  v1_cortical        : bool -- VISp units for V1 analysis
  v1_hippocampal     : bool -- CA1/SUB clean units

Output files:
  {p.LOGS_DIR}/RZ_msn_strict.csv        ({n1} units)
  {p.LOGS_DIR}/RZ_msn_waveform.csv      ({n2} units)  <- primary MSN
  {p.LOGS_DIR}/RZ_msn_conservative.csv  ({n3} units)
  {p.LOGS_DIR}/RZ_v1_cortical.csv       ({n_v1c} units)  <- primary V1
  {p.LOGS_DIR}/RZ_v1_hippocampal.csv    ({n_v1h} units)
  {PLOT_DIR}/relabeling_log.csv
  {PLOT_DIR}/cell_type_overview.png
  {PLOT_DIR}/msn_tiers_waterfall.png
  {PLOT_DIR}/rescued_msn_breakdown.png
  {PLOT_DIR}/msn_by_animal.png
  {PLOT_DIR}/units_per_region.png
""")