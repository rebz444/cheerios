"""
0j_track_deviation_analysis.py
===============================
Analyze STR probe tracks exported by brainreg.

Track CSV format (from brainreg segmentation export):
  Index | Distance from first position [um] | Region ID | Region acronym | Region name

Planned insertion parameters:
  - ML: 1.5 mm lateral (from midline)
  - AP: 0.9 mm anterior (from bregma)
  - DV: 4.0 mm depth
  - Angle: 15° anterior-to-posterior tilt

Expected regions along planned trajectory:
  MOp → cing → ccb → CP

This script:
  1. Loads actual track files (l_str.csv, r_str.csv) for each mouse
  2. Computes region profile along each track (which regions, how much coverage)
  3. Identifies whether CP was reached and at what depth
  4. Flags tracks that missed STR and summarizes likely causes
  5. Plots region profiles per mouse

Outputs (written to DATA_DIR/probe_tracks/track_analysis/):
  track_region_summary.csv
  region_profiles.png
  cp_coverage_summary.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from glob import glob

# ── Paths ─────────────────────────────────────────────────────────────────────
class Paths:
    DATA_DIR = Path("./data")

try:
    import paths as p
except ImportError:
    p = Paths()

TRACK_DIR = p.DATA_DIR / "probe_tracks"
PLOT_DIR = TRACK_DIR / "track_analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Track files live at:
#   {TRACK_DIR}/{mouse}/brainreg_output/segmentation/sample_space/tracks/{side}_str.csv
TRACK_SEARCH_PATTERN = TRACK_DIR / "*" / "brainreg_output" / "segmentation" / "sample_space" / "tracks" / "*_str.csv"

# ── Planned trajectory (for reference) ────────────────────────────────────────
PLANNED_ML_MM   = 1.5   # lateral from midline
PLANNED_AP_MM   = 0.9   # anterior from bregma
PLANNED_DV_MM   = 4.0   # depth from surface
PLANNED_ANGLE_DEG = 15  # anterior-to-posterior tilt

EXPECTED_REGIONS = [
    {"region": "MOp",  "depth_start_mm": 0.0, "depth_end_mm": 1.0},
    {"region": "cing", "depth_start_mm": 1.0, "depth_end_mm": 1.3},
    {"region": "ccb",  "depth_start_mm": 1.3, "depth_end_mm": 1.6},
    {"region": "CP",   "depth_start_mm": 1.6, "depth_end_mm": 4.0},
]

# Region groupings for color-coding
REGION_GROUPS = {
    "cortex":   {"prefixes": ["MO", "SS", "ACA", "PL", "ILA", "ORB", "AI"],
                 "color": "#4DAF4A"},
    "STR/CP":   {"exact": ["CP", "STR", "CPi"],
                 "color": "#E41A1C"},
    "PAL":      {"exact": ["PAL", "GPe", "GPi", "SI", "MA", "AAA"],
                 "color": "#FF7F00"},
    "WM/fiber": {"prefixes": ["ccb", "cing", "fi", "int", "ec", "scwm", "st", "fiber"],
                 "exact": ["fiber tracts"],
                 "color": "#984EA3"},
    "thalamus": {"prefixes": ["VL", "VM", "VPL", "VPM", "VAL", "MD", "LD", "LP",
                              "AV", "AD", "AM", "PO", "RT", "CL", "PCN", "CM",
                              "TH", "em"],
                 "color": "#377EB8"},
    "other":    {"color": "#A65628"},
}

def classify_region(acronym):
    """Return the group name for a region acronym."""
    for group, spec in REGION_GROUPS.items():
        if group == "other":
            continue
        if "exact" in spec and acronym in spec["exact"]:
            return group
        if "prefixes" in spec:
            for pfx in spec["prefixes"]:
                if acronym.startswith(pfx):
                    return group
    return "other"


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TRACKS
# ══════════════════════════════════════════════════════════════════════════════
def find_track_files():
    """
    Search for *_str.csv files under TRACK_DIR.
    Returns dict: {mouse: {side: filepath}}
    """
    track_files = {}
    matches = glob(str(TRACK_SEARCH_PATTERN))

    for fpath in matches:
        fpath = Path(fpath)
        # Skip macOS resource-fork files (._filename)
        if fpath.name.startswith("."):
            continue

        fname = fpath.name
        if fname.startswith("l_"):
            side = "left"
        elif fname.startswith("r_"):
            side = "right"
        else:
            continue

        # Mouse ID is the first directory under TRACK_DIR
        try:
            mouse = fpath.relative_to(TRACK_DIR).parts[0]
        except ValueError:
            mouse = fpath.parts[-5]  # fallback

        track_files.setdefault(mouse, {})[side] = fpath

    return track_files


def load_track_csv(filepath):
    """
    Load a brainreg track CSV.

    Expected columns:
      Index | Distance from first position [um] | Region ID | Region acronym | Region name

    Returns DataFrame with those columns, or None on failure.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"    Error loading {filepath}: {e}")
        return None

    required = {"Distance from first position [um]", "Region acronym"}
    if not required.issubset(df.columns):
        print(f"    Unexpected columns in {filepath.name}: {df.columns.tolist()}")
        return None

    df = df.rename(columns={"Distance from first position [um]": "distance_um",
                             "Region acronym": "region",
                             "Region name": "region_name",
                             "Region ID": "region_id"})
    df = df.dropna(subset=["distance_um", "region"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def analyze_track(df):
    """
    Compute region-coverage statistics for one track.

    The track is ordered by the brainreg export (tip-or-entry first varies).
    We detect the cortical entry end by looking for motor/sensory cortex at
    either extreme, then orient the track entry→tip.

    Returns dict of stats.
    """
    if df is None or len(df) < 2:
        return None

    total_length_um = df["distance_um"].max()
    n_points        = len(df)

    # ── orient: find which end has cortex ────────────────────────────────────
    # Check first and last 10% of points
    n_check = max(1, n_points // 10)
    head_regions = set(df["region"].iloc[:n_check])
    tail_regions = set(df["region"].iloc[-n_check:])

    def is_cortex(regions):
        return any(classify_region(r) == "cortex" for r in regions)

    if is_cortex(head_regions) and not is_cortex(tail_regions):
        # Already entry→tip
        oriented = df.copy()
        oriented["depth_um"] = oriented["distance_um"]
    elif is_cortex(tail_regions) and not is_cortex(head_regions):
        # Flip: tip→entry in the file, reverse so index 0 = entry
        oriented = df.iloc[::-1].reset_index(drop=True)
        oriented["depth_um"] = total_length_um - oriented["distance_um"]
    else:
        # Ambiguous — use as-is with distance_um as proxy for depth
        oriented = df.copy()
        oriented["depth_um"] = oriented["distance_um"]

    # ── region coverage ───────────────────────────────────────────────────────
    # Each point "owns" half the gap to its neighbors (use abs so flipped
    # tracks — where distance_um decreases — give positive lengths).
    dist_vals = oriented["distance_um"].values
    gaps = np.abs(np.diff(dist_vals))
    point_lengths = np.zeros(len(dist_vals))
    point_lengths[:-1] += gaps / 2
    point_lengths[1:]  += gaps / 2
    # End points get the full first/last gap
    if len(gaps) > 0:
        point_lengths[0]  = gaps[0] / 2
        point_lengths[-1] = gaps[-1] / 2

    coverage = {}
    for region, group_df in oriented.groupby("region"):
        idx = group_df.index
        coverage[region] = point_lengths[idx].sum()

    # ── CP stats ──────────────────────────────────────────────────────────────
    cp_rows = oriented[oriented["region"] == "CP"]
    hit_cp  = len(cp_rows) > 0

    cp_depth_start_um = float(cp_rows["depth_um"].min()) if hit_cp else np.nan
    cp_depth_end_um   = float(cp_rows["depth_um"].max()) if hit_cp else np.nan
    cp_length_um      = coverage.get("CP", 0.0)

    # Also count all STR-group regions
    str_group_regions = [r for r in coverage if classify_region(r) == "STR/CP"]
    str_total_um      = sum(coverage[r] for r in str_group_regions)

    # ── entry and deepest regions ─────────────────────────────────────────────
    entry_region  = oriented["region"].iloc[0]
    deepest_region = oriented["region"].iloc[-1]

    # Region sequence (run-length encoded)
    prev = None
    region_sequence = []
    for r in oriented["region"]:
        if r != prev:
            region_sequence.append(r)
            prev = r

    return {
        "total_length_um":    total_length_um,
        "n_points":           n_points,
        "hit_cp":             hit_cp,
        "cp_depth_start_um":  cp_depth_start_um,
        "cp_depth_end_um":    cp_depth_end_um,
        "cp_length_um":       cp_length_um,
        "str_total_um":       str_total_um,
        "entry_region":       entry_region,
        "deepest_region":     deepest_region,
        "region_sequence":    region_sequence,
        "coverage":           coverage,
        "oriented_df":        oriented,
    }


def diagnose_miss(stats):
    """
    Return a short reason string for why a track missed CP.
    """
    if stats["hit_cp"]:
        return "OK"
    seq = stats["region_sequence"]
    seq_str = "→".join(seq[:6])

    deep_groups = {classify_region(r) for r in seq[-5:]}
    if "thalamus" in deep_groups:
        return f"Overshot posteriorly into thalamus ({seq_str}…)"
    if "PAL" in deep_groups:
        return f"Reached pallidum but not CP ({seq_str}…)"
    if "cortex" in deep_groups and stats["total_length_um"] < 3000:
        return f"Insufficient depth — stayed in cortex ({stats['total_length_um']:.0f} µm)"
    return f"Missed CP — path: {seq_str}…"


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def plot_region_profiles(all_results, plot_dir):
    """
    Horizontal stacked bar: one bar per track, showing depth coverage by region group.
    """
    # Build rows: (label, {group: length_um})
    rows = []
    for mouse, sides in sorted(all_results.items()):
        for side in ("left", "right"):
            if side not in sides:
                continue
            stats = sides[side]
            if stats is None:
                continue
            group_cov = {}
            for region, length in stats["coverage"].items():
                g = classify_region(region)
                group_cov[group_cov.get] if False else None
                group_cov[g] = group_cov.get(g, 0) + length
            rows.append({
                "label": f"{mouse}\n{side[0].upper()}",
                "total": stats["total_length_um"],
                "hit_cp": stats["hit_cp"],
                "group_cov": group_cov,
            })

    if not rows:
        return

    group_order  = ["cortex", "WM/fiber", "STR/CP", "PAL", "thalamus", "other"]
    group_colors = {g: REGION_GROUPS[g]["color"] for g in group_order}

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.55 + 1.5)))
    y_pos = np.arange(len(rows))

    for i, row in enumerate(rows):
        left = 0
        for g in group_order:
            length = row["group_cov"].get(g, 0)
            if length <= 0:
                continue
            ax.barh(i, length / 1000, left=left / 1000, height=0.7,
                    color=group_colors[g], edgecolor="white", linewidth=0.5)
            if length > 100:
                ax.text(left / 1000 + length / 2000, i,
                        f"{length/1000:.1f}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
            left += length

        # Mark CP hit/miss
        marker = "✓" if row["hit_cp"] else "✗"
        color  = "#E41A1C" if row["hit_cp"] else "#999999"
        ax.text(row["total"] / 1000 + 0.05, i, marker,
                va="center", ha="left", fontsize=10, color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_xlabel("Track length (mm)")
    ax.set_title("Region Coverage Along Probe Track\n(✓ = hit CP, ✗ = missed CP)",
                 fontsize=13, fontweight="bold")

    # Add planned depth reference lines
    for er in EXPECTED_REGIONS:
        ax.axvline(er["depth_start_mm"], color="black", linewidth=0.8,
                   linestyle=":", alpha=0.4)
    ax.axvline(PLANNED_DV_MM, color="black", linewidth=1.2,
               linestyle="--", alpha=0.6, label=f"Planned depth ({PLANNED_DV_MM} mm)")

    # Legend
    patches = [mpatches.Patch(color=group_colors[g], label=g) for g in group_order]
    patches.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label=f"Planned depth ({PLANNED_DV_MM} mm)"))
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = plot_dir / "region_profiles.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def plot_cp_summary(results_df, plot_dir):
    """
    Bar chart of CP length per track, and CP entry depth.
    """
    cp_df = results_df[results_df["hit_cp"]].copy()
    miss_df = results_df[~results_df["hit_cp"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── CP length ─────────────────────────────────────────────────────────────
    ax = axes[0]
    labels_hit  = [f"{r['mouse']}\n{r['side'][0].upper()}" for _, r in cp_df.iterrows()]
    labels_miss = [f"{r['mouse']}\n{r['side'][0].upper()}" for _, r in miss_df.iterrows()]

    x_hit  = np.arange(len(cp_df))
    x_miss = np.arange(len(cp_df), len(cp_df) + len(miss_df))

    if len(cp_df):
        ax.bar(x_hit, cp_df["cp_length_um"] / 1000, color="#E41A1C",
               edgecolor="white", label="Hit CP")
    if len(miss_df):
        ax.bar(x_miss, [0] * len(miss_df), color="#CCCCCC",
               edgecolor="white", label="Missed CP")

    ax.set_xticks(np.concatenate([x_hit, x_miss]))
    ax.set_xticklabels(labels_hit + labels_miss, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("CP coverage (mm)")
    ax.set_title("CP Coverage Length")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── CP entry depth ────────────────────────────────────────────────────────
    ax = axes[1]
    if len(cp_df):
        ax.bar(x_hit, cp_df["cp_depth_start_um"] / 1000, color="#377EB8",
               edgecolor="white", label="CP entry depth")
        ax.axhline(EXPECTED_REGIONS[-1]["depth_start_mm"], color="red",
                   linestyle="--", label=f"Expected CP depth ({EXPECTED_REGIONS[-1]['depth_start_mm']} mm)")
    ax.set_xticks(x_hit if len(cp_df) else [])
    ax.set_xticklabels(labels_hit, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Depth from track start (mm)")
    ax.set_title("CP Entry Depth")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("CP Coverage Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = plot_dir / "cp_coverage_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("0j_track_deviation_analysis.py")
print("  Analyzing actual STR probe tracks (brainreg region export)")
print("=" * 70)

print(f"""
Planned insertion (for reference):
  ML: {PLANNED_ML_MM} mm, AP: {PLANNED_AP_MM} mm, DV: {PLANNED_DV_MM} mm, Angle: {PLANNED_ANGLE_DEG}°

Expected region sequence (entry → tip):
  {"  →  ".join(r["region"] for r in EXPECTED_REGIONS)}
""")

# ── Find tracks ───────────────────────────────────────────────────────────────
print("-" * 70)
print("SEARCHING FOR TRACK FILES")
print(f"  Pattern: {TRACK_SEARCH_PATTERN}")
print("-" * 70)

track_files = find_track_files()

if not track_files:
    print(f"\n  No track files found at:\n    {TRACK_SEARCH_PATTERN}")
    raise SystemExit(1)

print(f"\n  Found {sum(len(v) for v in track_files.values())} tracks for {len(track_files)} mice:")
for mouse in sorted(track_files):
    sides_str = ", ".join(sorted(track_files[mouse]))
    print(f"    {mouse}: {sides_str}")

# ── Analyze ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("ANALYZING TRACKS")
print("-" * 70)

all_results = {}
records     = []

for mouse in sorted(track_files):
    all_results[mouse] = {}
    for side in ("left", "right"):
        if side not in track_files[mouse]:
            continue
        fpath = track_files[mouse][side]
        print(f"\n  {mouse} {side}:  {fpath.name}")

        df    = load_track_csv(fpath)
        stats = analyze_track(df)

        if stats is None:
            print("    Failed to analyze")
            all_results[mouse][side] = None
            continue

        all_results[mouse][side] = stats

        print(f"    Points: {stats['n_points']}   Length: {stats['total_length_um']:.0f} µm")
        print(f"    Regions: {' → '.join(stats['region_sequence'])}")
        print(f"    CP hit: {'YES' if stats['hit_cp'] else 'NO'}")
        if stats["hit_cp"]:
            print(f"    CP depth: {stats['cp_depth_start_um']:.0f}–{stats['cp_depth_end_um']:.0f} µm  "
                  f"({stats['cp_length_um']:.0f} µm coverage)")
            print(f"    STR group total: {stats['str_total_um']:.0f} µm")
        else:
            print(f"    Diagnosis: {diagnose_miss(stats)}")

        records.append({
            "mouse":              mouse,
            "side":               side,
            "total_length_um":    stats["total_length_um"],
            "n_points":           stats["n_points"],
            "hit_cp":             stats["hit_cp"],
            "cp_depth_start_um":  stats["cp_depth_start_um"],
            "cp_depth_end_um":    stats["cp_depth_end_um"],
            "cp_length_um":       stats["cp_length_um"],
            "str_total_um":       stats["str_total_um"],
            "entry_region":       stats["entry_region"],
            "deepest_region":     stats["deepest_region"],
            "region_sequence":    " → ".join(stats["region_sequence"]),
            "diagnosis":          diagnose_miss(stats),
        })

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(records)

n_total = len(results_df)
n_hit   = results_df["hit_cp"].sum()
n_miss  = n_total - n_hit

print(f"\n  Total tracks: {n_total}")
print(f"  Hit CP:       {n_hit}  ({100*n_hit/n_total:.0f}%)")
print(f"  Missed CP:    {n_miss} ({100*n_miss/n_total:.0f}%)")

if n_hit > 0:
    cp_hits = results_df[results_df["hit_cp"]]
    print(f"\n  CP coverage (hits only):")
    print(f"    Length:      {cp_hits['cp_length_um'].mean():.0f} ± {cp_hits['cp_length_um'].std():.0f} µm")
    print(f"    Entry depth: {cp_hits['cp_depth_start_um'].mean():.0f} ± {cp_hits['cp_depth_start_um'].std():.0f} µm")

if n_miss > 0:
    print(f"\n  Missed tracks:")
    for _, row in results_df[~results_df["hit_cp"]].iterrows():
        print(f"    {row['mouse']} {row['side']}: {row['diagnosis']}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_csv = PLOT_DIR / "track_region_summary.csv"
results_df.drop(columns=["oriented_df"] if "oriented_df" in results_df.columns else [])
results_df.to_csv(out_csv, index=False)
print(f"\n  Saved: {out_csv}")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("GENERATING PLOTS")
print("-" * 70)

plot_region_profiles(all_results, PLOT_DIR)
plot_cp_summary(results_df, PLOT_DIR)

print("\n" + "=" * 70)
print(f"Output directory: {PLOT_DIR}")
print("=" * 70)
