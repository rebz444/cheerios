"""
0e_track_qc.py
==============
All track checks live here. Run BEFORE 0f_neuron_location_matching.py — units
belonging to tracks that fail this audit are excluded from region assignment
in 0f.

Four independent gates (a track passes only if all four are non-failure):

  1. Direction (DV sign vs population consensus)
       The DV axis of sample_space is detected automatically as the axis with
       the largest median |delta| across all tracks. The "correct" sign is
       the population majority. Tracks with the opposite sign are REVERSED.

  2. Region order (first vs last region in the CSV)
       A correct track starts at a cortical layer-1 region (probe enters
       through dorsal cortex) and ends deeper. If the *last* region is
       layer-1 and the first isn't, region_check = REVERSED.

  3. Depth consistency (track length vs in-vivo insertion depth, DV-corrected)
       For each session that uses the track, compare the track's path length
       to the session's insertion depth scaled by per-mouse DV shrinkage.
       Track fails if the worst-case |diff| / fixed_depth exceeds
       CONSISTENCY_THRESHOLD_PCT.

  4. Outside-brain content
       Fraction of trace voxels labelled "Not found in brain" by brainreg —
       i.e., the trace strayed outside the registered tissue. Track fails if
       this fraction exceeds OUTSIDE_BRAIN_THRESHOLD_PCT.

Wobble, tip-region info, and straightness (tortuosity + max perpendicular
deviation) are reported but not gating. The straightness metric flags tracks
with max bend > STRAIGHTNESS_THRESHOLD_UM for review.

Outputs:
  track_qc.csv                  — one row per track, written to LOGS_DIR
  qc_track_vs_insertion.png        — depth-consistency histograms
  qc_track_vs_insertion_scatter.png — track length vs insertion-depth scatter
"""

import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import paths as p

# ── Paths / constants ──────────────────────────────────────────────────────────
TRACKS_ROOT      = Path("/Volumes/T7 Shield/brain_stitching")
LOCAL_TRACKS_DIR = p.DATA_DIR / "probe_tracks"
PLOT_DIR         = p.DATA_DIR / "location_matching"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV          = p.LOGS_DIR / "track_qc.csv"

_DV_SCALE_CSV = p.LOGS_DIR / "dv_shrinkage.csv"

_RECORDING_LOG_SHEET_ID = "1xpb52EO7BII-_5zIcB2HaOpZqepFcYTvBYgxA4zLyNo"
RECORDING_LOG_URL = (
    f"https://docs.google.com/spreadsheets/d/{_RECORDING_LOG_SHEET_ID}"
    "/export?format=csv&gid=0"
)

# Flag tracks where |track_length - DV-corrected insertion_depth| / fixed_depth
# exceeds this fraction. 25% ≈ IBL 2-sigma upper bound for tissue shrinkage.
CONSISTENCY_THRESHOLD_PCT = 0.25

# Flag tracks where >this fraction of the trace falls outside the registered
# brain volume ("Not found in brain"). Small values (<5%) are typical when the
# user drags slightly past the dorsal pia; larger values mean the trace strayed
# into unsegmented tissue or empty space.
OUTSIDE_BRAIN_THRESHOLD_PCT = 1.0

# Informational only — flag tracks whose max perpendicular deviation from the
# first→last straight line exceeds this. Probes are near-rigid; >200 µm bend
# usually means sloppy clicking in brainreg, not a real probe trajectory.
STRAIGHTNESS_THRESHOLD_UM = 200.0

# Cortical layer-1 acronyms expected at the *start* of a correctly oriented track.
CORTICAL_L1_SUFFIXES = ("1",)


# ── Helpers (mirrored in 0f for now; consolidate later if needed) ──────────────

def copy_tracks_from_ssd():
    """Mirror per-mouse track files from SSD → local; wipes destination first."""
    if not TRACKS_ROOT.exists():
        print(f"Warning: SSD not found at {TRACKS_ROOT}. Using existing local tracks.")
        return
    LOCAL_TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    n_mice = n_files = 0
    for mouse_dir in TRACKS_ROOT.iterdir():
        if not mouse_dir.is_dir():
            continue
        src = mouse_dir / "brainreg_output" / "segmentation" / "sample_space" / "tracks"
        if not src.exists():
            continue
        dst = LOCAL_TRACKS_DIR / mouse_dir.name / "brainreg_output" / "segmentation" / "sample_space" / "tracks"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        for s in list(src.glob("*.csv")) + list(src.glob("*.points")):
            if s.name.startswith("._"):
                continue
            shutil.copy2(s, dst / s.name)
            n_files += 1
        n_mice += 1
    print(f"Tracks refreshed from SSD: {n_files} files across {n_mice} mice → {LOCAL_TRACKS_DIR}")


def load_points(pts_path: Path) -> np.ndarray | None:
    try:
        with h5py.File(pts_path, "r") as f:
            return f["df/block0_values"][:]
    except (OSError, KeyError):
        return None


def is_layer1(acronym: str) -> bool:
    a = str(acronym).strip()
    return a.endswith(CORTICAL_L1_SUFFIXES) and not a.endswith(("11", "21"))


def parse_depth(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    if s in ("all", "can't tell", "cant tell", ""):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def flip_hemisphere(h):
    return "r" if str(h).strip().lower() == "l" else "l"


def build_track_name(hemisphere, region):
    return f"{flip_hemisphere(hemisphere)}_{str(region).strip().lower()}"


def track_length_um(csv_path):
    return float(pd.read_csv(csv_path).iloc[:, 1].max())


# ── Region groupings (for coverage plot) ──────────────────────────────────────
REGION_GROUPS = {
    "cortex":   {"prefixes": ("MO", "SS", "ACA", "PL", "ILA", "ORB", "AI"),
                 "color":    "#4DAF4A"},
    "STR/CP":   {"exact":    ("CP", "STR", "CPi"),
                 "color":    "#E41A1C"},
    "PAL":      {"exact":    ("PAL", "GPe", "GPi", "SI", "MA", "AAA"),
                 "color":    "#FF7F00"},
    "WM/fiber": {"prefixes": ("ccb", "cing", "fi", "int", "ec", "scwm", "st", "fiber"),
                 "exact":    ("fiber tracts",),
                 "color":    "#984EA3"},
    "thalamus": {"prefixes": ("VL", "VM", "VPL", "VPM", "VAL", "MD", "LD", "LP",
                              "AV", "AD", "AM", "PO", "RT", "CL", "PCN", "CM",
                              "TH", "em"),
                 "color":    "#377EB8"},
    "other":    {"color":    "#A65628"},
}
GROUP_ORDER = ("cortex", "WM/fiber", "STR/CP", "PAL", "thalamus", "other")


def classify_region(acronym):
    """Return the broad-group name for an atlas acronym."""
    a = str(acronym)
    for group, spec in REGION_GROUPS.items():
        if group == "other":
            continue
        if "exact" in spec and a in spec["exact"]:
            return group
        if "prefixes" in spec and a.startswith(spec["prefixes"]):
            return group
    return "other"


def compute_straightness(pts: np.ndarray, csv_total_um: float):
    """Return (tortuosity, max_perp_deviation_um) for control points `pts`.

    tortuosity = piecewise path length / first→last euclidean distance (unitless)
    max_perp_deviation_um = greatest interior-point perpendicular distance from
    the first→last straight line, scaled from voxels to µm via the CSV's known
    total path length.
    Needs ≥3 points; returns (nan, nan) otherwise.
    """
    if pts is None or len(pts) < 3:
        return (float("nan"), float("nan"))
    diffs       = np.diff(pts, axis=0)
    path_vox    = float(np.linalg.norm(diffs, axis=1).sum())
    line_vec    = pts[-1] - pts[0]
    euclid_vox  = float(np.linalg.norm(line_vec))
    if euclid_vox == 0 or path_vox == 0:
        return (float("nan"), float("nan"))
    tortuosity = path_vox / euclid_vox
    # Perpendicular distance from each interior point to the first→last line.
    rel        = pts[1:-1] - pts[0]
    line_unit  = line_vec / euclid_vox
    along      = rel @ line_unit
    perp_vec   = rel - np.outer(along, line_unit)
    perp_vox   = np.linalg.norm(perp_vec, axis=1).max() if len(perp_vec) else 0.0
    um_per_vox = csv_total_um / path_vox
    return (round(tortuosity, 4), round(float(perp_vox) * um_per_vox, 1))


def find_track_csv(mouse, track_name):
    base = LOCAL_TRACKS_DIR / mouse / "brainreg_output" / "segmentation" / "sample_space" / "tracks"
    candidate = base / f"{track_name}.csv"
    if candidate.exists():
        return candidate
    for path in base.glob("*.csv"):
        if path.name.startswith("._"):
            continue
        if track_name in path.stem:
            return path
    return None


# ── 0. Sync local tracks from SSD ──────────────────────────────────────────────
copy_tracks_from_ssd()

# ── 1. Load DV shrinkage, recording log, and units ─────────────────────────────
if not _DV_SCALE_CSV.exists():
    print(f"{_DV_SCALE_CSV.name} not found — running compute_dv_shrinkage.py...")
    import subprocess, sys
    _shrink_script = Path(__file__).resolve().parent / "compute_dv_shrinkage.py"
    _r = subprocess.run([sys.executable, str(_shrink_script)],
                        cwd=str(_shrink_script.parent), check=False)
    if _r.returncode != 0 or not _DV_SCALE_CSV.exists():
        raise FileNotFoundError(
            f"{_DV_SCALE_CSV} still missing after compute_dv_shrinkage.py "
            f"(exit={_r.returncode}). Mount the brainreg SSD and retry."
        )
DV_SCALE = pd.read_csv(_DV_SCALE_CSV).set_index("mouse")["scale_DV"].to_dict()

log = pd.read_csv(RECORDING_LOG_URL)
log["date_only"] = pd.to_datetime(log["date"]).dt.strftime("%Y-%m-%d")
log["depth_um"]  = log["depth"].apply(parse_depth)
log["track_file"] = log.apply(
    lambda r: build_track_name(r["hemisphere"], r["region"]), axis=1
)

# Units: only audit tracks that actually have units recorded on them. Tracks
# that exist as files but have no downstream consumers don't need checking.
unit_props = pd.read_csv(p.LOGS_DIR / "unit_properties_with_qc.csv")
if "subject" in unit_props.columns:
    unit_props = unit_props.rename(columns={"subject": "mouse", "unit": "id",
                                            "session_datetime": "datetime"})
unit_props["date_only"] = pd.to_datetime(unit_props["datetime"]).dt.strftime("%Y-%m-%d")
_units_with_tracks = unit_props.merge(
    log[["mouse", "date_only", "insertion_number", "track_file"]],
    on=["mouse", "date_only", "insertion_number"], how="left"
)
_unit_counts = (
    _units_with_tracks.dropna(subset=["track_file"])
    .groupby(["mouse", "track_file"]).size().to_dict()
)
TRACKS_WITH_UNITS = set(_unit_counts.keys())

print(f"Loaded recording log: {len(log)} rows; DV scales for {len(DV_SCALE)} mice")
print(f"Units: {len(unit_props)} total; {len(TRACKS_WITH_UNITS)} unique (mouse, track) pairs have units")


# ── 2. Walk the tracks tree (direction + region check + tip dist) ──────────────
# Skip tracks that have no units recorded on them — those don't need auditing.
rows = []
n_skipped_no_units = 0
for mouse_dir in sorted(LOCAL_TRACKS_DIR.iterdir()):
    if not mouse_dir.is_dir():
        continue
    track_dir = mouse_dir / "brainreg_output" / "segmentation" / "sample_space" / "tracks"
    if not track_dir.exists():
        continue
    for csv_path in sorted(track_dir.glob("*.csv")):
        if csv_path.name.startswith("._"):
            continue
        if (mouse_dir.name, csv_path.stem) not in TRACKS_WITH_UNITS:
            n_skipped_no_units += 1
            continue
        pts_path = csv_path.with_suffix(".points")
        pts = load_points(pts_path) if pts_path.exists() else None

        df = pd.read_csv(csv_path)
        first_region = str(df["Region acronym"].iloc[0])
        last_region  = str(df["Region acronym"].iloc[-1])

        # Region sequence (run-length encoded): the ordered list of regions the
        # track passes through with consecutive duplicates collapsed. Answers
        # "does CP appear at all in this track?" without manual napari review.
        _regions = df["Region acronym"].astype(str).tolist()
        region_sequence = []
        for _r in _regions:
            if not region_sequence or region_sequence[-1] != _r:
                region_sequence.append(_r)

        # CP presence + dist range. The CSV's distance column lives at index 1
        # ("Distance from first position [um]"); track_length_um() relies on
        # the same convention.
        _dist_col = df.columns[1]
        _cp_mask  = df["Region acronym"] == "CP"
        cp_in_sequence = bool(_cp_mask.any())
        if cp_in_sequence:
            cp_dist_start_um = float(df.loc[_cp_mask, _dist_col].min())
            cp_dist_end_um   = float(df.loc[_cp_mask, _dist_col].max())
        else:
            cp_dist_start_um = np.nan
            cp_dist_end_um   = np.nan

        # Outside-brain content: rows where brainreg couldn't assign an atlas
        # region (trace passed through unsegmented voxels — typically past the
        # dorsal pia or below the registered tissue).
        is_outside = df["Region acronym"] == "Not found in brain"
        n_outside_brain = int(is_outside.sum())
        n_total_rows    = len(df)
        track_len       = track_length_um(csv_path)
        um_outside_brain = (n_outside_brain / n_total_rows) * track_len if n_total_rows else 0.0
        pct_outside_brain = 100.0 * n_outside_brain / n_total_rows if n_total_rows else 0.0

        tortuosity, max_dev_um = compute_straightness(pts, track_len)

        # Per-region-group coverage (µm). Each row "owns" half the gap to its
        # neighbours in dist_um. Used by the region-profiles plot below.
        _dist_vals  = df[_dist_col].to_numpy()
        _gaps       = np.abs(np.diff(_dist_vals))
        _row_lengths = np.zeros(len(_dist_vals))
        if len(_gaps):
            _row_lengths[:-1] += _gaps / 2
            _row_lengths[1:]  += _gaps / 2
            _row_lengths[0]    = _gaps[0]  / 2
            _row_lengths[-1]   = _gaps[-1] / 2
        _coverage = {g: 0.0 for g in GROUP_ORDER}
        for _i, _r in enumerate(_regions):
            _coverage[classify_region(_r)] += float(_row_lengths[_i])

        row = dict(
            mouse=mouse_dir.name,
            track=csv_path.stem,
            n_units=_unit_counts.get((mouse_dir.name, csv_path.stem), 0),
            n_points=len(pts) if pts is not None else 0,
            ax0_delta=np.nan, ax1_delta=np.nan, ax2_delta=np.nan,
            ax0_pct_reversed=np.nan, ax1_pct_reversed=np.nan, ax2_pct_reversed=np.nan,
            first_region=first_region,
            last_region=last_region,
            region_sequence=" → ".join(region_sequence),
            cp_in_sequence=cp_in_sequence,
            cp_dist_start_um=round(cp_dist_start_um, 1) if cp_in_sequence else np.nan,
            cp_dist_end_um=round(cp_dist_end_um, 1) if cp_in_sequence else np.nan,
            cp_length_um=round(cp_dist_end_um - cp_dist_start_um, 1) if cp_in_sequence else 0.0,
            cov_cortex_um=round(_coverage["cortex"],   1),
            cov_wm_um=round(_coverage["WM/fiber"],     1),
            cov_str_cp_um=round(_coverage["STR/CP"],   1),
            cov_pal_um=round(_coverage["PAL"],         1),
            cov_thal_um=round(_coverage["thalamus"],   1),
            cov_other_um=round(_coverage["other"],     1),
            track_length_um=track_len,
            n_outside_brain=n_outside_brain,
            pct_outside_brain=round(pct_outside_brain, 1),
            um_outside_brain=round(um_outside_brain, 1),
            tortuosity=tortuosity,
            max_dev_um=max_dev_um,
        )
        if pts is not None and len(pts) >= 2:
            d = pts[-1] - pts[0]
            row["ax0_delta"] = d[0]
            row["ax1_delta"] = d[1]
            row["ax2_delta"] = d[2]
            for ax in range(3):
                steps = np.diff(pts[:, ax])
                if d[ax] == 0 or len(steps) == 0:
                    continue
                wrong = (np.sign(steps) != np.sign(d[ax])).sum()
                row[f"ax{ax}_pct_reversed"] = round(100 * wrong / len(steps), 1)
        rows.append(row)

df_out = pd.DataFrame(rows)
if df_out.empty:
    raise SystemExit(f"No tracks found under {LOCAL_TRACKS_DIR}")

# Tracks that have units but whose CSV is missing locally — report so the user
# knows brainreg output is missing, not just being silently skipped.
audited_keys = set(zip(df_out["mouse"], df_out["track"]))
missing_csv = TRACKS_WITH_UNITS - audited_keys
print(f"\nTracks audited: {len(df_out)} (skipped {n_skipped_no_units} CSVs with no units recorded)")
if missing_csv:
    print(f"WARNING: {len(missing_csv)} (mouse, track) pairs have units but no CSV on disk:")
    for m, t in sorted(missing_csv):
        print(f"  {m} {t}  (n_units={_unit_counts[(m, t)]})")


# ── 3. Direction check (DV axis + consensus sign) ──────────────────────────────
median_abs = df_out[["ax0_delta", "ax1_delta", "ax2_delta"]].abs().median()
dv_axis    = int(median_abs.idxmax()[2])
dv_col     = f"ax{dv_axis}_delta"
consensus_sign = int(np.sign(df_out[dv_col].median()))

print(f"\nDV axis detected: axis {dv_axis}  (median |delta|={median_abs.iloc[dv_axis]:.0f} vox)")
print(f"Consensus DV sign: {consensus_sign:+d}  "
      f"(majority of tracks go {'+' if consensus_sign > 0 else '-'} along axis {dv_axis})")

df_out["dv_delta"] = df_out[dv_col]
_sign = np.sign(df_out["dv_delta"].fillna(0)).astype(int)
df_out["direction"] = np.where(
    df_out["dv_delta"].isna(), "no_points",
    np.where(_sign == consensus_sign, "OK",
             np.where(_sign == -consensus_sign, "REVERSED", "zero"))
)


# ── 4. Region-order check ──────────────────────────────────────────────────────
df_out["first_is_L1"] = df_out["first_region"].apply(is_layer1)
df_out["last_is_L1"]  = df_out["last_region"].apply(is_layer1)
df_out["region_check"] = np.where(
    df_out["first_is_L1"] & ~df_out["last_is_L1"], "OK",
    np.where(~df_out["first_is_L1"] & df_out["last_is_L1"], "REVERSED", "ambiguous")
)


# ── 5. Depth-consistency check (per-track, worst-case across sessions) ─────────
# For each track in df_out, find all sessions in the recording log that use it
# and compute the worst-case |diff_corrected| / fixed_depth ratio.
session_view_rows = []   # for the QC histogram + scatter
depth_status_per_track = {}

for _, r in df_out.iterrows():
    mouse = r["mouse"]
    track = r["track"]
    tl    = r["track_length_um"]
    scale = DV_SCALE.get(mouse)
    sessions = log[(log["mouse"] == mouse) & (log["track_file"] == track)]

    if scale is None:
        depth_status_per_track[(mouse, track)] = dict(
            passes_depth_check=False,
            depth_status=f"NO DV SCALE for {mouse}",
            worst_diff_corrected_um=np.nan,
            worst_frac=np.nan,
            n_sessions=len(sessions),
        )
        continue

    if len(sessions) == 0:
        depth_status_per_track[(mouse, track)] = dict(
            passes_depth_check=True,           # no usage → can't fail; downstream may still drop
            depth_status="no sessions in log",
            worst_diff_corrected_um=np.nan,
            worst_frac=np.nan,
            n_sessions=0,
        )
        continue

    worst_diff = 0.0
    worst_frac = 0.0
    any_unknown = False
    for _, s in sessions.iterrows():
        ins = s["depth_um"]
        if ins is None or (isinstance(ins, float) and np.isnan(ins)):
            any_unknown = True
            session_view_rows.append(dict(
                mouse=mouse, track=track, date=s["date_only"],
                insertion_depth_um=np.nan, insertion_depth_fixed_um=np.nan,
                track_length_um=tl, diff_um=np.nan, diff_corrected_um=np.nan,
            ))
            continue
        ins_fixed = ins * scale
        diff      = tl - ins
        diff_corr = tl - ins_fixed
        frac      = abs(diff_corr) / ins_fixed if ins_fixed else np.inf
        if abs(diff_corr) > abs(worst_diff):
            worst_diff = diff_corr
        if frac > worst_frac:
            worst_frac = frac
        session_view_rows.append(dict(
            mouse=mouse, track=track, date=s["date_only"],
            insertion_depth_um=ins, insertion_depth_fixed_um=ins_fixed,
            track_length_um=tl, diff_um=diff, diff_corrected_um=diff_corr,
        ))

    # Decision: pass if worst frac ≤ threshold; "depth unknown" rows alone don't fail.
    if worst_frac == 0.0 and any_unknown:
        depth_status_per_track[(mouse, track)] = dict(
            passes_depth_check=True,
            depth_status="depth unknown in log",
            length_verdict="",
            worst_diff_corrected_um=np.nan,
            worst_frac=np.nan,
            n_sessions=len(sessions),
        )
    else:
        passes = worst_frac <= CONSISTENCY_THRESHOLD_PCT
        # diff = track_length - corrected_insertion_depth. Positive => track is longer
        # than expected ("too long"); negative => track shorter ("too short").
        verdict = "ok" if passes else ("too long" if worst_diff > 0 else "too short")
        status = ("ok" if passes
                  else f"{verdict.upper()}: {abs(worst_diff):.0f} µm off ({worst_frac*100:.1f}%)")
        depth_status_per_track[(mouse, track)] = dict(
            passes_depth_check=passes,
            depth_status=status,
            length_verdict=verdict,
            worst_diff_corrected_um=worst_diff,
            worst_frac=worst_frac,
            n_sessions=len(sessions),
        )

depth_df = pd.DataFrame([
    dict(mouse=k[0], track=k[1], **v)
    for k, v in depth_status_per_track.items()
])
df_out = df_out.merge(depth_df, on=["mouse", "track"], how="left")
df_out["passes_depth_check"] = df_out["passes_depth_check"].fillna(False)


# ── 6. Final audit gate ────────────────────────────────────────────────────────
df_out["passes_outside_brain_check"] = (
    df_out["pct_outside_brain"] <= OUTSIDE_BRAIN_THRESHOLD_PCT
)
df_out["passes_audit"] = (
    (df_out["direction"] == "OK")
    & (df_out["region_check"] != "REVERSED")
    & df_out["passes_depth_check"]
    & df_out["passes_outside_brain_check"]
)

n_dir_bad     = (df_out["direction"] == "REVERSED").sum() + (df_out["direction"] == "no_points").sum()
n_region_bad  = (df_out["region_check"] == "REVERSED").sum()
n_depth_bad   = (~df_out["passes_depth_check"]).sum()
n_outside_bad = (~df_out["passes_outside_brain_check"]).sum()
print(f"\nAudit summary ({len(df_out)} tracks total):")
print(f"  Direction failed (REVERSED or no .points) : {n_dir_bad}")
print(f"  Region order REVERSED                     : {n_region_bad}")
print(f"  Depth-consistency failed                  : {n_depth_bad}")
print(f"  Outside-brain >{OUTSIDE_BRAIN_THRESHOLD_PCT:.0f}% of trace          : {n_outside_bad}")
print(f"  Passing audit                             : {df_out['passes_audit'].sum()}")


# ── CP-presence triage (for STR probes) ────────────────────────────────────────
# Tracks where CP never appears in the region sequence are genuine anatomical
# misses — the probe didn't reach striatum at all. Tracks where CP DOES appear
# but units are assigned elsewhere are depth/anchor issues, not anatomical ones.
df_out["probe_location"] = df_out["track"].apply(
    lambda t: "str" if "_str" in str(t) else ("v1" if "_v1" in str(t) else "other")
)
str_tracks = df_out[df_out["probe_location"] == "str"]
n_str = len(str_tracks)
n_str_with_cp = int(str_tracks["cp_in_sequence"].sum())
print(f"\nCP-in-sequence (STR probes only):")
print(f"  Tracks with CP in sequence    : {n_str_with_cp} / {n_str}")
print(f"  Tracks missing CP entirely    : {n_str - n_str_with_cp}  (genuine anatomical miss)")
if n_str_with_cp:
    cp_lens = str_tracks.loc[str_tracks["cp_in_sequence"], "cp_length_um"]
    print(f"  CP coverage µm  median/min/max: {cp_lens.median():.0f} / {cp_lens.min():.0f} / {cp_lens.max():.0f}")

cp_miss = str_tracks[~str_tracks["cp_in_sequence"]]
if len(cp_miss):
    print(f"\n  STR tracks with NO CP in region sequence ({len(cp_miss)}):")
    print(cp_miss[["mouse", "track", "n_units", "first_region", "last_region",
                   "track_length_um", "region_sequence"]].to_string(index=False))

dir_cols     = ["mouse", "track", "n_units", "dv_delta", "direction",
                "first_region", "last_region"]
region_cols  = ["mouse", "track", "n_units", "first_region", "last_region", "region_check"]
depth_cols   = ["mouse", "track", "n_units", "track_length_um",
                "worst_diff_corrected_um", "worst_frac",
                "length_verdict", "depth_status"]
outside_cols = ["mouse", "track", "n_units", "n_outside_brain",
                "pct_outside_brain", "um_outside_brain",
                "first_region", "last_region"]

dir_bad = df_out[df_out["direction"].isin(["REVERSED", "no_points"])]
if len(dir_bad):
    print(f"\nDirection discrepancy ({len(dir_bad)}):")
    print(dir_bad[dir_cols].to_string(index=False))

region_bad = df_out[df_out["region_check"] == "REVERSED"]
if len(region_bad):
    print(f"\nRegion-order discrepancy ({len(region_bad)}):")
    print(region_bad[region_cols].to_string(index=False))

outside_bad = df_out[~df_out["passes_outside_brain_check"]].sort_values(
    "pct_outside_brain", ascending=False
)
if len(outside_bad):
    print(f"\nOutside-brain discrepancy ({len(outside_bad)} tracks with >"
          f"{OUTSIDE_BRAIN_THRESHOLD_PCT:.0f}% outside the registered volume):")
    print(outside_bad[outside_cols].to_string(index=False))

depth_bad = df_out[~df_out["passes_depth_check"]]
if len(depth_bad):
    print(f"\nLength discrepancy ({len(depth_bad)}):")
    print(depth_bad[depth_cols].to_string(index=False))

# Straightness — informational only, not gating.
straight_cols = ["mouse", "track", "n_units", "tortuosity", "max_dev_um", "track_length_um"]
straight_flagged = df_out[df_out["max_dev_um"] > STRAIGHTNESS_THRESHOLD_UM].sort_values(
    "max_dev_um", ascending=False
)
if len(straight_flagged):
    print(f"\nStraightness (informational, {len(straight_flagged)} tracks "
          f"with max bend >{STRAIGHTNESS_THRESHOLD_UM:.0f} µm):")
    print(straight_flagged[straight_cols].to_string(index=False))


# ── 7. QC plots ────────────────────────────────────────────────────────────────
sess_df = pd.DataFrame(session_view_rows)
if len(sess_df):
    sess_df["probe_region"] = sess_df["track"].apply(
        lambda t: "str" if "_str" in str(t) else ("v1" if "_v1" in str(t) else "other")
    )

    # Histogram of (track length − insertion depth), raw and DV-corrected
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _diff_cols = [
        ("diff_um",           "track length − insertion depth (raw, µm)"),
        ("diff_corrected_um", "track length − DV-corrected depth (µm)"),
    ]
    for row_axes, (diff_col, xlabel) in zip(axes, _diff_cols):
        for ax, region in zip(row_axes, ["str", "v1"]):
            data = sess_df.loc[sess_df["probe_region"] == region, diff_col].dropna()
            ax.hist(data, bins=20, edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="red", linestyle="--", linewidth=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Session count")
            ax.set_title(region.upper())
            median_str = f"{data.median():.0f}" if len(data) else "n/a"
            ax.text(0.98, 0.97, f"n={len(data)}\nmedian={median_str} µm",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)
    plt.suptitle("Track length − insertion depth (top: raw  |  bottom: DV-corrected)", y=1.01)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "qc_track_vs_insertion.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Scatter: track length vs insertion depth (raw and DV-corrected)
    region_palette = {"str": "#2ca02c", "v1": "#9467bd", "other": "#7f7f7f"}
    _scatter_panels = [
        ("insertion_depth_um",       "Insertion depth (µm, raw)",          "Track length vs. insertion depth (raw)"),
        ("insertion_depth_fixed_um", "Insertion depth (µm, DV-corrected)", "Track length vs. insertion depth (DV-corrected)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (x_col, xlabel, title) in zip(axes, _scatter_panels):
        sd = sess_df[sess_df[x_col].notna() & sess_df["track_length_um"].notna()]
        if len(sd) == 0:
            continue
        depth_min = sd[x_col].min() * 0.85
        depth_max = max(sd[x_col].max(), sd["track_length_um"].max()) * 1.1
        ref = np.array([depth_min, depth_max])
        ax.fill_between(ref, ref * (1 - CONSISTENCY_THRESHOLD_PCT),
                        ref * (1 + CONSISTENCY_THRESHOLD_PCT),
                        color="gray", alpha=0.12,
                        label=f"±{int(CONSISTENCY_THRESHOLD_PCT*100)}% threshold")
        ax.plot(ref, ref, "k--", linewidth=1, label="y = x")
        for region, grp in sd.groupby("probe_region"):
            ax.scatter(grp[x_col], grp["track_length_um"],
                       color=region_palette.get(region, "gray"),
                       label=region, alpha=0.75, s=40, edgecolors="white", linewidths=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Track length (µm, from CSV)")
        ax.set_title(title)
        ax.set_xlim(depth_min, depth_max)
        ax.set_ylim(depth_min, depth_max)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "qc_track_vs_insertion_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 7b. Track region-profile + CP coverage plots ──────────────────────────────
# Stacked horizontal bar (one row per track) of region-group coverage, plus a
# secondary panel summarising CP coverage length and entry depth. Together
# these answer "did this track reach CP, and what did it pass through to get
# there?" in one figure.
_PROFILE_TRACKS = df_out.copy()
_PROFILE_TRACKS["probe_location"] = _PROFILE_TRACKS["track"].apply(
    lambda t: "str" if "_str" in str(t) else ("v1" if "_v1" in str(t) else "other")
)
_PROFILE_TRACKS = _PROFILE_TRACKS.sort_values(
    ["probe_location", "mouse", "track"]
).reset_index(drop=True)

if len(_PROFILE_TRACKS):
    fig, ax = plt.subplots(figsize=(14, max(4, len(_PROFILE_TRACKS) * 0.35 + 1.5)))
    _y = np.arange(len(_PROFILE_TRACKS))
    _cov_cols = [
        ("cov_cortex_um",  "cortex"),
        ("cov_wm_um",      "WM/fiber"),
        ("cov_str_cp_um",  "STR/CP"),
        ("cov_pal_um",     "PAL"),
        ("cov_thal_um",    "thalamus"),
        ("cov_other_um",   "other"),
    ]
    for i, row in _PROFILE_TRACKS.iterrows():
        left = 0.0
        for col, group in _cov_cols:
            length = float(row[col])
            if length <= 0:
                continue
            ax.barh(i, length / 1000, left=left / 1000, height=0.7,
                    color=REGION_GROUPS[group]["color"],
                    edgecolor="white", linewidth=0.4)
            if length > 100:
                ax.text(left / 1000 + length / 2000, i, f"{length/1000:.1f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")
            left += length
        marker = "✓" if row["cp_in_sequence"] else "✗"
        marker_color = "#E41A1C" if row["cp_in_sequence"] else "#999999"
        ax.text(row["track_length_um"] / 1000 + 0.05, i, marker,
                va="center", ha="left", fontsize=10, color=marker_color)

    ax.set_yticks(_y)
    ax.set_yticklabels(
        [f"{r['mouse']}  {r['track']}" for _, r in _PROFILE_TRACKS.iterrows()],
        fontsize=8,
    )
    ax.set_xlabel("Track length (mm)")
    ax.set_title("Region coverage along probe track\n(✓ = CP appears in sequence, ✗ = missed)",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=REGION_GROUPS[g]["color"], label=g)
                       for g in GROUP_ORDER],
              loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "region_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → region_profiles.png")

    # CP coverage summary: length + entry depth bars, hits only.
    _cp_hits = _PROFILE_TRACKS[_PROFILE_TRACKS["cp_in_sequence"]].reset_index(drop=True)
    _cp_miss = _PROFILE_TRACKS[~_PROFILE_TRACKS["cp_in_sequence"]].reset_index(drop=True)
    if len(_cp_hits) or len(_cp_miss):
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(_PROFILE_TRACKS) * 0.18)))
        _labels_hit  = [f"{r['mouse']} {r['track']}" for _, r in _cp_hits.iterrows()]
        _labels_miss = [f"{r['mouse']} {r['track']}" for _, r in _cp_miss.iterrows()]
        _x_hit  = np.arange(len(_cp_hits))
        _x_miss = np.arange(len(_cp_hits), len(_cp_hits) + len(_cp_miss))

        ax = axes[0]
        if len(_cp_hits):
            ax.bar(_x_hit, _cp_hits["cp_length_um"] / 1000, color="#E41A1C",
                   edgecolor="white", label="Hit CP")
        if len(_cp_miss):
            ax.bar(_x_miss, [0] * len(_cp_miss), color="#CCCCCC",
                   edgecolor="white", label="Missed CP")
        ax.set_xticks(np.concatenate([_x_hit, _x_miss]))
        ax.set_xticklabels(_labels_hit + _labels_miss, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("CP coverage (mm)")
        ax.set_title("CP coverage length per track")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1]
        if len(_cp_hits):
            ax.bar(_x_hit, _cp_hits["cp_dist_start_um"] / 1000, color="#377EB8",
                   edgecolor="white", label="CP entry depth")
            ax.axhline(1.6, color="red", linestyle="--",
                       label="Expected CP entry (1.6 mm)")
        ax.set_xticks(_x_hit if len(_cp_hits) else [])
        ax.set_xticklabels(_labels_hit, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Depth from track start (mm)")
        ax.set_title("CP entry depth per track")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle("CP coverage summary", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "cp_coverage_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → cp_coverage_summary.png")


# ── 8. Save ────────────────────────────────────────────────────────────────────
df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")
print(f"  Passing tracks: {df_out['passes_audit'].sum()} / {len(df_out)}")

print(f"Plots saved to: {PLOT_DIR}")
