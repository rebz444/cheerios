"""
compute_dv_shrinkage.py
========================
Compute per-mouse DV tissue-shrinkage scale factors from brainreg output and
write them to LOGS_DIR / RZ_dv_shrinkage.csv. Run this whenever a new mouse
finishes brainreg registration.

Method:
  brainreg's boundaries.tiff is the atlas region map warped into sample space.
  It is stored in CCF canonical order (AP, DV, ML) at the atlas voxel size
  (10 µm for allen_mouse_10um). Therefore:
      physical_DV_um = shape[1] * atlas_voxel_um
      scale_DV       = physical_DV_um / CCF_DV_8000um
  This formula is orientation-agnostic — input "ipr" and "ial" mice both work.

The output CSV is the single source of truth consumed by 0e_neuron_location_matching.py.

Inputs:
  TRACKS_ROOT/<mouse>/brainreg_output/boundaries.tiff
  TRACKS_ROOT/<mouse>/brainreg_output/brainreg.json   (for atlas name)

Outputs:
  LOGS_DIR / RZ_dv_shrinkage.csv
  PLOT_DIR / dv_shrinkage_per_animal.png   (sanity bar plot)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tifffile

import paths as p

# ── Config ─────────────────────────────────────────────────────────────────────
TRACKS_ROOT = Path("/Volumes/T7 Shield/brain_stitching")
CCF_DV_UM   = 8000.0
DV_AXIS     = 1   # CCF canonical: axis 0=AP, axis 1=DV, axis 2=ML

OUT_CSV  = p.LOGS_DIR / "RZ_dv_shrinkage.csv"
PLOT_DIR = p.DATA_DIR / "location_matching"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Free-form per-animal annotations carried into the output CSV.
NOTES = {
    "RZ052": "low confidence",
}


def atlas_voxel_um(brainreg_json_path: Path) -> float:
    """Return the atlas voxel size in µm.

    Parses the atlas name from brainreg.json (e.g. 'allen_mouse_10um' → 10.0).
    Raises if the atlas name doesn't end in '<N>um'.
    """
    cfg = json.loads(brainreg_json_path.read_text())
    atlas = cfg.get("atlas", "")
    # e.g. "allen_mouse_10um" → "10"
    if not atlas.endswith("um"):
        raise ValueError(
            f"Cannot infer atlas voxel size from atlas name '{atlas}' "
            f"(expected suffix like '10um'). File: {brainreg_json_path}"
        )
    digits = atlas.rsplit("_", 1)[-1].rstrip("um")
    return float(digits)


def compute_one(mouse_dir: Path) -> dict | None:
    """Compute shrinkage for a single mouse directory, or None if no brainreg output."""
    bdir  = mouse_dir / "brainreg_output"
    btiff = bdir / "boundaries.tiff"
    bjson = bdir / "brainreg.json"
    if not btiff.exists() or not bjson.exists():
        return None

    voxel_um = atlas_voxel_um(bjson)
    with tifffile.TiffFile(btiff) as tif:
        shape = tif.series[0].shape

    dv_planes   = shape[DV_AXIS]
    physical_dv = dv_planes * voxel_um
    scale_dv    = physical_dv / CCF_DV_UM

    cfg = json.loads(bjson.read_text())
    return {
        "mouse"           : mouse_dir.name,
        "dv_planes"       : dv_planes,
        "atlas_voxel_um"  : voxel_um,
        "physical_dv_um"  : round(physical_dv, 1),
        "scale_DV"        : round(scale_dv, 4),
        "shrinkage_pct"   : round((1 - scale_dv) * 100, 1),
        "atlas"           : cfg.get("atlas", ""),
        "input_orientation": cfg.get("orientation", ""),
        "notes"           : NOTES.get(mouse_dir.name, ""),
    }


def main():
    if not TRACKS_ROOT.exists():
        raise FileNotFoundError(f"TRACKS_ROOT not mounted: {TRACKS_ROOT}")

    rows = []
    skipped = []
    for mouse_dir in sorted(TRACKS_ROOT.iterdir()):
        if not mouse_dir.is_dir() or not mouse_dir.name.startswith("RZ"):
            continue
        rec = compute_one(mouse_dir)
        if rec is None:
            skipped.append(mouse_dir.name)
        else:
            rows.append(rec)

    df = pd.DataFrame(rows).sort_values("mouse").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(df.to_string(index=False))
    print(f"\n{len(df)} mice → {OUT_CSV}")
    if skipped:
        print(f"Skipped (no boundaries.tiff or brainreg.json): {skipped}")
    print(f"Mean DV shrinkage: {df['shrinkage_pct'].mean():.1f}% "
          f"± {df['shrinkage_pct'].std():.1f}%   "
          f"(IBL reference: 14% ± 5%)")

    # ── Sanity plot ────────────────────────────────────────────────────────────
    df_sorted = df.sort_values("shrinkage_pct").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(df))))
    ax.axvspan(4, 24, color="gray", alpha=0.15, label="IBL 2σ band (4–24%)")
    ax.axvline(14, color="gray", linestyle="--", linewidth=1, label="IBL median (14%)")
    bars = ax.barh(df_sorted["mouse"], df_sorted["shrinkage_pct"],
                   color="#1f77b4", edgecolor="white", linewidth=0.5)
    for bar, val, note in zip(bars, df_sorted["shrinkage_pct"], df_sorted["notes"]):
        label = f"{val:.1f}%" + (f"  ({note})" if note else "")
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8)
    ax.set_xlabel("DV shrinkage (%)")
    ax.set_title("DV tissue shrinkage per animal (computed from boundaries.tiff)")
    ax.set_xlim(0, max(df_sorted["shrinkage_pct"]) * 1.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_png = PLOT_DIR / "dv_shrinkage_per_animal.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out_png}")


if __name__ == "__main__":
    main()
