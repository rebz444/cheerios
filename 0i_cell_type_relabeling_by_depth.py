"""
0i_cell_type_relabeling_by_depth.py
===================================
Augments the canonical unit_properties_final.csv (output of 0h) with two
boolean columns marking depth-based MSN candidates. This is the depth-based
counterpart to 0h's waveform-based MSN classification — it's more robust to
track reconstruction failures because depth comes from peak channel + session
insertion depth, not the histology warp.

Pipeline ordering: 0h → 0i. 0i reads the canonical CSV, adds columns, and
writes back in place.

Zones (depth_from_surface = dist_along_track_um, fixed-tissue µm, from 0f):
  CORTEX     <1500 µm         → exclude (cortical neurons)
  BELOW_CTX  ≥1500 µm         → MSN candidate (if waveform + FR pass)

MSN gate within BELOW_CTX zone (STR probes only):
  PT > 0.40 ms       (excludes FSI and RT-broad-but-narrow)
  FR < 5 Hz strict   → is_msn_depth
  FR < 15 Hz permissive → is_msn_depth_permissive

Inputs:
  logs/unit_properties_final.csv   (output of 0h; carries waveform + depth)

Outputs:
  logs/unit_properties_final.csv   (overwritten with two new boolean columns)
"""

import numpy as np
import pandas as pd

import paths as p

# ── Zone boundaries (fixed-tissue µm from surface) ────────────────────────────
CORTEX_MAX = 1500

# ── MSN gate ──────────────────────────────────────────────────────────────────
PT_BROAD_MIN      = 0.40   # ms
FR_STRICT_MAX     = 5.0    # Hz — TANs/thalamic mostly fail this
FR_PERMISSIVE_MAX = 15.0   # Hz


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("0i_cell_type_relabeling_by_depth.py")
print("=" * 70)

csv_path = p.LOGS_DIR / "unit_properties_final.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Missing {csv_path}. Run 0h first.")

print(f"\nLoading: {csv_path.name}")
df = pd.read_csv(csv_path)

for required in ("dist_along_track_um", "pt_duration_ms", "firing_rate",
                 "probe_region", "bg_group"):
    if required not in df.columns:
        raise ValueError(f"{csv_path.name} missing column '{required}' — "
                         f"re-run 0h (which carries it forward from 0f).")

# ── Compute masks (as locals, not materialised on df) ─────────────────────────
depth_zone   = np.where(
    df["dist_along_track_um"].isna(), "unknown",
    np.where(df["dist_along_track_um"] < CORTEX_MAX, "CORTEX", "BELOW_CTX")
)
is_str_probe = df["probe_region"] == "str"
in_below_ctx = depth_zone == "BELOW_CTX"
pt_broad     = df["pt_duration_ms"] >= PT_BROAD_MIN
fr_strict    = df["firing_rate"] < FR_STRICT_MAX
fr_permiss   = df["firing_rate"] < FR_PERMISSIVE_MAX

df["is_msn_depth"]            = is_str_probe & in_below_ctx & pt_broad & fr_strict
df["is_msn_depth_permissive"] = is_str_probe & in_below_ctx & pt_broad & fr_permiss

# ── Reporting ─────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("Zone counts (STR probe only)")
print("-" * 70)
str_idx = is_str_probe
zone_series = pd.Series(depth_zone, index=df.index)
print(pd.crosstab(zone_series[str_idx], df.loc[str_idx, "bg_group"],
                  margins=True).to_string())

print("\n" + "-" * 70)
print(f"MSN candidates below cortex (STR probe; PT ≥ {PT_BROAD_MIN} ms)")
print("-" * 70)

n_below_str = int((is_str_probe & in_below_ctx).sum())
n_strict    = int(df["is_msn_depth"].sum())
n_permiss   = int(df["is_msn_depth_permissive"].sum())
print(f"  Below-cortex STR units            : {n_below_str}")
print(f"  Strict   (FR < {FR_STRICT_MAX:.0f} Hz) : {n_strict}")
print(f"  Permiss. (FR < {FR_PERMISSIVE_MAX:.0f} Hz) : {n_permiss}")

print("\n  Strict MSN per BG group:")
print(df.loc[df["is_msn_depth"], "bg_group"].value_counts().to_string())
print("\n  Permissive MSN per BG group:")
print(df.loc[df["is_msn_depth_permissive"], "bg_group"].value_counts().to_string())

_session_keys = (["mouse", "date_only", "insertion_number"]
                 if "date_only" in df.columns
                 else ["mouse", "datetime", "insertion_number"])

# Decoder eligibility (≥15 MSNs / session) by BG group, both tiers
DECODER_MIN_UNITS = 15
print("\n  Strict MSN per session (top 10 by count):")
strict_per_sess = (df[df["is_msn_depth"]].groupby(_session_keys + ["bg_group"]).size()
                   .rename("n_msn").reset_index()
                   .sort_values("n_msn", ascending=False))
print(strict_per_sess.head(10).to_string(index=False))

permiss_per_sess = (df[df["is_msn_depth_permissive"]].groupby(_session_keys + ["bg_group"]).size()
                    .rename("n_msn").reset_index()
                    .sort_values("n_msn", ascending=False))

for tier_name, per_sess in (("strict",     strict_per_sess),
                            ("permissive", permiss_per_sess)):
    per_sess["meets_threshold"] = per_sess["n_msn"] >= DECODER_MIN_UNITS
    n_qual = int(per_sess["meets_threshold"].sum())
    print(f"\n  Sessions ≥{DECODER_MIN_UNITS} {tier_name} MSNs: {n_qual}")
    for grp in ("Short BG", "Long BG"):
        sub = per_sess[per_sess["bg_group"] == grp]
        if len(sub):
            print(f"    {grp:<9}: {int(sub['meets_threshold'].sum())} / {len(sub)}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(csv_path, index=False)
print(f"\nSaved → {csv_path}  (added is_msn_depth + is_msn_depth_permissive)")
