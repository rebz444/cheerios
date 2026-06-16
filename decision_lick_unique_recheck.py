#!/usr/bin/env python3
"""
decision_lick_unique_recheck.py

Targeted re-fit of the decision-lick-specific units WITH unique ΔpR² on, to test
whether their lick_decision response survives partialling out WAIT/outcome/etc.

The full sweep (4c_simple_glm.py) ran with COMPUTE_UNIQUE_DPR2=False for speed, so
we only have MARGINAL ΔpR² (family-vs-null, which over-credits shared variance).
The decision lick coincides with the wait-ramp peak, consumption onset, and reward
delivery — so a high marginal lick_decision could be decision-TIME activity rather
than a genuine independent decision/action code. UNIQUE ΔpR² (full minus family)
answers that: how much does lick_decision explain that nothing else already does.

Selection (same as the interactive query): marginal lick_decision >= 0.002 AND
lick_decision > lick_bg AND > lick_cons. Reuses 4c_simple_glm's design + fitting so
the numbers are directly comparable to per_unit.csv.

Output: <DATA_DIR>/glm_simple/decision_lick_unique.csv  (one row per unit, marg + unique).
"""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import utils

spec = importlib.util.spec_from_file_location("sg", "4c_simple_glm.py")
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)

sg.COMPUTE_UNIQUE_DPR2 = True   # the whole point of this recheck

LD, LB, LC = ("delta_pr2_marg_lick_decision",
              "delta_pr2_marg_lick_bg", "delta_pr2_marg_lick_cons")
OUT_CSV = sg.OUT_DIR / "decision_lick_unique.csv"

d = pd.read_csv(sg.PER_UNIT_CSV)
mask = (d[LD] >= 0.002) & (d[LD] > d[LB]) & (d[LD] > d[LC])
targets = d[mask].copy()
print(f"[recheck] {len(targets)} decision-specific units across "
      f"{targets['session_id'].nunique()} sessions; unique ΔpR² ON", flush=True)

bg_lookup = sg.load_bg_group_lookup()
rows = []
n = 0
for sid, sub in targets.groupby("session_id"):
    try:
        events, trials, units = utils.get_session_data(sid)
    except Exception as e:
        print(f"[skip session] {sid}: {e}", flush=True)
        continue
    if "missed" in trials.columns:
        trials = trials.loc[~sg.coerce_bool_series(trials["missed"])].copy()
    trials_idx = trials.set_index("trial_id", drop=False)
    bg_basis, wait_basis = sg.basis_params_for_group(bg_lookup.get(sid))

    for _, ur in sub.iterrows():
        uk = ur["unit_key"]
        try:
            sp = sg.spikes_df_to_trial_map(units[uk])
        except KeyError:
            print(f"  [skip] {sid} {ur['unit_id']}: not in spikes dict", flush=True)
            continue
        Xraw, y, nmraw, tid, used, _ = sg.build_unit_design(
            sp, trials_idx, events, bg_basis=bg_basis, wait_basis=wait_basis)
        if Xraw is None:
            continue
        X, names, _ = sg.drop_zero_variance_columns(Xraw, nmraw)
        try:
            metrics, _, _ = sg.evaluate_unit(X, y, names, tid)
        except Exception as e:
            print(f"  [skip] {sid} {ur['unit_id']}: {e}", flush=True)
            continue
        rows.append({
            "session_id": sid, "unit_id": ur["unit_id"],
            "region_group": ur["region_group"], "cell_type": ur["cell_type"],
            "marg_lick_decision": float(ur[LD]),
            "uniq_lick_decision": metrics["delta_pr2_unique_lick_decision"],
            "uniq_lick_cons": metrics["delta_pr2_unique_lick_cons"],
            "uniq_WAIT_time": metrics["delta_pr2_unique_WAIT_time"],
            "uniq_outcome": metrics["delta_pr2_unique_outcome"],
            "uniq_spike_history": metrics["delta_pr2_unique_spike_history"],
        })
        n += 1
        print(f"  [{n}/{len(targets)}] {sid} {ur['unit_id']}: "
              f"marg_dec={ur[LD]:.4f}  uniq_dec={rows[-1]['uniq_lick_decision']:.4f}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)  # incremental

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}  ({len(out)} units)")

# ---- Summary ----
u = out["uniq_lick_decision"]
print("\n=== unique lick_decision ΔpR² distribution ===")
print(u.describe(percentiles=[.5, .9]).round(4).to_string())
for thr in (0.0, 0.001, 0.002):
    print(f"  uniq_lick_decision > {thr}: {(u > thr).sum()} ({(u > thr).mean():.0%})")
out["shared"] = out["marg_lick_decision"] - out["uniq_lick_decision"]
print(f"\nmedian marginal={out['marg_lick_decision'].median():.4f}  "
      f"median unique={u.median():.4f}  "
      f"median shared(marg-uniq)={out['shared'].median():.4f}")
print("\n=== genuine independent decision coders (uniq>=0.002) by region ===")
genuine = out[out["uniq_lick_decision"] >= 0.002]
print(genuine.groupby("region_group").size().sort_values(ascending=False).to_string())
print(f"\n{len(genuine)}/{len(out)} units keep a real independent decision-lick "
      f"signal after partialling out WAIT/outcome/cons/history.")
