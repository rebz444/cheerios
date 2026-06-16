"""Count qualifying sessions/mice per region at each MIN_UNITS threshold.

No decoding — just load_decoder_data once per region and count via
get_qualifying_sessions over a sweep. Cheap (~30 s per region).
"""
from __future__ import annotations

import pandas as pd

from population_decoder import (
    CELL_SETS, GROUP_DICT, MIN_TRIALS,
    load_decoder_data, get_qualifying_sessions,
)
import paths as p

SWEEP   = [3, 5, 8, 10, 12, 15]
REGIONS = ['val', 'po', 'vpm', 'thal', 'v1', 'ca1', 'hpf']


def _qual_summary(trials_df, units_df, mu):
    q = get_qualifying_sessions(trials_df, units_df,
                                min_units=mu, min_trials=MIN_TRIALS)
    if q.empty:
        return {}
    q = q.copy()
    q['mouse'] = q['session'].str.split('_').str[0]
    rows = {}
    for grp, gdf in q.groupby('group', sort=False):
        rows[grp] = {
            'n_sess'    : int(len(gdf)),
            'n_mice'    : int(gdf['mouse'].nunique()),
            'mice'      : sorted(gdf['mouse'].unique().tolist()),
            'mice_sess' : gdf.groupby('mouse').size().to_dict(),
        }
    return rows


def main():
    all_rows = []
    for region in REGIONS:
        cfg = CELL_SETS[region]
        print(f"\n{'='*72}\n  REGION: {region}    filter={cfg['filter_col']}  pickles={cfg['pickle_suffixes']}\n{'='*72}")
        trials_df, _, units_df = load_decoder_data(
            group_dict=GROUP_DICT,
            str_units_csv=cfg['csv'],
            filter_col=cfg['filter_col'],
            pickle_suffixes=cfg['pickle_suffixes'],
        )

        for mu in SWEEP:
            res = _qual_summary(trials_df, units_df, mu)
            for grp in ('Short BG', 'Long BG'):
                if grp in res:
                    r = res[grp]
                    all_rows.append({
                        'region':    region,
                        'group':     grp,
                        'min_units': mu,
                        'n_sess':    r['n_sess'],
                        'n_mice':    r['n_mice'],
                        'mice':      ','.join(r['mice']),
                    })
                else:
                    all_rows.append({
                        'region':    region, 'group': grp,
                        'min_units': mu, 'n_sess': 0, 'n_mice': 0, 'mice': '',
                    })

    df = pd.DataFrame(all_rows)
    out = p.DATA_DIR / 'region_comparison' / 'min_units_feasibility.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n\nSaved feasibility table → {out}")

    # Pivot tables for quick read: n_sess and n_mice per (region, group, mu)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    for metric in ('n_sess', 'n_mice'):
        print(f"\n=== {metric} per (region × group × min_units) ===")
        piv = df.pivot_table(index=['region', 'group'], columns='min_units',
                              values=metric, aggfunc='first', fill_value=0)
        print(piv.to_string())


if __name__ == '__main__':
    main()
