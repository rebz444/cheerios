#!/usr/bin/env python3
"""
cross_tab_test1_test2.py

Per-region cross-tabulation of Test 1 (time encoding, 4f) and Test 2
(reward history, 4d) per unit. CLAUDE.md step 3.

For each (region_group, anchor), count units in the 2x2 categories:
    only Test 1  (time-encoder, not reward-modulated)
    only Test 2  (reward-modulated, not time-encoder)
    both
    neither

Test 1 significance criteria:
    - 'lrt'        : p_wait_{anchor} < LRT_ALPHA  (nominal LRT only)
    - 'sustained'  : sustained_sig_{anchor} == True  (LRT + peak_lag >= 1s; recommended)
Test 2 significance criterion:
    - p_pr_cond < LRT_ALPHA  (always nominal; one statistic per unit)

Outputs
-------
  cross_tab_{anchor}_{criterion}.csv      one row per (region, category)
  cross_tab_summary.csv                   region x category counts (wide)
  cross_tab_overlap.png                   Venn-style stacked bars per region
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paths as p

LRT_ALPHA = 0.05
EXCLUDE_REGION_GROUPS = ('Excluded', 'Other')

T1_CSV = Path(p.DATA_DIR) / 'time_encoding_two_anchors' / 'per_unit.csv'
T2_CSV = Path(p.DATA_DIR) / 'simpler_reward_history_test' / 'per_unit.csv'
OUT_DIR = Path(p.DATA_DIR) / 'cross_tab_test1_test2'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_merge() -> pd.DataFrame:
    if not T1_CSV.exists():
        raise FileNotFoundError(f"Test 1 per_unit.csv not found at {T1_CSV}")
    if not T2_CSV.exists():
        raise FileNotFoundError(f"Test 2 per_unit.csv not found at {T2_CSV}")
    t1 = pd.read_csv(T1_CSV)
    t2 = pd.read_csv(T2_CSV)
    # Test 2 uses 'unit_id' as the human-readable string;
    # Test 1 keeps both 'unit_id' (string) and 'unit_key' (integer id).
    merged = t1.merge(
        t2[['session_id', 'unit_id', 'p_pr_cond', 'beta_pr_cond',
            'p_pr_uncond', 'beta_pr_uncond', 'p_interaction', 'fit_status']],
        on=['session_id', 'unit_id'], how='outer',
        suffixes=('_t1', '_t2'),
    )
    merged = merged.rename(columns={'fit_status': 'fit_status_t2'})
    return merged


def cross_tab_one(merged: pd.DataFrame, anchor: str, criterion: str) -> pd.DataFrame:
    """Per-region 2x2 counts for Test1(anchor) x Test2."""
    if criterion == 'lrt':
        t1_col = f'p_wait_{anchor}'
        t1_sig = merged[t1_col].notna() & (merged[t1_col] < LRT_ALPHA)
    elif criterion == 'sustained':
        col = f'sustained_sig_{anchor}'
        t1_sig = merged[col].fillna(False).astype(bool)
    else:
        raise ValueError(f"unknown criterion {criterion!r}")
    t2_sig = merged['p_pr_cond'].notna() & (merged['p_pr_cond'] < LRT_ALPHA)

    df = merged[merged['region_group'].notna() & (merged['region_group'] != '')]
    df = df[~df['region_group'].isin(EXCLUDE_REGION_GROUPS)]

    rows = []
    for region, sub in df.groupby('region_group'):
        s1 = t1_sig.loc[sub.index]
        s2 = t2_sig.loc[sub.index]
        n_both    = int((s1 & s2).sum())
        n_only_t1 = int((s1 & ~s2).sum())
        n_only_t2 = int((~s1 & s2).sum())
        n_neither = int((~s1 & ~s2).sum())
        n_total = n_both + n_only_t1 + n_only_t2 + n_neither
        rows.append({
            'region_group': region,
            'anchor': anchor,
            'criterion': criterion,
            'n_total': n_total,
            'n_both': n_both,
            'n_only_t1_time': n_only_t1,
            'n_only_t2_reward': n_only_t2,
            'n_neither': n_neither,
            'frac_both':    n_both / n_total    if n_total else np.nan,
            'frac_only_t1': n_only_t1 / n_total if n_total else np.nan,
            'frac_only_t2': n_only_t2 / n_total if n_total else np.nan,
            'frac_neither': n_neither / n_total if n_total else np.nan,
        })
    out = pd.DataFrame(rows).sort_values('n_total', ascending=False)
    return out


def plot_overlap(tabs: dict, out_path: Path) -> None:
    """Stacked bars per region showing only_t1 / both / only_t2 / neither fractions."""
    n_panels = len(tabs)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.6),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, (label, tab) in zip(axes, tabs.items()):
        if tab.empty:
            ax.set_title(f'{label}\n(no data)')
            continue
        regions = tab['region_group'].tolist()
        x = np.arange(len(regions))
        bottoms = np.zeros(len(regions))
        for col, color, name in [
            ('frac_only_t1',   'C0', 'only time (T1)'),
            ('frac_both',      'C2', 'both'),
            ('frac_only_t2',   'C3', 'only reward (T2)'),
            ('frac_neither',   '0.8', 'neither'),
        ]:
            vals = tab[col].values
            ax.bar(x, vals, bottom=bottoms, color=color, edgecolor='k',
                   linewidth=0.5, label=name)
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f'{r}\nn={n}' for r, n in
                            zip(regions, tab['n_total'])], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('fraction of units')
    axes[-1].legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.6, 1.0))
    fig.suptitle('Test 1 (time encoding) × Test 2 (reward history) cross-tab',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    merged = load_and_merge()
    print(f"[load] merged rows: {len(merged)} "
          f"(t1 only: {merged['fit_status_cue_off'].notna().sum() - merged['fit_status_t2'].notna().sum()}, "
          f"both fit: {(merged['fit_status_cue_off'].notna() & merged['fit_status_t2'].notna()).sum()})")

    tabs = {}
    for anchor in ('cue_off', 'cue_on'):
        for criterion in ('lrt', 'sustained'):
            tab = cross_tab_one(merged, anchor=anchor, criterion=criterion)
            tab.to_csv(OUT_DIR / f'cross_tab_{anchor}_{criterion}.csv', index=False)
            tabs[f'{anchor} ({criterion})'] = tab
            print(f"\n--- {anchor} / {criterion} ---")
            print(tab[['region_group', 'n_total', 'n_only_t1_time', 'n_both',
                       'n_only_t2_reward', 'n_neither']].to_string(index=False))

    # Wide summary file
    wide = pd.concat(
        [t.assign(anchor_criterion=k) for k, t in tabs.items()],
        ignore_index=True,
    )
    wide.to_csv(OUT_DIR / 'cross_tab_summary.csv', index=False)
    print(f"\nSaved → {OUT_DIR / 'cross_tab_summary.csv'}")

    plot_overlap(tabs, OUT_DIR / 'cross_tab_overlap.png')


if __name__ == '__main__':
    main()
