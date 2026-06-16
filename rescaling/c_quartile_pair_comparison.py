"""
Quartile-pair comparison for the cross-region rescaling analysis.

The headline rescaling r (Section 1 of analysis_results.md) is computed on
Q3 -> Q4. Q4 is the *top* wait-time quartile and is open-ended: pool_sessions
applies no outlier trimming — the only trial exclusions are miss trials (no
decision within the 60 s cutoff) and the >=20-trials-per-session gate, then
pd.qcut splits wait_length into 4 quantiles. Q4 therefore spans the 75th
percentile up to ~60 s and can carry a fat right tail of long / disengaged
trials, whereas Q2 and Q3 are bounded between the 25th and 75th percentiles.

This script recomputes the normalized peak-time rescaling metrics for every
adjacent quartile pair — Q1->Q2, Q2->Q3, Q3->Q4 — so the Q3->Q4 headline can
be checked against the tail-free middle pairs. If Q2->Q3 holds up, the
headline is not a Q4-outlier artifact; if Q3->Q4 is markedly stronger only
because of Q4, that shows up here.

All metrics come from the cached q_norm_spikes (per-trial normalized spike
times) via a_rescaling.compute_pair_full_metrics — no regenerate.
Q3->Q4 reproduces the cached metrics['r'] exactly.

Caveat: Q2 contains shorter waits and therefore fewer spikes per trial than
Q3/Q4, so its peak-time estimates are intrinsically noisier — Q3->Q4 was the
original choice because the long quartiles have the most spikes. The two
effects (Q4 tail vs Q2 sparsity) trade off; this table exposes both.

Anchors are reported group-matched: each group is shown at the SAME anchor
(short and long both at cue_on, both at cue_off, both at last_lick), never
short@last_lick vs long@cue_on. Mixing anchors across groups would conflate
the group difference with an anchor difference — cue_on, cue_off and last_lick
recruit different unit populations (cue-responsive vs lick-responsive), so a
cross-group comparison is only clean within a single anchor.

Outputs (data/rescaling/quartile_pair_comparison/)
    quartile_pair_table.csv          one row per cell_set x group x anchor x pair
    quartile_pair_comparison.png     r per quartile pair, per region — all three
                                     anchors, each group at the same anchor

Usage
    python rescaling/c_quartile_pair_comparison.py
"""
from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from a_rescaling import DIR_BASE, paths_for, compute_pair_full_metrics
from b_cross_region_rescaling import PANEL, GROUPS, ANCHORS


PAIRS = [('Q1', 'Q2'), ('Q2', 'Q3'), ('Q3', 'Q4')]
PAIR_COLORS = {'Q1→Q2': '#9ecae1', 'Q2→Q3': '#4292c6', 'Q3→Q4': '#084594'}
ANCHOR_ORDER = ['cue_on', 'cue_off', 'last_lick']

OUT_DIR = DIR_BASE / 'quartile_pair_comparison'


def collect():
    """Compute r / slope for every adjacent quartile pair across the panel."""
    rows = []
    for display, label, region in PANEL:
        _, _, _, cache_file = paths_for(label)
        if not cache_file.exists():
            print(f"  [skip] {label}: no cache")
            continue
        with open(cache_file, 'rb') as f:
            data_cache = pickle.load(f)['data_cache']

        for group in GROUPS:
            for anchor in ANCHORS:
                key = (group, f'{anchor}_time')
                if key not in data_cache:
                    continue
                entry      = data_cache[key]
                q_norm     = entry['q_norm_spikes']
                all_spikes = entry['all_spikes']
                q_med      = entry['q_medians']
                for qa, qb in PAIRS:
                    m = compute_pair_full_metrics(qa, qb, q_norm, all_spikes)
                    if m is None:
                        continue
                    rows.append({
                        'display': display, 'cell_set': label, 'region': region,
                        'group': group, 'anchor': anchor,
                        'pair': f'{qa}→{qb}',
                        'r': m['r'], 'p_shuffle': m['p_shuffle'],
                        'slope': m['slope'], 'slope_se': m['slope_se'],
                        'n_units': m['n_units'],
                        'med_qa': float(q_med.get(qa, np.nan)),
                        'med_qb': float(q_med.get(qb, np.nan)),
                        'dur_ratio': (float(q_med[qb] / q_med[qa])
                                      if qa in q_med.index and qb in q_med.index
                                      and q_med[qa] > 0 else np.nan),
                    })
        print(f"  done {label}")
    return pd.DataFrame(rows).sort_values(
        ['region', 'cell_set', 'group', 'anchor', 'pair']
    ).reset_index(drop=True)


def figure(df):
    """r per adjacent quartile pair, per region — all three anchors, each
    group reported at the same anchor (rows = anchor, columns = group)."""
    displays = [d for d, _, _ in PANEL if d in df['display'].unique()]
    pairs = list(PAIR_COLORS)
    width = 0.26

    fig, axes = plt.subplots(len(ANCHOR_ORDER), len(GROUPS),
                             figsize=(16, 13), squeeze=False)
    for ai, anchor in enumerate(ANCHOR_ORDER):
        for gi, group in enumerate(GROUPS):
            ax = axes[ai][gi]
            sub = df[(df['anchor'] == anchor) & (df['group'] == group)]
            x = np.arange(len(displays))
            for k, pair in enumerate(pairs):
                vals, ns = [], []
                for d in displays:
                    hit = sub[(sub['display'] == d) & (sub['pair'] == pair)]
                    vals.append(hit['r'].iloc[0] if len(hit) else np.nan)
                    ns.append(int(hit['n_units'].iloc[0]) if len(hit) else 0)
                off = (k - 1) * width
                ax.bar(x + off, vals, width, color=PAIR_COLORS[pair],
                       edgecolor='black', linewidth=0.6, label=pair)
                for xi, (v, n) in enumerate(zip(vals, ns)):
                    if not np.isnan(v):
                        ax.text(xi + off, v + 0.015, str(n), ha='center',
                                va='bottom', fontsize=5, rotation=90,
                                color='0.4')
            ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(displays, rotation=35, ha='right', fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('normalized peak-time r')
            ax.set_title(f'{anchor}  ·  {group.replace("_", " ")}', fontsize=10)
            if ai == 0 and gi == 1:
                ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Rescaling r by adjacent quartile pair — all three anchors, '
                 'each group reported at the same anchor\n'
                 'Q3→Q4 = current headline pair · Q2→Q3 / Q1→Q2 = tail-free '
                 'middle pairs',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / 'quartile_pair_comparison.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {path}")


def print_digest(df):
    """r for each adjacent pair, organized by anchor with both groups at the
    same anchor — no cross-group anchor mixing."""
    print()
    print("=" * 96)
    print("  Rescaling r by quartile pair — all three anchors, group-matched")
    print("  (each group reported at the SAME anchor; Q3→Q4 = current headline)")
    print("=" * 96)
    for anchor in ANCHOR_ORDER:
        sub = df[df['anchor'] == anchor]
        piv = sub.pivot_table(index=['region', 'cell_set', 'group'],
                              columns='pair', values='r')
        n_better = int((piv['Q2→Q3'] > piv['Q3→Q4']).sum())
        print(f"\n--- {anchor}   (Q2→Q3 > Q3→Q4 in {n_better}/{len(piv)} cells)"
              f" ---")
        print(f"  {'cell set':<13}{'group':<10}{'Q1→Q2':>8}{'Q2→Q3':>8}"
              f"{'Q3→Q4':>8}{'Δ(Q2Q3−Q3Q4)':>15}{'n':>6}")
        for (region, cell_set, group), row in piv.iterrows():
            nrow = sub[(sub['cell_set'] == cell_set) & (sub['group'] == group)]
            n = int(nrow['n_units'].max())
            delta = row['Q2→Q3'] - row['Q3→Q4']
            print(f"  {cell_set:<13}{group:<10}{row['Q1→Q2']:>8.2f}"
                  f"{row['Q2→Q3']:>8.2f}{row['Q3→Q4']:>8.2f}"
                  f"{delta:>+15.2f}{n:>6}")
        q23, q34 = piv['Q2→Q3'].median(), piv['Q3→Q4'].median()
        print(f"  {'median':<23}{'':>8}{q23:>8.2f}{q34:>8.2f}")
    print()


if __name__ == '__main__':
    print("=" * 94)
    print("  Quartile-pair comparison — Q1→Q2 / Q2→Q3 / Q3→Q4 rescaling r")
    print("=" * 94)
    df = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / 'quartile_pair_table.csv'
    df.to_csv(table_path, index=False)
    print(f"\n  {len(df)} rows  →  {table_path}")
    figure(df)
    print_digest(df)
    print("Done.")
