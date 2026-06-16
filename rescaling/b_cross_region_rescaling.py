"""
Cross-region rescaling comparison — headline figure.

Loads per-set caches written by ``a_rescaling.py`` and produces
a single comparison figure across the curated region/cell-type panel:

    Striatum:    str_all (aggregate), str_msn, str_fsi
    Cortex:      mc_all (aggregate), mc_l5l6_rs, mc_fsi, v1_all (aggregate), v1_rs
    Thalamus:    thal (aggregate), val, po, vpm
    Hippocampus: ca1, hpf

Notes
-----
- A whole cell-set is dropped at load time if **either** behavioral group
  has fewer than ``MIN_MICE`` mice. The cross-group comparison would be
  broken anyway, so partial single-mouse columns aren't worth showing.
  In practice this removes ``po`` (1 long-BG mouse) and ``vpm`` (1 long-BG
  mouse + 2 short-BG mice but only 8 units there).
- Individual (cell_set, group, anchor) cells are flagged ``headline_valid
  = False`` when ``frac_rescaling`` is NaN, which happens when ≤10 units
  have ``tau_b > 0.05``. Such cells are blanked in the figure (their r
  is correlating anchor-locked peaks at t≈0, not a real rescaling signal)
  but they are still written to the CSV with the flag visible.

Usage:
    python b_cross_region_rescaling.py
"""
from __future__ import annotations

import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from a_rescaling import paths_for, DIR_BASE


# Display label → cell-set label → region class. Order = panel column order.
PANEL = [
    # (display,     set_label,    region_class)
    ('Str (all)',   'str_all',    'Striatum'),
    ('Str MSN',     'str_msn',    'Striatum'),
    ('Str FSI',     'str_fsi',    'Striatum'),
    ('MC (all)',    'mc_all',     'Cortex'),
    ('MC RS',       'mc_l5l6_rs', 'Cortex'),
    ('MC FSI',      'mc_fsi',     'Cortex'),
    ('V1 (all)',    'v1_all',     'Cortex'),
    ('V1 RS',       'v1_rs',      'Cortex'),
    ('Thal (all)',  'thal',       'Thalamus'),
    ('VAL',         'val',        'Thalamus'),
    ('PO',          'po',         'Thalamus'),
    ('VPM',         'vpm',        'Thalamus'),
    ('CA1',         'ca1',        'Hippocampus'),
    ('HPF',         'hpf',        'Hippocampus'),
]

SKIP_CELLS = {('vpm', 'short_BG')}   # explicit drop (only 8 units / 2 mice)
MIN_MICE   = 2                        # drop whole cell-set if EITHER group has fewer mice

GROUPS          = ['short_BG', 'long_BG']
ANCHORS         = ['last_lick', 'cue_on', 'cue_off']
BEHAVIOR_ANCHOR = {'short_BG': 'last_lick', 'long_BG': 'cue_on'}

REGION_COLORS = {
    'Striatum':    '#2166AC',
    'Cortex':      '#D6604D',
    'Thalamus':    '#E08C24',
    'Hippocampus': '#8C564B',
}

OUT_DIR = DIR_BASE / 'cross_region'


def load_results():
    rows = []
    for display, label, region_class in PANEL:
        _, _, _, cache_file = paths_for(label)
        if not cache_file.exists():
            print(f"  [skip] {label}: no cache at {cache_file}")
            continue
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        df = cache['results_df'].copy()
        if 'cell_set' not in df.columns:
            df['cell_set'] = label
        df['display']      = display
        df['region_class'] = region_class
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            "No caches found — run a_rescaling.py first."
        )
    combined = pd.concat(rows, ignore_index=True)

    # Drop whole cell-sets if any behavioral group has < MIN_MICE mice
    # OR if any (cell_set, group) is in SKIP_CELLS — the cross-group
    # comparison is broken anyway, so don't show partial columns.
    bad_sets = set()
    for cs, sub in combined.groupby('cell_set'):
        for g in GROUPS:
            grp_rows = sub[sub['group'] == g]
            n_mice_max = (grp_rows['n_mice'].fillna(0).max()
                          if len(grp_rows) else 0)
            if n_mice_max < MIN_MICE:
                bad_sets.add((cs, f'{g} has n_mice={int(n_mice_max)} (< {MIN_MICE})'))
                break
            if (cs, g) in SKIP_CELLS:
                bad_sets.add((cs, f'{g} in SKIP_CELLS'))
                break

    for cs, reason in sorted(bad_sets):
        print(f"  drop  {cs:<14}  ({reason})")

    dropped = {cs for cs, _ in bad_sets}
    out = combined[~combined['cell_set'].isin(dropped)].copy()

    # Flag individual (cell_set, group, anchor) rows where frac_rescaling is
    # NaN. This signals that ≤10 units had a non-edge Q3 peak, which means
    # the r value is dominated by anchor-locked transients (peak at t≈0)
    # rather than real rescaling. Figure blanks these; CSV keeps them.
    out['headline_valid'] = out['frac_rescaling'].notna()
    invalid = out[~out['headline_valid']][['cell_set', 'group', 'alignment',
                                            'n_units', 'r']]
    if len(invalid):
        print()
        print(f"  Cells flagged headline_valid=False (frac_rescaling NaN, "
              f"figure blanks them):")
        for _, row in invalid.iterrows():
            print(f"    {row['cell_set']:<14} {row['group']:<10} "
                  f"{row['alignment']:<10}  n={int(row['n_units']):<3}  "
                  f"r={row['r']:.3f}")
    return out


def _lookup_value(df, display, group, anchor, col):
    """Return df[col] for the (display, group, anchor) row, or NaN if missing.

    Also returns NaN when the row is flagged headline_valid=False, so panels
    blank out cells whose metrics are unreliable (frac_rescaling NaN).
    """
    hit = df[(df['display'] == display) &
             (df['group'] == group) &
             (df['alignment'] == anchor)]
    if not len(hit):
        return np.nan
    row = hit.iloc[0]
    if 'headline_valid' in row.index and not row['headline_valid']:
        return np.nan
    return row[col]


def figure_cross_region(df):
    """Anchor-matched cross-region rescaling figure.

    Every panel reports the three anchors separately and compares short vs
    long BG only *within* a single anchor — never short@last_lick vs
    long@cue_on. cue_on / cue_off / last_lick recruit different unit
    populations (cue- vs lick-responsive), so a cross-group comparison is
    only clean like-for-like. No anchor is privileged and no anchor is
    selected by max r.
    """
    panel_labels = [d for d, _, _ in PANEL if d in df['display'].unique()]
    panel_colors = {d: REGION_COLORS[r] for d, _, r in PANEL}
    x = np.arange(len(panel_labels))

    fig = plt.figure(figsize=(21, 18))
    gs  = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 1.1], hspace=0.42)

    # ── Panel A: r heatmap (region × anchor) per BG group ────────────────────
    gs_a = gs[0].subgridspec(1, 2, wspace=0.12)
    for col, group in enumerate(GROUPS):
        ax = fig.add_subplot(gs_a[0, col])
        mat = np.full((len(panel_labels), len(ANCHORS)), np.nan)
        for i, d in enumerate(panel_labels):
            for j, a in enumerate(ANCHORS):
                mat[i, j] = _lookup_value(df, d, group, a, 'r')
        im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-0.8, vmax=0.8)
        ax.set_xticks(range(len(ANCHORS)))
        ax.set_xticklabels([a.replace('_', ' ') for a in ANCHORS])
        ax.set_yticks(range(len(panel_labels)))
        ax.set_yticklabels(panel_labels)
        for i in range(len(panel_labels)):
            for j in range(len(ANCHORS)):
                v = mat[i, j]
                if not np.isnan(v):
                    txt_color = 'white' if abs(v) > 0.45 else 'black'
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=9, color=txt_color)
        ax.set_title(f'A. {group.replace("_", " ")}  —  Q2→Q3 r, '
                     'region × anchor', fontsize=11)
        if col == 1:
            cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label('Q2→Q3 r')

    # ── Panel B: r per region, short vs long — one sub-panel per anchor ──────
    gs_b = gs[1].subgridspec(1, len(ANCHORS), wspace=0.16)
    width = 0.38
    legend_b = [mpatches.Patch(facecolor='0.7', edgecolor='black',
                               label='short BG'),
                mpatches.Patch(facecolor='0.7', edgecolor='black', hatch='///',
                               label='long BG')]
    for j, anchor in enumerate(ANCHORS):
        ax = fig.add_subplot(gs_b[0, j])
        for k, group in enumerate(GROUPS):
            r_vals  = [_lookup_value(df, d, group, anchor, 'r')
                       for d in panel_labels]
            n_units = [_lookup_value(df, d, group, anchor, 'n_units')
                       for d in panel_labels]
            offset = (-width / 2) if k == 0 else (width / 2)
            hatch  = '' if k == 0 else '///'
            ax.bar(x + offset, r_vals, width,
                   color=[panel_colors[d] for d in panel_labels],
                   edgecolor='black', linewidth=1.0, alpha=0.85, hatch=hatch)
            for i, (v, n) in enumerate(zip(r_vals, n_units)):
                if not np.isnan(v):
                    ax.text(i + offset, v + 0.02, f'{int(n)}', ha='center',
                            va='bottom', fontsize=6, rotation=90)
        ax.axhline(0.5, color='gray', ls='--', alpha=0.4, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(panel_labels, rotation=35, ha='right', fontsize=8)
        ax.set_ylim([0, 1.05])
        if j == 0:
            ax.set_ylabel('Q2→Q3 peak-time r')
        ax.set_title(f'B{j + 1}. {anchor.replace("_", " ")}', fontsize=10)
        if j == len(ANCHORS) - 1:
            ax.legend(handles=legend_b, loc='upper right', fontsize=8)

    # ── Panel C: frac_rescaling / frac_fixed — one sub-panel per anchor ──────
    gs_c = gs[2].subgridspec(1, len(ANCHORS), wspace=0.16)
    bw = 0.18
    offsets = {('short_BG', 'rescale'): -1.5 * bw,
               ('short_BG', 'fixed'):   -0.5 * bw,
               ('long_BG',  'rescale'):  0.5 * bw,
               ('long_BG',  'fixed'):    1.5 * bw}
    colors_c = {('short_BG', 'rescale'): '#27ae60',
                ('short_BG', 'fixed'):   '#1abc9c',
                ('long_BG',  'rescale'): '#2980b9',
                ('long_BG',  'fixed'):   '#5dade2'}
    for j, anchor in enumerate(ANCHORS):
        ax = fig.add_subplot(gs_c[0, j])
        for group in GROUPS:
            for metric, col_name in [('rescale', 'frac_rescaling'),
                                     ('fixed',   'frac_fixed')]:
                vals = [_lookup_value(df, d, group, anchor, col_name)
                        for d in panel_labels]
                ax.bar(x + offsets[(group, metric)], vals, bw,
                       color=colors_c[(group, metric)],
                       edgecolor='white', linewidth=0.7,
                       label=(f'{group.replace("_", " ")} · {metric}'
                              if j == len(ANCHORS) - 1 else None))
        ax.set_xticks(x)
        ax.set_xticklabels(panel_labels, rotation=35, ha='right', fontsize=8)
        ax.set_ylim([0, 1.05])
        if j == 0:
            ax.set_ylabel('Fraction of units')
        ax.set_title(f'C{j + 1}. {anchor.replace("_", " ")}', fontsize=10)
        if j == len(ANCHORS) - 1:
            ax.legend(loc='upper right', fontsize=7, ncol=2)

    fig.suptitle(
        'Cross-region rescaling — Q2→Q3 peak-time correlation, all three '
        'anchors, each group reported at the same anchor\n'
        'A: r heatmap region × anchor   ·   B: r per anchor (short vs long)   '
        '·   C: rescaling vs fixed-latency fractions per anchor   '
        f'(n_mice < {MIN_MICE} cell-sets omitted; frac_rescaling-NaN cells blanked)',
        fontsize=12, fontweight='bold', y=0.999,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'cross_region_rescaling.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


def table_sample_sizes(df):
    """Companion CSV: all metrics + sample sizes for the rows that survived
    load_results() filtering. Keeps headline_valid=False rows so you can
    audit what was hidden from the figure."""
    cols = ['display', 'cell_set', 'region_class', 'group', 'alignment',
            'r', 'p_shuffle', 'slope', 'frac_rescaling', 'frac_fixed', 'r_abs',
            'n_units', 'n_mice', 'n_sessions', 'n_trials', 'headline_valid']
    out_df = (df[cols]
              .rename(columns={'region_class': 'region', 'alignment': 'anchor'})
              .sort_values(['region', 'cell_set', 'group', 'anchor'])
              .reset_index(drop=True))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'cross_region_table.csv'
    out_df.to_csv(out_path, index=False)
    print(f"  Saved table  → {out_path}")
    return out_df


if __name__ == '__main__':
    print("=" * 70)
    print("  Cross-region rescaling comparison")
    print("=" * 70)
    df = load_results()
    print(f"  Loaded {df['cell_set'].nunique()} cell sets, {len(df)} rows")
    figure_cross_region(df)
    table_sample_sizes(df)
    print("\nDone.")
