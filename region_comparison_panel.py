"""Region-comparison panel for the "where is the clock?" question.

Builds a single summary figure comparing two metrics across brain regions:
    1. Rescaling r — from ``data/rescaling/results_summary_all_sets.csv``
    2. Reward-history Cohen's d — from
       ``data/population_decoding/results_summary_all_sets.csv``

For each (cell-set → display region) the rescaling r is the best-anchor row
per group (max r over cue_on / cue_off / last_lick). The decoder Cohen's d is
the per-session distribution from the pooled decoder.

Usage:
    python region_comparison_panel.py
    python region_comparison_panel.py --min-units 15

Outputs land in ``data/region_comparison/``.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths as p

OUT_DIR = p.DATA_DIR / 'region_comparison'

RESCALING_CSV = p.DATA_DIR / 'rescaling' / 'results_summary_all_sets.csv'
DECODER_CSV   = p.DATA_DIR / 'population_decoding' / 'results_summary_all_sets.csv'

# Cell-set → display region. Order = column order in the figures.
# `rescale_set` is the cell-set label in the rescaling CSV; `decoder_set` is
# the label in the decoder CSV. They differ for CP (the rescaling pipeline
# uses `str_msn` for waveform-validated MSNs, while the decoder labels the
# same selection `msn`).
REGION_ORDER = [
    # (display,     rescale_set, decoder_set)
    ('CP (MSN)',    'str_msn',   'msn'),
    ('M1/M2 L5/6',  'mc_l5l6',   'mc_l5l6'),
    ('VAL',         'val',       'val'),
    ('PO',          'po',        'po'),
    ('VPM',         'vpm',       'vpm'),
    ('V1',          'v1',        'v1'),
    ('CA1',         'ca1',       'ca1'),
    ('HPF',         'hpf',       'hpf'),
]
DISPLAY_REGIONS = [r[0] for r in REGION_ORDER]
RESCALE_LOOKUP  = {r[1]: r[0] for r in REGION_ORDER}
DECODER_LOOKUP  = {r[2]: r[0] for r in REGION_ORDER}

REGION_COLORS = {
    'CP (MSN)':    '#2166AC',
    'M1/M2 L5/6':  '#D6604D',
    'VAL':         '#E08C24',
    'PO':          '#F4B860',
    'VPM':         '#F7D58A',
    'V1':          '#9467BD',
    'CA1':         '#8C564B',
    'HPF':         '#B7886B',
}

GROUP_ORDER = ['short_BG', 'long_BG']        # rescaling CSV uses this
DECODER_GROUP_ORDER = ['Short BG', 'Long BG']  # decoder CSV uses this


def load_rescaling(min_groups=None):
    df = pd.read_csv(RESCALING_CSV)
    df = df[df['cell_set'].isin(RESCALE_LOOKUP)].copy()
    df['region'] = df['cell_set'].map(RESCALE_LOOKUP)

    # Best anchor per (cell_set, group) by max r.
    best = df.sort_values('r', ascending=False).drop_duplicates(['cell_set', 'group'])
    best = best.sort_values(['cell_set', 'group']).reset_index(drop=True)
    return best


def load_decoder(min_units=None):
    df = pd.read_csv(DECODER_CSV)
    df = df[df['cell_set'].isin(DECODER_LOOKUP)].copy()
    df['region'] = df['cell_set'].map(DECODER_LOOKUP)

    if 'min_units' not in df.columns:
        df['min_units'] = 15  # legacy runs predate the sweep column

    if min_units is not None:
        df = df[df['min_units'] == min_units].copy()
    return df


def plot_rescaling_r(ax, rescale_df):
    regions = DISPLAY_REGIONS
    width   = 0.4

    for i, group in enumerate(GROUP_ORDER):
        sub  = rescale_df[rescale_df['group'] == group]
        vals = []
        cis_lo = []
        cis_hi = []
        for region in regions:
            row = sub[sub['region'] == region]
            if row.empty:
                vals.append(np.nan); cis_lo.append(np.nan); cis_hi.append(np.nan)
                continue
            row = row.iloc[0]
            vals.append(row['r'])
            # Use slope_se as a proxy uncertainty band around r (the
            # bootstrap slope CI lives on the slope, not on r directly, but
            # they track each other closely).
            se = row.get('slope_se', np.nan)
            cis_lo.append(row['r'] - se if pd.notna(se) else np.nan)
            cis_hi.append(row['r'] + se if pd.notna(se) else np.nan)

        x = np.arange(len(regions)) + (i - 0.5) * width
        bars = ax.bar(x, vals, width=width,
                      color=[REGION_COLORS[r] for r in regions],
                      edgecolor='black', linewidth=0.5,
                      alpha=0.6 if group == 'long_BG' else 1.0,
                      label=group.replace('_BG', ' BG'))
        # Error bars from slope SE (proxy for r uncertainty).
        err_lo = [v - lo if not (np.isnan(v) or np.isnan(lo)) else 0 for v, lo in zip(vals, cis_lo)]
        err_hi = [hi - v if not (np.isnan(v) or np.isnan(hi)) else 0 for v, hi in zip(vals, cis_hi)]
        ax.errorbar(x, vals, yerr=[err_lo, err_hi], fmt='none', ecolor='black',
                    elinewidth=0.7, capsize=2)

        # Annotate the anchor on top of each bar.
        for xi, region in zip(x, regions):
            row = sub[sub['region'] == region]
            if row.empty:
                continue
            anchor = row.iloc[0]['alignment']
            yval   = row.iloc[0]['r']
            ax.text(xi, max(yval, 0) + 0.02, anchor.replace('_', ' '),
                    ha='center', va='bottom', fontsize=6, rotation=90, color='dimgray')

    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(regions, rotation=30, ha='right')
    ax.set_ylabel('Rescaling r  (best-anchor Q3 vs Q4 peak correlation)')
    ax.set_title('Temporal rescaling per region')
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend(loc='upper right', frameon=False, fontsize=8)


def per_mouse_summary(decode_df):
    """Collapse per-session d to per-mouse mean.

    Returns a DataFrame with one row per (region, group, mouse).
    """
    df = decode_df.copy()
    df['mouse'] = df['session_id'].str.split('_').str[0]
    return (df.groupby(['region', 'group', 'mouse'])['pooled_history_d']
              .mean().reset_index()
              .rename(columns={'pooled_history_d': 'd_per_mouse'}))


def plot_cohens_d(ax, decode_df, group, min_units, *,
                  exclude_mouse=None, per_mouse=False, title_suffix=''):
    regions = DISPLAY_REGIONS
    df = decode_df.copy()
    if 'mouse' not in df.columns:
        df['mouse'] = df['session_id'].str.split('_').str[0]
    if exclude_mouse is not None:
        df = df[df['mouse'] != exclude_mouse]
    if per_mouse:
        df = (df.groupby(['region', 'group', 'mouse'])['pooled_history_d']
                .mean().reset_index()
                .rename(columns={'pooled_history_d': 'pooled_history_d'}))

    data    = []
    labels  = []
    for region in regions:
        sub = df[(df['region'] == region) & (df['group'] == group)]
        d   = sub['pooled_history_d'].dropna().values
        data.append(d)
        unit = 'mice' if per_mouse else 'sess'
        labels.append(f"{region}\n(n={len(d)} {unit})")

    positions = np.arange(len(regions))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black'))
    for patch, region in zip(bp['boxes'], regions):
        patch.set_facecolor(REGION_COLORS[region])
        patch.set_alpha(0.7)

    for x, d in zip(positions, data):
        if len(d) == 0:
            continue
        jitter = np.random.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(x + jitter, d, color='black', s=10, zorder=3, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ylabel = "Reward-history Cohen's d"
    ylabel += "  (per mouse)" if per_mouse else "  (per session)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{group} — MIN_UNITS = {min_units}{title_suffix}")


def build_panel(min_units=15):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rescale_df = load_rescaling()
    decode_df  = load_decoder(min_units=min_units)
    decode_df['mouse'] = decode_df['session_id'].str.split('_').str[0]

    print(f"\nRescaling rows available: {len(rescale_df)}")
    print(rescale_df[['cell_set', 'region', 'group', 'alignment', 'r', 'n_units']].to_string(index=False))
    print(f"\nDecoder rows at min_units={min_units}: {len(decode_df)}")
    print(decode_df.groupby(['region', 'group']).size().to_string())

    # Combined headline figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                              gridspec_kw={'width_ratios': [1.2, 1, 1]})
    plot_rescaling_r(axes[0], rescale_df)
    plot_cohens_d(axes[1], decode_df, 'Short BG', min_units)
    plot_cohens_d(axes[2], decode_df, 'Long BG',  min_units)
    fig.suptitle("Where is the clock? — rescaling and reward-history effect per region",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out_combined = OUT_DIR / f'panel_combined_min_units_{min_units}.png'
    fig.savefig(out_combined, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Standalone rescaling
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_rescaling_r(ax, rescale_df)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'panel_rescaling_r.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Standalone Cohen's d (short + long)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_cohens_d(axes[0], decode_df, 'Short BG', min_units)
    plot_cohens_d(axes[1], decode_df, 'Long BG',  min_units)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'panel_cohens_d_min_units_{min_units}.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Robustness 2x2: per-session vs per-mouse × all data vs LOO (drop RZ059)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    plot_cohens_d(axes[0, 0], decode_df, 'Short BG', min_units,
                  per_mouse=False, title_suffix=' — all data, per session')
    plot_cohens_d(axes[0, 1], decode_df, 'Long BG',  min_units,
                  per_mouse=False, title_suffix=' — all data, per session')
    plot_cohens_d(axes[0, 2], decode_df, 'Short BG', min_units,
                  per_mouse=True,  title_suffix=' — all data, per mouse')
    plot_cohens_d(axes[0, 3], decode_df, 'Long BG',  min_units,
                  per_mouse=True,  title_suffix=' — all data, per mouse')
    plot_cohens_d(axes[1, 0], decode_df, 'Short BG', min_units,
                  exclude_mouse='RZ059', per_mouse=False,
                  title_suffix=' — drop RZ059, per session')
    plot_cohens_d(axes[1, 1], decode_df, 'Long BG',  min_units,
                  exclude_mouse='RZ059', per_mouse=False,
                  title_suffix=' — drop RZ059, per session')
    plot_cohens_d(axes[1, 2], decode_df, 'Short BG', min_units,
                  exclude_mouse='RZ059', per_mouse=True,
                  title_suffix=' — drop RZ059, per mouse')
    plot_cohens_d(axes[1, 3], decode_df, 'Long BG',  min_units,
                  exclude_mouse='RZ059', per_mouse=True,
                  title_suffix=' — drop RZ059, per mouse')
    fig.suptitle("Robustness — per-session vs per-mouse, with and without RZ059",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'panel_robustness_min_units_{min_units}.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Robustness table
    pm = per_mouse_summary(decode_df)
    pm_no059 = pm[pm['mouse'] != 'RZ059']
    rob_rows = []
    for display, _, decoder_set in REGION_ORDER:
        for group in ['Short BG', 'Long BG']:
            sub_sess = decode_df[(decode_df['cell_set'] == decoder_set) & (decode_df['group'] == group)]
            sub_mouse = pm[(pm['region'] == display) & (pm['group'] == group)]
            sub_sess_no059 = sub_sess[sub_sess['mouse'] != 'RZ059']
            sub_mouse_no059 = pm_no059[(pm_no059['region'] == display) & (pm_no059['group'] == group)]
            rob_rows.append({
                'region':              display,
                'group':               group,
                'd_per_session_mean':  sub_sess['pooled_history_d'].mean(),
                'n_sessions':          len(sub_sess),
                'd_per_mouse_mean':    sub_mouse['d_per_mouse'].mean(),
                'n_mice':              len(sub_mouse),
                'd_per_session_mean_no_rz059': sub_sess_no059['pooled_history_d'].mean(),
                'n_sessions_no_rz059':         len(sub_sess_no059),
                'd_per_mouse_mean_no_rz059':   sub_mouse_no059['d_per_mouse'].mean(),
                'n_mice_no_rz059':             len(sub_mouse_no059),
                'mice':                ','.join(sorted(sub_mouse['mouse'].tolist())),
            })
    rob_table = pd.DataFrame(rob_rows)
    rob_csv = OUT_DIR / f'robustness_table_min_units_{min_units}.csv'
    rob_table.to_csv(rob_csv, index=False)
    print(f"\nRobustness table → {rob_csv}")
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    print(rob_table.to_string(index=False))

    # Tidy table
    table_rows = []
    for display, rescale_set, decoder_set in REGION_ORDER:
        for group in GROUP_ORDER:
            rsub = rescale_df[(rescale_df['cell_set'] == rescale_set) & (rescale_df['group'] == group)]
            decode_group = ('Short BG' if group == 'short_BG' else 'Long BG')
            dsub = decode_df[(decode_df['cell_set'] == decoder_set) & (decode_df['group'] == decode_group)]
            row = {
                'region':             display,
                'rescaling_set':      rescale_set,
                'decoder_set':        decoder_set,
                'group':              group,
                'rescaling_r':        rsub['r'].iloc[0]                if not rsub.empty else np.nan,
                'rescaling_anchor':   rsub['alignment'].iloc[0]        if not rsub.empty else '',
                'rescaling_n_units':  rsub['n_units'].iloc[0]          if not rsub.empty else np.nan,
                'cohens_d_mean':      dsub['pooled_history_d'].mean()  if not dsub.empty else np.nan,
                'cohens_d_median':    dsub['pooled_history_d'].median()if not dsub.empty else np.nan,
                'cohens_d_n_sess':    len(dsub),
                'min_units':          min_units,
            }
            table_rows.append(row)
    table = pd.DataFrame(table_rows)
    out_csv = OUT_DIR / f'region_comparison_table_min_units_{min_units}.csv'
    table.to_csv(out_csv, index=False)

    print(f"\n  Saved figures and table to {OUT_DIR}")
    for f in [out_combined, OUT_DIR / 'panel_rescaling_r.png',
              OUT_DIR / f'panel_cohens_d_min_units_{min_units}.png', out_csv]:
        print(f"    {f.name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Region-comparison panel (rescaling r + Cohen's d).")
    parser.add_argument('--min-units', type=int, default=15,
                        help="Decoder MIN_UNITS threshold for the figure. Default 15.")
    args = parser.parse_args()
    build_panel(min_units=args.min_units)
