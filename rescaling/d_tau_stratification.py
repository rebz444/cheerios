"""
τ-stratification of the Q2→Q3 rescaling signal — edge-cell confound check.

Background
----------
The cross-region rescaling analysis (a_rescaling.py +
b_cross_region_rescaling.py) reports a Q2→Q3 peak-time correlation ``r`` per
region. That headline ``r`` can be inflated by units that fire only near the
interval edges — cue-locked onset transients or pre-motor / lick activity —
rather than genuinely tiling the interval. An edge unit's peak sits at τ≈0 or
τ≈1 in *both* quartiles simply because it is anchored to an event, so it
contributes a high-leverage point to the τ_Q2 vs τ_Q3 scatter without
reflecting real "stretch the code with the interval" timing.

This script does the first (fast, cache-only) confound check from
Open Questions.md → "Rescaling confounds": τ-stratified reporting.

What it does
------------
For each cell set × behavioral group × anchor it reads the cached normalized
peak times (``metrics['tau_a']`` / ``metrics['tau_b']`` in
data/rescaling/<set>/results_cache.pkl), bins units into 5 equal slices of the
normalized interval by τ_Q2 (0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8, 0.8–1.0), and
recomputes the rescaling metrics within each bin:

    r(τ_Q2, τ_Q3)   slope   n_units   frac_rescaling

If the rescaling signal is concentrated in the mid bins (0.2–0.8) the code is a
genuine interval-tiling signal. If it is concentrated in bin 0 / bin 4 — and
especially if r collapses once the edge bins are excluded — the headline r is
edge-cell contamination.

Caveat: r within a single 0.2-wide τ_Q2 bin is range-restricted and therefore
attenuated / noisy; slope is more stable but still restricted. The summary CSV
additionally reports r/slope over the *pooled* edge bins (0,4) vs mid bins
(1,2,3), which are wider and more reliable than any single bin.

Outputs (data/rescaling/tau_stratification/)
--------------------------------------------
    tau_stratification_table.csv     one row per cell_set × group × anchor ×
                                     τ-bin (plus an 'all' row) — the
                                     cross-region table.
    tau_stratification_summary.csv   one row per cell_set × group × anchor —
                                     edge-vs-mid digest answering "is the
                                     signal in mid-peak or edge-peak units?".
    tau_stratification_heatmap.png   region × τ-bin heatmap of r, faceted by
                                     anchor × group.
    tau_stratification_counts.png    region × τ-bin heatmap of the unit-count
                                     distribution (where the units live).
    tau_stratification_bin_distribution.png
                                     one facet per cell set — % of units per
                                     τ-bin, a line per anchor × group. A
                                     U-shaped curve is the edge-cell pile-up.
    tau_stratification_r_all_vs_mid.png
                                     r_all vs r_mid scatter per anchor. Points
                                     below the y=x diagonal lost correlation
                                     once edge units were dropped — the
                                     headline r was edge-cell-inflated.
    tau_stratification_scatter_{group}_{anchor}.png
                                     τ_Q2 vs τ_Q3 rescaling scatter, one facet
                                     per cell set — one figure for each of the
                                     6 group × anchor combinations. Edge-τ
                                     units in red; an all-unit OLS fit beside a
                                     mid-τ-only fit shows the edge corner
                                     clusters carrying the r.
    tau_stratification_dr_vs_edge.png
                                     Δr (r_all − r_mid) vs the edge-τ fraction,
                                     per anchor, against the uniform-edge
                                     baseline — nearly every cell set is both
                                     edge-over-represented and r-inflated.

Usage
-----
    python d_tau_stratification.py
"""
from __future__ import annotations

import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

from a_rescaling import DIR_BASE, paths_for
from b_cross_region_rescaling import (
    PANEL, GROUPS, ANCHORS, BEHAVIOR_ANCHOR, REGION_COLORS,
)


# ── Config ──────────────────────────────────────────────────────────────────

# τ_Q2 bin edges over the normalized interval [0, 1].
BIN_EDGES  = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
BIN_INNER  = BIN_EDGES[1:-1]                       # [.2,.4,.6,.8] for np.digitize
N_TAU_BINS = len(BIN_EDGES) - 1                    # 5
BIN_LABELS = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
EDGE_BINS  = (0, N_TAU_BINS - 1)                   # bin 0 + bin 4
MID_BINS   = tuple(range(1, N_TAU_BINS - 1))       # bins 1, 2, 3

# Minimum units in a stratum before r / slope are estimated. frac_rescaling is
# still reported for any non-empty stratum (n is always shown alongside).
MIN_BIN_N = 5

# frac_rescaling replicates a_rescaling.compute_metrics: units with
# τ_Q3 > TAU_FLOOR, scale factor τ_Q2/τ_Q3 within ±RESCALE_TOL of 1.0.
TAU_FLOOR = 0.05
RESCALE_TOL  = 0.2

OUT_DIR = DIR_BASE / 'tau_stratification'

# Per-anchor line colors and per-group line styles for the diagnostic plots.
ANCHOR_COLORS = {'last_lick': '#27ae60', 'cue_on': '#3498db', 'cue_off': '#e74c3c'}
GROUP_LS      = {'short_BG': '-', 'long_BG': '--'}


# ── Stratum metrics ─────────────────────────────────────────────────────────

def _safe_r(x, y):
    """Pearson r, or NaN if the stratum is too small / degenerate."""
    if len(x) < MIN_BIN_N or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _safe_slope(x, y):
    """OLS slope τ_Q3 ~ τ_Q2, or NaN if the stratum is too small / degenerate."""
    if len(x) < MIN_BIN_N or len(np.unique(x)) < 2:
        return np.nan
    return float(stats.linregress(x, y).slope)


def _frac_rescaling(tau_a, tau_b):
    """Fraction of units that hold their normalized peak position Q2→Q3
    (scale factor τ_Q2/τ_Q3 within ±RESCALE_TOL of 1.0). Matches
    a_rescaling.compute_metrics. Returns (frac, n_valid)."""
    valid = tau_b > TAU_FLOOR
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.nan, 0
    sf = tau_a[valid] / tau_b[valid]
    return float(np.mean(np.abs(sf - 1.0) < RESCALE_TOL)), n_valid


def stratum_stats(tau_a, tau_b):
    """All rescaling metrics for one stratum of units."""
    frac, n_valid = _frac_rescaling(tau_a, tau_b)
    return {
        'n_units':        len(tau_a),
        'n_valid':        n_valid,
        'r':              _safe_r(tau_a, tau_b),
        'slope':          _safe_slope(tau_a, tau_b),
        'frac_rescaling': frac,
    }


# ── Collect ─────────────────────────────────────────────────────────────────

def collect():
    """Walk the PANEL caches, returning (table_df, summary_df, raw_taus).

    table_df:   one row per cell_set × group × anchor × τ-bin (+ 'all').
    summary_df: one row per cell_set × group × anchor — edge-vs-mid digest.
    raw_taus:   {(display, group, anchor): (tau_a, tau_b)} — the raw normalized
                peak-time arrays, kept for the per-unit edge-scatter plots.
    """
    table_rows, summary_rows = [], []
    raw_taus = {}

    for display, label, region in PANEL:
        _, _, _, cache_file = paths_for(label)
        if not cache_file.exists():
            print(f"  [skip] {label}: no cache at {cache_file}")
            continue
        with open(cache_file, 'rb') as f:
            data_cache = pickle.load(f)['data_cache']

        for group in GROUPS:
            for anchor in ANCHORS:
                key = (group, f'{anchor}_time')
                if key not in data_cache:
                    continue
                metrics = data_cache[key]['metrics']
                tau_a = np.asarray(metrics.get('tau_a', []), dtype=float)
                tau_b = np.asarray(metrics.get('tau_b', []), dtype=float)
                if tau_a.size == 0:
                    continue
                raw_taus[(display, group, anchor)] = (tau_a, tau_b)
                n_total = tau_a.size
                bin_idx = np.digitize(tau_a, BIN_INNER)   # 0..4

                base = dict(display=display, cell_set=label, region=region,
                            group=group, anchor=anchor)

                # All-units row (reproduces the cached headline r / slope).
                all_st = stratum_stats(tau_a, tau_b)
                table_rows.append({
                    **base, 'tau_bin': 'all', 'tau_bin_idx': -1,
                    'tau_lo': 0.0, 'tau_hi': 1.0,
                    'frac_units_in_bin': 1.0, **all_st,
                })

                # Per τ-bin rows.
                for b in range(N_TAU_BINS):
                    sel = bin_idx == b
                    st = stratum_stats(tau_a[sel], tau_b[sel])
                    table_rows.append({
                        **base, 'tau_bin': BIN_LABELS[b], 'tau_bin_idx': b,
                        'tau_lo': BIN_EDGES[b], 'tau_hi': BIN_EDGES[b + 1],
                        'frac_units_in_bin': st['n_units'] / n_total, **st,
                    })

                # Edge-vs-mid digest (wider, more reliable than single bins).
                edge_sel = np.isin(bin_idx, EDGE_BINS)
                mid_sel  = np.isin(bin_idx, MID_BINS)
                edge_st = stratum_stats(tau_a[edge_sel], tau_b[edge_sel])
                mid_st  = stratum_stats(tau_a[mid_sel],  tau_b[mid_sel])
                summary_rows.append({
                    **base,
                    'n_total':            n_total,
                    'is_behavioral':      anchor == BEHAVIOR_ANCHOR[group],
                    'r_all':              all_st['r'],
                    'slope_all':          all_st['slope'],
                    'frac_rescaling_all': all_st['frac_rescaling'],
                    'n_edge':             edge_st['n_units'],
                    'frac_edge':          edge_st['n_units'] / n_total,
                    'r_edge':             edge_st['r'],
                    'slope_edge':         edge_st['slope'],
                    'frac_rescaling_edge': edge_st['frac_rescaling'],
                    'n_mid':              mid_st['n_units'],
                    'frac_mid':           mid_st['n_units'] / n_total,
                    'r_mid':              mid_st['r'],
                    'slope_mid':          mid_st['slope'],
                    'frac_rescaling_mid': mid_st['frac_rescaling'],
                })

    table_df = pd.DataFrame(table_rows).sort_values(
        ['region', 'cell_set', 'group', 'anchor', 'tau_bin_idx']
    ).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ['region', 'cell_set', 'group', 'anchor']
    ).reset_index(drop=True)
    return table_df, summary_df, raw_taus


# ── Heatmaps ────────────────────────────────────────────────────────────────

def _panel_matrix(table_df, displays, group, anchor, value_col):
    """(n_displays × N_TAU_BINS) value matrix + matching n_units matrix."""
    binned = table_df[table_df['tau_bin_idx'] >= 0]
    vals = np.full((len(displays), N_TAU_BINS), np.nan)
    ns   = np.zeros((len(displays), N_TAU_BINS), dtype=int)
    for i, disp in enumerate(displays):
        sub = binned[(binned['display'] == disp) &
                     (binned['group'] == group) &
                     (binned['anchor'] == anchor)]
        for _, row in sub.iterrows():
            j = int(row['tau_bin_idx'])
            vals[i, j] = row[value_col]
            ns[i, j]   = int(row['n_units'])
    return vals, ns


def _heatmap_figure(table_df, value_col, fname, suptitle, cmap_name,
                    vmin, vmax, fmt_value):
    """Generic anchor × group facet of region × τ-bin heatmaps."""
    displays = [d for d, _, _ in PANEL if d in table_df['display'].unique()]
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('0.85')

    fig, axes = plt.subplots(
        len(ANCHORS), len(GROUPS),
        figsize=(6.2 * len(GROUPS), 0.46 * len(displays) * len(ANCHORS) + 1.5),
        squeeze=False, constrained_layout=True,
    )

    im = None
    for ai, anchor in enumerate(ANCHORS):
        for gi, group in enumerate(GROUPS):
            ax = axes[ai][gi]
            vals, ns = _panel_matrix(table_df, displays, group, anchor, value_col)
            im = ax.imshow(np.ma.masked_invalid(vals), aspect='auto',
                           cmap=cmap, vmin=vmin, vmax=vmax)
            for i in range(len(displays)):
                for j in range(N_TAU_BINS):
                    v, n = vals[i, j], ns[i, j]
                    if np.isnan(v):
                        txt = f'n={n}' if n else ''
                        ax.text(j, i, txt, ha='center', va='center',
                                fontsize=6, color='0.4')
                    else:
                        rel = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        tc = 'white' if rel < 0.22 or rel > 0.78 else 'black'
                        ax.text(j, i, f'{fmt_value(v)}\nn={n}', ha='center',
                                va='center', fontsize=6, color=tc)
            ax.set_xticks(range(N_TAU_BINS))
            ax.set_xticklabels(BIN_LABELS, fontsize=7)
            ax.set_yticks(range(len(displays)))
            ax.set_yticklabels(displays, fontsize=8)
            anchor_tag = ' ★' if anchor == BEHAVIOR_ANCHOR[group] else ''
            ax.set_title(f'{group.replace("_", " ")}  ·  {anchor}{anchor_tag}',
                         fontsize=10)
            if ai == len(ANCHORS) - 1:
                ax.set_xlabel('τ_Q2 bin (normalized interval)', fontsize=8)

    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cb.set_label(value_col)
    fig.suptitle(suptitle + '   (★ = behavioral anchor for that group)',
                 fontsize=12, fontweight='bold')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / fname
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


def make_heatmaps(table_df):
    _heatmap_figure(
        table_df, value_col='r', fname='tau_stratification_heatmap.png',
        suptitle='τ-stratified Q2→Q3 peak-time correlation (region × τ-bin)',
        cmap_name='RdBu_r', vmin=-1.0, vmax=1.0,
        fmt_value=lambda v: f'{v:.2f}',
    )
    _heatmap_figure(
        table_df, value_col='frac_units_in_bin',
        fname='tau_stratification_counts.png',
        suptitle='Where the units live — unit-count distribution across τ-bins',
        cmap_name='magma', vmin=0.0, vmax=0.6,
        fmt_value=lambda v: f'{v * 100:.0f}%',
    )


# ── Diagnostic line / scatter plots ─────────────────────────────────────────

def make_bin_distribution_plot(table_df):
    """One facet per cell set: the % of units falling in each τ_Q2 bin, drawn
    as a line per anchor × group.

    A roughly flat line near the uniform 1/N_TAU_BINS level means units tile
    the interval evenly. A U-shape — bins 0 and 4 well above uniform, the
    middle below it — is the edge-cell pile-up this check is built to expose.
    """
    binned   = table_df[table_df['tau_bin_idx'] >= 0]
    displays = [d for d, _, _ in PANEL if d in binned['display'].unique()]
    if not displays:
        print("  [skip] bin-distribution plot: no data")
        return

    ncols = 4
    nrows = int(np.ceil(len(displays) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows),
                             squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)
    x           = np.arange(N_TAU_BINS)
    uniform_pct = 100.0 / N_TAU_BINS          # 20% for 5 bins

    for idx, disp in enumerate(displays):
        ax = axes[idx // ncols][idx % ncols]
        for b in EDGE_BINS:                    # shade the edge bins
            ax.axvspan(b - 0.5, b + 0.5, color='0.93', zorder=0)
        ax.axhline(uniform_pct, color='0.6', ls=':', lw=1, zorder=1)
        for anchor in ANCHORS:
            for group in GROUPS:
                sub = binned[(binned['display'] == disp) &
                             (binned['anchor'] == anchor) &
                             (binned['group'] == group)].sort_values('tau_bin_idx')
                if len(sub) < N_TAU_BINS:
                    continue
                y      = sub['frac_units_in_bin'].to_numpy() * 100.0
                is_beh = anchor == BEHAVIOR_ANCHOR[group]
                ax.plot(x, y, color=ANCHOR_COLORS[anchor], ls=GROUP_LS[group],
                        lw=2.6 if is_beh else 1.3,
                        marker='o' if is_beh else None, ms=4.5,
                        alpha=0.95 if is_beh else 0.65,
                        zorder=3 if is_beh else 2)
        ax.set_title(disp, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, fontsize=7, rotation=45, ha='right')
        ax.set_xlim(-0.5, N_TAU_BINS - 0.5)
        ax.set_ylim(bottom=0)
        if idx % ncols == 0:
            ax.set_ylabel('% of units in bin', fontsize=8)

    for idx in range(len(displays), nrows * ncols):     # blank unused axes
        axes[idx // ncols][idx % ncols].axis('off')

    handles = [Line2D([], [], color=ANCHOR_COLORS[a], lw=2.4, label=a)
               for a in ANCHORS]
    handles += [Line2D([], [], color='0.35', ls=GROUP_LS[g], lw=2,
                        label=g.replace('_', ' ')) for g in GROUPS]
    handles.append(Line2D([], [], color='0.6', ls=':', lw=1,
                          label=f'uniform ({uniform_pct:.0f}%)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle('τ_Q2 peak-time distribution across the normalized interval\n'
                 '(shaded = edge bins; bold line + dots = behavioral anchor; '
                 'U-shape = edge-cell pile-up)',
                 fontsize=12, fontweight='bold')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'tau_stratification_bin_distribution.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


def make_r_comparison_plot(summary_df):
    """r_all vs r_mid scatter, one panel per anchor.

    Each point is a (cell set × group). The dashed line is y = x: a point on
    it kept its full correlation after the edge bins were dropped. A point
    *below* it lost correlation once edge units were excluded — direct
    evidence the headline r was inflated by edge cells.
    """
    if summary_df.empty:
        print("  [skip] r_all-vs-r_mid plot: no data")
        return

    group_marker = {'short_BG': 'o', 'long_BG': '^'}
    lo, hi = -0.25, 1.0

    fig, axes = plt.subplots(1, len(ANCHORS),
                             figsize=(4.8 * len(ANCHORS), 5.2), squeeze=False,
                             constrained_layout=True)
    for ai, anchor in enumerate(ANCHORS):
        ax  = axes[0][ai]
        sub = summary_df[summary_df['anchor'] == anchor]

        ax.fill_between([lo, hi], [lo, hi], lo, color='#e74c3c', alpha=0.06,
                        zorder=0)
        ax.plot([lo, hi], [lo, hi], color='0.5', ls='--', lw=1.2, zorder=1)
        ax.text(0.97, 0.04, 'r drops without\nedge units',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7.5, color='#c0392b')

        drops = []
        for _, row in sub.iterrows():
            ra, rm = row['r_all'], row['r_mid']
            if pd.isna(ra) or pd.isna(rm):
                continue
            drops.append(ra - rm)
            is_beh = bool(row['is_behavioral'])
            ax.scatter(ra, rm, s=95 if is_beh else 48,
                       c=[REGION_COLORS[row['region']]],
                       marker=group_marker[row['group']],
                       edgecolors='black' if is_beh else 'white',
                       linewidths=1.4 if is_beh else 0.5,
                       alpha=0.9, zorder=3)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
        ax.set_xlabel('r_all  (all units)', fontsize=9)
        if ai == 0:
            ax.set_ylabel('r_mid  (mid-τ units, bins 0.2–0.8)', fontsize=9)
        n_below   = sum(d > 1e-9 for d in drops)
        mean_drop = float(np.mean(drops)) if drops else float('nan')
        ax.set_title(f'{anchor}\n{n_below}/{len(drops)} cells below diagonal'
                     f'   ·   mean Δr = {mean_drop:.2f}', fontsize=10)

    handles = [Line2D([], [], marker='o', color='0.35', ls='', label='short BG'),
               Line2D([], [], marker='^', color='0.35', ls='', label='long BG'),
               Line2D([], [], marker='s', color='black', ls='', mfc='none',
                      label='behavioral anchor')]
    handles += [mpatches.Patch(color=REGION_COLORS[r], label=r)
                for r in ['Striatum', 'Cortex', 'Thalamus', 'Hippocampus']]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Does the rescaling r survive dropping the edge-τ units?\n'
                 'r_all vs r_mid — points below the diagonal = headline r '
                 'inflated by edge cells',
                 fontsize=12, fontweight='bold')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'tau_stratification_r_all_vs_mid.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


def _fmt(x):
    """Format a metric for a plot title, 'nan' for missing."""
    return 'nan' if pd.isna(x) else f'{x:.2f}'


def make_edge_scatter_plots(raw_taus):
    """τ_Q2 vs τ_Q3 rescaling scatter per cell set, edge-τ units highlighted.

    Edge-τ units (τ_Q2 in bin 0 or 4) are drawn in red, mid-τ units in blue.
    Two OLS fits are overlaid: grey-dashed over *all* units, blue over mid-τ
    units only. When the grey fit hugs the identity line but the blue fit is
    much flatter, the red corner clusters are what carry the headline r — the
    rescaling correlation is edge-cell-driven.

    One figure per (group, anchor) — all three anchors, not just the
    behavioral one — each faceted by cell set.
    """
    for group in GROUPS:
        for anchor in ANCHORS:
            displays = [d for d, _, _ in PANEL
                        if len(raw_taus.get((d, group, anchor), ([],))[0])]
            if not displays:
                print(f"  [skip] edge scatter ({group}, {anchor}): no data")
                continue

            ncols = 4
            nrows = int(np.ceil(len(displays) / ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(3.6 * ncols, 3.6 * nrows),
                                     squeeze=False, constrained_layout=True)
            xs = np.array([0.0, 1.0])

            for idx, disp in enumerate(displays):
                ax = axes[idx // ncols][idx % ncols]
                tau_a, tau_b = raw_taus[(disp, group, anchor)]
                edge_sel = np.isin(np.digitize(tau_a, BIN_INNER), EDGE_BINS)
                mid_sel  = ~edge_sel

                ax.axvspan(0.0, BIN_EDGES[1],  color='0.93', zorder=0)
                ax.axvspan(BIN_EDGES[-2], 1.0, color='0.93', zorder=0)
                ax.plot([0, 1], [0, 1], color='0.55', ls=':', lw=1, zorder=1)

                ax.scatter(tau_a[mid_sel], tau_b[mid_sel], s=11, c='#2c7fb8',
                           alpha=0.5, edgecolors='none', zorder=2)
                ax.scatter(tau_a[edge_sel], tau_b[edge_sel], s=11, c='#e34a33',
                           alpha=0.55, edgecolors='none', zorder=2)

                if len(tau_a) >= MIN_BIN_N and len(np.unique(tau_a)) > 1:
                    fit = stats.linregress(tau_a, tau_b)
                    ax.plot(xs, fit.intercept + fit.slope * xs, color='0.35',
                            ls='--', lw=1.9, zorder=3)
                if mid_sel.sum() >= MIN_BIN_N and len(np.unique(tau_a[mid_sel])) > 1:
                    fitm = stats.linregress(tau_a[mid_sel], tau_b[mid_sel])
                    ax.plot(xs, fitm.intercept + fitm.slope * xs,
                            color='#2c7fb8', ls='-', lw=1.9, zorder=3)

                r_all = _safe_r(tau_a, tau_b)
                r_mid = _safe_r(tau_a[mid_sel], tau_b[mid_sel])
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_aspect('equal')
                ax.set_title(f'{disp}   r_all={_fmt(r_all)}  r_mid={_fmt(r_mid)}',
                             fontsize=8.5)
                if idx // ncols == nrows - 1:
                    ax.set_xlabel('τ_Q2', fontsize=8)
                if idx % ncols == 0:
                    ax.set_ylabel('τ_Q3', fontsize=8)

            for idx in range(len(displays), nrows * ncols):
                axes[idx // ncols][idx % ncols].axis('off')

            handles = [
                Line2D([], [], marker='o', ls='', color='#e34a33',
                       label='edge-τ unit (bin 0 or 4)'),
                Line2D([], [], marker='o', ls='', color='#2c7fb8',
                       label='mid-τ unit (bins 1–3)'),
                Line2D([], [], color='0.35', ls='--', lw=1.9,
                       label='OLS fit — all units'),
                Line2D([], [], color='#2c7fb8', ls='-', lw=1.9,
                       label='OLS fit — mid-τ only'),
                Line2D([], [], color='0.55', ls=':', lw=1,
                       label='identity (τ_Q2 = τ_Q3)'),
            ]
            fig.legend(handles=handles, loc='lower center', ncol=len(handles),
                       fontsize=8, bbox_to_anchor=(0.5, -0.02))
            beh = ('  ★ behavioral anchor'
                   if anchor == BEHAVIOR_ANCHOR[group] else '')
            fig.suptitle(f'τ_Q2 vs τ_Q3 rescaling scatter — '
                         f'{group.replace("_", " ")} at {anchor}{beh}\n'
                         '(edge-τ units in red; a flat blue fit beside a '
                         'diagonal-hugging grey fit = edge units carry r)',
                         fontsize=12, fontweight='bold')

            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = (OUT_DIR /
                        f'tau_stratification_scatter_{group}_{anchor}.png')
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved figure → {out_path}")


def make_dr_vs_edge_plot(summary_df):
    """Δr (= r_all − r_mid) vs the fraction of edge-τ units, one panel per
    anchor.

    Each point is a (cell set × group). Two reference lines split the panel:
    the vertical line at EDGE_UNIFORM (= len(EDGE_BINS)/N_TAU_BINS, the edge
    fraction expected if peaks tiled the interval uniformly) and Δr = 0. The
    shaded upper-right quadrant is the confounded regime — more edge units
    than uniform AND r that drops when they are removed.

    Note: frac_edge is saturated high for nearly every cell set, so there is
    no spread to support a frac_edge→Δr *trend*. The demonstration is the
    absolute position: almost every point sits right of the uniform line and
    above Δr = 0, i.e. edge dominance + r inflation are near-universal.
    """
    if summary_df.empty:
        print("  [skip] Δr-vs-frac_edge plot: no data")
        return

    edge_uniform = len(EDGE_BINS) / N_TAU_BINS       # 2/5 = 0.40 expected
    group_marker = {'short_BG': 'o', 'long_BG': '^'}
    ylo, yhi = -0.25, 0.95
    df = summary_df.copy()
    df['dr'] = df['r_all'] - df['r_mid']

    fig, axes = plt.subplots(1, len(ANCHORS),
                             figsize=(4.8 * len(ANCHORS), 5.0), squeeze=False,
                             constrained_layout=True)
    for ai, anchor in enumerate(ANCHORS):
        ax  = axes[0][ai]
        sub = df[(df['anchor'] == anchor) & df['dr'].notna()
                 & df['frac_edge'].notna()]

        # Confounded regime: more edge units than uniform AND r inflated.
        ax.fill_between([edge_uniform, 1.0], 0, yhi, color='#e74c3c',
                        alpha=0.06, zorder=0)
        ax.axhline(0, color='0.55', lw=1, zorder=1)
        ax.axvline(edge_uniform, color='0.55', lw=1, ls=':', zorder=1)
        ax.text(edge_uniform - 0.015, yhi - 0.03, f'uniform = {edge_uniform:.2f}',
                rotation=90, fontsize=7, color='0.4', va='top', ha='right')

        for _, row in sub.iterrows():
            is_beh = bool(row['is_behavioral'])
            ax.scatter(row['frac_edge'], row['dr'], s=95 if is_beh else 48,
                       c=[REGION_COLORS[row['region']]],
                       marker=group_marker[row['group']],
                       edgecolors='black' if is_beh else 'white',
                       linewidths=1.4 if is_beh else 0.5, alpha=0.9, zorder=3)

        n_conf   = int(((sub['frac_edge'] > edge_uniform) & (sub['dr'] > 0)).sum())
        med_edge = sub['frac_edge'].median() if len(sub) else float('nan')
        ax.set_xlim(0, 1); ax.set_ylim(ylo, yhi)
        ax.set_xlabel('fraction of edge-τ units', fontsize=9)
        if ai == 0:
            ax.set_ylabel('Δr  =  r_all − r_mid', fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_title(f'{anchor}\nmedian edge fraction = {_fmt(med_edge)}  ·  '
                     f'{n_conf}/{len(sub)} cells in confound zone', fontsize=9.5)

    handles = [Line2D([], [], marker='o', color='0.35', ls='', label='short BG'),
               Line2D([], [], marker='^', color='0.35', ls='', label='long BG'),
               Line2D([], [], marker='s', color='black', ls='', mfc='none',
                      label='behavioral anchor')]
    handles += [mpatches.Patch(color=REGION_COLORS[r], label=r)
                for r in ['Striatum', 'Cortex', 'Thalamus', 'Hippocampus']]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Edge units are over-represented and inflate r almost '
                 'everywhere\nΔr (drop in r when edge-τ units are removed) vs '
                 'the edge-τ fraction  ·  shaded = confound zone '
                 '(edge-over-represented AND r inflated)',
                 fontsize=12, fontweight='bold')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'tau_stratification_dr_vs_edge.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


# ── Console digest ──────────────────────────────────────────────────────────

def print_digest(summary_df):
    """Edge-vs-mid digest — all three anchors, both groups at each anchor."""
    print()
    print("=" * 80)
    print("  Edge vs mid concentration — all anchors, group-matched")
    print("  (frac_edge high + r_mid weak  ->  headline r is edge-cell driven)")
    print("=" * 80)
    hdr = (f"  {'cell set':<13}{'group':<10}{'n':>6}{'%edge':>7}"
           f"{'r_all':>8}{'r_edge':>8}{'r_mid':>8}")

    def fmt(x):
        return ' nan' if pd.isna(x) else f'{x:.2f}'

    for anchor in ['cue_on', 'cue_off', 'last_lick']:
        sub = summary_df[summary_df['anchor'] == anchor].sort_values(
            ['region', 'cell_set', 'group'])
        print(f"\n--- {anchor} ---")
        print(hdr)
        for _, row in sub.iterrows():
            print(f"  {row['cell_set']:<13}{row['group']:<10}"
                  f"{int(row['n_total']):>6}{row['frac_edge'] * 100:>6.0f}%"
                  f"{fmt(row['r_all']):>8}{fmt(row['r_edge']):>8}"
                  f"{fmt(row['r_mid']):>8}")
    print()


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 78)
    print("  τ-stratification of the Q2→Q3 rescaling signal")
    print("=" * 78)

    table_df, summary_df, raw_taus = collect()
    n_sets = table_df['cell_set'].nunique()
    print(f"  Collected {n_sets} cell sets, "
          f"{len(summary_df)} cell_set × group × anchor combinations")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path   = OUT_DIR / 'tau_stratification_table.csv'
    summary_path = OUT_DIR / 'tau_stratification_summary.csv'
    table_df.to_csv(table_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved table   → {table_path}")
    print(f"  Saved summary → {summary_path}")

    make_heatmaps(table_df)
    make_bin_distribution_plot(table_df)
    make_r_comparison_plot(summary_df)
    make_edge_scatter_plots(raw_taus)
    make_dr_vs_edge_plot(summary_df)
    print_digest(summary_df)
    print("Done.")
