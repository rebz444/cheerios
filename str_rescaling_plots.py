"""
DMS rescaling plots.

Loads the results cache saved by str_rescaling_analysis_fixed.py and
regenerates all figures. Edit plotting functions here and re-run without
re-running the slow analysis step.

Usage:
    python str_rescaling_plots.py
"""

import ast
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd

from str_rescaling_analysis_fixed import (
    build_population_matrix, compute_pair_r,
    QUARTILE_LABELS, N_BINS_NORM, N_SHUFFLE, TIME_STEP, SIGMA,
    T_MAX_SHORT, T_MAX_LONG, MIN_TRIALS,
    DIR_HEATMAP, DIR_ENHANCED, DIR_COMMITTEE,
    COLOR_SHORT, COLOR_LONG, COLOR_LAST_LICK, COLOR_CUE_ON, COLOR_CUE_OFF,
    CACHE_FILE,
)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _get_out_dir(label):
    if '__' in label:
        sub = 'per_session'
    elif label == 'all':
        sub = 'all_mice'
    else:
        sub = 'per_group'
    d = DIR_HEATMAP / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_sample_size_box(ax, n_mice, n_sessions, n_units, n_trials, loc='upper left'):
    text = f"n = {n_mice} mice\n{n_sessions} sessions\n{n_units} units\n{n_trials:,} trials"
    props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
    xy = {'upper left': (0.02, 0.98, 'left', 'top'),
          'upper right': (0.98, 0.98, 'right', 'top'),
          'lower left': (0.02, 0.02, 'left', 'bottom')}
    x, y, ha, va = xy.get(loc, xy['upper left'])
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9,
            verticalalignment=va, horizontalalignment=ha, bbox=props,
            family='monospace')


# ── Per-condition figures ─────────────────────────────────────────────────────

def render_rescaling(all_spikes, q_spikes, q_norm_spikes, q_medians,
                     t_start_col, label, quartiles, save, t_max=T_MAX_SHORT):
    """Heatmap figure: absolute-time (top row) and normalized-time (bottom row)."""
    sort_matrix, sort_uids, _ = build_population_matrix(
        q_spikes['Q3'], list(all_spikes.keys()),
        0.0, t_max, TIME_STEP, SIGMA
    )
    if sort_matrix is None:
        print("  Could not build sort matrix — skipping heatmap.")
        return
    print(f"  [{'+'.join(quartiles)}] units in heatmap: {sort_matrix.shape[0]}")

    q_abs  = {}
    q_norm = {}
    for q in quartiles:
        mat, _, bc = build_population_matrix(
            q_spikes[q], list(all_spikes.keys()),
            0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids
        )
        q_abs[q] = (mat, bc)
        mat, _, bc = build_population_matrix(
            q_norm_spikes[q], list(all_spikes.keys()),
            0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
        )
        q_norm[q] = (mat, bc)

    n_cols = len(quartiles)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 5.5, 10), squeeze=False)
    fig.suptitle(
        f'{label}  |  align: {t_start_col} → decision  '
        f'[{"+".join(quartiles)}]  (n={sort_matrix.shape[0]} MSN units)',
        fontsize=13, y=0.93
    )

    for col, q in enumerate(quartiles):
        med_t = q_medians.get(q, np.nan)

        ax = axes[0, col]
        mat, bc = q_abs[q]
        if mat is not None and mat.size > 0:
            vmax = np.percentile(np.abs(mat), 95)
            ax.imshow(mat, aspect='auto', cmap='viridis',
                      vmin=-vmax, vmax=vmax, origin='lower',
                      extent=[bc[0], bc[-1], 0, mat.shape[0]])
            if pd.notna(med_t):
                ax.axvline(med_t, color='white', lw=1.5, ls='--', alpha=0.8)
        ax.set_title(f'{q}  (median wait = {med_t:.2f}s)', fontsize=10)
        ax.set_xlabel(f'Time from {t_start_col.replace("_time", "")} (s)')
        if col == 0:
            ax.set_ylabel('Unit (sorted by Q3 peak time)')

        ax = axes[1, col]
        mat, bc = q_norm[q]
        if mat is not None and mat.size > 0:
            vmax = np.percentile(np.abs(mat), 95)
            ax.imshow(mat, aspect='auto', cmap='viridis',
                      vmin=-vmax, vmax=vmax, origin='lower',
                      extent=[0, 1, 0, mat.shape[0]])
            ax.axvline(1.0, color='white', lw=1.5, ls='--', alpha=0.8)
        ax.set_xlabel('Normalised time (0=start, 1=decision)')
        if col == 0:
            ax.set_ylabel('Unit (sorted by peak time)')

    plt.tight_layout()
    if save:
        out_dir  = _get_out_dir(label)
        qs_suffix = '' if quartiles == QUARTILE_LABELS else f'__{"+".join(quartiles)}'
        fname    = out_dir / f'rescaling__{label}__{t_start_col}{qs_suffix}.png'
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
    else:
        plt.show()


def plot_rescaling_summary(metrics, q_medians, label, t_start_col,
                           exclude_impulsive, save=True):
    """2×2 enhanced summary: scatter, scale factor dist., fraction bar, stats table."""
    if metrics is None:
        return

    imp_tag  = 'excl_impulsive' if exclude_impulsive else 'all_trials'
    tau_q3   = metrics['tau_q3']
    tau_q4   = metrics['tau_q4']

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(
        f'{label}  |  {t_start_col}  |  {imp_tag}\n'
        f'Q3 median: {q_medians.get("Q3", np.nan):.2f}s  →  '
        f'Q4 median: {q_medians.get("Q4", np.nan):.2f}s  '
        f'(n={metrics["n_units"]} units)',
        fontsize=12, y=0.93
    )

    # Panel A: correlation scatter
    ax = axes[0, 0]
    ax.scatter(tau_q3, tau_q4, s=20, alpha=0.6, edgecolors='none', c='steelblue')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect rescaling (slope=1)')
    x_fit = np.linspace(0, 1, 100)
    ax.plot(x_fit, metrics['slope'] * x_fit + metrics['intercept'], 'r-', lw=2,
            label=f'Fit: slope={metrics["slope"]:.2f}±{metrics["slope_se"]:.2f}')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel('Normalized peak time (Q3)')
    ax.set_ylabel('Normalized peak time (Q4)')
    ax.set_title(f'A. Peak time correlation\n'
                 f'r = {metrics["r"]:.3f}, '
                 f'p < {max(metrics["p_shuffle"], 0.001):.3f}')
    ax.legend(loc='lower right', fontsize=9)

    # Panel B: scale factor distribution
    ax = axes[0, 1]
    valid_mask = tau_q4 > 0.05
    if valid_mask.sum() > 5:
        sf = np.clip(tau_q3[valid_mask] / tau_q4[valid_mask], 0, 3)
        ax.hist(sf, bins=30, density=True, alpha=0.7,
                color='steelblue', edgecolor='white')
        ax.axvline(1.0, color='k', ls='--', lw=2, label='Expected (=1.0)')
        ax.axvline(metrics['scale_factor_median'], color='red', lw=2,
                   label=f'Observed: {metrics["scale_factor_median"]:.2f}')
        ax.set_xlabel('Scale factor (τ_Q3 / τ_Q4)')
        ax.set_ylabel('Density')
        ax.set_title(f'B. Scale factor distribution\n'
                     f'Median={metrics["scale_factor_median"]:.2f}, '
                     f'IQR={metrics["scale_factor_iqr"]:.2f}')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim([0, 3])
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('B. Scale factor distribution')

    # Panel C: fraction of rescaling neurons
    ax = axes[1, 0]
    categories = ['Rescaling\n(normalized)', 'Fixed latency\n(absolute)']
    fractions  = [metrics['frac_rescaling'], metrics['frac_rescaling_abs']]
    colors_bar = ['steelblue', 'coral']
    valid = [(cat, f, col) for cat, f, col in zip(categories, fractions, colors_bar)
             if pd.notna(f)]
    if valid:
        cats, fracs, cols = zip(*valid)
        bars = ax.bar(cats, fracs, color=cols, alpha=0.7, edgecolor='white', linewidth=2)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Fraction of neurons')
        title_str = f'C. Neurons within 20–30% of expected\nRescaling: {metrics["frac_rescaling"]:.0%}'
        if pd.notna(metrics['frac_rescaling_abs']):
            title_str += f' | Fixed: {metrics["frac_rescaling_abs"]:.0%}'
        ax.set_title(title_str)
        for bar, frac in zip(bars, fracs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{frac:.0%}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('C. Fraction of rescaling neurons')

    # Panel D: summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_rows = [
        ['Metric', 'Value', 'Interpretation'],
        ['─' * 15, '─' * 12, '─' * 25],
        ['r (correlation)', f'{metrics["r"]:.3f}',             'Neurons maintain rank order'],
        ['Slope',           f'{metrics["slope"]:.2f} ± {metrics["slope_se"]:.2f}',
                                                                '1.0 = perfect rescaling'],
        ['Scale factor',    f'{metrics["scale_factor_median"]:.2f}', '1.0 = perfect rescaling'],
        ['Frac. rescaling', f'{metrics["frac_rescaling"]:.0%}', 'Within 20% of expected'],
        ['─' * 15, '─' * 12, '─' * 25],
        ['r (absolute)',    f'{metrics["r_abs"]:.3f}',          'Fixed latency correlation'],
        ['Frac. fixed',
         f'{metrics["frac_rescaling_abs"]:.0%}' if pd.notna(metrics["frac_rescaling_abs"]) else 'N/A',
         'Within 30% of expected'],
    ]
    y = 0.95
    for row in table_rows:
        ax.text(0.02, y, row[0], fontsize=10, fontfamily='monospace',
                transform=ax.transAxes, va='top')
        ax.text(0.40, y, row[1], fontsize=10, fontfamily='monospace',
                transform=ax.transAxes, va='top')
        ax.text(0.60, y, row[2], fontsize=10, fontfamily='monospace',
                transform=ax.transAxes, va='top')
        y -= 0.10
    ax.set_title('D. Summary statistics', loc='left', fontsize=11)

    plt.tight_layout()
    if save:
        DIR_ENHANCED.mkdir(parents=True, exist_ok=True)
        fname = DIR_ENHANCED / f'summary__{label}__{t_start_col}__{imp_tag}.png'
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
    else:
        plt.show()


# ── Committee figures ─────────────────────────────────────────────────────────

def figure1_heatmaps(data_short, data_long,
                     q_spikes_short, q_spikes_long,
                     q_norm_spikes_short, q_norm_spikes_long,
                     q_medians_short, q_medians_long,
                     counts_short, counts_long,
                     t_max_short=T_MAX_SHORT, t_max_long=T_MAX_LONG):
    """Population heatmaps with sample sizes (absolute + normalized, Q3 & Q4).

    Sort order is derived from absolute-time Q3 peak so all panels
    share consistent row ordering and the absolute panels show a clean diagonal.
    """
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.25)
    fig.suptitle(
        'DMS MSN populations show temporal rescaling\n'
        '(neurons sorted by absolute Q3 peak time, same order applied to Q4)',
        fontsize=14, fontweight='bold', y=0.97
    )

    datasets = [
        ('Short BG', 'aligned to last lick',
         data_short, q_spikes_short, q_norm_spikes_short,
         q_medians_short, counts_short, COLOR_SHORT, 0, t_max_short),
        ('Long BG', 'aligned to cue onset',
         data_long,  q_spikes_long,  q_norm_spikes_long,
         q_medians_long,  counts_long,  COLOR_LONG,  1, t_max_long),
    ]

    for group_name, align_desc, data, q_spk, q_norm_spk, q_med, counts, _, row, t_max \
            in datasets:
        if data is None:
            continue

        # Sort by absolute-time Q3 peak; apply same order to Q4 and normalized panels.
        all_unit_ids = list(q_spk['Q3'].keys())
        _, sort_uids, _ = build_population_matrix(
            q_spk['Q3'], all_unit_ids, 0.0, t_max, TIME_STEP, SIGMA
        )
        if sort_uids is None:
            sort_uids = data['sort_uids']

        mat_q3_abs, _, _ = build_population_matrix(
            q_spk['Q3'], all_unit_ids,
            0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids
        )
        mat_q4_abs, _, _ = build_population_matrix(
            q_spk['Q4'], all_unit_ids,
            0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids
        )
        mat_q3_norm, _, _ = build_population_matrix(
            q_norm_spk['Q3'], all_unit_ids,
            0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
        )
        mat_q4_norm, _, _ = build_population_matrix(
            q_norm_spk['Q4'], all_unit_ids,
            0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
        )

        n_units = mat_q3_abs.shape[0] if mat_q3_abs is not None else data['n_units']

        matrices = [mat_q3_abs, mat_q4_abs, mat_q3_norm, mat_q4_norm]
        titles   = [
            f"Q3 absolute\n(median {q_med.get('Q3', 0):.1f}s)",
            f"Q4 absolute\n(median {q_med.get('Q4', 0):.1f}s)",
            "Q3 normalized", "Q4 normalized",
        ]
        extents  = [
            [0, t_max, 0, n_units], [0, t_max, 0, n_units],
            [0, 1, 0, n_units],     [0, 1, 0, n_units],
        ]
        xlabels = ['Time (s)', 'Time (s)', 'Normalized time', 'Normalized time']
        vlines  = [q_med.get('Q3', 0), q_med.get('Q4', 0), None, None]

        for col in range(4):
            ax  = fig.add_subplot(gs[row, col])
            mat = matrices[col]
            if mat is not None:
                vmax = np.percentile(np.abs(mat), 95)
                ax.imshow(mat, aspect='auto', cmap='viridis',
                          vmin=-vmax, vmax=vmax, origin='lower',
                          extent=extents[col])
                if vlines[col] is not None:
                    ax.axvline(vlines[col], color='white', lw=2, ls='--', alpha=0.9)
            ax.set_xlabel(xlabels[col], fontsize=10)
            ax.set_title(titles[col], fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(
                    f'{group_name}  ({align_desc})\n'
                    f'n={counts["n_mice"]} mice · {counts["n_sessions"]} sess · '
                    f'{n_units} units · {counts["n_trials"]:,} trials\n\n'
                    'Unit #',
                    fontsize=9
                )
            else:
                ax.set_yticklabels([])

    fig.savefig(DIR_COMMITTEE / 'fig1_population_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 1 → {DIR_COMMITTEE / 'fig1_population_heatmaps.png'}")


def figure1b_heatmaps(data_short, data_long,
                      q_spikes_short, q_spikes_long,
                      q_norm_spikes_short, q_norm_spikes_long,
                      q_medians_short, q_medians_long,
                      counts_short, counts_long,
                      t_max_short=T_MAX_SHORT, t_max_long=T_MAX_LONG):
    """Same as figure1_heatmaps but sorted by normalized-time Q3 peak instead of absolute."""
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.25)
    fig.suptitle(
        'DMS MSN populations show temporal rescaling\n'
        '(neurons sorted by normalized Q3 peak time, same order applied to Q4)',
        fontsize=14, fontweight='bold', y=0.97
    )

    datasets = [
        ('Short BG', 'aligned to last lick',
         data_short, q_spikes_short, q_norm_spikes_short,
         q_medians_short, counts_short, COLOR_SHORT, 0, t_max_short),
        ('Long BG', 'aligned to cue onset',
         data_long,  q_spikes_long,  q_norm_spikes_long,
         q_medians_long,  counts_long,  COLOR_LONG,  1, t_max_long),
    ]

    for group_name, align_desc, data, q_spk, q_norm_spk, q_med, counts, _, row, t_max \
            in datasets:
        if data is None:
            continue

        # Sort by normalized-time Q3 peak; apply same order to all panels.
        all_unit_ids = list(q_norm_spk['Q3'].keys())
        _, sort_uids, _ = build_population_matrix(
            q_norm_spk['Q3'], all_unit_ids, 0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA
        )
        if sort_uids is None:
            sort_uids = data['sort_uids']

        mat_q3_abs, _, _ = build_population_matrix(
            q_spk['Q3'], all_unit_ids,
            0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids
        )
        mat_q4_abs, _, _ = build_population_matrix(
            q_spk['Q4'], all_unit_ids,
            0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids
        )
        mat_q3_norm, _, _ = build_population_matrix(
            q_norm_spk['Q3'], all_unit_ids,
            0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
        )
        mat_q4_norm, _, _ = build_population_matrix(
            q_norm_spk['Q4'], all_unit_ids,
            0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
        )

        n_units = mat_q3_norm.shape[0] if mat_q3_norm is not None else data['n_units']

        matrices = [mat_q3_abs, mat_q4_abs, mat_q3_norm, mat_q4_norm]
        titles   = [
            f"Q3 absolute\n(median {q_med.get('Q3', 0):.1f}s)",
            f"Q4 absolute\n(median {q_med.get('Q4', 0):.1f}s)",
            "Q3 normalized", "Q4 normalized",
        ]
        extents  = [
            [0, t_max, 0, n_units], [0, t_max, 0, n_units],
            [0, 1, 0, n_units],     [0, 1, 0, n_units],
        ]
        xlabels = ['Time (s)', 'Time (s)', 'Normalized time', 'Normalized time']
        vlines  = [q_med.get('Q3', 0), q_med.get('Q4', 0), None, None]

        for col in range(4):
            ax  = fig.add_subplot(gs[row, col])
            mat = matrices[col]
            if mat is not None:
                vmax = np.percentile(np.abs(mat), 95)
                ax.imshow(mat, aspect='auto', cmap='viridis',
                          vmin=-vmax, vmax=vmax, origin='lower',
                          extent=extents[col])
                if vlines[col] is not None:
                    ax.axvline(vlines[col], color='white', lw=2, ls='--', alpha=0.9)
            ax.set_xlabel(xlabels[col], fontsize=10)
            ax.set_title(titles[col], fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(
                    f'{group_name}  ({align_desc})\n'
                    f'n={counts["n_mice"]} mice · {counts["n_sessions"]} sess · '
                    f'{n_units} units · {counts["n_trials"]:,} trials\n\n'
                    'Unit #',
                    fontsize=9
                )
            else:
                ax.set_yticklabels([])

    fig.savefig(DIR_COMMITTEE / 'fig1b_population_heatmaps_normsort.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 1b → {DIR_COMMITTEE / 'fig1b_population_heatmaps_normsort.png'}")


def figure2_all_alignments(results_df):
    """Bar chart comparing all alignment × group × filter combinations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    alignments = ['last_lick', 'cue_on', 'cue_off']
    colors     = {'last_lick': COLOR_LAST_LICK,
                  'cue_on':    COLOR_CUE_ON,
                  'cue_off':   COLOR_CUE_OFF}
    behavioral_anchor = {'short_BG': 'last_lick', 'long_BG': 'cue_on'}
    groups   = ['short_BG', 'long_BG']
    filters  = ['all', 'excl_imp']
    g_labels = {'short_BG': 'Short BG', 'long_BG': 'Long BG'}
    f_labels = {'all': 'All trials', 'excl_imp': 'Excl. impulsive'}

    for row, group in enumerate(groups):
        for col, filt in enumerate(filters):
            ax     = axes[row, col]
            subset = results_df[(results_df['group'] == group) &
                                 (results_df['filter'] == filt)]
            x      = np.arange(len(alignments))
            r_vals = []
            for a in alignments:
                rd = subset[subset['alignment'] == a]
                r_vals.append(rd['r'].values[0] if len(rd) > 0 else 0)

            bars = ax.bar(x, r_vals,
                          color=[colors[a] for a in alignments],
                          alpha=0.85, edgecolor='white', linewidth=2)
            anchor_idx = alignments.index(behavioral_anchor[group])
            bars[anchor_idx].set_edgecolor('black')
            bars[anchor_idx].set_linewidth(3)
            bars[anchor_idx].set_hatch('///')

            for i, (a, v) in enumerate(zip(alignments, r_vals)):
                rd = subset[subset['alignment'] == a]
                if len(rd) > 0:
                    p     = rd['p_shuffle'].values[0]
                    p_str = 'p<.001' if p < 0.001 else f'p={p:.3f}'
                    ax.text(i, v + 0.02, f'{v:.2f}\n({p_str})',
                            ha='center', fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(['Last lick', 'Cue onset', 'Cue offset'], fontsize=10)
            ax.set_ylim([0.5, 0.8])
            ax.axhline(0.5, color='gray', ls='--', alpha=0.4, lw=1)
            ax.set_ylabel('Peak time correlation (r)', fontsize=10)
            ax.set_title(f'{g_labels[group]} — {f_labels[filt]}',
                         fontsize=11)

            n_units = subset['n_units'].values[0] if len(subset) > 0 else 0
            ax.text(0.98, 0.98, f'n={n_units} units', transform=ax.transAxes,
                    fontsize=9, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8))

    legend_elements = [
        mpatches.Patch(facecolor='gray', edgecolor='black', linewidth=3,
                       hatch='///', label='Behavioral anchor'),
        Line2D([0], [0], color='gray', linestyle='--', label='r = 0.5 (moderate)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(DIR_COMMITTEE / 'fig2_all_alignments.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 2 → {DIR_COMMITTEE / 'fig2_all_alignments.png'}")


def figure3_scatter_with_uncertainty(data_short, data_long, counts_short, counts_long):
    """Scatter plots with bootstrap CI bands and shuffle distribution insets."""
    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.35, wspace=0.3)
    fig.suptitle(
        'Rescaling quality: correlation, slope, and statistical significance',
        fontsize=13, y=0.97)

    datasets = [
        ('Short BG (last lick)', data_short, counts_short, COLOR_SHORT, 0),
        ('Long BG (cue onset)',  data_long,  counts_long,  COLOR_LONG,  1),
    ]

    for title, data, counts, color, col in datasets:
        if data is None:
            continue

        tau_q3    = data['tau_q3']
        tau_q4    = data['tau_q4']
        r         = data['r']
        slope     = data['slope']
        slope_se  = data['slope_se']
        slope_ci  = data['slope_ci']
        intercept = data['intercept']
        n         = data['n_units']
        p         = data['p_shuffle']

        ax_main = fig.add_subplot(gs[0, col])
        ax_main.scatter(tau_q3, tau_q4, s=40, alpha=0.6, c=color,
                        edgecolors='white', linewidth=0.5)
        ax_main.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.4,
                     label='Perfect rescaling')
        x_fit   = np.linspace(0, 1, 100)
        y_fit   = slope * x_fit + intercept
        y_lower = slope_ci[0] * x_fit + intercept
        y_upper = slope_ci[1] * x_fit + intercept
        ax_main.fill_between(x_fit, y_lower, y_upper, color='red', alpha=0.15)
        ax_main.plot(x_fit, y_fit, color='darkred', lw=2.5,
                     label=(f'slope = {slope:.2f}\n'
                            f'95% CI: [{slope_ci[0]:.2f}, {slope_ci[1]:.2f}]'))
        ax_main.set_xlim([0, 1]); ax_main.set_ylim([0, 1])
        ax_main.set_xlabel('Normalized peak time (Q3)', fontsize=11)
        ax_main.set_ylabel('Normalized peak time (Q4)', fontsize=11)
        p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
        ax_main.set_title(
            f'{title}\n'
            f'r = {r:.3f} ({p_str})  ·  slope = {slope:.2f} ± {slope_se:.2f}  ·  n = {n} units',
            fontsize=11        )
        ax_main.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax_main.grid(True, alpha=0.3)

        ax_shuf = fig.add_subplot(gs[1, col])
        r_shuffle = data['r_shuffle']
        ax_shuf.hist(r_shuffle, bins=50, density=True, alpha=0.7,
                     color='gray', edgecolor='white', label='Shuffle distribution')
        ax_shuf.axvline(r, color=color, lw=3, label=f'Observed r = {r:.3f}')
        ax_shuf.axvline(np.percentile(r_shuffle, 95), color='gray', lw=1.5, ls='--',
                        label='95th percentile')
        ax_shuf.set_xlabel('Correlation (r)', fontsize=10)
        ax_shuf.set_ylabel('Density', fontsize=10)
        ax_shuf.set_title(f'Shuffle test (n={N_SHUFFLE} permutations)', fontsize=10)
        ax_shuf.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1, 1),
                       bbox_transform=ax_shuf.transAxes)
        ax_shuf.set_xlim([-0.3, 0.9])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DIR_COMMITTEE / 'fig3_scatter_with_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 3 → {DIR_COMMITTEE / 'fig3_scatter_with_uncertainty.png'}")


def figure4_strategy_dissociation(results_df):
    """Effect of excluding impulsive trials, short BG vs long BG."""
    fig, ax_main = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        'Effect of excluding impulsive trials on rescaling\n'
        '(impulsive = decision < 0.5s after cue offset)',
        fontsize=13, fontweight='bold', y=0.93
    )

    def _get_r(group, alignment, filt):
        rd = results_df[(results_df['group'] == group) &
                        (results_df['alignment'] == alignment) &
                        (results_df['filter'] == filt)]
        return rd['r'].values[0] if len(rd) > 0 else np.nan

    r_short_all  = _get_r('short_BG', 'last_lick', 'all')
    r_short_excl = _get_r('short_BG', 'last_lick', 'excl_imp')
    r_long_all   = _get_r('long_BG',  'cue_on',    'all')
    r_long_excl  = _get_r('long_BG',  'cue_on',    'excl_imp')

    x      = np.array([0, 1, 3, 4])
    heights = [r_short_all, r_short_excl, r_long_all, r_long_excl]
    colors  = [COLOR_SHORT, COLOR_SHORT, COLOR_LONG, COLOR_LONG]
    alphas  = [0.5, 1.0, 1.0, 0.5]
    labels  = ['All', 'Excl.\nimp.', 'All', 'Excl.\nimp.']

    for xi, h, clr, a, lab in zip(x, heights, colors, alphas, labels):
        ax_main.bar(xi, h, width=0.8, color=clr, alpha=a, edgecolor='white', linewidth=2)
        ax_main.text(xi, h + 0.015, f'{h:.3f}', ha='center',
                     fontsize=11, fontweight='bold')

    delta_short = r_short_excl - r_short_all
    delta_long  = r_long_excl  - r_long_all

    ax_main.text(0.5, max(r_short_all, r_short_excl) + 0.015,
                 f'Δ = {delta_short:+.3f}', ha='center',
                 fontsize=12, color='darkgreen', fontweight='bold')
    ax_main.text(3.5, max(r_long_all, r_long_excl) + 0.015,
                 f'Δ = {delta_long:+.3f}', ha='center',
                 fontsize=12, color='darkred', fontweight='bold')

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(labels, fontsize=10)
    ax_main.set_ylabel('Peak time correlation (r)', fontsize=12)
    valid_heights = [h for h in heights if not np.isnan(h)]
    if valid_heights:
        y_min = min(valid_heights) - 0.08
        y_max = max(valid_heights) + 0.08
        ax_main.set_ylim([max(0, y_min), min(1, y_max)])
    ax_main.axhline(0.5, color='gray', ls='--', alpha=0.4, lw=1)
    ax_main.text(0.5, -0.12, 'Short BG\n(last lick)', ha='center',
                 transform=ax_main.get_xaxis_transform(),
                 fontsize=12, fontweight='bold', color=COLOR_SHORT)
    ax_main.text(3.5, -0.12, 'Long BG\n(cue onset)', ha='center',
                 transform=ax_main.get_xaxis_transform(),
                 fontsize=12, fontweight='bold', color=COLOR_LONG)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DIR_COMMITTEE / 'fig4_strategy_dissociation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 4 → {DIR_COMMITTEE / 'fig4_strategy_dissociation.png'}")


def figure5_summary_table(results_df, all_counts):
    """Comprehensive results table with all metrics and sample sizes."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    cols = ['Group', 'Alignment', 'Filter', 'n mice', 'n sess', 'n units', 'n trials',
            'r', 'p(shuffle)', 'Slope', '95% CI']
    table_data = []
    for _, row in results_df.iterrows():
        key    = (row['group'], row['alignment'] + '_time', row['filter'] == 'excl_imp')
        counts = all_counts.get(key, {})
        p      = row['p_shuffle']
        p_str  = '<.001' if p < 0.001 else f'{p:.3f}'

        # Handle slope_ci which may be list, ndarray, or string (if reloaded from CSV)
        ci = row.get('slope_ci', [np.nan, np.nan])
        if isinstance(ci, str):
            try:
                ci = ast.literal_eval(ci)
            except (ValueError, SyntaxError):
                ci = [np.nan, np.nan]
        ci_str = (f'[{ci[0]:.2f}, {ci[1]:.2f}]'
                  if isinstance(ci, (list, np.ndarray)) and len(ci) == 2
                  and not (np.isnan(ci[0]) and np.isnan(ci[1])) else 'N/A')

        table_data.append([
            row['group'].replace('_', ' ').title(),
            row['alignment'].replace('_', ' ').title(),
            'Excl. imp.' if row['filter'] == 'excl_imp' else 'All',
            counts.get('n_mice', '?'),
            counts.get('n_sessions', '?'),
            row['n_units'],
            f"{counts.get('n_trials', 0):,}",
            f"{row['r']:.3f}", p_str,
            f"{row['slope']:.2f} ± {row['slope_se']:.2f}",
            ci_str,
        ])

    table = ax.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)
    for i in range(len(cols)):
        table[(0, i)].set_facecolor('#4a4a4a')
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=8)
    for row_idx, row_data in enumerate(table_data, start=1):
        if 'Short' in row_data[0] and 'Last' in row_data[1] and 'Excl' in row_data[2]:
            for j in range(len(cols)):
                table[(row_idx, j)].set_facecolor('#d5f5e3')
        if 'Long' in row_data[0] and 'Cue On' in row_data[1] and 'All' in row_data[2]:
            for j in range(len(cols)):
                table[(row_idx, j)].set_facecolor('#d6eaf8')
    ax.set_title('Complete results summary\n'
                 '(green = best for Short BG, blue = best for Long BG)',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DIR_COMMITTEE / 'fig5_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 5 → {DIR_COMMITTEE / 'fig5_summary_table.png'}")


def figure6_all_quartile_pairs(d_short, d_long):
    """Scatter plots for all 6 pairwise quartile combinations, both groups."""
    pairs    = [('Q1', 'Q2'), ('Q1', 'Q3'), ('Q1', 'Q4'),
                ('Q2', 'Q3'), ('Q2', 'Q4'), ('Q3', 'Q4')]
    datasets = [
        ('Short BG (last lick)', d_short, COLOR_SHORT),
        ('Long BG (cue onset)',  d_long,  COLOR_LONG),
    ]

    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    fig.suptitle(
        'Pairwise quartile correlations — normalized peak time\n'
        '(Q3→Q4 is the standard rescaling check; all 6 pairs shown)',
        fontsize=13, fontweight='bold', y=0.96
    )

    for row, (group_name, data, color) in enumerate(datasets):
        if not data:
            for col in range(6):
                axes[row, col].set_visible(False)
            continue

        q_norm = data['q_norm_spikes']
        all_sp = data['all_spikes']

        for col, (qa, qb) in enumerate(pairs):
            from scipy import stats as _stats
            ax     = axes[row, col]
            result = compute_pair_r(qa, qb, q_norm, all_sp)
            r      = result['r']
            p      = result['p_shuffle']
            tau_a  = result.get('tau_a', np.array([]))
            tau_b  = result.get('tau_b', np.array([]))

            if len(tau_a) > 1:
                ax.scatter(tau_a, tau_b, s=8, alpha=0.45, c=color, edgecolors='none')
                x_fit = np.linspace(0, 1, 50)
                lr    = _stats.linregress(tau_a, tau_b)
                ax.plot(x_fit, lr.slope * x_fit + lr.intercept, 'r-', lw=1.5)

            ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
            ax.set_xlabel(f'τ {qa}', fontsize=8)
            ax.set_ylabel(f'τ {qb}' if col > 0 else f'{group_name}\nτ {qb}', fontsize=8)
            if col == 0:
                ax.yaxis.label.set_fontsize(9)

            p_str  = 'p<.001' if p < 0.001 else f'p={p:.3f}'
            is_std = (qa == 'Q3' and qb == 'Q4')
            ax.set_title(f'{qa}→{qb}\nr={r:.2f} ({p_str})',
                         fontsize=9, fontweight='normal')
            if is_std:
                for spine in ax.spines.values():
                    spine.set_edgecolor('darkred')
                    spine.set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(DIR_COMMITTEE / 'fig6_all_quartile_pairs.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 6 → {DIR_COMMITTEE / 'fig6_all_quartile_pairs.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("=" * 70)
    print("  DMS RESCALING PLOTS")
    print("=" * 70)

    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_FILE}\n"
            "Run str_rescaling_analysis_fixed.py first to generate it."
        )

    print(f"  Loading cache from {CACHE_FILE} ...")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)

    data_cache = cache['data_cache']
    results_df = cache['results_df']
    all_counts = cache['all_counts']

    DIR_COMMITTEE.mkdir(parents=True, exist_ok=True)

    key_short = ('short_BG', 'last_lick_time', True)
    key_long  = ('long_BG',  'cue_on_time',    False)

    d_short = data_cache.get(key_short, {})
    d_long  = data_cache.get(key_long,  {})

    # ── Per-condition heatmaps + summaries ────────────────────────────────────
    conditions = [
        ('short_BG', 'last_lick_time', False),
        ('short_BG', 'last_lick_time', True),
        ('short_BG', 'cue_on_time',    False),
        ('short_BG', 'cue_on_time',    True),
        ('short_BG', 'cue_off_time',   False),
        ('short_BG', 'cue_off_time',   True),
        ('long_BG',  'cue_on_time',    False),
        ('long_BG',  'cue_on_time',    True),
        ('long_BG',  'cue_off_time',   False),
        ('long_BG',  'cue_off_time',   True),
        ('long_BG',  'last_lick_time', False),
        ('long_BG',  'last_lick_time', True),
    ]
    for group, alignment, excl_imp in conditions:
        key = (group, alignment, excl_imp)
        d   = data_cache.get(key)
        if d is None:
            continue
        render_rescaling(
            d['all_spikes'], d['q_spikes'], d['q_norm_spikes'], d['q_medians'],
            alignment, group, QUARTILE_LABELS, save=True,
        )
        render_rescaling(
            d['all_spikes'], d['q_spikes'], d['q_norm_spikes'], d['q_medians'],
            alignment, group, ['Q3', 'Q4'], save=True,
        )
        plot_rescaling_summary(
            d['metrics'], d['q_medians'], group, alignment, excl_imp, save=True
        )

    # ── Committee figures ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMMITTEE FIGURES")
    print("=" * 70)

    print("\n  Figure 1: Population heatmaps...")
    if d_short and d_long:
        figure1_heatmaps(
            d_short['metrics'],       d_long['metrics'],
            d_short['q_spikes'],      d_long['q_spikes'],
            d_short['q_norm_spikes'], d_long['q_norm_spikes'],
            d_short['q_medians'],     d_long['q_medians'],
            d_short['counts'],        d_long['counts'],
        )
    else:
        print("    → Skipping (missing data)")

    print("\n  Figure 1b: Population heatmaps (sorted by normalized Q3)...")
    if d_short and d_long:
        figure1b_heatmaps(
            d_short['metrics'],       d_long['metrics'],
            d_short['q_spikes'],      d_long['q_spikes'],
            d_short['q_norm_spikes'], d_long['q_norm_spikes'],
            d_short['q_medians'],     d_long['q_medians'],
            d_short['counts'],        d_long['counts'],
        )
    else:
        print("    → Skipping (missing data)")

    print("\n  Figure 2: All alignments...")
    figure2_all_alignments(results_df)

    print("\n  Figure 3: Scatter with uncertainty...")
    if d_short and d_long:
        figure3_scatter_with_uncertainty(
            d_short['metrics'], d_long['metrics'],
            d_short['counts'],  d_long['counts'],
        )
    else:
        print("    → Skipping (missing data)")

    print("\n  Figure 4: Strategy dissociation...")
    figure4_strategy_dissociation(results_df)

    print("\n  Figure 5: Summary table...")
    figure5_summary_table(results_df, all_counts)

    print("\n  Figure 6: All quartile pairs...")
    if d_short and d_long:
        figure6_all_quartile_pairs(d_short, d_long)
    else:
        print("    → Skipping (missing data)")

    print(f"\n  Done. Figures → {DIR_COMMITTEE}")
