"""
DMS rescaling analysis — enhanced metrics + committee figures.

Metrics:
  r, slope (+ bootstrap CI), scale factor, fraction rescaling
  computed in both normalized and absolute time.

Figures produced per condition:
  - Population heatmaps (all Q / Q3+Q4)
  - 2×2 enhanced summary (scatter, scale factor dist., bar, table)

Committee figures (saved once at the end):
  Fig 1  Population heatmaps with sample sizes
  Fig 2  All alignments compared
  Fig 3  Scatter with shuffle insets and CI
  Fig 4  Strategy dissociation (impulsive filter effect)
  Fig 5  Comprehensive summary table
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy import stats

import utils
import paths as p
import constants as c

# ── Config ────────────────────────────────────────────────────────────────────

UNIT_CSV            = p.LOGS_DIR / "RZ_unit_properties_final.csv"
MSN_TIER            = "msn_tier2"
TIME_STEP           = 0.05
SIGMA               = 2
T_MAX_SHORT         = 15.0
T_MAX_LONG          = 22.0
MIN_TRIALS          = 5
N_BINS_NORM         = 100
N_SHUFFLE           = 1000
IMPULSIVE_THRESHOLD = 0.5

QUARTILE_LABELS = ['Q1', 'Q2', 'Q3', 'Q4']

DIR_HEATMAP   = p.DATA_DIR / 'rescaling'
DIR_ENHANCED  = p.DATA_DIR / 'rescaling' / 'enhanced'
DIR_COMMITTEE = p.DATA_DIR / 'rescaling' / 'committee'

COLOR_SHORT     = '#27ae60'
COLOR_LONG      = '#2980b9'
COLOR_LAST_LICK = '#27ae60'
COLOR_CUE_ON    = '#3498db'
COLOR_CUE_OFF   = '#e74c3c'

# ── Data helpers ──────────────────────────────────────────────────────────────

_msn_id_cache = {}


def get_msn_unit_ids_for_session(session_id, unit_df):
    if session_id not in _msn_id_cache:
        mask = (unit_df['session_id'] == session_id) & unit_df[MSN_TIER]
        if 'qc_pass_all' in unit_df.columns:
            mask &= unit_df['qc_pass_all'] == True
        _msn_id_cache[session_id] = set(unit_df.loc[mask, 'id'].tolist())
    return _msn_id_cache[session_id]


def get_spikes(unit_spk_df, trial_subset, t_start_col):
    aligned, normalised = [], []
    for _, row in trial_subset.iterrows():
        t0 = row.get(t_start_col, np.nan)
        t1 = row.get('decision_time', np.nan)
        if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
            continue
        trial_start_abs = row['event_start_time']
        t0_abs = trial_start_abs + t0
        t1_abs = trial_start_abs + t1
        spk = unit_spk_df[
            (unit_spk_df['spike_time'] >= t0_abs) &
            (unit_spk_df['spike_time'] <  t1_abs)
        ]['spike_time'].values
        aligned.append(spk - t0_abs)
        normalised.append((spk - t0_abs) / (t1 - t0))
    return aligned, normalised


def compute_psth(spike_times_list, t_min, t_max, time_step, sigma):
    bin_edges   = np.arange(t_min, t_max + time_step, time_step)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if len(spike_times_list) == 0:
        return bin_centers, None
    counts = np.zeros(len(bin_centers))
    for spk in spike_times_list:
        counts += np.histogram(spk, bins=bin_edges)[0]
    rate        = counts / (len(spike_times_list) * time_step)
    rate_smooth = gaussian_filter1d(rate, sigma=sigma)
    sd = rate_smooth.std()
    if sd == 0:
        return bin_centers, None
    return bin_centers, (rate_smooth - rate_smooth.mean()) / sd


def build_population_matrix(spike_dict, unit_ids, t_min, t_max,
                             time_step, sigma, sort_uids=None):
    psths = {}
    bin_centers = None
    for uid in unit_ids:
        spk_list = spike_dict.get(uid, [])
        if len(spk_list) < MIN_TRIALS:
            continue
        bc, norm = compute_psth(spk_list, t_min, t_max, time_step, sigma)
        bin_centers = bc
        if norm is not None:
            psths[uid] = norm
    if not psths:
        return None, None, None
    uid_list = list(psths.keys())
    matrix   = np.array([psths[u] for u in uid_list])
    if sort_uids is None:
        sort_uids = [uid_list[i] for i in np.argsort(np.argmax(matrix, axis=1))]
    rows = [psths[uid] for uid in sort_uids if uid in psths]
    if not rows:
        return None, sort_uids, bin_centers
    return np.array(rows), sort_uids, bin_centers


# ── Pooling ───────────────────────────────────────────────────────────────────

def pool_sessions(session_ids, unit_df, t_start_col, exclude_impulsive=False):
    """
    Collect spike data across sessions.
    Returns all_spikes, q_spikes, q_norm_spikes, q_medians, counts.
    counts = {n_mice, n_sessions, n_trials, n_trials_excluded}
    """
    all_spikes    = {}
    q_spikes      = {q: {} for q in QUARTILE_LABELS}
    q_norm_spikes = {q: {} for q in QUARTILE_LABELS}
    q_medians_list = []

    n_sessions_used   = 0
    n_trials_total    = 0
    n_trials_excluded = 0
    mice_used         = set()

    for session_id in session_ids:
        _, trials, units = utils.get_session_data(session_id)
        str_ids = get_msn_unit_ids_for_session(session_id, unit_df)
        units   = {uid: spk for uid, spk in units.items() if uid in str_ids}
        if not units:
            continue

        missed_bool = trials['missed'].replace({'True': True, 'False': False}).astype(bool)
        non_missed  = trials[~missed_bool].copy()

        if exclude_impulsive:
            cue_off_to_decision = non_missed['decision_time'] - non_missed['cue_off_time']
            impulsive_mask      = cue_off_to_decision < IMPULSIVE_THRESHOLD
            n_trials_excluded  += impulsive_mask.sum()
            non_missed          = non_missed[~impulsive_mask].copy()

        if len(non_missed) < 20:
            continue

        n_sessions_used += 1
        n_trials_total  += len(non_missed)
        mice_used.add(session_id.split('_')[0])

        non_missed['wait_length'] = non_missed['decision_time'] - non_missed[t_start_col]
        non_missed['quartile']    = pd.qcut(
            non_missed['wait_length'], q=4, labels=QUARTILE_LABELS
        )
        trials = trials.merge(
            non_missed[['trial_id', 'quartile']], on='trial_id', how='left'
        )
        q_medians_list.append(
            non_missed.groupby('quartile', observed=True)['wait_length'].median()
        )

        for uid, unit_spk_df in units.items():
            gid = f"{session_id}__{uid}"
            all_spikes[gid], _ = get_spikes(unit_spk_df, non_missed, t_start_col)
            for q in QUARTILE_LABELS:
                q_trials = trials[trials['quartile'] == q]
                q_spikes[q][gid], q_norm_spikes[q][gid] = get_spikes(
                    unit_spk_df, q_trials, t_start_col
                )

    q_medians = (pd.concat(q_medians_list).groupby(level=0, observed=True).mean()
                 if q_medians_list else pd.Series())
    counts = {
        'n_mice':             len(mice_used),
        'n_sessions':         n_sessions_used,
        'n_trials':           n_trials_total,
        'n_trials_excluded':  n_trials_excluded,
    }
    return all_spikes, q_spikes, q_norm_spikes, q_medians, counts


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(q_spikes, q_norm_spikes, all_spikes, q_medians, t_max=T_MAX_SHORT):
    """
    Compute all rescaling metrics comparing Q3 and Q4.

    Normalized-time metrics
    -----------------------
    r, p_shuffle, r_shuffle          peak-time correlation + shuffle distribution
    slope, slope_se, slope_ci        linear fit τ_Q4 ~ τ_Q3 with bootstrap 95% CI
    intercept
    scale_factor_median/iqr          τ_Q3 / τ_Q4; expected = 1.0 for perfect rescaling
    frac_rescaling                   fraction of neurons within 20% of expected
    tau_q3, tau_q4                   normalized peak times (used for scatter plots)
    mat_q3, mat_q4, sort_uids        population matrices (used for heatmaps)
    n_units

    Absolute-time metrics
    ---------------------
    r_abs, tau_q3_abs, tau_q4_abs
    expected_ratio_abs               median_Q4 / median_Q3
    abs_ratio_median, frac_rescaling_abs
    """
    # Normalized-time matrices
    mat_q3, sort_uids, _ = build_population_matrix(
        q_norm_spikes['Q3'], list(all_spikes.keys()),
        0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA
    )
    mat_q4, _, _ = build_population_matrix(
        q_norm_spikes['Q4'], list(all_spikes.keys()),
        0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
    )
    if mat_q3 is None or mat_q4 is None:
        return None

    tau_q3  = np.argmax(mat_q3, axis=1) / N_BINS_NORM
    tau_q4  = np.argmax(mat_q4, axis=1) / N_BINS_NORM
    n_units = len(tau_q3)

    # Correlation + shuffle distribution
    r   = np.corrcoef(tau_q3, tau_q4)[0, 1]
    rng = np.random.default_rng(42)
    r_shuffle = np.array([
        np.corrcoef(tau_q3, rng.permutation(tau_q4))[0, 1]
        for _ in range(N_SHUFFLE)
    ])
    p_shuffle = np.mean(r_shuffle >= r)

    # Slope + bootstrap CI
    slope_result = stats.linregress(tau_q3, tau_q4)
    slope        = slope_result.slope
    slope_se     = slope_result.stderr
    intercept    = slope_result.intercept
    slopes_boot  = []
    for _ in range(N_SHUFFLE):
        idx = rng.choice(n_units, n_units, replace=True)
        if len(np.unique(tau_q3[idx])) > 2:
            slopes_boot.append(stats.linregress(tau_q3[idx], tau_q4[idx]).slope)
    slopes_boot = np.array(slopes_boot)
    slope_ci = (np.percentile(slopes_boot, [2.5, 97.5]).tolist()
                if len(slopes_boot) > 0 else [np.nan, np.nan])

    # Scale factors (normalized time; expected = 1.0)
    valid_mask = tau_q4 > 0.05
    if valid_mask.sum() > 10:
        sf                   = tau_q3[valid_mask] / tau_q4[valid_mask]
        scale_factor_median  = np.median(sf)
        scale_factor_iqr     = np.percentile(sf, 75) - np.percentile(sf, 25)
        frac_rescaling       = np.mean(np.abs(sf - 1.0) < 0.2)
    else:
        scale_factor_median = scale_factor_iqr = frac_rescaling = np.nan

    # Absolute-time metrics
    mat_q3_abs, sort_uids_abs, _ = build_population_matrix(
        q_spikes['Q3'], list(all_spikes.keys()),
        0.0, t_max, TIME_STEP, SIGMA
    )
    mat_q4_abs, _, _ = build_population_matrix(
        q_spikes['Q4'], list(all_spikes.keys()),
        0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids_abs
    )
    if mat_q3_abs is not None and mat_q4_abs is not None:
        n_bins_abs  = int(t_max / TIME_STEP)
        tau_q3_abs  = np.argmax(mat_q3_abs, axis=1) / n_bins_abs * t_max
        tau_q4_abs  = np.argmax(mat_q4_abs, axis=1) / n_bins_abs * t_max
        r_abs       = np.corrcoef(tau_q3_abs, tau_q4_abs)[0, 1]
        med_q3      = q_medians.get('Q3', np.nan)
        med_q4      = q_medians.get('Q4', np.nan)
        expected_ratio_abs = (med_q4 / med_q3
                              if pd.notna(med_q3) and med_q3 > 0 else np.nan)
        valid_abs = tau_q3_abs > 0.5
        if valid_abs.sum() > 10 and pd.notna(expected_ratio_abs):
            abs_ratios         = tau_q4_abs[valid_abs] / tau_q3_abs[valid_abs]
            abs_ratio_median   = np.median(abs_ratios)
            frac_rescaling_abs = np.mean(
                np.abs(abs_ratios - expected_ratio_abs) < 0.3 * expected_ratio_abs
            )
        else:
            abs_ratio_median = frac_rescaling_abs = np.nan
    else:
        r_abs = np.nan
        tau_q3_abs = tau_q4_abs = np.array([])
        expected_ratio_abs = abs_ratio_median = frac_rescaling_abs = np.nan

    return {
        # Normalized time
        'r': r, 'p_shuffle': p_shuffle, 'r_shuffle': r_shuffle,
        'slope': slope, 'slope_se': slope_se, 'slope_ci': slope_ci,
        'intercept': intercept,
        'scale_factor_median': scale_factor_median,
        'scale_factor_iqr': scale_factor_iqr,
        'frac_rescaling': frac_rescaling,
        'tau_q3': tau_q3, 'tau_q4': tau_q4,
        'mat_q3': mat_q3, 'mat_q4': mat_q4, 'sort_uids': sort_uids,
        'n_units': n_units,
        # Absolute time
        'r_abs': r_abs, 'tau_q3_abs': tau_q3_abs, 'tau_q4_abs': tau_q4_abs,
        'expected_ratio_abs': expected_ratio_abs,
        'abs_ratio_median': abs_ratio_median,
        'frac_rescaling_abs': frac_rescaling_abs,
    }


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

def _render_rescaling(all_spikes, q_spikes, q_norm_spikes, q_medians,
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
        fontsize=13
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
        fontsize=12
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

    Sort order is derived from absolute-time Q3+Q4 combined so all panels
    share consistent row ordering and the absolute panels show a clean diagonal.
    """
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.25)
    fig.suptitle(
        'DMS MSN populations show temporal rescaling\n'
        '(neurons sorted by absolute Q3 peak time, same order applied to Q4)',
        fontsize=14, fontweight='bold', y=0.98
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


def figure2_all_alignments(results_df):
    """Bar chart comparing all alignment × group × filter combinations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Rescaling correlation (r) across all alignment and filter conditions',
        fontsize=13, fontweight='bold', y=0.98
    )

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
                            ha='center', fontsize=9, fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(['Last lick', 'Cue onset', 'Cue offset'], fontsize=10)
            ax.set_ylim([0.5, 0.8])
            ax.axhline(0.5, color='gray', ls='--', alpha=0.4, lw=1)
            ax.set_ylabel('Peak time correlation (r)', fontsize=10)
            ax.set_title(f'{g_labels[group]} — {f_labels[filt]}',
                         fontsize=11, fontweight='bold')

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
        fontsize=13, fontweight='bold', y=0.98
    )

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
            fontsize=11, fontweight='bold'
        )
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
        fontsize=13, fontweight='bold', y=0.98
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
    ax_main.set_ylim([0.6, 0.82])
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
        ci     = row.get('slope_ci', [np.nan, np.nan])
        ci_str = (f'[{ci[0]:.2f}, {ci[1]:.2f}]'
                  if isinstance(ci, (list, np.ndarray)) and len(ci) == 2 else 'N/A')
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


def compute_pair_r(q_a, q_b, q_norm_spikes, all_spikes):
    """Compute normalized peak-time correlation between any two quartiles."""
    mat_a, sort_uids, _ = build_population_matrix(
        q_norm_spikes[q_a], list(all_spikes.keys()),
        0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA
    )
    if mat_a is None:
        return {'r': np.nan, 'p_shuffle': np.nan, 'n_units': 0}
    mat_b, _, _ = build_population_matrix(
        q_norm_spikes[q_b], list(all_spikes.keys()),
        0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
    )
    if mat_b is None:
        return {'r': np.nan, 'p_shuffle': np.nan, 'n_units': 0}

    tau_a = np.argmax(mat_a, axis=1) / N_BINS_NORM
    tau_b = np.argmax(mat_b, axis=1) / N_BINS_NORM

    r   = np.corrcoef(tau_a, tau_b)[0, 1]
    rng = np.random.default_rng(42)
    r_shuffle = np.array([
        np.corrcoef(tau_a, rng.permutation(tau_b))[0, 1]
        for _ in range(N_SHUFFLE)
    ])
    p_shuffle = np.mean(r_shuffle >= r)

    return {'r': r, 'p_shuffle': p_shuffle, 'n_units': len(tau_a),
            'tau_a': tau_a, 'tau_b': tau_b}


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
        fontsize=13, fontweight='bold', y=1.01
    )

    for row, (group_name, data, color) in enumerate(datasets):
        if not data:
            for col in range(6):
                axes[row, col].set_visible(False)
            continue

        q_norm = data['q_norm_spikes']
        all_sp = data['all_spikes']

        for col, (qa, qb) in enumerate(pairs):
            ax     = axes[row, col]
            result = compute_pair_r(qa, qb, q_norm, all_sp)
            r      = result['r']
            p      = result['p_shuffle']
            tau_a  = result.get('tau_a', np.array([]))
            tau_b  = result.get('tau_b', np.array([]))

            if len(tau_a) > 1:
                ax.scatter(tau_a, tau_b, s=8, alpha=0.45, c=color, edgecolors='none')
                x_fit     = np.linspace(0, 1, 50)
                lr        = stats.linregress(tau_a, tau_b)
                ax.plot(x_fit, lr.slope * x_fit + lr.intercept, 'r-', lw=1.5)

            ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
            ax.set_xlabel(f'τ {qa}', fontsize=8)
            ax.set_ylabel(f'τ {qb}' if col > 0 else f'{group_name}\nτ {qb}', fontsize=8)
            if col == 0:
                ax.yaxis.label.set_fontsize(9)

            p_str  = 'p<.001' if p < 0.001 else f'p={p:.3f}'
            is_std = (qa == 'Q3' and qb == 'Q4')
            ax.set_title(f'{qa}→{qb}\nr={r:.2f} ({p_str})',
                         fontsize=9, fontweight='bold' if is_std else 'normal')
            if is_std:
                for spine in ax.spines.values():
                    spine.set_edgecolor('darkred')
                    spine.set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(DIR_COMMITTEE / 'fig6_all_quartile_pairs.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 6 → {DIR_COMMITTEE / 'fig6_all_quartile_pairs.png'}")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_analysis(session_ids, unit_df, t_start_col, label,
                 exclude_impulsive=False, save=True, t_max=T_MAX_SHORT):
    """
    Pool sessions, compute metrics, and save per-condition figures.
    Returns (metrics, q_spikes, q_norm_spikes, all_spikes, q_medians, counts)
    so that the caller can cache data for committee figures.
    """
    imp_tag = 'excl_impulsive' if exclude_impulsive else 'all_trials'
    print(f"\n{'═'*70}")
    print(f"  {label}  |  {t_start_col}  |  {imp_tag}")
    print(f"{'═'*70}")

    all_spikes, q_spikes, q_norm_spikes, q_medians, counts = pool_sessions(
        session_ids, unit_df, t_start_col, exclude_impulsive=exclude_impulsive
    )
    if not all_spikes:
        print("  No units — skipping.")
        return None, None, None, None, None, None

    if exclude_impulsive:
        n_excl = counts['n_trials_excluded']
        n_kept = counts['n_trials']
        print(f"  Trials: {n_kept} kept, {n_excl} excluded "
              f"({100 * n_excl / max(n_excl + n_kept, 1):.1f}%)")

    med_str = '  '.join(
        f"{q}: {q_medians.get(q, float('nan')):.2f}s" for q in QUARTILE_LABELS
    )
    print(f"  Median wait times — {med_str}")

    metrics = compute_metrics(q_spikes, q_norm_spikes, all_spikes, q_medians, t_max=t_max)
    if metrics:
        print(f"  r={metrics['r']:.3f}  slope={metrics['slope']:.2f}±{metrics['slope_se']:.2f}"
              f"  SF={metrics['scale_factor_median']:.2f}  "
              f"frac_resc={metrics['frac_rescaling']:.0%}  r_abs={metrics['r_abs']:.3f}")
        _render_rescaling(all_spikes, q_spikes, q_norm_spikes, q_medians,
                          t_start_col, label, QUARTILE_LABELS, save, t_max=t_max)
        _render_rescaling(all_spikes, q_spikes, q_norm_spikes, q_medians,
                          t_start_col, label, ['Q3', 'Q4'], save, t_max=t_max)
        plot_rescaling_summary(metrics, q_medians, label, t_start_col,
                               exclude_impulsive, save)

    return metrics, q_spikes, q_norm_spikes, all_spikes, q_medians, counts


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("=" * 70)
    print("  DMS RESCALING ANALYSIS")
    print("=" * 70)

    unit_df = pd.read_csv(UNIT_CSV)
    unit_df['session_id'] = (unit_df['mouse'] + '_' +
                             unit_df['date_only'] + '_' +
                             unit_df['probe_region'])
    mouse_to_group = {mouse: grp for grp, mice in c.GROUP_DICT.items() for mouse in mice}
    unit_df['group'] = unit_df['mouse'].map(mouse_to_group)

    _msn_mask = unit_df[MSN_TIER]
    if 'qc_pass_all' in unit_df.columns:
        _msn_mask &= unit_df['qc_pass_all'] == True
    all_msn_sessions = unit_df[_msn_mask]['session_id'].unique().tolist()

    long_sessions  = [s for s in all_msn_sessions
                      if any(s.startswith(m) for m in c.GROUP_DICT['l'])]
    short_sessions = [s for s in all_msn_sessions
                      if any(s.startswith(m) for m in c.GROUP_DICT['s'])]

    print(f"Long BG sessions:  {len(long_sessions)}")
    print(f"Short BG sessions: {len(short_sessions)}")

    DIR_COMMITTEE.mkdir(parents=True, exist_ok=True)

    # ── Run all conditions ────────────────────────────────────────────────────
    conditions = [
        (short_sessions, 'short_BG', 'last_lick_time', False, T_MAX_SHORT),
        (short_sessions, 'short_BG', 'last_lick_time', True,  T_MAX_SHORT),
        (short_sessions, 'short_BG', 'cue_on_time',    False, T_MAX_SHORT),
        (short_sessions, 'short_BG', 'cue_on_time',    True,  T_MAX_SHORT),
        (short_sessions, 'short_BG', 'cue_off_time',   False, T_MAX_SHORT),
        (short_sessions, 'short_BG', 'cue_off_time',   True,  T_MAX_SHORT),
        (long_sessions,  'long_BG',  'cue_on_time',    False, T_MAX_LONG),
        (long_sessions,  'long_BG',  'cue_on_time',    True,  T_MAX_LONG),
        (long_sessions,  'long_BG',  'cue_off_time',   False, T_MAX_LONG),
        (long_sessions,  'long_BG',  'cue_off_time',   True,  T_MAX_LONG),
        (long_sessions,  'long_BG',  'last_lick_time', False, T_MAX_LONG),
        (long_sessions,  'long_BG',  'last_lick_time', True,  T_MAX_LONG),
    ]

    all_results = []
    data_cache  = {}
    all_counts  = {}

    for sessions, group, alignment, excl_imp, t_max in conditions:
        if not sessions:
            print(f"\n[{group} / {alignment}] No sessions — skipping.")
            continue

        metrics, q_spikes, q_norm_spikes, all_spikes, q_medians, counts = run_analysis(
            sessions, unit_df, alignment, group,
            exclude_impulsive=excl_imp, save=True, t_max=t_max
        )

        key      = (group, alignment, excl_imp)
        filt_str = 'excl_imp' if excl_imp else 'all'
        all_counts[key] = counts or {}

        if metrics:
            all_results.append({
                'group':        group,
                'alignment':    alignment.replace('_time', ''),
                'filter':       filt_str,
                'r':            metrics['r'],
                'p_shuffle':    metrics['p_shuffle'],
                'slope':        metrics['slope'],
                'slope_se':     metrics['slope_se'],
                'slope_ci':     metrics['slope_ci'],
                'scale_factor': metrics['scale_factor_median'],
                'frac_rescaling': metrics['frac_rescaling'],
                'r_abs':        metrics['r_abs'],
                'frac_fixed':   metrics['frac_rescaling_abs'],
                'n_units':      metrics['n_units'],
            })
            data_cache[key] = {
                'metrics':      metrics,
                'q_spikes':     q_spikes,
                'q_norm_spikes': q_norm_spikes,
                'all_spikes':   all_spikes,
                'q_medians':    q_medians,
                'counts':       counts,
            }

    # ── Summary CSV ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    DIR_ENHANCED.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DIR_ENHANCED / 'results_summary.csv', index=False)
    print(f"\n  Summary CSV → {DIR_ENHANCED / 'results_summary.csv'}")

    # ── Committee figures ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMMITTEE FIGURES")
    print("=" * 70)

    key_short = ('short_BG', 'last_lick_time', True)
    key_long  = ('long_BG',  'cue_on_time',    False)

    d_short = data_cache.get(key_short, {})
    d_long  = data_cache.get(key_long,  {})

    print("\n  Figure 1: Population heatmaps...")
    if d_short and d_long:
        figure1_heatmaps(
            d_short['metrics'],      d_long['metrics'],
            d_short['q_spikes'],     d_long['q_spikes'],
            d_short['q_norm_spikes'], d_long['q_norm_spikes'],
            d_short['q_medians'],    d_long['q_medians'],
            d_short['counts'],       d_long['counts'],
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
