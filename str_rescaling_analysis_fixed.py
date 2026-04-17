"""
DMS rescaling analysis — data pooling and metrics.

Run this script to pool spike data across sessions, compute all metrics,
and save a results cache (pickle + CSV) for use by str_rescaling_plots.py.

Cache saved to: <DATA_DIR>/rescaling/results_cache.pkl
"""

import pickle
import numpy as np
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

# Threshold for classifying neurons as "rescaling" in absolute time.
# A neuron is considered rescaling if its absolute peak time ratio (Q4/Q3)
# is within this fraction of the expected ratio (based on median trial durations).
# E.g., 0.3 means within ±30% of expected.
ABS_RESCALING_TOLERANCE = 0.3

QUARTILE_LABELS = ['Q1', 'Q2', 'Q3', 'Q4']

DIR_HEATMAP   = p.DATA_DIR / 'rescaling'
DIR_ENHANCED  = p.DATA_DIR / 'rescaling' / 'enhanced'
DIR_COMMITTEE = p.DATA_DIR / 'rescaling' / 'committee'

COLOR_SHORT     = '#27ae60'
COLOR_LONG      = '#2980b9'
COLOR_LAST_LICK = '#27ae60'
COLOR_CUE_ON    = '#3498db'
COLOR_CUE_OFF   = '#e74c3c'

CACHE_FILE = DIR_HEATMAP / 'results_cache.pkl'

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
            non_missed['wait_length'], q=4, labels=QUARTILE_LABELS, duplicates='drop'
        )
        # Drop trials whose wait_length fell exactly on a bin edge (qcut returns NaN).
        n_before = len(non_missed)
        non_missed = non_missed[non_missed['quartile'].notna()].copy()
        n_trials_total -= (n_before - len(non_missed))  # adjust for qcut-dropped trials

        # Assign quartile labels back into full trials DataFrame via index.
        # Missed trials will have quartile=NaN and won't be selected in q_trials below.
        trials['quartile'] = None
        trials.loc[non_missed.index, 'quartile'] = non_missed['quartile'].astype(object)

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
    scale_factor_median/iqr          τ_Q3 / τ_Q4 (ratio of normalized peak times)
                                     Expected = 1.0 for perfect rescaling (same relative position)
    frac_rescaling                   fraction of neurons within 20% of expected (i.e., 0.8–1.2)
    tau_q3, tau_q4                   normalized peak times (used for scatter plots)
    mat_q3, mat_q4, sort_uids        population matrices (used for heatmaps)
    n_units

    Absolute-time metrics (uses DIFFERENT sort order — sorted by absolute Q3 peaks)
    ---------------------
    r_abs, tau_q3_abs, tau_q4_abs
    expected_ratio_abs               median_Q4 / median_Q3 (expected peak time ratio if rescaling)
    abs_ratio_median                 actual median of τ_Q4_abs / τ_Q3_abs across neurons
    frac_rescaling_abs               fraction within ABS_RESCALING_TOLERANCE of expected ratio
    """
    # ── Normalized-time matrices ──────────────────────────────────────────────
    # Sort by normalized Q3 peak times; apply same order to Q4
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

    # Correlation + shuffle distribution (null = random pairing)
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

    # Peak time ratio in normalized time (expected = 1.0 for perfect rescaling)
    # This measures whether neurons maintain the same RELATIVE position within the interval.
    # τ_Q3 / τ_Q4 ≈ 1.0 means neuron fires at same fraction of interval in both conditions.
    valid_mask = tau_q4 > 0.05  # Avoid division by near-zero
    if valid_mask.sum() > 10:
        sf                   = tau_q3[valid_mask] / tau_q4[valid_mask]
        scale_factor_median  = np.median(sf)
        scale_factor_iqr     = np.percentile(sf, 75) - np.percentile(sf, 25)
        frac_rescaling       = np.mean(np.abs(sf - 1.0) < 0.2)  # Within ±20% of expected
    else:
        scale_factor_median = scale_factor_iqr = frac_rescaling = np.nan

    # ── Absolute-time metrics ─────────────────────────────────────────────────
    # NOTE: Uses a DIFFERENT sort order (sorted by absolute Q3 peaks, not normalized).
    # This is intentional — absolute-time metrics test whether peak latencies scale
    # with interval duration, which requires sorting in absolute time.
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
        valid_abs = tau_q3_abs > 0.5  # Avoid very early peaks (likely noise)
        if valid_abs.sum() > 10 and pd.notna(expected_ratio_abs):
            abs_ratios         = tau_q4_abs[valid_abs] / tau_q3_abs[valid_abs]
            abs_ratio_median   = np.median(abs_ratios)
            # Fraction of neurons with peak ratio within tolerance of expected
            frac_rescaling_abs = np.mean(
                np.abs(abs_ratios - expected_ratio_abs) < ABS_RESCALING_TOLERANCE * expected_ratio_abs
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


# ── Runner ────────────────────────────────────────────────────────────────────

def run_analysis(session_ids, unit_df, t_start_col, label,
                 exclude_impulsive=False, t_max=T_MAX_SHORT):
    """
    Pool sessions and compute metrics.
    Returns (metrics, q_spikes, q_norm_spikes, all_spikes, q_medians, counts).
    Plotting is handled separately by str_rescaling_plots.py.
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
            exclude_impulsive=excl_imp, t_max=t_max
        )

        key      = (group, alignment, excl_imp)
        filt_str = 'excl_imp' if excl_imp else 'all'
        all_counts[key] = counts or {}

        if metrics:
            all_results.append({
                'group':          group,
                'alignment':      alignment.replace('_time', ''),
                'filter':         filt_str,
                'r':              metrics['r'],
                'p_shuffle':      metrics['p_shuffle'],
                'slope':          metrics['slope'],
                'slope_se':       metrics['slope_se'],
                'slope_ci':       metrics['slope_ci'],
                'scale_factor':   metrics['scale_factor_median'],
                'frac_rescaling': metrics['frac_rescaling'],
                'r_abs':          metrics['r_abs'],
                'frac_fixed':     metrics['frac_rescaling_abs'],
                'n_units':        metrics['n_units'],
            })
            data_cache[key] = {
                'metrics':       metrics,
                'q_spikes':      q_spikes,
                'q_norm_spikes': q_norm_spikes,
                'all_spikes':    all_spikes,
                'q_medians':     q_medians,
                'counts':        counts,
            }

    # ── Save CSV + cache ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    DIR_ENHANCED.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DIR_ENHANCED / 'results_summary.csv', index=False)
    print(f"\n  Summary CSV → {DIR_ENHANCED / 'results_summary.csv'}")

    DIR_HEATMAP.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({
            'data_cache':  data_cache,
            'results_df':  results_df,
            'all_counts':  all_counts,
        }, f)
    print(f"  Results cache → {CACHE_FILE}")
    print("\n  Run str_rescaling_plots.py to generate figures.")
