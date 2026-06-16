"""
Cross-region rescaling — analysis + plots in one script.

Single-script pipeline: pools spike data across sessions, computes all
rescaling metrics, caches results per cell-type set, then renders every
figure for that set.

    python a_rescaling.py                      # all cell sets: analysis + plots
    python a_rescaling.py <cell_set>           # one cell set only
    python a_rescaling.py remetric             # recompute cached metrics for QUARTILE_PAIR
    python a_rescaling.py --regenerate         # force re-run analysis AND plots
    python a_rescaling.py --regenerate-analysis   # force re-pool + re-metric
    python a_rescaling.py --regenerate-plots      # force re-render figures

Skip behavior (the point of merging the two scripts):
  - Analysis is skipped for a cell set when its results_cache.pkl already
    exists. Pass --regenerate-analysis (or --regenerate) to force a rerun.
  - Plots are skipped for a cell set when its figures already exist. Pass
    --regenerate-plots (or --regenerate) to force. If analysis actually
    re-ran for a set, its plots are re-rendered regardless (stale figures).

Loops over the cell-type selections in CELL_SETS and saves a separate cache
per set. Each selection is a row-mask over the canonical
unit_properties_final.csv (0h+0i output).

Headline panel (used by b_cross_region_rescaling.py):
  Striatum:    str_all (aggregate), str_msn, str_fsi
  Cortex:      mc_all (aggregate), mc_l5l6_rs, mc_fsi, v1_all (aggregate), v1_rs
  Thalamus:    thal (aggregate), val, po, vpm
  Hippocampus: ca1, hpf

Note: vpm short-BG has only 8 units / 2 mice; po and vpm long-BG are each
single-mouse — flagged in the cross-region figure but not excluded here.

Caches saved to: <DATA_DIR>/rescaling/<set_label>/results_cache.pkl
Figures to:      <DATA_DIR>/rescaling/<set_label>/{per_group,enhanced,committee}/
"""

import ast
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# This module lives in rescaling/ but imports project-root modules
# (utils, paths, constants). Add the repo root to sys.path so they resolve
# regardless of the directory the script is launched from.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils
import paths as p
import constants as c

# ── Config ────────────────────────────────────────────────────────────────────

# Cell-type selection sets. Each CSV must contain mouse, date_only,
# probe_region, and id columns; rescaling pools the listed units per session.
# Optional config keys:
#   'filter_col' — name of a boolean column to gate rows by True
#   'where'      — callable (df) -> df for arbitrary row filtering
UPF = p.LOGS_DIR / 'unit_properties_final.csv'   # canonical (0h + 0i)

CELL_SETS = {
    # ── Striatum ──────────────────────────────────────────────────────────────
    # 'str_all' = every striatal unit regardless of cell type; 'str_msn' /
    # 'str_fsi' are the cell-type-specific subsets.
    'str_all': {'csv': UPF, 'filter_col': 'is_str_unit'},
    'str_msn': {'csv': UPF, 'filter_col': 'is_str_msn'},
    'str_fsi': {
        'csv':   UPF,
        'where': lambda df: df[df['is_str_unit'] & (df['cell_type'] == 'FSI')],
    },
    # ── Cortex ────────────────────────────────────────────────────────────────
    # 'mc_all' / 'v1_all' = every unit in the region regardless of cell type or
    # layer; the cell-type-specific subsets follow each aggregate.
    'mc_all': {
        'csv':   UPF,
        'where': lambda df: df[df['region_group'] == 'Motor cortex'],
    },
    'mc_l5l6_rs': {
        'csv':   UPF,
        'where': lambda df: df[df['is_mc_l5l6'] & (df['cell_type'] == 'RS')],
    },
    'mc_fsi': {
        'csv':   UPF,
        'where': lambda df: df[(df['region_group'] == 'Motor cortex')
                               & (df['cell_type'] == 'FSI')],
    },
    'v1_all': {
        'csv':   UPF,
        'where': lambda df: df[df['region_group'] == 'Visual cortex'],
    },
    'v1_rs': {
        'csv':   UPF,
        'where': lambda df: df[(df['region_group'] == 'Visual cortex')
                               & (df['cell_type'] == 'RS')],
    },
    # ── Thalamus ──────────────────────────────────────────────────────────────
    'thal': {'csv': UPF, 'filter_col': 'is_thal'},
    'val':  {'csv': UPF, 'filter_col': 'is_val'},
    'po':   {'csv': UPF, 'filter_col': 'is_po'},
    'vpm':  {'csv': UPF, 'filter_col': 'is_vpm'},
    # ── Hippocampus ───────────────────────────────────────────────────────────
    'ca1':  {'csv': UPF, 'filter_col': 'is_ca1'},
    'hpf':  {'csv': UPF, 'filter_col': 'is_hpf'},

    # ── Deprecated (kept commented for reference; prior caches still on disk) ─
    # 'str_units':                {'csv': UPF, 'filter_col': 'is_str_unit'},
    # 'str_msn_depth':            {'csv': UPF, 'filter_col': 'is_msn_depth'},
    # 'str_msn_depth_permissive': {'csv': UPF, 'filter_col': 'is_msn_depth_permissive'},
    # 'mc_l5l6':                  {'csv': UPF, 'filter_col': 'is_mc_l5l6'},
    # 'mop_mos_rs_final':         <equivalent to mc_l5l6_rs in practice>
    # 'msn_tier2_first_batch':    {'csv': p.LOGS_DIR / 'unit_properties_first_batch.csv',
    #                              'filter_col': 'msn_tier2'},
    # 'v1':                       {'csv': UPF, 'filter_col': 'is_visp'},
}
DEFAULT_CELL_SET = 'str_msn'

# Force re-run even when cached outputs exist. Override per-invocation with
# --regenerate-analysis / --regenerate-plots / --regenerate on the CLI.
REGENERATE_ANALYSIS = False
REGENERATE_PLOTS    = False

TIME_STEP   = 0.05
SIGMA       = 2
T_MAX_SHORT = 15.0
T_MAX_LONG  = 22.0
MIN_TRIALS  = 5
N_BINS_NORM = 100
N_SHUFFLE   = 1000

# Drop a unit if any quartile has fewer than this many total spikes summed
# across its trials. Smoothed PSTHs from sparse spiking produce unreliable
# peak times — typically landing at the basis edge, which then propagates
# as anchor-locked artifacts (high r but NaN frac_rescaling in small-n sets).
MIN_SPIKES_PER_QUARTILE = 30

# Threshold for classifying neurons as "rescaling" in absolute time.
# A neuron is considered rescaling if its absolute peak time ratio (Q4/Q3)
# is within this fraction of the expected ratio (based on median trial durations).
# E.g., 0.3 means within ±30% of expected.
ABS_RESCALING_TOLERANCE = 0.3

QUARTILE_LABELS = ['Q1', 'Q2', 'Q3', 'Q4']

# Quartile pair compared for the headline rescaling metrics. Q2→Q3 replaced
# Q3→Q4: Q4 is the open-ended top wait-time quartile (no outlier trimming) and
# adds noise to the peak estimate, whereas Q2 and Q3 are both bounded between
# the 25th–75th percentiles. compute_metrics takes a `pair` argument; this is
# the project-wide default imported by the downstream rescaling scripts.
# See c_quartile_pair_comparison.py for the pair-by-pair justification.
QUARTILE_PAIR = ('Q2', 'Q3')
QA, QB = QUARTILE_PAIR

DIR_BASE = p.DATA_DIR / 'rescaling'   # per-set output paths via paths_for()

COLOR_SHORT     = '#27ae60'
COLOR_LONG      = '#2980b9'
COLOR_LAST_LICK = '#27ae60'
COLOR_CUE_ON    = '#3498db'
COLOR_CUE_OFF   = '#e74c3c'

# Committee figure rendered last by make_plots_for_set(); its presence is the
# marker that a cell set's plots already exist (see make_plots_for_set).
PLOTS_DONE_MARKER = 'fig5_summary_table.png'


def paths_for(set_label):
    """Return (set_dir, enhanced_dir, committee_dir, cache_file) for a cell set."""
    base = DIR_BASE / set_label
    return base, base / 'enhanced', base / 'committee', base / 'results_cache.pkl'


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# ── Data helpers ──────────────────────────────────────────────────────────────

# Populated by load_unit_selection() before pool_sessions() is called.
_unit_selection = {}      # session_id -> set of unit ids
_cp_id_cache    = {}      # currently a thin pass-through to _unit_selection


def load_unit_selection(csv_path, filter_col=None, where=None):
    """Load a unit-selection CSV (e.g., str_msn.csv) and return the unit DataFrame.

    If `filter_col` is given, only rows where that boolean column is True are kept.
    If `where` is given (callable: df -> df), it is applied for arbitrary row
    filtering (e.g., compound region/cell-type masks).

    Side effect: populates the module-level _unit_selection dict so that
    get_cp_unit_ids_for_session returns the listed units for each session.
    Resets the per-session cache so prior selections don't leak across runs.
    """
    global _unit_selection, _cp_id_cache
    sel = pd.read_csv(csv_path)
    sel['session_id'] = (sel['mouse'] + '_'
                         + sel['date_only'] + '_'
                         + sel['probe_region'])
    if 'qc_pass_all' in sel.columns:
        sel = sel[sel['qc_pass_all'] == True].copy()
    if filter_col is not None:
        sel = sel[sel[filter_col].astype(bool)].copy()
    if where is not None:
        sel = where(sel).copy()
    _unit_selection = {sid: set(grp['id'].tolist())
                       for sid, grp in sel.groupby('session_id')}
    _cp_id_cache = {}
    return sel


def get_cp_unit_ids_for_session(session_id, unit_df=None):
    """Return the set of selected unit IDs for a session.

    The selection is determined by load_unit_selection(); unit_df is accepted
    for backward compatibility but ignored.
    """
    return _unit_selection.get(session_id, set())


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

def pool_sessions(session_ids, unit_df, t_start_col):
    """
    Collect spike data across sessions.
    Returns all_spikes, q_spikes, q_norm_spikes, q_medians, counts.
    counts = {n_mice, n_sessions, n_trials}
    """
    all_spikes    = {}
    q_spikes      = {q: {} for q in QUARTILE_LABELS}
    q_norm_spikes = {q: {} for q in QUARTILE_LABELS}
    q_medians_list = []

    n_sessions_used = 0
    n_trials_total  = 0
    mice_used       = set()

    for session_id in session_ids:
        _, trials, units = utils.get_session_data(session_id)
        cp_ids = get_cp_unit_ids_for_session(session_id, unit_df)
        units  = {uid: spk for uid, spk in units.items() if uid in cp_ids}
        if not units:
            continue

        missed_bool = trials['missed'].replace({'True': True, 'False': False}).astype(bool)
        non_missed  = trials[~missed_bool].copy()

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

    # Min-spikes-per-quartile filter: drop any unit whose sparsest quartile
    # has < MIN_SPIKES_PER_QUARTILE total spikes. This is a unit-level filter
    # (no other thresholds change). Eliminates the small-sample peak-time
    # artifacts that produce high-r-but-NaN-frac_rescaling cells in small sets.
    n_before = len(all_spikes)
    all_spikes, q_spikes, q_norm_spikes = _filter_low_spike_units(
        all_spikes, q_spikes, q_norm_spikes, MIN_SPIKES_PER_QUARTILE
    )
    n_after = len(all_spikes)
    if n_before:
        print(f"  Min-spikes filter (≥{MIN_SPIKES_PER_QUARTILE}/quartile): "
              f"kept {n_after}/{n_before} units")

    counts = {
        'n_mice':     len(mice_used),
        'n_sessions': n_sessions_used,
        'n_trials':   n_trials_total,
    }
    return all_spikes, q_spikes, q_norm_spikes, q_medians, counts


def _filter_low_spike_units(all_spikes, q_spikes, q_norm_spikes, min_spikes):
    """Drop any unit whose sparsest quartile has fewer than min_spikes
    total spikes summed across its trials.

    Returns filtered (all_spikes, q_spikes, q_norm_spikes) with the same
    schema as the inputs but restricted to units that pass the threshold
    in every quartile.
    """
    valid = set()
    for gid in all_spikes:
        ok = True
        for q in QUARTILE_LABELS:
            trial_lists = q_spikes[q].get(gid, [])
            n_spk = sum(len(s) for s in trial_lists)
            if n_spk < min_spikes:
                ok = False
                break
        if ok:
            valid.add(gid)
    all_spikes_f    = {g: v for g, v in all_spikes.items() if g in valid}
    q_spikes_f      = {q: {g: v for g, v in d.items() if g in valid}
                       for q, d in q_spikes.items()}
    q_norm_spikes_f = {q: {g: v for g, v in d.items() if g in valid}
                       for q, d in q_norm_spikes.items()}
    return all_spikes_f, q_spikes_f, q_norm_spikes_f


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(q_spikes, q_norm_spikes, all_spikes, q_medians,
                    t_max=T_MAX_SHORT, pair=QUARTILE_PAIR):
    """
    Compute all rescaling metrics comparing the two quartiles in ``pair``
    (default QUARTILE_PAIR = Q2→Q3). Output keys are pair-generic: ``tau_a`` /
    ``mat_a`` / ``tau_a_abs`` carry the first quartile of ``pair`` (qa), the
    ``_b`` keys the second (qb); ``pair`` itself is stored under
    ``quartile_pair``.

    Normalized-time metrics
    -----------------------
    r, p_shuffle, r_shuffle          peak-time correlation + shuffle distribution
    slope, slope_se, slope_ci        linear fit τ_b ~ τ_a with bootstrap 95% CI
    intercept
    scale_factor_median/iqr          τ_a / τ_b (ratio of normalized peak times)
                                     Expected = 1.0 for perfect rescaling (same relative position)
    frac_rescaling                   fraction of neurons within 20% of expected (i.e., 0.8–1.2)
    tau_a, tau_b                     normalized peak times (used for scatter plots)
    mat_a, mat_b, sort_uids          population matrices (used for heatmaps)
    n_units

    Absolute-time metrics (uses DIFFERENT sort order — sorted by absolute qa peaks)
    ---------------------
    r_abs, tau_a_abs, tau_b_abs
    expected_ratio_abs               median_qb / median_qa (expected peak time ratio if rescaling)
    abs_ratio_median                 actual median of τ_b_abs / τ_a_abs across neurons
    frac_rescaling_abs               fraction within ABS_RESCALING_TOLERANCE of expected ratio
    """
    qa, qb = pair

    # ── Normalized-time matrices ──────────────────────────────────────────────
    # Sort by normalized qa peak times; apply the same order to qb.
    mat_a, sort_uids, _ = build_population_matrix(
        q_norm_spikes[qa], list(all_spikes.keys()),
        0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA
    )
    mat_b, _, _ = build_population_matrix(
        q_norm_spikes[qb], list(all_spikes.keys()),
        0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
    )
    if mat_a is None or mat_b is None:
        return None

    tau_a  = np.argmax(mat_a, axis=1) / N_BINS_NORM
    tau_b  = np.argmax(mat_b, axis=1) / N_BINS_NORM
    n_units = len(tau_a)

    # Correlation + shuffle distribution (null = random pairing)
    r   = np.corrcoef(tau_a, tau_b)[0, 1]
    rng = np.random.default_rng(42)
    r_shuffle = np.array([
        np.corrcoef(tau_a, rng.permutation(tau_b))[0, 1]
        for _ in range(N_SHUFFLE)
    ])
    p_shuffle = np.mean(r_shuffle >= r)

    # Slope + bootstrap CI
    slope_result = stats.linregress(tau_a, tau_b)
    slope        = slope_result.slope
    slope_se     = slope_result.stderr
    intercept    = slope_result.intercept
    slopes_boot  = []
    for _ in range(N_SHUFFLE):
        idx = rng.choice(n_units, n_units, replace=True)
        if len(np.unique(tau_a[idx])) > 2:
            slopes_boot.append(stats.linregress(tau_a[idx], tau_b[idx]).slope)
    slopes_boot = np.array(slopes_boot)
    slope_ci = (np.percentile(slopes_boot, [2.5, 97.5]).tolist()
                if len(slopes_boot) > 0 else [np.nan, np.nan])

    # Peak time ratio in normalized time (expected = 1.0 for perfect rescaling)
    # This measures whether neurons maintain the same RELATIVE position within the interval.
    # τ_a / τ_b ≈ 1.0 means neuron fires at same fraction of interval in both conditions.
    valid_mask = tau_b > 0.05  # Avoid division by near-zero
    if valid_mask.sum() > 10:
        sf                   = tau_a[valid_mask] / tau_b[valid_mask]
        scale_factor_median  = np.median(sf)
        scale_factor_iqr     = np.percentile(sf, 75) - np.percentile(sf, 25)
        frac_rescaling       = np.mean(np.abs(sf - 1.0) < 0.2)  # Within ±20% of expected
    else:
        scale_factor_median = scale_factor_iqr = frac_rescaling = np.nan

    # ── Absolute-time metrics ─────────────────────────────────────────────────
    # NOTE: Uses a DIFFERENT sort order (sorted by absolute Q3 peaks, not normalized).
    # This is intentional — absolute-time metrics test whether peak latencies scale
    # with interval duration, which requires sorting in absolute time.
    mat_a_abs, sort_uids_abs, _ = build_population_matrix(
        q_spikes[qa], list(all_spikes.keys()),
        0.0, t_max, TIME_STEP, SIGMA
    )
    mat_b_abs, _, _ = build_population_matrix(
        q_spikes[qb], list(all_spikes.keys()),
        0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids_abs
    )
    if mat_a_abs is not None and mat_b_abs is not None:
        n_bins_abs  = int(t_max / TIME_STEP)
        tau_a_abs  = np.argmax(mat_a_abs, axis=1) / n_bins_abs * t_max
        tau_b_abs  = np.argmax(mat_b_abs, axis=1) / n_bins_abs * t_max
        r_abs       = np.corrcoef(tau_a_abs, tau_b_abs)[0, 1]
        med_a      = q_medians.get(qa, np.nan)
        med_b      = q_medians.get(qb, np.nan)
        expected_ratio_abs = (med_b / med_a
                              if pd.notna(med_a) and med_a > 0 else np.nan)
        valid_abs = tau_a_abs > 0.5  # Avoid very early peaks (likely noise)
        if valid_abs.sum() > 10 and pd.notna(expected_ratio_abs):
            abs_ratios         = tau_b_abs[valid_abs] / tau_a_abs[valid_abs]
            abs_ratio_median   = np.median(abs_ratios)
            # Fraction of neurons with peak ratio within tolerance of expected
            frac_rescaling_abs = np.mean(
                np.abs(abs_ratios - expected_ratio_abs) < ABS_RESCALING_TOLERANCE * expected_ratio_abs
            )
        else:
            abs_ratio_median = frac_rescaling_abs = np.nan
    else:
        r_abs = np.nan
        tau_a_abs = tau_b_abs = np.array([])
        expected_ratio_abs = abs_ratio_median = frac_rescaling_abs = np.nan

    return {
        'quartile_pair': pair,
        # Normalized time
        'r': r, 'p_shuffle': p_shuffle, 'r_shuffle': r_shuffle,
        'slope': slope, 'slope_se': slope_se, 'slope_ci': slope_ci,
        'intercept': intercept,
        'scale_factor_median': scale_factor_median,
        'scale_factor_iqr': scale_factor_iqr,
        'frac_rescaling': frac_rescaling,
        'tau_a': tau_a, 'tau_b': tau_b,
        'mat_a': mat_a, 'mat_b': mat_b, 'sort_uids': sort_uids,
        'n_units': n_units,
        # Absolute time
        'r_abs': r_abs, 'tau_a_abs': tau_a_abs, 'tau_b_abs': tau_b_abs,
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


def compute_pair_full_metrics(q_a, q_b, q_norm_spikes, all_spikes):
    """Compute the full metric bundle for an arbitrary (q_a, q_b) pair, with
    the same statistics as ``compute_metrics`` (slope, slope_se, slope_ci,
    intercept, r_shuffle, tau_a, tau_b, r, p_shuffle, n_units).

    Used by figure3 to render scatter+CI+shuffle inset for the best-fit
    quartile pair when that differs from the cache's default Q3→Q4."""
    mat_a, sort_uids, _ = build_population_matrix(
        q_norm_spikes[q_a], list(all_spikes.keys()),
        0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA
    )
    if mat_a is None:
        return None
    mat_b, _, _ = build_population_matrix(
        q_norm_spikes[q_b], list(all_spikes.keys()),
        0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
    )
    if mat_b is None:
        return None

    tau_a   = np.argmax(mat_a, axis=1) / N_BINS_NORM
    tau_b   = np.argmax(mat_b, axis=1) / N_BINS_NORM
    n_units = len(tau_a)

    r   = np.corrcoef(tau_a, tau_b)[0, 1]
    rng = np.random.default_rng(42)
    r_shuffle = np.array([
        np.corrcoef(tau_a, rng.permutation(tau_b))[0, 1]
        for _ in range(N_SHUFFLE)
    ])
    p_shuffle = np.mean(r_shuffle >= r)

    slope_result = stats.linregress(tau_a, tau_b)
    slope        = slope_result.slope
    slope_se     = slope_result.stderr
    intercept    = slope_result.intercept

    slopes_boot = []
    for _ in range(N_SHUFFLE):
        idx = rng.choice(n_units, n_units, replace=True)
        if len(np.unique(tau_a[idx])) > 2:
            slopes_boot.append(stats.linregress(tau_a[idx], tau_b[idx]).slope)
    slopes_boot = np.array(slopes_boot)
    slope_ci = (np.percentile(slopes_boot, [2.5, 97.5]).tolist()
                if len(slopes_boot) > 0 else [np.nan, np.nan])

    return {
        'r': r, 'p_shuffle': p_shuffle, 'r_shuffle': r_shuffle,
        'slope': slope, 'slope_se': slope_se, 'slope_ci': slope_ci,
        'intercept': intercept,
        'tau_a': tau_a, 'tau_b': tau_b,
        'n_units': n_units,
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_analysis(session_ids, unit_df, t_start_col, label, t_max=T_MAX_SHORT):
    """
    Pool sessions and compute metrics.
    Returns (metrics, q_spikes, q_norm_spikes, all_spikes, q_medians, counts).
    Plotting is handled by make_plots_for_set() further down this file.
    """
    print(f"\n  {label}  |  {t_start_col}")

    all_spikes, q_spikes, q_norm_spikes, q_medians, counts = pool_sessions(
        session_ids, unit_df, t_start_col
    )
    if not all_spikes:
        print("  No units — skipping.")
        return None, None, None, None, None, None

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


def run_for_cell_set(set_label, config, regenerate=False):
    """Pool spikes, compute metrics, and save cache+CSV for one cell-type set.

    `config` is a dict with keys:
        'csv'        — path to the unit-selection CSV (required)
        'filter_col' — optional name of a boolean column to gate rows by True
        'where'      — optional callable (df) -> df for arbitrary row filtering

    If the cache pickle already exists and `regenerate` is False, the existing
    cache is loaded and its results_df is returned without re-running pooling.
    Pass `regenerate=True` to force a rerun.
    """
    csv_path   = config['csv']
    filter_col = config.get('filter_col')
    where      = config.get('where')

    print("\n" + "=" * 70)
    print(f"  CELL SET: {set_label}")
    print(f"  Source:   {csv_path}")
    if filter_col:
        print(f"  Filter:   {filter_col} == True")
    if where:
        print(f"  Filter:   custom `where` callable")
    print("=" * 70)

    out_base, out_enhanced, _, cache_file = paths_for(set_label)

    if cache_file.exists() and not regenerate:
        print(f"  [CACHED] {cache_file} — pass regenerate=True to rerun.")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        return cache.get('results_df')

    if not csv_path.exists():
        print(f"  [SKIP] CSV not found: {csv_path}")
        return None

    unit_df = load_unit_selection(csv_path, filter_col=filter_col, where=where)
    mouse_to_group = {m: g for g, ms in c.GROUP_DICT.items() for m in ms}
    unit_df['group'] = unit_df['mouse'].map(mouse_to_group)

    all_sessions   = sorted(_unit_selection.keys())
    long_sessions  = [s for s in all_sessions
                      if any(s.startswith(m) for m in c.GROUP_DICT['l'])]
    short_sessions = [s for s in all_sessions
                      if any(s.startswith(m) for m in c.GROUP_DICT['s'])]

    short_units = sum(len(_unit_selection[s]) for s in short_sessions)
    long_units  = sum(len(_unit_selection[s]) for s in long_sessions)
    print(f"  Units: short BG={short_units}, long BG={long_units}  "
          f"(total {short_units + long_units})")

    conditions = [
        (short_sessions, 'short_BG', 'last_lick_time', T_MAX_SHORT),
        (short_sessions, 'short_BG', 'cue_on_time',    T_MAX_SHORT),
        (short_sessions, 'short_BG', 'cue_off_time',   T_MAX_SHORT),
        (long_sessions,  'long_BG',  'cue_on_time',    T_MAX_LONG),
        (long_sessions,  'long_BG',  'cue_off_time',   T_MAX_LONG),
        (long_sessions,  'long_BG',  'last_lick_time', T_MAX_LONG),
    ]

    all_results = []
    data_cache  = {}
    all_counts  = {}

    for sessions, group, alignment, t_max in conditions:
        if not sessions:
            print(f"\n[{group} / {alignment}] No sessions — skipping.")
            continue

        metrics, q_spikes, q_norm_spikes, all_spikes, q_medians, counts = run_analysis(
            sessions, unit_df, alignment, group, t_max=t_max
        )

        key = (group, alignment)
        all_counts[key] = counts or {}

        if metrics:
            all_results.append({
                'cell_set':       set_label,
                'group':          group,
                'alignment':      alignment.replace('_time', ''),
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
                'n_mice':         (counts or {}).get('n_mice'),
                'n_sessions':     (counts or {}).get('n_sessions'),
                'n_trials':       (counts or {}).get('n_trials'),
            })
            data_cache[key] = {
                'metrics':       metrics,
                'q_spikes':      q_spikes,
                'q_norm_spikes': q_norm_spikes,
                'all_spikes':    all_spikes,
                'q_medians':     q_medians,
                'counts':        counts,
            }

    results_df = pd.DataFrame(all_results)
    out_enhanced.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_enhanced / 'results_summary.csv', index=False)
    print(f"\n  Summary CSV  → {out_enhanced / 'results_summary.csv'}")

    out_base.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'data_cache':  data_cache,
            'results_df':  results_df,
            'all_counts':  all_counts,
            'cell_set':    set_label,
            'source_csv':  str(csv_path),
        }, f)
    print(f"  Results cache → {cache_file}")

    return results_df


def remetric_all_caches(pair=QUARTILE_PAIR):
    """Recompute the cached ``metrics`` + ``results_df`` for ``pair`` from the
    raw per-quartile spikes already stored in each cache — no session
    re-pooling. Run after changing QUARTILE_PAIR:

        python a_rescaling.py remetric

    The cached q_spikes / q_norm_spikes hold all four quartiles, so switching
    the compared pair only needs the (cheap) metric step re-run, not the
    (overnight) session pooling.
    """
    print(f"  Re-deriving cached metrics for pair {pair[0]}→{pair[1]}")
    per_set = []
    for set_label in CELL_SETS:
        _, out_enhanced, _, cache_file = paths_for(set_label)
        if not cache_file.exists():
            print(f"  [skip] {set_label}: no cache")
            continue
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        rows = []
        for (group, alignment), entry in cache['data_cache'].items():
            t_max = T_MAX_SHORT if group == 'short_BG' else T_MAX_LONG
            metrics = compute_metrics(
                entry['q_spikes'], entry['q_norm_spikes'],
                entry['all_spikes'], entry['q_medians'],
                t_max=t_max, pair=pair,
            )
            if metrics is None:
                continue
            entry['metrics'] = metrics
            counts = entry.get('counts') or {}
            rows.append({
                'cell_set':       set_label,
                'group':          group,
                'alignment':      alignment.replace('_time', ''),
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
                'n_mice':         counts.get('n_mice'),
                'n_sessions':     counts.get('n_sessions'),
                'n_trials':       counts.get('n_trials'),
            })
        cache['results_df'] = pd.DataFrame(rows)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        out_enhanced.mkdir(parents=True, exist_ok=True)
        cache['results_df'].to_csv(out_enhanced / 'results_summary.csv',
                                   index=False)
        per_set.append(cache['results_df'])
        print(f"  remetric {set_label:<14}  {len(rows)} rows")
    if per_set:
        combined = pd.concat(per_set, ignore_index=True)
        combined.to_csv(DIR_BASE / 'results_summary_all_sets.csv', index=False)
        print(f"  Cross-set summary → {DIR_BASE / 'results_summary_all_sets.csv'}")
    print("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Plotting helpers ──────────────────────────────────────────────────────────

def _get_out_dir(label, set_dir):
    if '__' in label:
        sub = 'per_session'
    elif label == 'all':
        sub = 'all_mice'
    else:
        sub = 'per_group'
    d = set_dir / sub
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
                     t_start_col, label, quartiles, save, set_dir,
                     t_max=T_MAX_SHORT):
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
        f'[{"+".join(quartiles)}]  (n={sort_matrix.shape[0]} units)',
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
        out_dir  = _get_out_dir(label, set_dir)
        qs_suffix = '' if quartiles == QUARTILE_LABELS else f'__{"+".join(quartiles)}'
        fname    = out_dir / f'rescaling__{label}__{t_start_col}{qs_suffix}.png'
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
    else:
        plt.show()


def plot_rescaling_summary(metrics, q_medians, label, t_start_col, enhanced_dir,
                           save=True):
    """2×2 enhanced summary: scatter, scale factor dist., fraction bar, stats table."""
    if metrics is None:
        return

    tau_a = metrics['tau_a']
    tau_b = metrics['tau_b']

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(
        f'{label}  |  {t_start_col}\n'
        f'{QA} median: {q_medians.get(QA, np.nan):.2f}s  →  '
        f'{QB} median: {q_medians.get(QB, np.nan):.2f}s  '
        f'(n={metrics["n_units"]} units)',
        fontsize=12, y=0.93
    )

    # Panel A: correlation scatter
    ax = axes[0, 0]
    ax.scatter(tau_a, tau_b, s=20, alpha=0.6, edgecolors='none', c='steelblue')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect rescaling (slope=1)')
    x_fit = np.linspace(0, 1, 100)
    ax.plot(x_fit, metrics['slope'] * x_fit + metrics['intercept'], 'r-', lw=2,
            label=f'Fit: slope={metrics["slope"]:.2f}±{metrics["slope_se"]:.2f}')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel(f'Normalized peak time ({QA})')
    ax.set_ylabel(f'Normalized peak time ({QB})')
    ax.set_title(f'A. Peak time correlation\n'
                 f'r = {metrics["r"]:.3f}, '
                 f'p < {max(metrics["p_shuffle"], 0.001):.3f}')
    ax.legend(loc='lower right', fontsize=9)

    # Panel B: scale factor distribution
    ax = axes[0, 1]
    valid_mask = tau_b > 0.05
    if valid_mask.sum() > 5:
        sf = np.clip(tau_a[valid_mask] / tau_b[valid_mask], 0, 3)
        ax.hist(sf, bins=30, density=True, alpha=0.7,
                color='steelblue', edgecolor='white')
        ax.axvline(1.0, color='k', ls='--', lw=2, label='Expected (=1.0)')
        ax.axvline(metrics['scale_factor_median'], color='red', lw=2,
                   label=f'Observed: {metrics["scale_factor_median"]:.2f}')
        ax.set_xlabel(f'Scale factor (τ_{QA} / τ_{QB})')
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
        ax.set_ylim([0, 1.12])
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
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        fname = enhanced_dir / f'summary__{label}__{t_start_col}.png'
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
    else:
        plt.show()


# ── Committee figures ─────────────────────────────────────────────────────────

def find_best_config(data_cache, group):
    """Scan all (alignment × quartile pair) combinations for ``group`` and
    return the configuration with the highest peak-time correlation r.

    Returns a dict with keys:
        r, p_shuffle, n_units, anchor (display label, no _time suffix),
        alignment (full data_cache key), q_a, q_b, data (data_cache entry)
        and t_max (T_MAX_SHORT or T_MAX_LONG by group).
    Or ``None`` if no valid configuration was found.
    """
    quartile_pairs = [(qa, qb)
                      for i, qa in enumerate(QUARTILE_LABELS)
                      for qb in QUARTILE_LABELS[i + 1:]]
    t_max_for_group = T_MAX_SHORT if group == 'short_BG' else T_MAX_LONG

    best = None
    for (g, alignment), d in data_cache.items():
        if g != group:
            continue
        for qa, qb in quartile_pairs:
            result = compute_pair_r(qa, qb, d['q_norm_spikes'], d['all_spikes'])
            r = result['r']
            if not np.isfinite(r):
                continue
            if best is None or r > best['r']:
                best = {
                    'r':         r,
                    'p_shuffle': result['p_shuffle'],
                    'n_units':   result['n_units'],
                    'anchor':    alignment.replace('_time', ''),
                    'alignment': alignment,
                    'q_a':       qa,
                    'q_b':       qb,
                    'data':      d,
                    't_max':     t_max_for_group,
                }
    return best


def figure1_heatmaps(best_short, best_long, committee_dir, set_label=''):
    """Population heatmaps for the best-fit (anchor × quartile pair) per group.

    Sort order is derived from the absolute-time Qa peak so all four panels in
    a row share consistent unit ordering.

    Parameters
    ----------
    best_short, best_long : dict | None
        Output of ``find_best_config`` for the short_BG / long_BG groups.
    set_label : str
        Cell-set label shown in the suptitle.
    """
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.25)
    fig.suptitle(
        f'[{set_label}]  Best-fit population heatmaps per behavioral group\n'
        '(anchor and quartile pair selected by max r over all '
        '{anchor × Qa→Qb} combinations; rows sorted by absolute-time Qa peak)',
        fontsize=14, fontweight='bold', y=0.97
    )

    datasets = [
        ('Short BG', best_short, 0),
        ('Long BG',  best_long,  1),
    ]

    for group_name, cfg, row in datasets:
        if cfg is None:
            continue

        d         = cfg['data']
        q_spk     = d['q_spikes']
        q_norm_spk = d['q_norm_spikes']
        q_med     = d['q_medians']
        counts    = d['counts']
        q_a, q_b  = cfg['q_a'], cfg['q_b']
        anchor    = cfg['anchor']
        r_best    = cfg['r']
        t_max     = cfg['t_max']

        # Sort by absolute-time Qa peak; apply same order to all panels.
        all_unit_ids = list(q_spk[q_a].keys())
        _, sort_uids, _ = build_population_matrix(
            q_spk[q_a], all_unit_ids, 0.0, t_max, TIME_STEP, SIGMA
        )

        mat_a_abs, _, _ = build_population_matrix(
            q_spk[q_a], all_unit_ids,
            0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids
        )
        mat_b_abs, _, _ = build_population_matrix(
            q_spk[q_b], all_unit_ids,
            0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids
        )
        mat_a_norm, _, _ = build_population_matrix(
            q_norm_spk[q_a], all_unit_ids,
            0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
        )
        mat_b_norm, _, _ = build_population_matrix(
            q_norm_spk[q_b], all_unit_ids,
            0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
        )

        n_units = mat_a_abs.shape[0] if mat_a_abs is not None else cfg['n_units']

        matrices = [mat_a_abs, mat_b_abs, mat_a_norm, mat_b_norm]
        titles   = [
            f"{q_a} absolute\n(median {q_med.get(q_a, 0):.1f}s)",
            f"{q_b} absolute\n(median {q_med.get(q_b, 0):.1f}s)",
            f"{q_a} normalized", f"{q_b} normalized",
        ]
        extents  = [
            [0, t_max, 0, n_units], [0, t_max, 0, n_units],
            [0, 1, 0, n_units],     [0, 1, 0, n_units],
        ]
        xlabels = ['Time (s)', 'Time (s)', 'Normalized time', 'Normalized time']
        vlines  = [q_med.get(q_a, 0), q_med.get(q_b, 0), None, None]

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
            ax.set_xlabel(xlabels[col], fontsize=8)
            ax.set_title(titles[col], fontsize=8)
            if col == 0:
                ax.set_ylabel(
                    f'{group_name}  (anchor: {anchor.replace("_", " ")}, '
                    f'{q_a}→{q_b}  r={r_best:.2f})\n'
                    f'n={counts["n_mice"]} mice · {counts["n_sessions"]} sess · '
                    f'{n_units} units · {counts["n_trials"]:,} trials\n\n'
                    'Unit #',
                    fontsize=9
                )
            else:
                ax.set_yticklabels([])

    fig.savefig(committee_dir / 'fig1_population_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 1 → {committee_dir / 'fig1_population_heatmaps.png'}")


def figure1b_heatmaps(best_short, best_long, committee_dir, set_label=''):
    """Same as figure1_heatmaps but sorted by normalized-time Qa peak.

    Parameters
    ----------
    best_short, best_long : dict | None
        Output of ``find_best_config`` for the short_BG / long_BG groups.
    set_label : str
        Cell-set label shown in the suptitle.
    """
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.25)
    fig.suptitle(
        f'[{set_label}]  Best-fit population heatmaps per behavioral group\n'
        '(anchor and quartile pair selected by max r; '
        'rows sorted by normalized-time Qa peak)',
        fontsize=14, fontweight='bold', y=0.97
    )

    datasets = [
        ('Short BG', best_short, 0),
        ('Long BG',  best_long,  1),
    ]

    for group_name, cfg, row in datasets:
        if cfg is None:
            continue

        d          = cfg['data']
        q_spk      = d['q_spikes']
        q_norm_spk = d['q_norm_spikes']
        q_med      = d['q_medians']
        counts     = d['counts']
        q_a, q_b   = cfg['q_a'], cfg['q_b']
        anchor     = cfg['anchor']
        r_best     = cfg['r']
        t_max      = cfg['t_max']

        # Sort by normalized-time Qa peak; apply same order to all panels.
        all_unit_ids = list(q_norm_spk[q_a].keys())
        _, sort_uids, _ = build_population_matrix(
            q_norm_spk[q_a], all_unit_ids, 0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA
        )

        mat_a_abs, _, _ = build_population_matrix(
            q_spk[q_a], all_unit_ids,
            0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids
        )
        mat_b_abs, _, _ = build_population_matrix(
            q_spk[q_b], all_unit_ids,
            0.0, t_max, TIME_STEP, SIGMA, sort_uids=sort_uids
        )
        mat_a_norm, _, _ = build_population_matrix(
            q_norm_spk[q_a], all_unit_ids,
            0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
        )
        mat_b_norm, _, _ = build_population_matrix(
            q_norm_spk[q_b], all_unit_ids,
            0.0, 1.0, 1.0 / N_BINS_NORM, SIGMA, sort_uids=sort_uids
        )

        n_units = mat_a_norm.shape[0] if mat_a_norm is not None else cfg['n_units']

        matrices = [mat_a_abs, mat_b_abs, mat_a_norm, mat_b_norm]
        titles   = [
            f"{q_a} absolute\n(median {q_med.get(q_a, 0):.1f}s)",
            f"{q_b} absolute\n(median {q_med.get(q_b, 0):.1f}s)",
            f"{q_a} normalized", f"{q_b} normalized",
        ]
        extents  = [
            [0, t_max, 0, n_units], [0, t_max, 0, n_units],
            [0, 1, 0, n_units],     [0, 1, 0, n_units],
        ]
        xlabels = ['Time (s)', 'Time (s)', 'Normalized time', 'Normalized time']
        vlines  = [q_med.get(q_a, 0), q_med.get(q_b, 0), None, None]

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
            ax.set_xlabel(xlabels[col], fontsize=8)
            ax.set_title(titles[col], fontsize=8)
            if col == 0:
                ax.set_ylabel(
                    f'{group_name}  (anchor: {anchor.replace("_", " ")}, '
                    f'{q_a}→{q_b}  r={r_best:.2f})\n'
                    f'n={counts["n_mice"]} mice · {counts["n_sessions"]} sess · '
                    f'{n_units} units · {counts["n_trials"]:,} trials\n\n'
                    'Unit #',
                    fontsize=9
                )
            else:
                ax.set_yticklabels([])

    fig.savefig(committee_dir / 'fig1b_population_heatmaps_normsort.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 1b → {committee_dir / 'fig1b_population_heatmaps_normsort.png'}")


def figure2_all_alignments(results_df, committee_dir, set_label=''):
    """Bar chart comparing all alignment × group combinations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'[{set_label}]  Q3→Q4 r by alignment × group',
                 fontsize=13, fontweight='bold', y=1.02)

    alignments = ['last_lick', 'cue_on', 'cue_off']
    colors     = {'last_lick': COLOR_LAST_LICK,
                  'cue_on':    COLOR_CUE_ON,
                  'cue_off':   COLOR_CUE_OFF}
    behavioral_anchor = {'short_BG': 'last_lick', 'long_BG': 'cue_on'}
    groups   = ['short_BG', 'long_BG']
    g_labels = {'short_BG': 'Short BG', 'long_BG': 'Long BG'}

    for col, group in enumerate(groups):
        ax     = axes[col]
        subset = results_df[results_df['group'] == group]
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
        ax.axhline(0.5, color='gray', ls='--', alpha=0.4, lw=1)
        ax.set_ylim([0, 1.18])
        ax.set_ylabel('Peak time correlation (r)', fontsize=10)
        ax.set_title(g_labels[group], fontsize=11)

        n_units = subset['n_units'].values[0] if len(subset) > 0 else 0
        ax.text(0.98, 0.02, f'n={n_units} units', transform=ax.transAxes,
                fontsize=9, ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8))

    legend_elements = [
        mpatches.Patch(facecolor='gray', edgecolor='black', linewidth=3,
                       hatch='///', label='Behavioral anchor'),
        Line2D([0], [0], color='gray', linestyle='--', label='r = 0.5 (moderate)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(committee_dir / 'fig2_all_alignments.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 2 → {committee_dir / 'fig2_all_alignments.png'}")


def figure3_scatter_with_uncertainty(best_short, best_long, committee_dir, set_label=''):
    """Scatter plots with bootstrap CI bands and shuffle distribution insets,
    for the best-fit (anchor × quartile pair) per group.

    Recomputes slope/CI/r_shuffle for the chosen (q_a, q_b) on the fly, since
    the cache only stores those stats for Q3→Q4."""
    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.35, wspace=0.3)
    fig.suptitle(
        f'[{set_label}]  Rescaling quality at best-fit (anchor × Qa→Qb) per group',
        fontsize=13, y=0.97)

    datasets = [
        ('Short BG', best_short, COLOR_SHORT, 0),
        ('Long BG',  best_long,  COLOR_LONG,  1),
    ]

    for group_name, cfg, color, col in datasets:
        if cfg is None:
            continue

        d      = cfg['data']
        q_a    = cfg['q_a']
        q_b    = cfg['q_b']
        anchor = cfg['anchor']

        metrics = compute_pair_full_metrics(
            q_a, q_b, d['q_norm_spikes'], d['all_spikes']
        )
        if metrics is None:
            continue

        tau_a     = metrics['tau_a']
        tau_b     = metrics['tau_b']
        r         = metrics['r']
        slope     = metrics['slope']
        slope_se  = metrics['slope_se']
        slope_ci  = metrics['slope_ci']
        intercept = metrics['intercept']
        n         = metrics['n_units']
        p         = metrics['p_shuffle']
        r_shuffle = metrics['r_shuffle']

        title = f"{group_name} ({anchor.replace('_', ' ')}, {q_a}→{q_b})"

        ax_main = fig.add_subplot(gs[0, col])
        ax_main.scatter(tau_a, tau_b, s=40, alpha=0.6, c=color,
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
        ax_main.set_xlabel(f'Normalized peak time ({q_a})', fontsize=11)
        ax_main.set_ylabel(f'Normalized peak time ({q_b})', fontsize=11)
        p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
        ax_main.set_title(
            f'{title}\n'
            f'r = {r:.3f} ({p_str})  ·  slope = {slope:.2f} ± {slope_se:.2f}  ·  n = {n} units',
            fontsize=11)
        ax_main.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax_main.grid(True, alpha=0.3)

        ax_shuf = fig.add_subplot(gs[1, col])
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
    fig.savefig(committee_dir / 'fig3_scatter_with_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 3 → {committee_dir / 'fig3_scatter_with_uncertainty.png'}")


def figure5_summary_table(results_df, all_counts, committee_dir, set_label=''):
    """Comprehensive results table with all metrics and sample sizes."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    cols = ['Group', 'Alignment', 'n mice', 'n sess', 'n units', 'n trials',
            'r', 'p(shuffle)', 'Slope', '95% CI']
    table_data = []
    for _, row in results_df.iterrows():
        key    = (row['group'], row['alignment'] + '_time')
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
        if 'Short' in row_data[0] and 'Last' in row_data[1]:
            for j in range(len(cols)):
                table[(row_idx, j)].set_facecolor('#d5f5e3')
        if 'Long' in row_data[0] and 'Cue On' in row_data[1]:
            for j in range(len(cols)):
                table[(row_idx, j)].set_facecolor('#d6eaf8')
    ax.set_title(f'[{set_label}]  Complete results summary\n'
                 '(green = behavioral anchor for Short BG, blue = anchor for Long BG)',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(committee_dir / 'fig5_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 5 → {committee_dir / 'fig5_summary_table.png'}")


def figure6_all_quartile_pairs(data_cache, committee_dir, set_label=''):
    """Scatter plots for all 6 pairwise quartile combinations × 3 anchors,
    both groups. 6 rows (group × anchor) × 6 columns (quartile pair) = 36 panels.

    Behavioral-anchor rows (last_lick for short_BG, cue_on for long_BG) are
    highlighted with a bolded row label. The QUARTILE_PAIR column has a
    darkred border."""
    from scipy import stats as _stats

    pairs = [('Q1', 'Q2'), ('Q1', 'Q3'), ('Q1', 'Q4'),
             ('Q2', 'Q3'), ('Q2', 'Q4'), ('Q3', 'Q4')]
    anchors = [('last_lick', 'last_lick_time'),
               ('cue_on',    'cue_on_time'),
               ('cue_off',   'cue_off_time')]
    groups = [('Short BG', 'short_BG', COLOR_SHORT, 'last_lick'),
              ('Long BG',  'long_BG',  COLOR_LONG,  'cue_on')]

    # Row order: (group, anchor) pairs, group-major
    row_specs = []
    for g_label, g_key, g_color, behavioral_anchor in groups:
        for a_label, a_key in anchors:
            row_specs.append({
                'group_label':       g_label,
                'group_key':         g_key,
                'color':             g_color,
                'anchor_label':      a_label,
                'anchor_key':        a_key,
                'is_behavioral':     a_label == behavioral_anchor,
            })

    # Pre-compute best (anchor × pair) per group for highlighting.
    best_short = find_best_config(data_cache, 'short_BG')
    best_long  = find_best_config(data_cache, 'long_BG')
    best_lookup = {}  # (group_key, anchor_key, qa, qb) → True
    for g_key, cfg in [('short_BG', best_short), ('long_BG', best_long)]:
        if cfg is not None:
            best_lookup[(g_key, cfg['alignment'], cfg['q_a'], cfg['q_b'])] = True

    n_rows = len(row_specs)
    n_cols = len(pairs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    fig.suptitle(
        f'[{set_label}]  Pairwise quartile correlations × anchors — normalized peak time\n'
        '(Q3→Q4 in darkred; per-group max r in gold; behavioral anchor in bold)',
        fontsize=13, fontweight='bold', y=0.995,
    )

    for row, spec in enumerate(row_specs):
        d = data_cache.get((spec['group_key'], spec['anchor_key']))
        if not d:
            for col in range(n_cols):
                axes[row, col].set_visible(False)
            continue

        q_norm = d['q_norm_spikes']
        all_sp = d['all_spikes']

        for col, (qa, qb) in enumerate(pairs):
            ax     = axes[row, col]
            result = compute_pair_r(qa, qb, q_norm, all_sp)
            r      = result['r']
            p      = result['p_shuffle']
            tau_a  = result.get('tau_a', np.array([]))
            tau_b  = result.get('tau_b', np.array([]))

            if len(tau_a) > 1:
                ax.scatter(tau_a, tau_b, s=8, alpha=0.45, c=spec['color'],
                           edgecolors='none')
                x_fit = np.linspace(0, 1, 50)
                lr    = _stats.linregress(tau_a, tau_b)
                ax.plot(x_fit, lr.slope * x_fit + lr.intercept, 'r-', lw=1.5)

            ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
            ax.set_xlabel(f'τ {qa}', fontsize=8)
            if col == 0:
                row_lbl_weight = 'bold' if spec['is_behavioral'] else 'normal'
                ax.set_ylabel(
                    f"{spec['group_label']}\n{spec['anchor_label']}\nτ {qb}",
                    fontsize=9, fontweight=row_lbl_weight,
                )
            else:
                ax.set_ylabel(f'τ {qb}', fontsize=8)

            p_str  = 'p<.001' if p < 0.001 else f'p={p:.3f}'
            is_std  = ((qa, qb) == QUARTILE_PAIR)
            is_best = (spec['group_key'], spec['anchor_key'], qa, qb) in best_lookup
            title_weight = 'bold' if is_best else 'normal'
            ax.set_title(f'{qa}→{qb}\nr={r:.2f} ({p_str})',
                         fontsize=9, fontweight=title_weight)
            # Best-fit border wins over the QUARTILE_PAIR standard border.
            if is_best:
                for spine in ax.spines.values():
                    spine.set_edgecolor('gold')
                    spine.set_linewidth(3)
            elif is_std:
                for spine in ax.spines.values():
                    spine.set_edgecolor('darkred')
                    spine.set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(committee_dir / 'fig6_all_quartile_pairs.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Figure 6 → {committee_dir / 'fig6_all_quartile_pairs.png'}")


# ── Cross-cell-set comparison ─────────────────────────────────────────────────

def figure_compare_cell_sets(set_labels, save=True):
    """Bar chart comparing rescaling metrics across all cell-type sets.

    Loads each set's results_cache.pkl, then plots `r` faceted by
    group × alignment with one bar per cell set. Saves a comparison CSV
    alongside the figure under DIR_BASE.
    """
    rows = []
    for label in set_labels:
        _, _, _, cache_file = paths_for(label)
        if not cache_file.exists():
            print(f"  [skip] {label}: cache not found at {cache_file}")
            continue
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        df = cache['results_df'].copy()
        if 'cell_set' not in df.columns:
            df['cell_set'] = label
        rows.append(df)

    if not rows:
        print("  [SKIP] No caches available — run the analysis step first.")
        return

    combined = pd.concat(rows, ignore_index=True)

    # Save the merged table for easy off-line comparison.
    if save:
        DIR_BASE.mkdir(parents=True, exist_ok=True)
        out_csv = DIR_BASE / 'compare_cell_sets.csv'
        combined.to_csv(out_csv, index=False)
        print(f"  Saved comparison CSV → {out_csv}")

    groups     = ['short_BG', 'long_BG']
    alignments = ['last_lick', 'cue_on', 'cue_off']
    cell_sets  = [s for s in set_labels if s in combined['cell_set'].unique()]
    palette    = plt.cm.tab10(np.arange(len(cell_sets)) % 10)
    color_map  = dict(zip(cell_sets, palette))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    fig.suptitle('Rescaling correlation (r) across cell-type sets',
                 fontsize=14, fontweight='bold', y=0.98)

    for r_idx, group in enumerate(groups):
        for c_idx, alignment in enumerate(alignments):
            ax  = axes[r_idx, c_idx]
            sub = combined[(combined['group'] == group) &
                           (combined['alignment'] == alignment)]
            x       = np.arange(len(cell_sets))
            r_vals  = []
            n_units = []
            p_vals  = []
            for cs in cell_sets:
                hit = sub[sub['cell_set'] == cs]
                if len(hit):
                    r_vals.append(hit['r'].values[0])
                    n_units.append(int(hit['n_units'].values[0]))
                    p_vals.append(hit['p_shuffle'].values[0])
                else:
                    r_vals.append(np.nan)
                    n_units.append(0)
                    p_vals.append(np.nan)

            ax.bar(x, r_vals, color=[color_map[cs] for cs in cell_sets],
                   alpha=0.85, edgecolor='white', linewidth=2)
            for i, (v, n, p) in enumerate(zip(r_vals, n_units, p_vals)):
                if np.isnan(v):
                    continue
                p_str = 'p<.001' if p < 0.001 else f'p={p:.2f}'
                ax.text(i, v + 0.02, f'{v:.2f}\nn={n}\n{p_str}',
                        ha='center', fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(cell_sets, rotation=30, ha='right', fontsize=8)
            ax.axhline(0.5, color='gray', ls='--', alpha=0.4, lw=1)
            ax.set_ylim([0, 1.25])
            if c_idx == 0:
                ax.set_ylabel(f'{group}\npeak time correlation (r)', fontsize=10)
            if r_idx == 0:
                ax.set_title(alignment.replace('_', ' '), fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        out_path = DIR_BASE / 'compare_cell_sets.png'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved cross-set comparison → {out_path}")
    else:
        plt.show()


def make_plots_for_set(set_label, regenerate=False):
    """Render all per-condition + committee figures for a single cell set.

    Resolves the per-set output paths via paths_for() and passes them
    explicitly to every plot function.

    If the cell set's figures already exist (detected via the committee
    PLOTS_DONE_MARKER) and `regenerate` is False, plotting is skipped.
    Pass `regenerate=True` to force a re-render.
    """
    set_dir, enhanced_dir, committee_dir, cache_file = paths_for(set_label)

    print("\n" + "█" * 70)
    print(f"  CELL SET: {set_label}")
    print(f"  Cache:    {cache_file}")
    print(f"  Output:   {set_dir}")
    print("█" * 70)

    if not cache_file.exists():
        print(f"  [SKIP] cache not found — run the analysis step first.")
        return

    marker = committee_dir / PLOTS_DONE_MARKER
    if marker.exists() and not regenerate:
        print(f"  [CACHED] plots exist ({marker}) — pass regenerate=True to remake.")
        return

    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)

    data_cache = cache['data_cache']
    results_df = cache['results_df']
    all_counts = cache['all_counts']

    committee_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-condition heatmaps + summaries ────────────────────────────────────
    conditions = [
        ('short_BG', 'last_lick_time'),
        ('short_BG', 'cue_on_time'),
        ('short_BG', 'cue_off_time'),
        ('long_BG',  'cue_on_time'),
        ('long_BG',  'cue_off_time'),
        ('long_BG',  'last_lick_time'),
    ]
    for group, alignment in conditions:
        key = (group, alignment)
        d   = data_cache.get(key)
        if d is None:
            continue
        render_rescaling(
            d['all_spikes'], d['q_spikes'], d['q_norm_spikes'], d['q_medians'],
            alignment, group, QUARTILE_LABELS, save=True, set_dir=set_dir,
        )
        render_rescaling(
            d['all_spikes'], d['q_spikes'], d['q_norm_spikes'], d['q_medians'],
            alignment, group, list(QUARTILE_PAIR), save=True, set_dir=set_dir,
        )
        plot_rescaling_summary(
            d['metrics'], d['q_medians'], group, alignment, enhanced_dir,
            save=True,
        )

    # ── Committee figures ─────────────────────────────────────────────────────
    print("\n  COMMITTEE FIGURES")

    print("  Scanning for best (anchor × quartile pair) per group...")
    best_short = find_best_config(data_cache, 'short_BG')
    best_long  = find_best_config(data_cache, 'long_BG')
    for label, cfg in [('short_BG', best_short), ('long_BG', best_long)]:
        if cfg is None:
            print(f"    {label}: no valid configuration")
        else:
            print(f"    {label}: anchor={cfg['anchor']:<10} "
                  f"{cfg['q_a']}→{cfg['q_b']}  r={cfg['r']:.3f}  "
                  f"n={cfg['n_units']}")

    print("  Figure 1: Best-fit population heatmaps (abs-sort)...")
    if best_short or best_long:
        figure1_heatmaps(best_short, best_long, committee_dir, set_label=set_label)
    else:
        print("    → Skipping (no valid configurations)")

    print("  Figure 1b: Best-fit population heatmaps (norm-sort)...")
    if best_short or best_long:
        figure1b_heatmaps(best_short, best_long, committee_dir, set_label=set_label)
    else:
        print("    → Skipping (no valid configurations)")

    print("  Figure 2: All alignments...")
    figure2_all_alignments(results_df, committee_dir, set_label=set_label)

    print("  Figure 3: Scatter with uncertainty (best-fit)...")
    if best_short or best_long:
        figure3_scatter_with_uncertainty(best_short, best_long, committee_dir, set_label=set_label)
    else:
        print("    → Skipping (no valid configurations)")

    print("  Figure 5: Summary table...")
    figure5_summary_table(results_df, all_counts, committee_dir, set_label=set_label)

    print("  Figure 6: All quartile pairs × all anchors...")
    if data_cache:
        figure6_all_quartile_pairs(data_cache, committee_dir, set_label=set_label)
    else:
        print("    → Skipping (missing data)")

    print(f"\n  [{set_label}] Done. Figures → {set_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Parse CLI ─────────────────────────────────────────────────────────────
    # Flags toggle regeneration; the lone positional is a cell-set label or
    # the `remetric` subcommand.
    regen_analysis = REGENERATE_ANALYSIS
    regen_plots    = REGENERATE_PLOTS
    positional     = []
    for arg in sys.argv[1:]:
        if arg in ('--regenerate', '--regenerate-all'):
            regen_analysis = regen_plots = True
        elif arg == '--regenerate-analysis':
            regen_analysis = True
        elif arg == '--regenerate-plots':
            regen_plots = True
        else:
            positional.append(arg)

    # `remetric` re-derives cached metrics for QUARTILE_PAIR without re-pooling.
    if positional and positional[0] == 'remetric':
        remetric_all_caches()
        sys.exit(0)

    if positional:
        labels = [positional[0]]
        if labels[0] not in CELL_SETS:
            raise ValueError(
                f"Unknown cell set '{labels[0]}'. "
                f"Available: {list(CELL_SETS.keys())}"
            )
    else:
        labels = list(CELL_SETS.keys())

    print("=" * 70)
    print("  CROSS-REGION RESCALING — analysis + plots")
    print(f"  cell sets:  {labels}")
    print(f"  regenerate: analysis={regen_analysis}  plots={regen_plots}")
    print("=" * 70)

    # ── Pass 1/2 — analysis for every cell set ────────────────────────────────
    # All analysis runs to completion before any plotting starts, so every
    # cache is in place first and plotting never races ahead of a pending run.
    print("\n" + "─" * 70)
    print("  PASS 1/2 — ANALYSIS")
    print("─" * 70)
    per_set_results = []
    analysis_ran    = {}   # set_label -> did pooling actually (re)run this call?
    for set_label in labels:
        config = CELL_SETS[set_label]
        _, _, _, cache_file = paths_for(set_label)
        cache_existed = cache_file.exists()

        df = run_for_cell_set(set_label, config, regenerate=regen_analysis)

        # Analysis re-ran if it was forced, or no cache existed beforehand.
        analysis_ran[set_label] = regen_analysis or not cache_existed
        if df is not None and not df.empty:
            per_set_results.append(df)

    # ── Cross-set summary CSV (analysis output) ───────────────────────────────
    if per_set_results:
        DIR_BASE.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(per_set_results, ignore_index=True)
        combined_csv = DIR_BASE / 'results_summary_all_sets.csv'
        combined.to_csv(combined_csv, index=False)
        print(f"\n  Cross-set summary CSV → {combined_csv}")

    # ── Pass 2/2 — plots for every cell set ───────────────────────────────────
    # A set whose analysis re-ran in pass 1 has stale figures, so it is
    # re-rendered regardless of regen_plots.
    print("\n" + "─" * 70)
    print("  PASS 2/2 — PLOTS")
    print("─" * 70)
    for set_label in labels:
        make_plots_for_set(set_label,
                           regenerate=regen_plots or analysis_ran[set_label])

    # ── Cross-cell-set comparison figure ──────────────────────────────────────
    if len(labels) > 1:
        print("\n" + "█" * 70)
        print("  CROSS-CELL-SET COMPARISON")
        print("█" * 70)
        figure_compare_cell_sets(labels)

    print("\n" + "=" * 70)
    print("  All cell sets complete.")
    print(f"    caches  → {DIR_BASE}/<set_label>/results_cache.pkl")
    print(f"    figures → {DIR_BASE}/<set_label>/{{per_group,enhanced,committee}}/")
    print("=" * 70)
