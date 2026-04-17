"""
Population Temporal Decoder — Tier 2 MSN units
================================================
• Aligns to group-specific timing reference
    Long BG  → cue_on
    Short BG → last_lick_time
• Leave-one-trial-out Ridge regression per session
• Trial filter: all non-miss trials (bg_repeats NOT excluded — no bg carryover
  effect on waiting behavior, and excluding them reduces usable trial count
  without improving decoder validity)
• Runs on: (a) all non-miss trials, (b) Q3+Q4 only — for comparison
• Behavioral comparison variable is configurable via BEHAVIOR_VAR:
    'time_waited'       – raw wait time (start here; direct Mello analog)
    'dev_from_median'   – |time_waited - session median| (trial-type-agnostic error)
    'delta_t'           – R2M ΔT: time_waited(n+1) - time_waited(n)
                          requires trials to be sorted by trial order within session
• Outputs:
    Fig 1 – Decoding trajectory: mean decoded time vs. true time (both groups)
    Fig 2 – Decoding error over elapsed time
    Fig 3 – Per-trial decoding error vs. behavioral variable (Panel C)
    Fig 4 – Summary: mean absolute error per session, both filter modes
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─── USER: point these at your actual objects ────────────────────────────────
# spikes_df   : pd.DataFrame with columns:
#               ['session', 'unit_id', 'trial_id', 'spike_times']
#               where spike_times is a 1-D np.array of ABSOLUTE spike times (s)
#
# trials_df   : pd.DataFrame with columns:
#               ['session', 'trial_id', 'mouse_id',
#                'cue_on', 'cue_off', 'last_lick_time',
#                'time_waited',            # seconds from cue_off
#                'miss_trial', 'bg_repeats', 'bg_length']
#
# units_df    : pd.DataFrame with columns:
#               ['session', 'unit_id', 'probe_region', 'mouse_id']
#               already filtered to CP/STR units
#
# GROUP_DICT  : {'Long BG': [list of mouse_ids], 'Short BG': [...]}
# ─────────────────────────────────────────────────────────────────────────────

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BIN_SIZE    = 0.10   # s  — 100 ms bins
SMOOTH_SD   = 1.0    # bins — Gaussian kernel σ
MAX_DECODE  = 8.0    # s  — clip decoding window to this max
MIN_WAIT    = 0.3    # s  — skip bins before this (reaction-time artefact)
MIN_UNITS   = 5      # sessions with fewer STR units are skipped
MIN_TRIALS  = 15     # sessions with fewer usable trials are skipped
RIDGE_ALPHA = [0.1, 1, 10, 100, 1000, 10000]  # cross-validated
FIXED_WINDOW = 3.0   # s — Panel C: evaluate MAE only within first 3 s post-reference

# Timing reference per group
T_REF_COL   = {'Long BG': 'cue_on', 'Short BG': 'last_lick_time'}

# Behavioural comparison variable for Panel C
# Options: 'time_waited' | 'dev_from_median' | 'delta_t'
BEHAVIOR_VAR = 'time_waited'
BEHAVIOR_LABEL = {
    'time_waited'     : 'Time waited (s)',
    'dev_from_median' : '|time_waited − session median| (s)',
    'delta_t'         : 'ΔT = time_waited(n+1) − time_waited(n)  (s)',
}

# Colours
GROUP_COLOR = {'Long BG': '#2166AC', 'Short BG': '#D6604D'}
FILTER_LS   = {'all': '-', 'q3q4': '--'}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_good_trials(trials, mode='all'):
    """
    Return trial subset.
    mode: 'all'   — all non-miss trials (bg_repeats NOT filtered;
                     no bg carryover effect on waiting behavior has been found)
          'q3q4'  — top half of wait-time distribution only
    """
    good = trials[~trials['miss_trial']].copy()
    if mode == 'q3q4':
        tw = good['time_waited']
        q2 = tw.quantile(0.5)
        good = good[tw > q2]
    return good


def compute_behavior_var(good_trials, var=BEHAVIOR_VAR):
    """
    Compute per-trial behavioural comparison value.

    Parameters
    ----------
    good_trials : pd.DataFrame  already filtered trials, sorted by trial order
    var         : str  one of 'time_waited' | 'dev_from_median' | 'delta_t'

    Returns
    -------
    pd.Series indexed like good_trials.
    NaN entries arise for 'delta_t' (last trial in session) — caller should
    propagate and drop NaN rows when building scatter.
    """
    tw = good_trials['time_waited']
    if var == 'time_waited':
        return tw
    elif var == 'dev_from_median':
        return (tw - tw.median()).abs()
    elif var == 'delta_t':
        # ΔT(n) = tw(n+1) − tw(n); last trial per session → NaN
        return tw.shift(-1) - tw
    else:
        raise ValueError(f"Unknown BEHAVIOR_VAR: {var!r}. "
                         "Choose 'time_waited', 'dev_from_median', or 'delta_t'.")


def compute_fixed_window_mae(y_true, y_pred, min_t=MIN_WAIT, max_t=FIXED_WINDOW):
    """
    MAE restricted to bins within [min_t, max_t], so every trial contributes
    equally regardless of total duration (removes trial-length confound).
    Returns NaN if no bins fall in the window.
    """
    mask = (y_true >= min_t) & (y_true <= max_t)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_pred[mask] - y_true[mask]))


def bin_spikes(spike_times_abs, t_ref, t_end, bin_size=BIN_SIZE, smooth_sd=SMOOTH_SD):
    """
    Bin and smooth a single unit's spike train for one trial.

    Parameters
    ----------
    spike_times_abs : array-like  absolute spike times (s)
    t_ref           : float       timing reference (cue_on or last_lick_time)
    t_end           : float       end of wait window (= t_ref + time_waited, capped)
    Returns
    -------
    rates      : np.ndarray  shape (n_bins,)  spikes/s
    bin_edges  : np.ndarray  shape (n_bins+1,)
    """
    duration = min(t_end - t_ref, MAX_DECODE)
    if duration <= 0:
        return np.array([]), np.array([])
    edges = np.arange(0, duration + bin_size, bin_size)
    rel   = np.asarray(spike_times_abs) - t_ref
    counts, _ = np.histogram(rel, bins=edges)
    rates = gaussian_filter1d(counts.astype(float) / bin_size, sigma=smooth_sd)
    return rates, edges


def build_session_matrix(session_id, group, trials_df, spikes_df, units_df, mode):
    """
    Build the population rate matrix for one session.

    Returns
    -------
    X_all   : list of np.ndarray, one per trial, shape (n_bins_t, n_units)
    y_all   : list of np.ndarray, one per trial, shape (n_bins_t,) — elapsed time
    tw_all  : list of float, time_waited per trial
    t_ids   : list of trial ids
    """
    # ── filter trials ──────────────────────────────────────────────────────
    sess_trials = trials_df[trials_df['session'] == session_id]
    good        = get_good_trials(sess_trials, mode)
    if len(good) < MIN_TRIALS:
        return None

    t_ref_col   = T_REF_COL[group]

    # ── filter units ───────────────────────────────────────────────────────
    sess_units  = units_df[units_df['session'] == session_id]['unit_id'].values
    if len(sess_units) < MIN_UNITS:
        return None

    # ── spike lookup: {unit_id: {trial_id: spike_times}} ──────────────────
    sess_spikes = spikes_df[
        (spikes_df['session'] == session_id) &
        (spikes_df['unit_id'].isin(sess_units))
    ]
    spike_lookup = (
        sess_spikes
        .groupby(['unit_id', 'trial_id'])['spike_times']
        .first()
    )

    X_all, y_all, tw_all, t_ids, bvar_vals = [], [], [], [], []

    bvar_series = compute_behavior_var(good)

    for _, row in good.iterrows():
        tid   = row['trial_id']
        t_ref = row[t_ref_col]
        t_end = t_ref + row['time_waited']

        # ── build [n_bins, n_units] matrix for this trial ─────────────────
        trial_rates = []
        n_bins_ref  = None

        for uid in sess_units:
            try:
                st = spike_lookup.loc[(uid, tid)]
            except KeyError:
                st = np.array([])

            rates, edges = bin_spikes(st, t_ref, t_end)
            if len(rates) == 0:
                continue

            if n_bins_ref is None:
                n_bins_ref = len(rates)
            else:
                # align lengths (rounding artefact at boundaries)
                rates = rates[:n_bins_ref]
                if len(rates) < n_bins_ref:
                    rates = np.pad(rates, (0, n_bins_ref - len(rates)))

            trial_rates.append(rates)

        if not trial_rates or n_bins_ref is None or n_bins_ref < 2:
            continue

        pop_matrix = np.stack(trial_rates, axis=1)  # (n_bins, n_units)
        bin_centers = (np.arange(n_bins_ref) + 0.5) * BIN_SIZE  # elapsed time

        # skip very short waits
        valid = bin_centers >= MIN_WAIT
        if valid.sum() < 2:
            continue

        X_all.append(pop_matrix[valid])
        y_all.append(bin_centers[valid])
        tw_all.append(row['time_waited'])
        bvar_vals.append(bvar_series.loc[row.name])
        t_ids.append(tid)

    if len(X_all) < MIN_TRIALS:
        return None

    return {'X': X_all, 'y': y_all, 'tw': np.array(tw_all),
            'bvar': np.array(bvar_vals, dtype=float), 'trial_ids': t_ids,
            'n_units': len(sess_units), 'n_trials': len(X_all)}


def loto_decode(session_data):
    """
    Leave-one-trial-out Ridge regression decoder.

    Returns
    -------
    results : dict with per-trial arrays
        'y_true'        : list of true elapsed time arrays
        'y_pred'        : list of decoded elapsed time arrays
        'mae_per_trial' : mean |error| for each trial
        'bias_per_trial': mean signed error (decoded - true)
        'tw'            : time_waited per trial
    """
    X_all = session_data['X']
    y_all = session_data['y']
    n     = len(X_all)

    y_true_all, y_pred_all = [], []
    mae_trials, bias_trials = [], []

    for i in range(n):
        # training set: all trials except i
        X_train = np.vstack([X_all[j] for j in range(n) if j != i])
        y_train = np.concatenate([y_all[j] for j in range(n) if j != i])

        # test set: trial i
        X_test  = X_all[i]
        y_test  = y_all[i]

        # fit
        model = RidgeCV(alphas=RIDGE_ALPHA, fit_intercept=True)
        model.fit(X_train, y_train)

        y_hat = model.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_hat)

        err = y_hat - y_test
        mae_trials.append(np.mean(np.abs(err)))
        bias_trials.append(np.mean(err))

    return {
        'y_true'        : y_true_all,
        'y_pred'        : y_pred_all,
        'mae_per_trial' : np.array(mae_trials),
        'bias_per_trial': np.array(bias_trials),
        'tw'            : session_data['tw'],
        'bvar'          : session_data['bvar'],
    }


def loto_decode_shuffle(session_data, n_shuffles=5, seed=0):
    """
    Within-trial time-bin permutation null: for each shuffle, independently
    permute the elapsed-time labels inside every trial (keeping bin counts
    matched to their X matrices).  Between-trial label swaps cannot be used
    directly because trials have unequal durations.

    Returns the mean per-trial MAE across n_shuffles as a null baseline.
    """
    rng   = np.random.default_rng(seed)
    y_all = session_data['y']

    all_maes = []
    for _ in range(n_shuffles):
        y_shuf    = [rng.permutation(y) for y in y_all]
        shuf_data = {**session_data, 'y': y_shuf}
        res       = loto_decode(shuf_data)
        all_maes.append(res['mae_per_trial'])

    return np.mean(all_maes, axis=0)   # shape: (n_trials,)


def error_vs_time(y_true_all, y_pred_all, bin_size=BIN_SIZE, max_t=MAX_DECODE):
    """
    Compute mean signed error and MAE as a function of elapsed time,
    averaging across all trials in a session.
    Returns: bin_centers, mean_error, mean_mae
    """
    bins = np.arange(0, max_t + bin_size, bin_size)
    centers = (bins[:-1] + bins[1:]) / 2
    err_by_bin = defaultdict(list)

    for y_t, y_p in zip(y_true_all, y_pred_all):
        for t, e in zip(y_t, y_p - y_t):
            b = int(t / bin_size)
            if 0 <= b < len(centers):
                err_by_bin[b].append(e)

    mean_err = np.full(len(centers), np.nan)
    mean_mae = np.full(len(centers), np.nan)
    for b, errs in err_by_bin.items():
        mean_err[b] = np.mean(errs)
        mean_mae[b] = np.mean(np.abs(errs))

    return centers, mean_err, mean_mae


# ─── DATA LOADER ─────────────────────────────────────────────────────────────

def load_decoder_data(group_dict=None, unit_csv=None, msn_tier_csv=None, pickle_dir=None):
    """
    Build trials_df, spikes_df, units_df from per-session pickle files.

    Parameters
    ----------
    group_dict : dict
        {'Long BG': [mouse_ids], 'Short BG': [mouse_ids]}
        Defaults to constants.GROUP_DICT with 'l' → 'Long BG', 's' → 'Short BG'.
    unit_csv : str or Path, optional
        Path to RZ_unit_properties_with_qc_and_regions.csv (output of 0e script).
        If provided, only QC-passing CP/STR units are included; otherwise all units
        are used. Ignored when msn_tier_csv is provided.
    msn_tier_csv : str or Path, optional
        Path to a tier CSV from 0h_cell_type_relabeling.py (e.g. RZ_msn_waveform.csv
        for Tier 2). When provided, only units listed in this CSV are included — the
        tier CSV is already pre-filtered by waveform type, so no additional
        region/QC conditions are applied. Takes precedence over unit_csv.
    pickle_dir : str or Path, optional
        Directory containing per-session .pkl files. Defaults to paths.PICKLE_DIR.

    Returns
    -------
    trials_df, spikes_df, units_df  — as expected by run_decoder()

    Notes
    -----
    `time_waited` is set to (decision_time − t_ref) so the decode window covers
    the reference point through to the decision:
        Long BG  → decision_time − cue_on_time  (= background_length + wait_length)
        Short BG → decision_time − last_lick_time
    Trials where the reference time is NaN or the window is non-positive are
    marked as missed and filtered out by the decoder automatically.
    """
    import pickle as _pkl
    import paths as p
    import constants as c

    if pickle_dir is None:
        pickle_dir = p.PICKLE_DIR

    if group_dict is None:
        group_dict = {
            'Long BG' : c.GROUP_DICT['l'],
            'Short BG': c.GROUP_DICT['s'],
        }

    mouse_to_group = {m: g for g, mice in group_dict.items() for m in mice}
    t_ref_col_map  = {'Long BG': 'cue_on_time', 'Short BG': 'last_lick_time'}

    # Build session_id → session_key mapping from the sessions log
    sessions_log_path = p.LOGS_DIR / 'sessions_official_raw.csv'
    session_to_key = {}
    if os.path.exists(sessions_log_path):
        sess_log = pd.read_csv(sessions_log_path)
        for _, row in sess_log.iterrows():
            sid = row['id']
            key = f"{row['mouse']}|{row['date']}|{int(row['insertion_number'])}"
            session_to_key[sid] = key

    msn_df = None
    if msn_tier_csv is not None and os.path.exists(str(msn_tier_csv)):
        msn_df = pd.read_csv(msn_tier_csv)

    qc_df = None
    if msn_df is None and unit_csv is not None and os.path.exists(str(unit_csv)):
        qc_df = pd.read_csv(unit_csv)

    all_trials, all_spikes, all_units = [], [], []

    for fname in sorted(os.listdir(pickle_dir)):
        if not fname.endswith('_str.pkl'):
            continue
        session_id = fname[:-4]

        with open(os.path.join(pickle_dir, fname), 'rb') as f:
            sess = _pkl.load(f)

        mouse = sess['mouse']
        group = mouse_to_group.get(mouse)
        if group is None:
            continue

        ref_col    = t_ref_col_map[group]
        trials     = sess['trials']
        units_dict = sess['units']  # {unit_id: spikes_df}

        # ── determine which units to include ──────────────────────────────
        sess_key = session_to_key.get(session_id, '')
        if msn_df is not None:
            # Tier CSV filtered by waveform type + qc_pass_all; match by session key
            mask = msn_df['session_key'] == sess_key
            if 'qc_pass_all' in msn_df.columns:
                mask &= msn_df['qc_pass_all'] == True
            good_uids = set(msn_df.loc[mask, 'id'].tolist())
            sample_key = next(iter(units_dict), None)
            if sample_key is not None and good_uids:
                good_uids = {type(sample_key)(u) for u in good_uids}
        elif qc_df is not None:
            good_uids = set(
                qc_df.loc[
                    (qc_df['session_key'] == sess_key) &
                    (qc_df['qc_pass_all'] == True) &
                    (qc_df['region_acronym'].isin(['CP', 'STR'])),
                    'id'
                ].tolist()
            )
            # coerce type to match units_dict key type
            sample_key = next(iter(units_dict), None)
            if sample_key is not None and good_uids:
                good_uids = {type(sample_key)(u) for u in good_uids}
        else:
            good_uids = set(units_dict.keys())

        good_uids &= set(units_dict.keys())

        # ── units_df rows ──────────────────────────────────────────────────
        for uid in good_uids:
            all_units.append({
                'session'     : session_id,
                'unit_id'     : uid,
                'probe_region': 'STR',
                'mouse_id'    : mouse,
            })

        # ── spikes_df rows (one row per unit-trial pair) ───────────────────
        for uid in good_uids:
            spk_df  = units_dict[uid]
            grouped = spk_df.groupby('trial_id')['trial_time'].apply(np.array)
            for tid, st_arr in grouped.items():
                all_spikes.append({
                    'session'    : session_id,
                    'unit_id'    : uid,
                    'trial_id'   : int(tid),
                    'spike_times': st_arr,
                })

        # ── trials_df rows ─────────────────────────────────────────────────
        for _, row in trials.iterrows():
            ref_time = row.get(ref_col)
            dec_time = row.get('decision_time')

            valid_window = (
                not pd.isna(ref_time) and
                not pd.isna(dec_time) and
                (dec_time - ref_time) > 0
            )
            time_waited = (dec_time - ref_time) if valid_window else np.nan
            if not pd.isna(time_waited) and time_waited > 30:
                time_waited = np.nan
                valid_window = False
            miss_trial  = bool(row.get('missed', True)) or not valid_window

            all_trials.append({
                'session'        : session_id,
                'trial_id'       : int(row['trial_id']),
                'mouse_id'       : mouse,
                'cue_on'         : row.get('cue_on_time',   np.nan),
                'cue_off'        : row.get('cue_off_time',  np.nan),
                'last_lick_time' : row.get('last_lick_time', np.nan),
                'decision_time'  : dec_time if valid_window else np.nan,
                'time_waited'    : time_waited,
                'miss_trial'     : miss_trial,
                'bg_repeats'     : row.get('num_bg_repeat', 0),
                'bg_length'      : row.get('background_length', np.nan),
            })

    trials_df = pd.DataFrame(all_trials)
    spikes_df = pd.DataFrame(all_spikes)
    units_df  = pd.DataFrame(all_units)

    n_sess = trials_df['session'].nunique() if not trials_df.empty else 0
    unit_src = "Tier 2 MSN" if msn_df is not None else ("QC STR" if qc_df is not None else "all")
    print(f"Loaded {n_sess} sessions | {len(units_df)} {unit_src} unit records | "
          f"{len(trials_df)} trials")
    return trials_df, spikes_df, units_df


# ─── DATA OVERVIEW ───────────────────────────────────────────────────────────

def plot_data_overview(trials_df, units_df, GROUP_DICT):
    """
    Plot a summary of the dataset: sessions per mouse, units per session,
    valid trials per session, colored by group.

    Parameters
    ----------
    trials_df, units_df : DataFrames from load_decoder_data()
    GROUP_DICT : dict mapping group name → list of mouse IDs

    Returns
    -------
    fig
    """
    mouse_to_group = {m: g for g, mice in GROUP_DICT.items() for m in mice}
    group_colors   = {g: c for g, c in zip(GROUP_DICT.keys(), ['#2166AC', '#D6604D'])}

    # ── per-session summaries ─────────────────────────────────────────────
    n_units = (
        units_df.groupby('session')['unit_id']
        .nunique()
        .rename('n_units')
    )
    n_trials_all = (
        trials_df.groupby('session')['trial_id']
        .nunique()
        .rename('n_trials_all')
    )
    n_trials_valid = (
        trials_df[~trials_df['miss_trial']]
        .groupby('session')['trial_id']
        .nunique()
        .rename('n_trials_valid')
    )

    sess_df = pd.concat([n_units, n_trials_all, n_trials_valid], axis=1).fillna(0)
    sess_df.index.name = 'session'
    sess_df = sess_df.reset_index()
    sess_df['mouse']  = sess_df['session'].str.split('_').str[0]
    sess_df['group']  = sess_df['mouse'].map(mouse_to_group)
    sess_df['date']   = sess_df['session'].str.extract(r'(\d{4}-\d{2}-\d{2})')

    # Sort: group order → mouse → date
    group_order = list(GROUP_DICT.keys())
    sess_df['group_order'] = sess_df['group'].map({g: i for i, g in enumerate(group_order)})
    sess_df = sess_df.sort_values(['group_order', 'mouse', 'date']).reset_index(drop=True)

    # ── per-mouse summaries ───────────────────────────────────────────────
    mouse_df = (
        sess_df.groupby(['mouse', 'group'])
        .agg(n_sessions=('session', 'count'),
             total_units=('n_units', 'sum'),
             mean_units=('n_units', 'mean'),
             total_valid_trials=('n_trials_valid', 'sum'))
        .reset_index()
    )
    mouse_df['group_order'] = mouse_df['group'].map({g: i for i, g in enumerate(group_order)})
    mouse_df = mouse_df.sort_values(['group_order', 'mouse']).reset_index(drop=True)

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    ax_units  = fig.add_subplot(gs[0, :2])   # units per session — wide
    ax_trials = fig.add_subplot(gs[1, :2])   # valid trials per session — wide
    ax_mouse  = fig.add_subplot(gs[0, 2])    # sessions per mouse
    ax_summary= fig.add_subplot(gs[1, 2])    # text summary

    x     = np.arange(len(sess_df))
    colors = [group_colors[g] for g in sess_df['group']]

    # Panel 1 — units per session
    bars = ax_units.bar(x, sess_df['n_units'], color=colors, alpha=0.85, edgecolor='none')
    ax_units.set_xticks(x)
    ax_units.set_xticklabels(sess_df['mouse'], rotation=45, ha='right', fontsize=7)
    ax_units.set_ylabel('Units per session')
    ax_units.set_title('Units per session (Tier 2 MSN)')
    ax_units.axhline(sess_df['n_units'].mean(), color='k', lw=1, ls='--',
                     label=f'Mean = {sess_df["n_units"].mean():.1f}')
    ax_units.legend(fontsize=8)

    # Add mouse boundary lines
    mouse_boundaries = sess_df.groupby('mouse', sort=False).apply(lambda g: g.index[-1]).values
    for b in mouse_boundaries[:-1]:
        ax_units.axvline(b + 0.5, color='gray', lw=0.5, ls=':')

    # Group labels at top
    for g, grp_df in sess_df.groupby('group', sort=False):
        mid = grp_df.index.values.mean()
        ax_units.text(mid, ax_units.get_ylim()[1] if ax_units.get_ylim()[1] > 0 else 1,
                      g, ha='center', va='bottom', fontsize=8,
                      color=group_colors[g], fontweight='bold')

    # Panel 2 — valid trials per session
    ax_trials.bar(x, sess_df['n_trials_valid'], color=colors, alpha=0.85,
                  edgecolor='none', label='Valid')
    ax_trials.bar(x, sess_df['n_trials_all'] - sess_df['n_trials_valid'],
                  bottom=sess_df['n_trials_valid'], color=colors, alpha=0.25,
                  edgecolor='none', label='Missed')
    ax_trials.set_xticks(x)
    ax_trials.set_xticklabels(sess_df['mouse'], rotation=45, ha='right', fontsize=7)
    ax_trials.set_ylabel('Trials per session')
    ax_trials.set_title('Trials per session (solid = valid, faded = missed)')
    ax_trials.legend(fontsize=8)
    for b in mouse_boundaries[:-1]:
        ax_trials.axvline(b + 0.5, color='gray', lw=0.5, ls=':')

    # Panel 3 — sessions and total units per mouse
    mx = np.arange(len(mouse_df))
    mcolors = [group_colors[g] for g in mouse_df['group']]
    ax2 = ax_mouse.twinx()
    ax_mouse.bar(mx - 0.2, mouse_df['n_sessions'], width=0.35,
                 color=mcolors, alpha=0.85, edgecolor='none', label='Sessions')
    ax2.bar(mx + 0.2, mouse_df['mean_units'], width=0.35,
            color=mcolors, alpha=0.4, edgecolor='none', label='Mean units/sess')
    ax_mouse.set_xticks(mx)
    ax_mouse.set_xticklabels(mouse_df['mouse'], rotation=45, ha='right', fontsize=7)
    ax_mouse.set_ylabel('# Sessions', fontsize=9)
    ax2.set_ylabel('Mean units/session', fontsize=9)
    ax_mouse.set_title('Per-mouse overview')
    lines1, labels1 = ax_mouse.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_mouse.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')

    # Panel 4 — text summary
    ax_summary.axis('off')
    total_sess  = len(sess_df)
    total_units = int(sess_df['n_units'].sum())
    total_valid = int(sess_df['n_trials_valid'].sum())
    total_all   = int(sess_df['n_trials_all'].sum())

    lines = ['Dataset Summary', '=' * 32, '']
    for g in group_order:
        gdf = sess_df[sess_df['group'] == g]
        gmdf = mouse_df[mouse_df['group'] == g]
        lines += [
            f'{g}',
            f'  Mice:     {len(gmdf)}',
            f'  Sessions: {len(gdf)}',
            f'  Units:    {int(gdf["n_units"].sum())} '
            f'(μ={gdf["n_units"].mean():.1f}/sess)',
            f'  Valid Δ:  {int(gdf["n_trials_valid"].sum())} trials',
            '',
        ]
    lines += [
        '─' * 32,
        f'Total sessions:  {total_sess}',
        f'Total units:     {total_units}',
        f'Total trials:    {total_all}  ({total_valid} valid)',
    ]
    ax_summary.text(0.05, 0.97, '\n'.join(lines), transform=ax_summary.transAxes,
                    fontsize=9, va='top', fontfamily='monospace')

    # Group color legend
    from matplotlib.patches import Patch
    handles = [Patch(color=group_colors[g], label=g) for g in group_order]
    fig.legend(handles=handles, loc='upper right', fontsize=9, title='Group',
               bbox_to_anchor=(0.98, 0.98))

    plt.suptitle('Dataset Overview', fontsize=13, fontweight='bold')
    return fig


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def run_decoder(trials_df, spikes_df, units_df, GROUP_DICT):
    """
    Run per-session LOTO decoder for both groups and both trial filters.

    Returns
    -------
    results : nested dict
        results[group][mode] = list of per-session dicts, each with:
            'decoded'  : output of loto_decode()
            'err_time' : (centers, mean_err, mean_mae) from error_vs_time()
            'n_units', 'n_trials'
    """
    results = {}

    for group, mice in GROUP_DICT.items():
        results[group] = {}
        group_sessions = trials_df[trials_df['mouse_id'].isin(mice)]['session'].unique()

        for mode in ('all', 'q3q4'):
            session_results = []

            for sess in group_sessions:
                data = build_session_matrix(sess, group, trials_df, spikes_df,
                                            units_df, mode)
                if data is None:
                    continue

                decoded      = loto_decode(data)
                err_time     = error_vs_time(decoded['y_true'], decoded['y_pred'])
                shuffle_mae  = loto_decode_shuffle(data)

                session_results.append({
                    'session'    : sess,
                    'decoded'    : decoded,
                    'err_time'   : err_time,
                    'shuffle_mae': shuffle_mae,
                    'n_units'    : data['n_units'],
                    'n_trials'   : data['n_trials'],
                })

            results[group][mode] = session_results
            print(f"{group} | {mode:9s} | {len(session_results)} sessions decoded")

    return results


# ─── FIGURES ─────────────────────────────────────────────────────────────────

def plot_decoder_results(results):
    """
    Four-panel summary figure.
    Call after run_decoder().
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    ax_traj   = fig.add_subplot(gs[0, 0])  # decoded vs true time
    ax_err_t  = fig.add_subplot(gs[0, 1])  # MAE over elapsed time
    ax_corr   = fig.add_subplot(gs[1, 0])  # decoding error vs time_waited
    ax_summ   = fig.add_subplot(gs[1, 1])  # mean MAE per session, both modes

    # ── Panel A: mean decoded time vs true time ──────────────────────────
    for group in results:
        c = GROUP_COLOR[group]
        for mode, ls in FILTER_LS.items():
            sess_list = results[group][mode]
            if not sess_list:
                continue

            # pool all (y_true, y_pred) pairs, then bin
            all_true, all_pred = [], []
            for s in sess_list:
                for yt, yp in zip(s['decoded']['y_true'], s['decoded']['y_pred']):
                    all_true.extend(yt)
                    all_pred.extend(yp)

            all_true  = np.array(all_true)
            all_pred  = np.array(all_pred)
            bins      = np.arange(0, MAX_DECODE + BIN_SIZE, BIN_SIZE)
            bcs       = (bins[:-1] + bins[1:]) / 2
            mean_pred = np.full(len(bcs), np.nan)
            sem_pred  = np.full(len(bcs), np.nan)
            for bi in range(len(bcs)):
                mask = (all_true >= bins[bi]) & (all_true < bins[bi+1])
                if mask.sum() > 5:
                    mean_pred[bi] = np.mean(all_pred[mask])
                    sem_pred[bi]  = np.std(all_pred[mask]) / np.sqrt(mask.sum())

            valid = ~np.isnan(mean_pred)
            label = f"{group} ({mode.replace('_',' ')})"
            ax_traj.plot(bcs[valid], mean_pred[valid], color=c, ls=ls,
                         lw=1.8, label=label, alpha=0.85)
            ax_traj.fill_between(bcs[valid],
                                  mean_pred[valid] - sem_pred[valid],
                                  mean_pred[valid] + sem_pred[valid],
                                  color=c, alpha=0.12)

    ax_traj.plot([0, MAX_DECODE], [0, MAX_DECODE], 'k--', lw=1, label='Identity')
    ax_traj.set_xlabel('True elapsed time (s)')
    ax_traj.set_ylabel('Decoded elapsed time (s)')
    ax_traj.set_title('A  Decoded vs. true time')
    ax_traj.legend(fontsize=7, loc='upper left')
    ax_traj.set_xlim(0, MAX_DECODE); ax_traj.set_ylim(0, MAX_DECODE)

    # ── Panel B: MAE over elapsed time ───────────────────────────────────
    for group in results:
        c = GROUP_COLOR[group]
        for mode, ls in FILTER_LS.items():
            sess_list = results[group][mode]
            if not sess_list:
                continue

            all_mae_by_bin = defaultdict(list)
            for s in sess_list:
                centers, _, mae_t = s['err_time']
                for bi, m in enumerate(mae_t):
                    if not np.isnan(m):
                        all_mae_by_bin[bi].append(m)

            if not all_mae_by_bin:
                continue
            max_b = max(all_mae_by_bin.keys()) + 1
            mean_mae = np.full(max_b, np.nan)
            sem_mae  = np.full(max_b, np.nan)
            for bi, vals in all_mae_by_bin.items():
                if len(vals) > 1:
                    mean_mae[bi] = np.mean(vals)
                    sem_mae[bi]  = np.std(vals) / np.sqrt(len(vals))

            centers = np.arange(max_b) * BIN_SIZE + BIN_SIZE / 2
            valid   = ~np.isnan(mean_mae)
            ax_err_t.plot(centers[valid], mean_mae[valid], color=c, ls=ls,
                          lw=1.8, alpha=0.85)
            ax_err_t.fill_between(centers[valid],
                                   mean_mae[valid] - sem_mae[valid],
                                   mean_mae[valid] + sem_mae[valid],
                                   color=c, alpha=0.12)

    # shuffle baselines — flat horizontal dashed lines per group
    for group in results:
        c = GROUP_COLOR[group]
        sess_list = results[group].get('all', [])
        if not sess_list:
            continue
        mean_shuf = np.mean([np.mean(s['shuffle_mae']) for s in sess_list])
        ax_err_t.axhline(mean_shuf, color=c, ls=':', lw=1.5, alpha=0.7,
                         label=f'{group} shuffle')

    ax_err_t.axhline(0, color='k', lw=0.8, ls='--')
    ax_err_t.set_xlabel('Elapsed time since reference (s)')
    ax_err_t.set_ylabel('Mean absolute decoding error (s)')
    ax_err_t.set_title('B  Decoding error over time')
    ax_err_t.set_xlim(0, 5)  # truncate: data sparse beyond ~5 s

    # ── Panel C: fixed-window decoding error vs behavioral variable ───────
    # MAE is computed only over [MIN_WAIT, FIXED_WINDOW] so every trial
    # contributes equally regardless of total duration (removes length confound).
    ax_corr.axhline(0, color='k', lw=0.8, ls='--')
    for group in results:
        c = GROUP_COLOR[group]
        sess_list = results[group].get('all', [])
        if not sess_list:
            continue

        all_bvar, all_mae_fixed = [], []
        for s in sess_list:
            decoded    = s['decoded']
            y_true_list = decoded['y_true']
            y_pred_list = decoded['y_pred']
            bvar        = decoded['bvar']

            for i, (yt, yp) in enumerate(zip(y_true_list, y_pred_list)):
                mae_i = compute_fixed_window_mae(yt, yp)
                if not np.isnan(mae_i) and not np.isnan(bvar[i]):
                    all_bvar.append(bvar[i])
                    all_mae_fixed.append(mae_i)

        all_bvar      = np.array(all_bvar)
        all_mae_fixed = np.array(all_mae_fixed)

        # filter implausible time_waited values (coordinate-mismatch trials)
        valid         = all_bvar < 15
        all_bvar      = all_bvar[valid]
        all_mae_fixed = all_mae_fixed[valid]

        if len(all_bvar) < 10:
            continue

        # scatter (semi-transparent, small)
        ax_corr.scatter(all_bvar, all_mae_fixed, c=c, s=8, alpha=0.3, linewidths=0)

        # running mean
        order   = np.argsort(all_bvar)
        bvar_s  = all_bvar[order]
        mae_s   = all_mae_fixed[order]
        win     = max(1, len(bvar_s) // 20)
        run_mae = np.convolve(mae_s, np.ones(win)/win, mode='valid')
        run_bvar = bvar_s[win//2 : win//2 + len(run_mae)]
        ax_corr.plot(run_bvar, run_mae, color=c, lw=2.0, label=group)

        # correlation
        r, p = pearsonr(all_bvar, all_mae_fixed)
        y_pos = 0.95 if group == 'Long BG' else 0.88
        ax_corr.text(0.97, y_pos, f"{group}: r={r:.2f}, p={p:.3f}",
                     transform=ax_corr.transAxes, ha='right', fontsize=9, color=c)

    ax_corr.set_xlabel(BEHAVIOR_LABEL.get(BEHAVIOR_VAR, BEHAVIOR_VAR))
    ax_corr.set_ylabel(f'MAE in first {FIXED_WINDOW:.0f}s (s)')
    ax_corr.set_title(f'C  Decoding error (0–{FIXED_WINDOW:.0f}s) vs. {BEHAVIOR_VAR}')
    ax_corr.legend(fontsize=8, loc='upper left')
    ax_corr.set_xlim(0, 12)

    # ── Panel D: session-level MAE, both modes ────────────────────────────
    positions = {'Long BG': {'all': 1, 'q3q4': 2},
                 'Short BG': {'all': 4, 'q3q4': 5}}
    labels    = {1: 'Long BG\nall non-miss', 2: 'Long BG\nQ3+Q4',
                 4: 'Short BG\nall non-miss', 5: 'Short BG\nQ3+Q4'}

    SHUF_OFFSET = 0.3
    for group in results:
        c = GROUP_COLOR[group]
        for mode, pos in positions[group].items():
            sess_list = results[group].get(mode, [])
            if not sess_list:
                continue
            # per-session mean MAE (real)
            session_maes = [np.mean(s['decoded']['mae_per_trial']) for s in sess_list]
            ax_summ.scatter([pos] * len(session_maes), session_maes,
                            color=c, s=30, alpha=0.6, zorder=3,
                            ls=FILTER_LS[mode])
            ax_summ.errorbar(pos, np.mean(session_maes),
                             yerr=np.std(session_maes)/np.sqrt(len(session_maes)),
                             fmt='o', color=c, markersize=8, linewidth=2, zorder=4,
                             capsize=4)
            # per-session mean MAE (shuffle baseline)
            shuf_maes = [np.mean(s['shuffle_mae']) for s in sess_list]
            ax_summ.scatter([pos + SHUF_OFFSET] * len(shuf_maes), shuf_maes,
                            facecolors='none', edgecolors='gray', s=30, alpha=0.6,
                            zorder=3)
            ax_summ.errorbar(pos + SHUF_OFFSET, np.mean(shuf_maes),
                             yerr=np.std(shuf_maes)/np.sqrt(len(shuf_maes)),
                             fmt='o', color='gray', markersize=8, linewidth=2,
                             zorder=4, capsize=4, mfc='none')

    ax_summ.set_xticks(list(labels.keys()))
    ax_summ.set_xticklabels(list(labels.values()), fontsize=8)
    ax_summ.set_ylabel('Mean absolute decoding error (s)')
    ax_summ.set_title('D  Session-level MAE\n(solid=all, dashed=Q3+Q4)')
    ax_summ.axvline(3, color='gray', lw=0.8, ls=':')

    # ── custom legend for line styles ─────────────────────────────────────
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='gray', ls='-',  lw=2, label='All non-miss trials'),
        Line2D([0], [0], color='gray', ls='--', lw=2, label='Q3+Q4 only'),
        Line2D([0], [0], color='gray', ls=':',  lw=1.5, label='Shuffle baseline (B)'),
        Line2D([0], [0], marker='o', color='gray', lw=0, markersize=7,
               markerfacecolor='none', label='Shuffle baseline (D)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    plt.suptitle('DMS Population Decoder — Tier 2 MSN units (waveform-primary)',
                 fontsize=12, fontweight='bold', y=1.01)
    return fig


# ─── MULTI-ANCHOR COMPARISON ─────────────────────────────────────────────────

ANCHORS = {
    'cue_on'   : 'cue_on',
    'cue_off'  : 'cue_off',
    'last_lick': 'last_lick_time',
}


def build_session_matrix_anchor(session_id, group, trials_df, spikes_df,
                                 units_df, mode, anchor_key):
    """
    Same as build_session_matrix but with configurable alignment anchor.

    anchor_key : str  one of 'cue_on', 'cue_off', 'last_lick'
    """
    sess_trials = trials_df[trials_df['session'] == session_id]
    good        = get_good_trials(sess_trials, mode)
    if len(good) < MIN_TRIALS:
        return None

    anchor_col = ANCHORS[anchor_key]

    sess_units = units_df[units_df['session'] == session_id]['unit_id'].values
    if len(sess_units) < MIN_UNITS:
        return None

    sess_spikes = spikes_df[
        (spikes_df['session'] == session_id) &
        (spikes_df['unit_id'].isin(sess_units))
    ]
    spike_lookup = sess_spikes.groupby(['unit_id', 'trial_id'])['spike_times'].first()

    X_all, y_all, tw_all, t_ids = [], [], [], []

    for _, row in good.iterrows():
        tid   = row['trial_id']
        t_ref = row.get(anchor_col, np.nan)
        if pd.isna(t_ref):
            continue
        t_end = row.get('decision_time', np.nan)
        if pd.isna(t_end) or t_end <= t_ref:
            continue

        trial_rates = []
        n_bins_ref  = None

        for uid in sess_units:
            try:
                st = spike_lookup.loc[(uid, tid)]
            except KeyError:
                st = np.array([])

            rates, _ = bin_spikes(st, t_ref, t_end)
            if len(rates) == 0:
                continue

            if n_bins_ref is None:
                n_bins_ref = len(rates)
            else:
                rates = rates[:n_bins_ref]
                if len(rates) < n_bins_ref:
                    rates = np.pad(rates, (0, n_bins_ref - len(rates)))

            trial_rates.append(rates)

        if not trial_rates or n_bins_ref is None or n_bins_ref < 2:
            continue

        pop_matrix  = np.stack(trial_rates, axis=1)
        bin_centers = (np.arange(n_bins_ref) + 0.5) * BIN_SIZE

        valid = bin_centers >= MIN_WAIT
        if valid.sum() < 2:
            continue

        X_all.append(pop_matrix[valid])
        y_all.append(bin_centers[valid])
        tw_all.append(t_end - t_ref)
        t_ids.append(tid)

    if len(X_all) < MIN_TRIALS:
        return None

    return {'X': X_all, 'y': y_all, 'tw': np.array(tw_all),
            'bvar': np.full(len(X_all), np.nan),
            'trial_ids': t_ids, 'n_units': len(sess_units), 'n_trials': len(X_all)}


def run_anchor_comparison(trials_df, spikes_df, units_df, GROUP_DICT):
    """
    Run LOTO decoder for every (group, anchor) combination.

    Returns
    -------
    results : dict
        results[group][anchor] = {
            'mae_sessions': list of per-session mean MAE,
            'mean_mae', 'sem_mae', 'n_sessions'
        }
    """
    results = {}

    for group, mice in GROUP_DICT.items():
        results[group] = {}
        group_sessions = trials_df[trials_df['mouse_id'].isin(mice)]['session'].unique()

        for anchor_key in ANCHORS:
            session_maes = []

            for sess in group_sessions:
                data = build_session_matrix_anchor(
                    sess, group, trials_df, spikes_df, units_df,
                    mode='all', anchor_key=anchor_key,
                )
                if data is None:
                    continue
                decoded = loto_decode(data)
                session_maes.append(np.mean(decoded['mae_per_trial']))

            if session_maes:
                results[group][anchor_key] = {
                    'mae_sessions': session_maes,
                    'mean_mae'    : np.mean(session_maes),
                    'sem_mae'     : np.std(session_maes) / np.sqrt(len(session_maes)),
                    'n_sessions'  : len(session_maes),
                }
            print(f"{group:10s} | {anchor_key:12s} | "
                  f"MAE = {np.mean(session_maes):.3f} ± "
                  f"{np.std(session_maes)/np.sqrt(len(session_maes)):.3f} "
                  f"({len(session_maes)} sessions)")

    return results


def plot_anchor_comparison(results):
    """
    Bar plot: MAE by alignment anchor, grouped by Long/Short BG.
    A gold star marks the predicted best anchor for each group
    (cue_on for Long BG, last_lick for Short BG).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    anchors = list(ANCHORS.keys())
    x       = np.arange(len(anchors))
    width   = 0.35

    for i, (group, color) in enumerate([('Long BG', '#2166AC'), ('Short BG', '#D6604D')]):
        means = [results[group].get(a, {}).get('mean_mae', np.nan) for a in anchors]
        sems  = [results[group].get(a, {}).get('sem_mae',  0)      for a in anchors]

        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=sems,
               label=group, color=color, alpha=0.8, capsize=4)

        predicted_best = 'cue_on' if group == 'Long BG' else 'last_lick'
        best_idx = anchors.index(predicted_best)
        ax.scatter(best_idx + offset, means[best_idx] - 0.05,
                   marker='*', s=150, color='gold', edgecolor='k', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(anchors)
    ax.set_ylabel('Mean absolute decoding error (s)')
    ax.set_xlabel('Timing anchor')
    ax.set_title('Decoding accuracy by alignment anchor\n(★ = predicted best anchor)')
    ax.legend()
    ax.set_ylim(bottom=0)

    return fig


# ─── PCA TRAJECTORY CHECKS ───────────────────────────────────────────────────

def plot_single_trial_trajectories(session_data, n_trials_to_plot=15, seed=42):
    """
    Plot individual trial trajectories in PC space, colored by elapsed time.

    Reveals whether single trials track time better than the average suggests.
    """
    from sklearn.decomposition import PCA
    import matplotlib.cm as cm

    X_all = session_data['X']
    y_all = session_data['y']

    X_stacked = np.vstack(X_all)
    y_stacked = np.concatenate(y_all)

    pca = PCA(n_components=3)
    X_pca_all = pca.fit_transform(X_stacked)
    var_explained = pca.explained_variance_ratio_
    print(f"PC1: {var_explained[0]*100:.1f}%, PC2: {var_explained[1]*100:.1f}%, "
          f"PC3: {var_explained[2]*100:.1f}%")

    # Reconstruct per-trial PC trajectories from the stacked projection
    trial_pcs = []
    cumsum = 0
    for X_trial in X_all:
        n_bins = X_trial.shape[0]
        trial_pcs.append(X_pca_all[cumsum:cumsum + n_bins])
        cumsum += n_bins

    rng = np.random.default_rng(seed)
    trial_idx = rng.choice(len(X_all), min(n_trials_to_plot, len(X_all)), replace=False)

    max_time = max(y.max() for y in y_all)
    cmap = cm.viridis

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: single-trial trajectories in PC1–PC2
    ax1 = axes[0]
    for i in trial_idx:
        pc  = trial_pcs[i]
        t   = y_all[i]
        ax1.scatter(pc[:, 0], pc[:, 1], c=t, cmap='viridis',
                    s=15, alpha=0.6, vmin=0, vmax=max_time)
        ax1.plot(pc[:, 0], pc[:, 1], 'k-', alpha=0.15, lw=0.5)
        ax1.scatter(pc[0, 0], pc[0, 1], c='green', s=40,
                    marker='o', edgecolor='k', zorder=5, alpha=0.7)
    ax1.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
    ax1.set_title(f'Single-trial trajectories (n={len(trial_idx)})\nGreen = trial start')

    # Panel 2: trial-averaged trajectory
    ax2 = axes[1]
    time_bins      = np.arange(0, max_time + 0.2, 0.2)
    mean_trajectory, mean_times = [], []
    for t0, t1 in zip(time_bins[:-1], time_bins[1:]):
        mask = (y_stacked >= t0) & (y_stacked < t1)
        if mask.sum() > 10:
            mean_trajectory.append(X_pca_all[mask].mean(axis=0))
            mean_times.append((t0 + t1) / 2)
    mean_trajectory = np.array(mean_trajectory)
    mean_times      = np.array(mean_times)

    sc = ax2.scatter(mean_trajectory[:, 0], mean_trajectory[:, 1],
                     c=mean_times, cmap='viridis', s=60, edgecolor='k',
                     linewidth=0.5, vmin=0, vmax=max_time)
    ax2.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], 'k-', alpha=0.4, lw=1.5)
    ax2.scatter(mean_trajectory[0, 0], mean_trajectory[0, 1], c='green', s=100,
                marker='o', edgecolor='k', zorder=5)
    ax2.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
    ax2.set_title('Trial-averaged trajectory\nGreen = t=0')
    plt.colorbar(sc, ax=ax2).set_label('Elapsed time (s)')

    # Panel 3: PC1 vs elapsed time
    ax3 = axes[2]
    for i in trial_idx:
        ax3.plot(y_all[i], trial_pcs[i][:, 0], 'b-', alpha=0.15, lw=0.8)
    ax3.plot(mean_times, mean_trajectory[:, 0], 'b-', lw=2.5, label='Trial average')
    ax3.scatter(mean_times, mean_trajectory[:, 0], c='blue', s=30, edgecolor='k', zorder=5)
    ax3.set_xlabel('Elapsed time (s)')
    ax3.set_ylabel(f'PC1 ({var_explained[0]*100:.1f}%)')
    ax3.set_title('PC1 vs. elapsed time\n(thin = single trials, thick = average)')
    ax3.legend()

    plt.tight_layout()
    return fig, pca


def plot_trajectory_spread(session_data, time_points=None):
    """
    At fixed time points, show the spread of PC locations across trials.

    If spread is large, the average trajectory understates single-trial precision.
    """
    from sklearn.decomposition import PCA
    from matplotlib.patches import Ellipse

    if time_points is None:
        time_points = [1.0, 2.0, 3.0, 4.0]

    X_all = session_data['X']
    y_all = session_data['y']

    pca = PCA(n_components=2)
    pca.fit(np.vstack(X_all))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(time_points)))

    for t_target, color in zip(time_points, colors):
        pc_at_t = []
        for X_trial, y_trial in zip(X_all, y_all):
            idx = np.argmin(np.abs(y_trial - t_target))
            if np.abs(y_trial[idx] - t_target) < 0.15:
                pc_at_t.append(pca.transform(X_trial[idx:idx+1])[0])

        if len(pc_at_t) < 5:
            continue

        pc_at_t = np.array(pc_at_t)
        ax.scatter(pc_at_t[:, 0], pc_at_t[:, 1], color=color, s=40,
                   alpha=0.6, label=f't = {t_target}s (n={len(pc_at_t)})')

        cov = np.cov(pc_at_t.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle  = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(np.abs(eigenvalues))
        ax.add_patch(Ellipse(xy=pc_at_t.mean(axis=0), width=width, height=height,
                             angle=angle, facecolor=color, alpha=0.2,
                             edgecolor=color, lw=2))

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Population state spread at fixed time points\n(ellipse = 1 SD)')
    ax.legend()
    ax.set_aspect('equal')
    return fig


# ─── RUN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import paths as p
    import constants as c

    OUT_DIR = p.DATA_DIR / 'population_decoding'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    GROUP_DICT_DECODER = {
        'Long BG' : c.GROUP_DICT['l'],
        'Short BG': c.GROUP_DICT['s'],
    }

    TIER2_CSV = p.LOGS_DIR / 'RZ_msn_waveform.csv'  # Tier 2 MSN — output of 0h

    trials_df, spikes_df, units_df = load_decoder_data(
        group_dict=GROUP_DICT_DECODER,
        msn_tier_csv=TIER2_CSV,
    )

    fig_overview = plot_data_overview(trials_df, units_df, GROUP_DICT_DECODER)
    fig_overview.savefig(OUT_DIR / 'data_overview.pdf', bbox_inches='tight')
    plt.close(fig_overview)

    results = run_decoder(trials_df, spikes_df, units_df, GROUP_DICT_DECODER)
    fig_main = plot_decoder_results(results)
    fig_main.savefig(OUT_DIR / 'decoder_results_tier2_msn.pdf', bbox_inches='tight')

    anchor_results = run_anchor_comparison(trials_df, spikes_df, units_df, GROUP_DICT_DECODER)
    fig_anchor = plot_anchor_comparison(anchor_results)
    fig_anchor.savefig(OUT_DIR / 'decoder_anchor_comparison_tier2_msn.pdf', bbox_inches='tight')

    # ── PCA trajectory checks: one representative session per group ───────
    for group, mice in GROUP_DICT_DECODER.items():
        group_sessions = trials_df[trials_df['mouse_id'].isin(mice)]['session'].unique()

        # pick session with most usable trials
        best_sess, best_data = None, None
        for sess in group_sessions:
            d = build_session_matrix(sess, group, trials_df, spikes_df, units_df, mode='all')
            if d is not None and (best_data is None or d['n_trials'] > best_data['n_trials']):
                best_sess, best_data = sess, d

        if best_data is None:
            continue

        group_tag = group.replace(' ', '_').lower()
        print(f"\n{group} — PCA checks on session {best_sess} "
              f"({best_data['n_trials']} trials, {best_data['n_units']} units)")

        fig_traj, _ = plot_single_trial_trajectories(best_data)
        fig_traj.suptitle(f'{group} — {best_sess}', fontsize=11, fontweight='bold')
        fig_traj.savefig(OUT_DIR / f'pca_trajectories_{group_tag}.pdf', bbox_inches='tight')

        fig_spread = plot_trajectory_spread(best_data)
        fig_spread.suptitle(f'{group} — {best_sess}', fontsize=11, fontweight='bold')
        fig_spread.savefig(OUT_DIR / f'pca_spread_{group_tag}.pdf', bbox_inches='tight')

    plt.show()