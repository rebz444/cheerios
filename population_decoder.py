"""
population_decoder.py
─────────────────────
Run the full population-clock decoder pipeline on every session that meets
the inclusion thresholds:
    • > MIN_UNITS  MSN units (from RZ_str_msn.csv)
    • > MIN_TRIALS valid (non-miss) trials

For each qualifying session the pipeline runs:
  1. Build population firing-rate matrices (Ridge regression)
  2. Shuffle control
  3. Per-trial clock-speed extraction
  4. History-effect analysis (clock speed ~ previous outcome)
  5. Time-matched confound control
  6. PCA trajectory visualisation

Output (per session, saved to OUT_DIR/results/<session>/):
  decoder_performance_<session>.png
  clock_speed_analysis_<session>.png
  pca_trajectories_<session>.png
  trial_results_<session>.csv
"""

import pickle
import warnings
from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

warnings.filterwarnings('ignore')

import os

import constants as c
import paths as p

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR = p.DATA_DIR / 'population_decoding'
MSN_CSV = p.LOGS_DIR / 'RZ_str_msn.csv'  # MSN units — output of 0g_cell_type_relabeling

# Inclusion thresholds
MIN_UNITS  = 15
MIN_TRIALS = 150

GROUP_DICT = {
    'Long BG' : c.GROUP_DICT['l'],
    'Short BG': c.GROUP_DICT['s'],
}

# Per-group timing anchor and plot colour
GROUP_CONFIG = {
    'Short BG': {'t_ref_col': 'last_lick_time', 'color': '#ffb400'},
    'Long BG':  {'t_ref_col': 'cue_on',          'color': '#9080ff'},
}

# Decoder parameters
BIN_SIZE    = 0.10
SMOOTH_SD   = 1.0
MAX_DECODE  = 10.0
MIN_WAIT    = 0.3
RIDGE_ALPHA = [0.1, 1, 10, 100, 1000, 10000]

# Clock speed
MIN_BINS_FOR_SLOPE = 5

# Shuffle control (always uses pooled for speed)
N_SHUFFLES = 50

# Decoder mode
# False → pooled only (fast, ~2-3 min/session)
# True  → pooled + LOTO with comparison figures (~12-18 min/session)
USE_LOTO = True

# Set True to skip analysis and regenerate summary plots from saved CSVs only
PLOT_ONLY = False

# Sessions to exclude from decoding
EXCLUDE_SESSIONS = set()

# Alignment anchors for cross-anchor accuracy comparison
ANCHOR_COLS   = {
    'cue_on'   : 'cue_on',
    'cue_off'  : 'cue_off',
    'last_lick': 'last_lick_time',
}
ANCHOR_LABELS = {
    'cue_on'   : 'Cue On',
    'cue_off'  : 'Cue Off',
    'last_lick': 'Last Lick',
}
ANCHOR_COLORS = {
    'cue_on'   : '#4C8BE8',
    'cue_off'  : '#E8674C',
    'last_lick': '#3BAA6E',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_decoder_data(group_dict=None, unit_csv=None, msn_csv=None, pickle_dir=None):
    """
    Build trials_df, spikes_df, units_df from per-session pickle files.

    Parameters
    ----------
    group_dict : dict
        {'Long BG': [mouse_ids], 'Short BG': [mouse_ids]}
        Defaults to constants.GROUP_DICT with 'l' → 'Long BG', 's' → 'Short BG'.
    unit_csv : str or Path, optional
        Path to RZ_unit_properties_with_qc_and_regions.csv (output of 0e).
        If provided, only QC-passing CP/STR units are included; otherwise all
        units are used. Ignored when msn_csv is provided.
    msn_csv : str or Path, optional
        Path to an MSN unit list CSV (e.g. RZ_str_msn.csv from 0g_cell_type_relabeling).
        When provided, only units listed in this CSV are included — the CSV is
        already pre-filtered by cell_type==MSN, so no additional region/QC
        conditions are applied. Takes precedence over unit_csv.
    pickle_dir : str or Path, optional
        Directory containing per-session .pkl files. Defaults to paths.PICKLE_DIR.

    Returns
    -------
    trials_df, spikes_df, units_df

    Notes
    -----
    `time_waited` is set to (decision_time − t_ref) so the decode window covers
    the reference point through to the decision:
        Long BG  → decision_time − cue_on_time
        Short BG → decision_time − last_lick_time
    Trials where the reference time is NaN or the window is non-positive are
    marked as missed and filtered out by the decoder automatically.
    """
    if pickle_dir is None:
        pickle_dir = p.PICKLE_DIR

    if group_dict is None:
        group_dict = {
            'Long BG' : c.GROUP_DICT['l'],
            'Short BG': c.GROUP_DICT['s'],
        }

    mouse_to_group = {m: g for g, mice in group_dict.items() for m in mice}
    t_ref_col_map  = {'Long BG': 'cue_on_time', 'Short BG': 'last_lick_time'}

    sessions_log_path = p.LOGS_DIR / 'sessions_official_raw.csv'
    session_to_key = {}
    if os.path.exists(sessions_log_path):
        sess_log = pd.read_csv(sessions_log_path)
        for _, row in sess_log.iterrows():
            sid = row['id']
            key = f"{row['mouse']}|{row['date']}|{int(row['insertion_number'])}"
            session_to_key[sid] = key

    msn_df = None
    if msn_csv is not None and os.path.exists(str(msn_csv)):
        msn_df = pd.read_csv(msn_csv)

    qc_df = None
    if msn_df is None and unit_csv is not None and os.path.exists(str(unit_csv)):
        qc_df = pd.read_csv(unit_csv)

    all_trials, all_spikes, all_units = [], [], []

    for fname in sorted(os.listdir(pickle_dir)):
        if not fname.endswith('_str.pkl'):
            continue
        session_id = fname[:-4]

        with open(os.path.join(pickle_dir, fname), 'rb') as f:
            sess = pickle.load(f)

        mouse = sess['mouse']
        group = mouse_to_group.get(mouse)
        if group is None:
            continue

        ref_col    = t_ref_col_map[group]
        trials     = sess['trials']
        units_dict = sess['units']  # {unit_id: spikes_df}

        sess_key = session_to_key.get(session_id, '')
        if msn_df is not None:
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
            sample_key = next(iter(units_dict), None)
            if sample_key is not None and good_uids:
                good_uids = {type(sample_key)(u) for u in good_uids}
        else:
            good_uids = set(units_dict.keys())

        good_uids &= set(units_dict.keys())

        for uid in good_uids:
            all_units.append({
                'session'     : session_id,
                'unit_id'     : uid,
                'probe_region': 'STR',
                'mouse_id'    : mouse,
            })

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
    unit_src = "MSN" if msn_df is not None else ("QC STR" if qc_df is not None else "all")
    print(f"Loaded {n_sess} sessions | {len(units_df)} {unit_src} unit records | "
          f"{len(trials_df)} trials")
    return trials_df, spikes_df, units_df


# ── Session summary & threshold filtering ─────────────────────────────────────

def get_qualifying_sessions(trials_df, units_df,
                             min_units=MIN_UNITS, min_trials=MIN_TRIALS,
                             group_dict=GROUP_DICT):
    """
    Print all sessions for both groups and return those that meet the
    unit and trial thresholds.
    """
    rows = []
    for group, mice in group_dict.items():
        sess_list = trials_df[trials_df['mouse_id'].isin(mice)]['session'].unique()

        n_units = (
            units_df[units_df['session'].isin(sess_list)]
            .groupby('session')['unit_id'].nunique()
        )
        n_valid = (
            trials_df[(trials_df['session'].isin(sess_list)) & (~trials_df['miss_trial'])]
            .groupby('session')['trial_id'].nunique()
        )
        n_all = (
            trials_df[trials_df['session'].isin(sess_list)]
            .groupby('session')['trial_id'].nunique()
        )

        for sess in sess_list:
            rows.append({
                'group'         : group,
                'session'       : sess,
                'mouse'         : sess.split('_')[0],
                'n_units'       : int(n_units.get(sess, 0)),
                'n_valid_trials': int(n_valid.get(sess, 0)),
                'n_trials_all'  : int(n_all.get(sess, 0)),
            })

    df = pd.DataFrame(rows).sort_values(
        ['group', 'n_units', 'n_valid_trials'], ascending=[True, False, False]
    ).reset_index(drop=True)

    qualifies = ((df['n_units'] > min_units) &
                 (df['n_valid_trials'] > min_trials) &
                 (~df['session'].isin(EXCLUDE_SESSIONS)))

    print("\n" + "=" * 76)
    print(f"SESSION SUMMARY  (threshold: >{min_units} units, >{min_trials} valid trials)")
    print("=" * 76)
    for group, gdf in df.groupby('group', sort=False):
        print(f"\n  {group}")
        print(f"  {'session':<35} {'units':>6} {'valid':>7} {'all':>6}  {'':>4}")
        print(f"  {'-'*35} {'-'*6} {'-'*7} {'-'*6}  {'-'*4}")
        for _, r in gdf.iterrows():
            passes = qualifies[r.name]
            excluded = r['session'] in EXCLUDE_SESSIONS
            flag = '  excl.' if excluded else ('  ✓' if passes else '')
            print(f"  {r['session']:<35} {r['n_units']:>6} "
                  f"{r['n_valid_trials']:>7} {r['n_trials_all']:>6}{flag}")

    qualifying = df[qualifies].copy()
    print(f"\n  → {len(qualifying)} sessions qualify for decoding")
    print("=" * 76)
    return qualifying


# ── Reward enrichment ─────────────────────────────────────────────────────────

def enrich_with_reward(trials_df, session):
    """Add 'rewarded' and 'prev_rewarded' from the session pickle."""
    pickle_path = p.PICKLE_DIR / f'{session}.pkl'
    with open(pickle_path, 'rb') as f:
        raw = pickle.load(f)

    events  = raw['events']
    cons    = events[events['event_type'].isin(['cons_reward', 'cons_no_reward'])]
    rew_map = cons.groupby('trial_id')['event_type'].first().map(
        lambda x: 1 if x == 'cons_reward' else 0
    )

    t = trials_df[trials_df['session'] == session].copy()
    t['rewarded']      = t['trial_id'].map(rew_map).fillna(0).astype(int)
    t = t.sort_values('trial_id').reset_index(drop=True)
    t['prev_rewarded'] = t['rewarded'].shift(1)
    return t


# ── Spike processing ──────────────────────────────────────────────────────────

def bin_spikes(spike_times, t_ref, t_end):
    duration = min(t_end - t_ref, MAX_DECODE)
    if duration <= 0:
        return np.array([]), np.array([])
    edges       = np.arange(0, duration + BIN_SIZE, BIN_SIZE)
    rel_spikes  = np.asarray(spike_times) - t_ref
    counts, _   = np.histogram(rel_spikes, bins=edges)
    rates       = gaussian_filter1d(counts.astype(float) / BIN_SIZE, sigma=SMOOTH_SD)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    return rates, bin_centers


def build_population_matrix(trials_df, spikes_df, t_ref_col):
    """
    Build population-rate matrices for all non-miss trials.
    spikes_df rows have 'spike_times' as an array (from load_decoder_data).
    """
    good     = trials_df[~trials_df['miss_trial']].copy()
    unit_ids = sorted(spikes_df['unit_id'].unique())

    spike_lookup = {
        (row['unit_id'], row['trial_id']): np.asarray(row['spike_times'])
        for _, row in spikes_df.iterrows()
    }

    X_all, y_all = [], []
    trial_ids, tw_all, rewarded_all, prev_rewarded_all, bg_repeats_all = [], [], [], [], []

    for _, row in good.iterrows():
        tid   = row['trial_id']
        t_ref = row[t_ref_col]
        t_end = row['decision_time']

        if pd.isna(t_ref) or pd.isna(t_end) or t_end <= t_ref:
            continue

        trial_rates = []
        n_bins_ref  = None
        bin_times   = None

        for uid in unit_ids:
            spikes = spike_lookup.get((uid, tid), np.array([]))
            rates, bin_centers = bin_spikes(spikes, t_ref, t_end)
            if len(rates) == 0:
                continue
            if n_bins_ref is None:
                n_bins_ref = len(rates)
                bin_times  = bin_centers
            else:
                rates = rates[:n_bins_ref]
                if len(rates) < n_bins_ref:
                    rates = np.pad(rates, (0, n_bins_ref - len(rates)))
            trial_rates.append(rates)

        if not trial_rates or n_bins_ref is None or n_bins_ref < 2:
            continue

        pop_matrix = np.stack(trial_rates, axis=1)
        valid_bins = bin_times >= MIN_WAIT

        if valid_bins.sum() < MIN_BINS_FOR_SLOPE:
            continue

        X_all.append(pop_matrix[valid_bins])
        y_all.append(bin_times[valid_bins])
        trial_ids.append(tid)
        tw_all.append(row['time_waited'])
        rewarded_all.append(row.get('rewarded', np.nan))
        prev_rewarded_all.append(row.get('prev_rewarded', np.nan))
        bg_repeats_all.append(row.get('bg_repeats', np.nan))

    return {
        'X'            : X_all,
        'y'            : y_all,
        'trial_ids'    : trial_ids,
        'tw'           : np.array(tw_all),
        'rewarded'     : np.array(rewarded_all),
        'prev_rewarded': np.array(prev_rewarded_all),
        'bg_repeats'   : np.array(bg_repeats_all),
        'n_units'      : len(unit_ids),
        'n_trials'     : len(X_all),
    }


# ── Decoder ───────────────────────────────────────────────────────────────────

def loto_decode(data):
    X_all, y_all = data['X'], data['y']
    n = len(X_all)
    y_true_all, y_pred_all, mae_trials, r2_trials, bias_trials = [], [], [], [], []

    print(f"  Running LOTO decoder on {n} trials...")
    for i in range(n):
        if (i + 1) % 50 == 0:
            print(f"    Trial {i+1}/{n}")
        X_train = np.vstack([X_all[j] for j in range(n) if j != i])
        y_train = np.concatenate([y_all[j] for j in range(n) if j != i])
        model   = RidgeCV(alphas=RIDGE_ALPHA, fit_intercept=True)
        model.fit(X_train, y_train)
        y_hat   = model.predict(X_all[i])
        y_test  = y_all[i]
        err     = y_hat - y_test
        y_true_all.append(y_test);  y_pred_all.append(y_hat)
        mae_trials.append(np.mean(np.abs(err)))
        bias_trials.append(np.mean(err))
        ss_res = np.sum(err**2)
        ss_tot = np.sum((y_test - y_test.mean())**2)
        r2_trials.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)

    return dict(y_true=y_true_all, y_pred=y_pred_all,
                mae_per_trial=np.array(mae_trials), r2_per_trial=np.array(r2_trials),
                bias_per_trial=np.array(bias_trials),
                trial_ids=data['trial_ids'], tw=data['tw'],
                rewarded=data['rewarded'], prev_rewarded=data['prev_rewarded'],
                bg_repeats=data['bg_repeats'])


def pooled_decode(data, test_frac=0.2, seed=42):
    X_all, y_all = data['X'], data['y']
    n   = len(X_all)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_idx = idx[max(1, int(n * test_frac)):]
    X_train   = np.vstack([X_all[i] for i in train_idx])
    y_train   = np.concatenate([y_all[i] for i in train_idx])
    model     = RidgeCV(alphas=RIDGE_ALPHA, fit_intercept=True)
    model.fit(X_train, y_train)

    y_true_all, y_pred_all, mae_trials, r2_trials, bias_trials = [], [], [], [], []
    for i in range(n):
        y_hat  = model.predict(X_all[i])
        y_test = y_all[i]
        err    = y_hat - y_test
        y_true_all.append(y_test);  y_pred_all.append(y_hat)
        mae_trials.append(np.mean(np.abs(err)))
        bias_trials.append(np.mean(err))
        ss_res = np.sum(err**2)
        ss_tot = np.sum((y_test - y_test.mean())**2)
        r2_trials.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)

    return dict(y_true=y_true_all, y_pred=y_pred_all,
                mae_per_trial=np.array(mae_trials), r2_per_trial=np.array(r2_trials),
                bias_per_trial=np.array(bias_trials), model=model,
                trial_ids=data['trial_ids'], tw=data['tw'],
                rewarded=data['rewarded'], prev_rewarded=data['prev_rewarded'],
                bg_repeats=data['bg_repeats'])


def shuffle_decode(data, n_shuffles=N_SHUFFLES, seed=42, use_loto=False):
    rng = np.random.default_rng(seed)
    shuffle_maes, shuffle_r2s = [], []
    print(f"  Running shuffle control ({n_shuffles} permutations)...")
    for s in range(n_shuffles):
        if (s + 1) % 10 == 0:
            print(f"    Shuffle {s+1}/{n_shuffles}")
        y_shuffled = [rng.permutation(y) for y in data['y']]
        results    = (loto_decode({**data, 'y': y_shuffled}) if use_loto
                      else pooled_decode({**data, 'y': y_shuffled}, seed=seed + s))
        shuffle_maes.append(results['mae_per_trial'].mean())
        shuffle_r2s.append(np.nanmean(results['r2_per_trial']))
    return dict(shuffle_maes=np.array(shuffle_maes), shuffle_r2s=np.array(shuffle_r2s))


# ── Clock speed ───────────────────────────────────────────────────────────────

def compute_clock_speed(y_true, y_pred):
    if len(y_true) < MIN_BINS_FOR_SLOPE:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    y_fit  = slope * y_true + intercept
    ss_res = np.sum((y_pred - y_fit)**2)
    ss_tot = np.sum((y_pred - y_pred.mean())**2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2


def extract_clock_speeds(decoder_results):
    slopes, intercepts, r2s = [], [], []
    for y_t, y_p in zip(decoder_results['y_true'], decoder_results['y_pred']):
        s, ic, r2 = compute_clock_speed(y_t, y_p)
        slopes.append(s);  intercepts.append(ic);  r2s.append(r2)
    return dict(clock_speed=np.array(slopes), intercept=np.array(intercepts),
                r2=np.array(r2s))


# ── History analysis ──────────────────────────────────────────────────────────

def analyze_history_effect(decoder_results, clock_speeds):
    speed    = clock_speeds['clock_speed']
    prev_rew = decoder_results['prev_rewarded']
    tw       = decoder_results['tw']
    valid    = ~np.isnan(prev_rew) & ~np.isnan(speed)
    speed_v, prev_rew_v, tw_v = speed[valid], prev_rew[valid], tw[valid]

    after_reward    = speed_v[prev_rew_v == 1]
    after_no_reward = speed_v[prev_rew_v == 0]

    t_stat, p_ttest = ttest_ind(after_reward, after_no_reward)
    _, p_mwu        = mannwhitneyu(after_reward, after_no_reward, alternative='two-sided')
    pooled_std = np.sqrt((after_reward.std()**2 + after_no_reward.std()**2) / 2)
    cohens_d   = ((after_reward.mean() - after_no_reward.mean()) / pooled_std
                  if pooled_std > 0 else np.nan)
    r_own_tw, p_own_tw = pearsonr(speed_v, tw_v)

    return dict(after_reward_mean=after_reward.mean(), after_reward_std=after_reward.std(),
                after_reward_n=len(after_reward),
                after_no_reward_mean=after_no_reward.mean(), after_no_reward_std=after_no_reward.std(),
                after_no_reward_n=len(after_no_reward),
                t_stat=t_stat, p_ttest=p_ttest, p_mwu=p_mwu, cohens_d=cohens_d,
                r_own_tw=r_own_tw, p_own_tw=p_own_tw,
                speed=speed_v, prev_rewarded=prev_rew_v, tw=tw_v)


def time_matched_history_analysis(decoder_results, clock_speeds, n_quantiles=4):
    speed    = clock_speeds['clock_speed']
    prev_rew = decoder_results['prev_rewarded']
    tw       = decoder_results['tw']
    valid    = ~np.isnan(prev_rew) & ~np.isnan(speed)
    speed_v, prev_rew_v, tw_v = speed[valid], prev_rew[valid], tw[valid]

    quantiles = np.percentile(tw_v, np.linspace(0, 100, n_quantiles + 1))
    results   = []
    for i in range(n_quantiles):
        q_low, q_high = quantiles[i], quantiles[i + 1]
        mask = (tw_v >= q_low) & (tw_v <= q_high if i == n_quantiles - 1 else tw_v < q_high)
        speed_q, prev_rew_q = speed_v[mask], prev_rew_v[mask]
        after_rew = speed_q[prev_rew_q == 1]
        after_no  = speed_q[prev_rew_q == 0]
        t, p = (ttest_ind(after_rew, after_no)
                if len(after_rew) >= 5 and len(after_no) >= 5
                else (np.nan, np.nan))
        results.append(dict(quantile=i + 1, tw_range=(q_low, q_high),
                            after_rew_mean=after_rew.mean() if len(after_rew) else np.nan,
                            after_rew_n=len(after_rew),
                            after_no_rew_mean=after_no.mean() if len(after_no) else np.nan,
                            after_no_rew_n=len(after_no),
                            t_stat=t, p_value=p))
    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_decoder_performance(decoder_results, shuffle_results, data, config, save_dir, method='pooled'):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    y_true_pool = np.concatenate(decoder_results['y_true'])
    y_pred_pool = np.concatenate(decoder_results['y_pred'])
    r, _        = pearsonr(y_true_pool, y_pred_pool)
    r2_overall  = r**2
    r2_trials   = decoder_results['r2_per_trial']
    real_mae    = np.mean(decoder_results['mae_per_trial'])
    shuffle_maes = shuffle_results['shuffle_maes']
    p_val        = (shuffle_maes <= real_mae).mean()

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true_pool, y_pred_pool, alpha=0.1, s=5, c=config['color'])
    ax1.plot([0, MAX_DECODE], [0, MAX_DECODE], 'k--', lw=1.5, label='Unity')
    ax1.set_xlabel('Actual elapsed time (s)')
    ax1.set_ylabel('Decoded time (s)')
    ax1.set_title(f'Decoded vs. Actual\nR² = {r2_overall:.3f}, r = {r:.3f}')
    ax1.set_xlim(0, 8);  ax1.set_ylim(0, 8)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    time_bins = np.arange(0.5, 7, 0.2)
    mean_dec, std_dec = [], []
    for t in time_bins:
        mask = np.abs(y_true_pool - t) < 0.15
        mean_dec.append(y_pred_pool[mask].mean() if mask.sum() > 10 else np.nan)
        std_dec.append(y_pred_pool[mask].std()   if mask.sum() > 10 else np.nan)
    mean_dec, std_dec = np.array(mean_dec), np.array(std_dec)
    ax2.fill_between(time_bins, mean_dec - std_dec, mean_dec + std_dec,
                     alpha=0.3, color=config['color'])
    ax2.plot(time_bins, mean_dec, '-', color=config['color'], lw=2)
    ax2.plot([0, 7], [0, 7], 'k--', lw=1, alpha=0.5)
    ax2.set_xlabel('Actual elapsed time (s)')
    ax2.set_ylabel('Mean decoded time (s)')
    ax2.set_title('Mean Decoding Trajectory (±1 SD)')
    ax2.set_xlim(0, 7);  ax2.set_ylim(0, 7)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(shuffle_maes, bins=20, color='gray', alpha=0.6, edgecolor='white')
    ax3.axvline(real_mae, color='red', lw=2, label=f'Real: {real_mae:.3f}')
    ax3.set_xlabel('Mean Absolute Error (s)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Shuffle Control (n={len(shuffle_maes)})\np < {p_val:.4f}')
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 0])
    err_by_time = defaultdict(list)
    for y_t, y_p in zip(decoder_results['y_true'], decoder_results['y_pred']):
        for t, e in zip(y_t, y_p - y_t):
            err_by_time[int(t / 0.5) * 0.5].append(e)
    times     = sorted(err_by_time)
    mean_errs = [np.mean(err_by_time[t]) for t in times]
    std_errs  = [np.std(err_by_time[t])  for t in times]
    ax4.fill_between(times, np.array(mean_errs) - np.array(std_errs),
                     np.array(mean_errs) + np.array(std_errs), alpha=0.3, color=config['color'])
    ax4.plot(times, mean_errs, '-o', color=config['color'], markersize=4)
    ax4.axhline(0, color='gray', ls='--', lw=1)
    ax4.set_xlabel('Elapsed time (s)')
    ax4.set_ylabel('Decoding error (s)')
    ax4.set_title('Signed Error vs. Time')

    ax5 = fig.add_subplot(gs[1, 1])
    valid_r2 = r2_trials[~np.isnan(r2_trials)]
    ax5.hist(valid_r2, bins=25, color=config['color'], alpha=0.7, edgecolor='white')
    ax5.axvline(np.median(valid_r2), color='red', lw=2, ls='--',
                label=f'Median: {np.median(valid_r2):.2f}')
    ax5.set_xlabel('Per-trial R²')
    ax5.set_ylabel('Count')
    ax5.set_title('Per-Trial Decoding Quality')
    ax5.legend()

    ax6 = fig.add_subplot(gs[1, 2])
    valid_idx  = np.where(~np.isnan(r2_trials))[0]
    sorted_idx = valid_idx[np.argsort(r2_trials[valid_idx])]
    ex_idx     = [sorted_idx[len(sorted_idx)//4], sorted_idx[len(sorted_idx)//2],
                  sorted_idx[3*len(sorted_idx)//4]]
    for idx, color, label in zip(ex_idx, ['#E41A1C', '#377EB8', '#4DAF4A'],
                                 ['Low R²', 'Med R²', 'High R²']):
        ax6.plot(decoder_results['y_true'][idx], decoder_results['y_pred'][idx],
                 '-', color=color, lw=1.5, alpha=0.8,
                 label=f'{label}: {r2_trials[idx]:.2f}')
    ax6.plot([0, 7], [0, 7], 'k--', lw=1, alpha=0.5)
    ax6.set_xlabel('Actual time (s)');  ax6.set_ylabel('Decoded time (s)')
    ax6.set_title('Example Single Trials')
    ax6.legend(loc='lower right')
    ax6.set_xlim(0, 7);  ax6.set_ylim(0, 7)

    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    summary = (
        f"\n{'═'*76}\n"
        f"DECODER PERFORMANCE: {config['session_id']} ({config['group']})\n"
        f"{'═'*76}\n"
        f"Data:  {data['n_trials']} trials  |  {data['n_units']} units  "
        f"|  anchor: {config['t_ref_col']}\n\n"
        f"Overall:   Pooled R² = {r2_overall:.4f}  (r = {r:.4f})   "
        f"Mean MAE = {real_mae:.3f} s\n"
        f"Shuffle:   MAE = {shuffle_maes.mean():.3f} ± {shuffle_maes.std():.3f} s   "
        f"p = {p_val:.6f}\n"
        f"Per-trial: median R² = {np.nanmedian(r2_trials):.3f}   "
        f"R²>0.5: {(r2_trials > 0.5).sum()}/{len(r2_trials)} "
        f"({100*(r2_trials > 0.5).mean():.1f}%)\n"
        f"{'═'*76}"
    )
    ax7.text(0.05, 0.95, summary, transform=ax7.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top')

    plt.suptitle(f'{config["group"]} — Decoder Performance [{method.upper()}]: {config["session_id"]}',
                 fontsize=13)
    plt.savefig(str(save_dir / f'decoder_performance_{method}_{config["session_id"]}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    return r2_overall, real_mae, p_val


def plot_clock_speed_analysis(decoder_results, clock_speeds, history_stats,
                               time_matched_results, config, save_dir, method='pooled'):
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    speed       = clock_speeds['clock_speed']
    valid_speed = speed[~np.isnan(speed)]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(valid_speed, bins=30, color=config['color'], alpha=0.7, edgecolor='white')
    ax1.axvline(1.0, color='black', lw=2, ls='--', label='Unity')
    ax1.axvline(valid_speed.mean(), color='red', lw=2, label=f'Mean: {valid_speed.mean():.3f}')
    ax1.set_xlabel('Clock speed (slope)')
    ax1.set_ylabel('Count')
    ax1.set_title('Per-Trial Clock Speed Distribution')
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    tw    = decoder_results['tw']
    valid = ~np.isnan(speed)
    ax2.scatter(tw[valid], speed[valid], alpha=0.4, s=20, c=config['color'])
    z      = np.polyfit(tw[valid], speed[valid], 1)
    tw_rng = np.linspace(tw[valid].min(), tw[valid].max(), 100)
    ax2.plot(tw_rng, np.poly1d(z)(tw_rng), 'k-', lw=2)
    ax2.axhline(1.0, color='gray', ls='--', lw=1)
    ax2.set_xlabel('Time waited (s)')
    ax2.set_ylabel('Clock speed')
    ax2.set_title(f'Clock Speed vs. Own Wait Time\n(r = {history_stats["r_own_tw"]:.3f})')

    ax3 = fig.add_subplot(gs[0, 2])
    after_rew    = history_stats['speed'][history_stats['prev_rewarded'] == 1]
    after_no_rew = history_stats['speed'][history_stats['prev_rewarded'] == 0]
    bp = ax3.boxplot([after_no_rew, after_rew], positions=[0, 1], widths=0.6, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightgray', 'lightgreen']):
        patch.set_facecolor(color)
    ax3.axhline(1.0, color='gray', ls='--', lw=1)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['After\nNo Reward', 'After\nReward'])
    ax3.set_ylabel('Clock speed')
    ax3.set_title(f'Clock Speed by Previous Outcome\n'
                  f't={history_stats["t_stat"]:.2f}, p={history_stats["p_ttest"]:.4f}, '
                  f'd={history_stats["cohens_d"]:.2f}')

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(history_stats['tw'], history_stats['speed'],
                c=['green' if pr == 1 else 'gray' for pr in history_stats['prev_rewarded']],
                alpha=0.5, s=20)
    ax4.axhline(1.0, color='gray', ls='--', lw=1)
    ax4.set_xlabel('Time waited (s)')
    ax4.set_ylabel('Clock speed')
    ax4.set_title('Clock Speed vs. TW (by prev outcome)')

    ax5 = fig.add_subplot(gs[1, 1])
    q_labels = [f"Q{r['quantile']}\n{r['tw_range'][0]:.1f}-{r['tw_range'][1]:.1f}s"
                for r in time_matched_results]
    q_diffs  = [(r['after_rew_mean'] - r['after_no_rew_mean'])
                if not np.isnan(r.get('after_rew_mean', np.nan) or
                                r.get('after_no_rew_mean', np.nan)) else np.nan
                for r in time_matched_results]
    q_pvals  = [r['p_value'] for r in time_matched_results]
    colors5  = ['#D6604D' if (p is not None and not np.isnan(p) and p < 0.05) else '#AAAAAA'
                for p in q_pvals]
    ax5.bar(range(len(q_labels)), q_diffs, color=colors5, alpha=0.8)
    ax5.set_xticks(range(len(q_labels)))
    ax5.set_xticklabels(q_labels, fontsize=8)
    ax5.axhline(0, color='gray', ls='--', lw=1)
    ax5.set_ylabel('Δ clock speed (after rew − no rew)')
    ax5.set_title('Time-Matched History Effect\n(red = p<0.05)')

    ax6 = fig.add_subplot(gs[1, 2])
    r2_cs  = clock_speeds['r2']
    valid6 = ~np.isnan(r2_cs)
    ax6.hist(r2_cs[valid6], bins=25, color=config['color'], alpha=0.7, edgecolor='white')
    ax6.axvline(np.median(r2_cs[valid6]), color='red', lw=2, ls='--',
                label=f'Median: {np.median(r2_cs[valid6]):.2f}')
    ax6.set_xlabel('Linear fit R² (clock speed)')
    ax6.set_ylabel('Count')
    ax6.set_title('Clock Speed Fit Quality')
    ax6.legend()

    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    pred_dir = ("CONSISTENT" if history_stats['after_reward_mean'] <
                history_stats['after_no_reward_mean'] else "OPPOSITE")
    summary = (
        f"\n{'═'*90}\n"
        f"CLOCK SPEED: {config['session_id']} ({config['group']})\n"
        f"{'═'*90}\n"
        f"Distribution:  mean={valid_speed.mean():.4f} ± {valid_speed.std():.4f}   "
        f"median={np.median(valid_speed):.4f}\n"
        f"Confound (vs own TW):  r={history_stats['r_own_tw']:.4f}, "
        f"p={history_stats['p_own_tw']:.6f}\n\n"
        f"After REWARD    (n={history_stats['after_reward_n']}): "
        f"{history_stats['after_reward_mean']:.4f} ± {history_stats['after_reward_std']:.4f}\n"
        f"After NO REWARD (n={history_stats['after_no_reward_n']}): "
        f"{history_stats['after_no_reward_mean']:.4f} ± {history_stats['after_no_reward_std']:.4f}\n"
        f"t={history_stats['t_stat']:.3f}, p={history_stats['p_ttest']:.6f}, "
        f"d={history_stats['cohens_d']:.3f}   → {pred_dir}\n"
        f"{'═'*90}"
    )
    ax7.text(0.02, 0.95, summary, transform=ax7.transAxes, fontsize=9,
             fontfamily='monospace', verticalalignment='top')

    plt.suptitle(f'{config["group"]} — Clock Speed [{method.upper()}]: {config["session_id"]}',
                 fontsize=13)
    plt.savefig(str(save_dir / f'clock_speed_analysis_{method}_{config["session_id"]}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca_trajectories(data, decoder_results, config, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    X_stacked = np.vstack(data['X'])
    pca       = PCA(n_components=3)
    pca.fit(X_stacked)
    var_exp   = pca.explained_variance_ratio_
    trial_pcs = [pca.transform(X) for X in data['X']]
    prev_rew  = decoder_results['prev_rewarded']

    ax1 = axes[0]
    for pc, pr in zip(trial_pcs, prev_rew):
        if np.isnan(pr):
            continue
        ax1.plot(pc[:, 0], pc[:, 1], '-',
                 color='green' if pr == 1 else 'gray',
                 alpha=0.4 if pr == 1 else 0.2, lw=0.5)
    ax1.set_xlabel(f'PC1 ({100*var_exp[0]:.1f}%)')
    ax1.set_ylabel(f'PC2 ({100*var_exp[1]:.1f}%)')
    ax1.set_title('Single-Trial Trajectories')

    ax2 = axes[1]
    for pc, y, pr in zip(trial_pcs, data['y'], prev_rew):
        if np.isnan(pr):
            continue
        ax2.plot(y, pc[:, 0], '-',
                 color='green' if pr == 1 else 'gray',
                 alpha=0.4 if pr == 1 else 0.2, lw=0.5)
    ax2.set_xlabel('Elapsed time (s)')
    ax2.set_ylabel(f'PC1 ({100*var_exp[0]:.1f}%)')
    ax2.set_title('PC1 vs. Time')

    ax3 = axes[2]
    time_bins = np.arange(0.5, 5, 0.2)
    for label, cond_val, color in [('After Reward', 1, 'green'), ('After No Reward', 0, 'gray')]:
        pc1_by_time = []
        for t in time_bins:
            vals = []
            for pc, y, pr in zip(trial_pcs, data['y'], prev_rew):
                if np.isnan(pr) or pr != cond_val:
                    continue
                idx = np.argmin(np.abs(y - t))
                if np.abs(y[idx] - t) < 0.15:
                    vals.append(pc[idx, 0])
            pc1_by_time.append(np.mean(vals) if vals else np.nan)
        ax3.plot(time_bins, pc1_by_time, '-o', color=color, lw=2, label=label, markersize=4)
    ax3.set_xlabel('Elapsed time (s)')
    ax3.set_ylabel('Mean PC1')
    ax3.set_title('Mean PC1 by Condition')
    ax3.legend()

    plt.suptitle(f'{config["group"]}: PCA Trajectories — {config["session_id"]}',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(str(save_dir / f'pca_trajectories_{config["session_id"]}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ── Method comparison plot ────────────────────────────────────────────────────

def plot_method_comparison(pooled_dec, loto_dec, pooled_speeds, loto_speeds,
                            pooled_hist, loto_hist, shuffle_results, config, save_dir):
    """
    Side-by-side comparison of Pooled vs LOTO decoder:
      Row 1 — decoder accuracy (R² distributions, MAE bar, shuffle null)
      Row 2 — clock speed (distributions, per-trial correlation, history effect)
    """
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.38)

    clr_pooled = '#4393C3'
    clr_loto   = '#D6604D'
    shuffle_maes = shuffle_results['shuffle_maes']

    # ── Row 1: decoder accuracy ───────────────────────────────────────────────

    # Panel A: per-trial R² distributions
    ax1 = fig.add_subplot(gs[0, 0])
    r2_pooled = pooled_dec['r2_per_trial']
    r2_loto   = loto_dec['r2_per_trial']
    bins = np.linspace(min(np.nanmin(r2_pooled), np.nanmin(r2_loto)),
                       max(np.nanmax(r2_pooled), np.nanmax(r2_loto)), 30)
    ax1.hist(r2_pooled[~np.isnan(r2_pooled)], bins=bins, alpha=0.6,
             color=clr_pooled, label=f'Pooled (med={np.nanmedian(r2_pooled):.2f})')
    ax1.hist(r2_loto[~np.isnan(r2_loto)], bins=bins, alpha=0.6,
             color=clr_loto,   label=f'LOTO   (med={np.nanmedian(r2_loto):.2f})')
    ax1.set_xlabel('Per-trial R²')
    ax1.set_ylabel('Count')
    ax1.set_title('Per-Trial R²: Pooled vs LOTO')
    ax1.legend(fontsize=8)

    # Panel B: MAE — real vs shuffle for both decoders
    ax2 = fig.add_subplot(gs[0, 1])
    mae_pooled = pooled_dec['mae_per_trial'].mean()
    mae_loto   = loto_dec['mae_per_trial'].mean()
    mae_shuf   = shuffle_maes.mean()
    bars = ax2.bar(['Pooled', 'LOTO', 'Shuffle\n(null)'],
                   [mae_pooled, mae_loto, mae_shuf],
                   color=[clr_pooled, clr_loto, '#AAAAAA'],
                   alpha=0.85, edgecolor='none')
    ax2.errorbar([2], [mae_shuf], yerr=[shuffle_maes.std()],
                 fmt='none', color='k', capsize=4)
    p_pooled = (shuffle_maes <= mae_pooled).mean()
    p_loto   = (shuffle_maes <= mae_loto).mean()
    ax2.text(0, mae_pooled + 0.01, f'p={p_pooled:.3f}', ha='center', fontsize=8)
    ax2.text(1, mae_loto   + 0.01, f'p={p_loto:.3f}',   ha='center', fontsize=8)
    ax2.set_ylabel('Mean Absolute Error (s)')
    ax2.set_title('MAE vs Shuffle Null')

    # Panel C: pooled MAE per trial vs LOTO MAE per trial (scatter)
    ax3 = fig.add_subplot(gs[0, 2])
    shared_tids  = set(pooled_dec['trial_ids']) & set(loto_dec['trial_ids'])
    p_idx = {tid: i for i, tid in enumerate(pooled_dec['trial_ids'])}
    l_idx = {tid: i for i, tid in enumerate(loto_dec['trial_ids'])}
    shared = sorted(shared_tids)
    mae_p  = np.array([pooled_dec['mae_per_trial'][p_idx[t]] for t in shared])
    mae_l  = np.array([loto_dec['mae_per_trial'][l_idx[t]]   for t in shared])
    ax3.scatter(mae_p, mae_l, alpha=0.3, s=15, color='k')
    lim = max(mae_p.max(), mae_l.max()) * 1.05
    ax3.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5)
    r_mae, _ = pearsonr(mae_p, mae_l)
    ax3.set_xlabel('Pooled MAE (s)')
    ax3.set_ylabel('LOTO MAE (s)')
    ax3.set_title(f'Per-Trial MAE Correlation\nr = {r_mae:.3f}')

    # Panel D: overall R² bar
    ax4 = fig.add_subplot(gs[0, 3])
    r2_vals = [np.nanmean(r2_pooled), np.nanmean(r2_loto)]
    ax4.bar(['Pooled', 'LOTO'], r2_vals, color=[clr_pooled, clr_loto], alpha=0.85, edgecolor='none')
    ax4.set_ylabel('Mean R²')
    ax4.set_title('Mean Per-Trial R²')
    ax4.set_ylim(0, max(r2_vals) * 1.25)
    for i, v in enumerate(r2_vals):
        ax4.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    # ── Row 2: clock speed ────────────────────────────────────────────────────

    # Panel E: clock speed distributions
    ax5 = fig.add_subplot(gs[1, 0])
    cs_p = pooled_speeds['clock_speed']
    cs_l = loto_speeds['clock_speed']
    bins_cs = np.linspace(min(np.nanmin(cs_p), np.nanmin(cs_l)),
                          max(np.nanmax(cs_p), np.nanmax(cs_l)), 35)
    ax5.hist(cs_p[~np.isnan(cs_p)], bins=bins_cs, alpha=0.6, color=clr_pooled,
             label=f'Pooled (μ={np.nanmean(cs_p):.3f})')
    ax5.hist(cs_l[~np.isnan(cs_l)], bins=bins_cs, alpha=0.6, color=clr_loto,
             label=f'LOTO   (μ={np.nanmean(cs_l):.3f})')
    ax5.axvline(1.0, color='k', ls='--', lw=1.5)
    ax5.set_xlabel('Clock speed (slope)')
    ax5.set_ylabel('Count')
    ax5.set_title('Clock Speed Distribution')
    ax5.legend(fontsize=8)

    # Panel F: per-trial clock speed correlation (Pooled vs LOTO)
    ax6 = fig.add_subplot(gs[1, 1])
    cs_p_shared = np.array([cs_p[p_idx[t]] for t in shared])
    cs_l_shared = np.array([cs_l[l_idx[t]] for t in shared])
    valid_both  = ~np.isnan(cs_p_shared) & ~np.isnan(cs_l_shared)
    ax6.scatter(cs_p_shared[valid_both], cs_l_shared[valid_both],
                alpha=0.3, s=15, color='k')
    lim6 = max(np.nanmax(np.abs(cs_p_shared)), np.nanmax(np.abs(cs_l_shared))) * 1.1
    ax6.plot([-lim6, lim6], [-lim6, lim6], 'k--', lw=1, alpha=0.5)
    r_cs, _ = pearsonr(cs_p_shared[valid_both], cs_l_shared[valid_both])
    ax6.set_xlabel('Pooled clock speed')
    ax6.set_ylabel('LOTO clock speed')
    ax6.set_title(f'Per-Trial Clock Speed Correlation\nr = {r_cs:.3f}')

    # Panel G: history effect — after reward vs no reward, both decoders
    ax7 = fig.add_subplot(gs[1, 2])
    groups_data = [
        pooled_hist['speed'][pooled_hist['prev_rewarded'] == 0],
        pooled_hist['speed'][pooled_hist['prev_rewarded'] == 1],
        loto_hist['speed'][loto_hist['prev_rewarded'] == 0],
        loto_hist['speed'][loto_hist['prev_rewarded'] == 1],
    ]
    bp = ax7.boxplot(groups_data, positions=[0, 1, 2.5, 3.5],
                     widths=0.6, patch_artist=True)
    face_colors = ['#CCCCCC', '#90EE90', '#AACCEE', '#D6604D']
    for patch, fc in zip(bp['boxes'], face_colors):
        patch.set_facecolor(fc)
    ax7.axhline(1.0, color='gray', ls='--', lw=1)
    ax7.set_xticks([0, 1, 2.5, 3.5])
    ax7.set_xticklabels(['No Rew\nPooled', 'Rew\nPooled',
                         'No Rew\nLOTO', 'Rew\nLOTO'], fontsize=8)
    ax7.set_ylabel('Clock speed')
    ax7.set_title('History Effect: Pooled vs LOTO\n'
                  f'Pooled p={pooled_hist["p_ttest"]:.3f} d={pooled_hist["cohens_d"]:.2f}  '
                  f'LOTO p={loto_hist["p_ttest"]:.3f} d={loto_hist["cohens_d"]:.2f}')

    # Panel H: Cohen's d and p-value comparison
    ax8 = fig.add_subplot(gs[1, 3])
    metrics = ['Cohen\'s d', '-log10(p)']
    pooled_vals = [pooled_hist['cohens_d'],
                   -np.log10(max(pooled_hist['p_ttest'], 1e-10))]
    loto_vals   = [loto_hist['cohens_d'],
                   -np.log10(max(loto_hist['p_ttest'], 1e-10))]
    x_pos = np.arange(len(metrics))
    w = 0.3
    ax8.bar(x_pos - w/2, pooled_vals, width=w, color=clr_pooled, alpha=0.85,
            label='Pooled', edgecolor='none')
    ax8.bar(x_pos + w/2, loto_vals,   width=w, color=clr_loto,   alpha=0.85,
            label='LOTO',   edgecolor='none')
    ax8.axhline(0, color='gray', ls='--', lw=1)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(metrics)
    ax8.set_title('History Effect Size')
    ax8.legend(fontsize=8)

    sid = config['session_id']
    plt.suptitle(f'{config["group"]} — Pooled vs LOTO Comparison: {sid}',
                 fontsize=13)
    plt.savefig(str(save_dir / f'method_comparison_{sid}.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ── Committee figures ─────────────────────────────────────────────────────────

def plot_committee_point1(raw_cache, config, save_dir):
    """
    Point 1: DMS encodes elapsed time.
    Panels: (A) decoded vs actual scatter, (B) mean trajectory ± SD, (C) shuffle control.
    Uses pooled decoder output from raw_cache.
    """
    dec   = raw_cache['pooled']
    clr   = config['color']
    sid   = config['session_id']
    grp   = config['group']

    y_true_pool = np.concatenate(dec['y_true'])
    y_pred_pool = np.concatenate(dec['y_pred'])
    shuffle_maes = raw_cache['shuffle_maes']
    real_mae     = dec['mae_per_trial'].mean()
    r, _         = pearsonr(y_true_pool, y_pred_pool)
    r2_pooled    = r ** 2
    p_val        = (shuffle_maes <= real_mae).mean()

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))

    # Panel A: decoded vs actual scatter
    ax = axes[0]
    ax.scatter(y_true_pool, y_pred_pool, alpha=0.04, s=3, color=clr, rasterized=True)
    lim = max(y_true_pool.max(), y_pred_pool.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1.2, label='Unity')
    ax.set_xlabel('Actual elapsed time (s)', fontsize=11)
    ax.set_ylabel('Decoded time (s)', fontsize=11)
    ax.set_title(f'Decoded vs. Actual\nr = {r:.3f},  R² = {r2_pooled:.3f}', fontsize=10)
    ax.set_xlim(0, lim);  ax.set_ylim(0, lim)
    ax.legend(fontsize=9)

    # Panel B: mean trajectory ± SD
    ax = axes[1]
    bin_edges = np.arange(0, MAX_DECODE + 0.2, 0.2)
    bin_ctrs  = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_dec, std_dec = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_true_pool >= lo) & (y_true_pool < hi)
        if mask.sum() >= 5:
            mean_dec.append(y_pred_pool[mask].mean())
            std_dec.append(y_pred_pool[mask].std())
        else:
            mean_dec.append(np.nan)
            std_dec.append(np.nan)
    mean_dec, std_dec = np.array(mean_dec), np.array(std_dec)
    valid = ~np.isnan(mean_dec)
    ax.fill_between(bin_ctrs[valid], (mean_dec - std_dec)[valid], (mean_dec + std_dec)[valid],
                    alpha=0.25, color=clr)
    ax.plot(bin_ctrs[valid], mean_dec[valid], '-', color=clr, lw=2)
    ax.plot([0, MAX_DECODE], [0, MAX_DECODE], 'k--', lw=1.2, alpha=0.6)
    ax.set_xlabel('Actual elapsed time (s)', fontsize=11)
    ax.set_ylabel('Mean decoded time (s)', fontsize=11)
    ax.set_title('Mean Decoding Trajectory (±1 SD)', fontsize=10)
    ax.set_xlim(0, MAX_DECODE);  ax.set_ylim(0, MAX_DECODE)

    # Panel C: error over time
    ax = axes[2]
    abs_err = np.abs(y_pred_pool - y_true_pool)
    mean_err, sem_err = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_true_pool >= lo) & (y_true_pool < hi)
        if mask.sum() >= 5:
            errs = abs_err[mask]
            mean_err.append(errs.mean())
            sem_err.append(errs.std() / np.sqrt(mask.sum()))
        else:
            mean_err.append(np.nan)
            sem_err.append(np.nan)
    mean_err = np.array(mean_err)
    sem_err  = np.array(sem_err)
    valid_e  = ~np.isnan(mean_err)
    ax.fill_between(bin_ctrs[valid_e],
                    (mean_err - sem_err)[valid_e],
                    (mean_err + sem_err)[valid_e],
                    alpha=0.25, color=clr)
    ax.plot(bin_ctrs[valid_e], mean_err[valid_e], '-', color=clr, lw=2)
    ax.axhline(real_mae, color='gray', ls='--', lw=1.2, alpha=0.8, label=f'Mean MAE = {real_mae:.2f} s')
    ax.set_xlabel('Actual elapsed time (s)', fontsize=11)
    ax.set_ylabel('Mean absolute error (s)', fontsize=11)
    ax.set_title('Decoding Error over Time (±1 SEM)', fontsize=10)
    ax.set_xlim(0, MAX_DECODE)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

    # Panel D: shuffle control
    ax = axes[3]
    counts, _, _ = ax.hist(
        shuffle_maes, bins=25, color='#888888', alpha=0.9, edgecolor='#555555', lw=0.5,
        label=f'Shuffle (n={len(shuffle_maes)})')
    ax.axvline(real_mae, color=clr, lw=3, zorder=5,
               label=f'Real MAE = {real_mae:.3f} s')
    # Annotate with an arrow if the real MAE is inside the shuffle distribution
    ax.annotate('', xy=(real_mae, counts.max() * 0.6),
                xytext=(real_mae, counts.max() * 0.85),
                arrowprops=dict(arrowstyle='->', color=clr, lw=2))
    ax.set_xlabel('Mean Absolute Error (s)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Shuffle Control\np = {p_val:.4f}', fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(str(save_dir / f'committee_point1_elapsed_time_{sid}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_committee_point2(raw_cache, config, save_dir):
    """
    Point 2: Clock speed is extractable.
    Panels: (A) example single trials with slope labeled, (B) clock speed distribution.
    """
    dec    = raw_cache['pooled']
    speeds = raw_cache['pooled_speeds']['clock_speed']
    clr    = config['color']
    sid    = config['session_id']
    grp    = config['group']

    valid_idx = np.where(~np.isnan(speeds))[0]
    if len(valid_idx) < 4:
        print(f"  [skip committee_point2] fewer than 4 valid clock speeds for {sid}")
        return

    # Pick 4 trials at 25th, 50th, 75th, 90th percentile of slope
    pct_vals  = np.percentile(speeds[valid_idx], [25, 50, 75, 90])
    ex_idx    = [valid_idx[np.argmin(np.abs(speeds[valid_idx] - p))] for p in pct_vals]
    ex_colors = ['#E41A1C', '#FF7F00', '#4DAF4A', '#984EA3']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: example trials
    ax = axes[0]
    lim = 0
    for idx, ec, pct_label in zip(ex_idx, ex_colors,
                                  ['25th pct', '50th pct', '75th pct', '90th pct']):
        y_t = dec['y_true'][idx]
        y_p = dec['y_pred'][idx]
        slope = speeds[idx]
        ax.plot(y_t, y_p, '-', color=ec, lw=1.5, alpha=0.85,
                label=f'{pct_label}: slope = {slope:.2f}')
        # linear fit overlay
        fit = np.polyfit(y_t, y_p, 1)
        t_rng = np.linspace(y_t.min(), y_t.max(), 50)
        ax.plot(t_rng, np.poly1d(fit)(t_rng), '--', color=ec, lw=1, alpha=0.6)
        lim = max(lim, y_t.max(), y_p.max())
    lim *= 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='Unity')
    ax.set_xlabel('Actual elapsed time (s)', fontsize=11)
    ax.set_ylabel('Decoded time (s)', fontsize=11)
    ax.set_title('Example Trials with Clock Speed', fontsize=10)
    ax.set_xlim(0, lim);  ax.set_ylim(0, lim)
    ax.legend(fontsize=8, loc='upper left')

    # Panel B: clock speed distribution
    ax = axes[1]
    valid_speeds = speeds[valid_idx]
    ax.hist(valid_speeds, bins=30, color=clr, alpha=0.75, edgecolor='white')
    ax.axvline(1.0, color='k', lw=1.5, ls='--', label='Unity (slope = 1)')
    ax.axvline(valid_speeds.mean(), color=clr, lw=2,
               label=f'Mean = {valid_speeds.mean():.3f}')
    ax.set_xlabel('Clock speed (slope of decoded vs. actual)', fontsize=11)
    ax.set_ylabel('Trial count', fontsize=11)
    ax.set_title(f'Clock Speed Distribution\n(n = {len(valid_speeds)} trials)', fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(str(save_dir / f'committee_point2_clock_speed_{sid}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_committee_point3_session(raw_cache, config, save_dir):
    """
    Point 3 (per session): Clock speed differs by reward history.
    Single panel: violin + jittered dots split by prev_rewarded.
    """
    dec    = raw_cache['pooled']
    speeds = raw_cache['pooled_speeds']['clock_speed']
    hist   = raw_cache['pooled_hist']
    clr    = config['color']
    sid    = config['session_id']
    grp    = config['group']

    prev_rew = dec['prev_rewarded']
    valid    = ~np.isnan(prev_rew) & ~np.isnan(speeds)
    spd_v    = speeds[valid]
    prw_v    = prev_rew[valid]

    after_rew  = spd_v[prw_v == 1]
    after_nrew = spd_v[prw_v == 0]

    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(5, 5))

    parts = ax.violinplot([after_nrew, after_rew], positions=[0, 1],
                          showmedians=True, showextrema=False)
    for pc, fc in zip(parts['bodies'], ['#CCCCCC', clr]):
        pc.set_facecolor(fc);  pc.set_alpha(0.6)
    parts['cmedians'].set_color('k');  parts['cmedians'].set_linewidth(2)

    for pos, arr in zip([0, 1], [after_nrew, after_rew]):
        jitter = rng.uniform(-0.08, 0.08, size=len(arr))
        ax.scatter(pos + jitter, arr, s=18, alpha=0.5, color='k', zorder=3)

    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'After No Reward\n(n={len(after_nrew)})',
                        f'After Reward\n(n={len(after_rew)})'], fontsize=10)
    ax.set_ylabel('Clock speed', fontsize=11)
    ax.set_title(f"p = {hist['p_ttest']:.4f},  d = {hist['cohens_d']:.3f}", fontsize=10)

    plt.tight_layout()
    plt.savefig(str(save_dir / f'committee_point3_history_{sid}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_committee_point3_summary(all_raw_caches, save_dir):
    """
    Point 3 (cross-session summary): grouped bar of mean clock speed by reward history
    and Cohen's d per session.
    """
    if not all_raw_caches:
        return

    grp_color = {'Long BG': '#9080ff', 'Short BG': '#ffb400'}
    sessions, groups = [], []
    mean_rew, sem_rew, mean_nrew, sem_nrew, cohens_ds, pvals = [], [], [], [], [], []

    for rc in all_raw_caches:
        hist = rc['pooled_hist']
        sessions.append(rc['session_id'])
        groups.append(rc['group'])
        n_r  = hist['after_reward_n']
        n_nr = hist['after_no_reward_n']
        mean_rew.append(hist['after_reward_mean'])
        sem_rew.append(hist['after_reward_std'] / np.sqrt(max(n_r, 1)))
        mean_nrew.append(hist['after_no_reward_mean'])
        sem_nrew.append(hist['after_no_reward_std'] / np.sqrt(max(n_nr, 1)))
        cohens_ds.append(hist['cohens_d'])
        pvals.append(hist['p_ttest'])

    x    = np.arange(len(sessions))
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(sessions) * 1.8), 5))
    fig.suptitle('Clock Speed by Reward History (All Sessions)',
                 fontsize=12)

    # Panel A: grouped bar — after no-reward vs after-reward
    ax = axes[0]
    w  = 0.35
    ax.bar(x - w/2, mean_nrew, width=w, color='#CCCCCC', alpha=0.85,
           label='After No Reward', edgecolor='none',
           yerr=sem_nrew, capsize=4, error_kw={'elinewidth': 1.2})
    for i, (mr, sr, grp) in enumerate(zip(mean_rew, sem_rew, groups)):
        ax.bar(x[i] + w/2, mr, width=w, color=grp_color.get(grp, 'steelblue'),
               alpha=0.85, edgecolor='none',
               yerr=sr, capsize=4, error_kw={'elinewidth': 1.2})
    ax.axhline(1.0, color='gray', ls='--', lw=1, label='Unity')
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean clock speed', fontsize=11)
    ax.set_title('Mean Clock Speed: After Reward vs. No Reward\n(colored = after reward)',
                 fontsize=9)

    # Panel B: Cohen's d per session with significance markers
    ax = axes[1]
    for i, (d, p, grp) in enumerate(zip(cohens_ds, pvals, groups)):
        clr = grp_color.get(grp, 'steelblue')
        ax.scatter(i, d, color=clr, s=80, zorder=3,
                   alpha=0.95 if not np.isnan(d) else 0.2)
        if not np.isnan(p) and p < 0.05:
            ax.text(i, d + 0.03, '*', ha='center', va='bottom', fontsize=14, color=clr)
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Cohen's d  (reward − no reward)", fontsize=11)
    ax.set_title("History Effect Size per Session\n(* = p < 0.05)", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(save_dir / 'committee_point3_history_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir / 'committee_point3_history_summary.png'}")


def plot_committee_point1_summary(all_raw_caches, save_dir):
    """
    Point 1 (cross-session summary): DMS encodes elapsed time in ALL sessions.

    Panel A: Real MAE vs Shuffle MAE (grouped bars) — shows decoder beats chance
    Panel B: Decoder correlation (r) per session — shows decoding quality
    Panel C: Mean decoded trajectory overlaid for all sessions
    """
    if not all_raw_caches:
        return

    grp_color = {'Long BG': '#9080ff', 'Short BG': '#ffb400'}

    # Extract metrics from each session
    sessions, groups = [], []
    real_maes, shuffle_means, shuffle_stds = [], [], []
    correlations, r2s = [], []
    p_vals = []

    for rc in all_raw_caches:
        dec = rc['pooled']
        sessions.append(rc['session_id'])
        groups.append(rc['group'])

        # MAE
        real_mae = dec['mae_per_trial'].mean()
        real_maes.append(real_mae)
        shuffle_maes = rc['shuffle_maes']
        shuffle_means.append(shuffle_maes.mean())
        shuffle_stds.append(shuffle_maes.std())

        # Correlation
        y_true_pool = np.concatenate(dec['y_true'])
        y_pred_pool = np.concatenate(dec['y_pred'])
        r, _ = pearsonr(y_true_pool, y_pred_pool)
        correlations.append(r)
        r2s.append(r ** 2)

        # p-value vs shuffle
        p_val = (shuffle_maes <= real_mae).mean()
        p_vals.append(p_val)

    x = np.arange(len(sessions))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel A: Real MAE vs Shuffle MAE ──────────────────────────────────────
    ax = axes[0]
    w = 0.35

    ax.bar(x - w/2, shuffle_means, width=w, color='#CCCCCC', alpha=0.8,
           label='Shuffle MAE', edgecolor='none',
           yerr=shuffle_stds, capsize=4, error_kw={'elinewidth': 1.2})

    for i, (mae, grp) in enumerate(zip(real_maes, groups)):
        clr = grp_color.get(grp, 'steelblue')
        ax.bar(x[i] + w/2, mae, width=w, color=clr, alpha=0.9, edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_str', '') for s in sessions],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Absolute Error (s)', fontsize=11)
    ax.set_title('Decoder MAE vs. Shuffle\n(colored = real, gray = shuffle ± SD)', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')

    for i, p in enumerate(p_vals):
        y_pos = max(shuffle_means[i] + shuffle_stds[i], real_maes[i]) + 0.05
        if p < 0.0001:
            ax.text(x[i], y_pos, '***', ha='center', va='bottom', fontsize=10, fontweight='bold')
        elif p < 0.001:
            ax.text(x[i], y_pos, '**', ha='center', va='bottom', fontsize=10)
        elif p < 0.05:
            ax.text(x[i], y_pos, '*', ha='center', va='bottom', fontsize=10)

    # ── Panel B: Correlation per session ──────────────────────────────────────
    ax = axes[1]

    for i, (r, grp) in enumerate(zip(correlations, groups)):
        clr = grp_color.get(grp, 'steelblue')
        ax.bar(x[i], r, width=0.6, color=clr, alpha=0.85, edgecolor='none')
        ax.text(x[i], r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_str', '') for s in sessions],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Pearson r (decoded vs. actual)', fontsize=11)
    ax.set_title('Decoding Correlation per Session', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color='gray', ls='-', lw=0.5)

    # ── Panel C: Mean trajectory overlay (all sessions) ───────────────────────
    ax = axes[2]

    for rc in all_raw_caches:
        dec = rc['pooled']
        grp = rc['group']
        sid = rc['session_id']
        clr = grp_color.get(grp, 'steelblue')

        y_true_pool = np.concatenate(dec['y_true'])
        y_pred_pool = np.concatenate(dec['y_pred'])

        bin_edges = np.arange(0, MAX_DECODE + 0.25, 0.25)
        bin_ctrs = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_dec = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_true_pool >= lo) & (y_true_pool < hi)
            if mask.sum() >= 5:
                mean_dec.append(y_pred_pool[mask].mean())
            else:
                mean_dec.append(np.nan)
        mean_dec = np.array(mean_dec)
        valid = ~np.isnan(mean_dec)

        label = f"{sid.split('_')[0]} ({grp.split()[0]})"
        ax.plot(bin_ctrs[valid], mean_dec[valid], '-', color=clr, lw=2, alpha=0.7, label=label)

    ax.plot([0, MAX_DECODE], [0, MAX_DECODE], 'k--', lw=1.5, alpha=0.7, label='Unity')
    ax.set_xlabel('Actual elapsed time (s)', fontsize=11)
    ax.set_ylabel('Mean decoded time (s)', fontsize=11)
    ax.set_title('Mean Decoding Trajectory (All Sessions)', fontsize=10)
    ax.set_xlim(0, MAX_DECODE)
    ax.set_ylim(0, MAX_DECODE)
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    out_path = save_dir / 'committee_point1_summary.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Anchor comparison (per session) ──────────────────────────────────────────

def run_anchor_decoders(t_sess, s_sess):
    """
    Run pooled decoder for each alignment anchor on the same session data.

    Returns
    -------
    dict keyed by anchor key ('cue_on', 'cue_off', 'last_lick'), each value:
        {'mae': float, 'r': float, 'r2': float, 'n_trials': int}
        or None if the anchor column is missing / too few trials.
    """
    results = {}
    for key, col in ANCHOR_COLS.items():
        try:
            data = build_population_matrix(t_sess, s_sess, col)
        except KeyError:
            results[key] = None
            continue
        if data is None or data['n_trials'] < 10:
            results[key] = None
            continue
        dec = pooled_decode(data)
        y_true = np.concatenate(dec['y_true'])
        y_pred = np.concatenate(dec['y_pred'])
        r, _   = pearsonr(y_true, y_pred)
        results[key] = {
            'mae'     : dec['mae_per_trial'].mean(),
            'r'       : r,
            'r2'      : r ** 2,
            'n_trials': data['n_trials'],
        }
    return results


def plot_anchor_comparison_session(anchor_results, shuffle_mae, config, save_dir):
    """
    Per-session decoding accuracy across all three alignment anchors.

    Panel A: MAE per anchor (bar) with shuffle MAE reference line
    Panel B: Pearson r per anchor (bar)
    """
    sid = config['session_id']
    grp = config['group']

    keys    = list(ANCHOR_COLS.keys())
    labels  = [ANCHOR_LABELS[k] for k in keys]
    colors  = [ANCHOR_COLORS[k]  for k in keys]
    maes    = [anchor_results[k]['mae'] if anchor_results[k] else np.nan for k in keys]
    rs      = [anchor_results[k]['r']   if anchor_results[k] else np.nan for k in keys]
    n_valid = [anchor_results[k]['n_trials'] if anchor_results[k] else 0  for k in keys]

    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle(f'{grp} — Anchor Comparison\n{sid}', fontsize=12)

    # Panel A: MAE
    ax = axes[0]
    bars = ax.bar(x, maes, color=colors, alpha=0.85, edgecolor='none', width=0.55)
    ax.axhline(shuffle_mae, color='#888888', ls='--', lw=1.5,
               label=f'Shuffle MAE = {shuffle_mae:.2f} s')
    for bar, mae, n in zip(bars, maes, n_valid):
        if not np.isnan(mae):
            ax.text(bar.get_x() + bar.get_width() / 2, mae + 0.02,
                    f'{mae:.2f}\n(n={n})', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Mean Absolute Error (s)', fontsize=11)
    ax.set_title('MAE by Alignment Anchor', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

    # Panel B: Pearson r
    ax = axes[1]
    bars = ax.bar(x, rs, color=colors, alpha=0.85, edgecolor='none', width=0.55)
    for bar, r in zip(bars, rs):
        if not np.isnan(r):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                    f'{r:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Pearson r', fontsize=11)
    ax.set_title('Decoding Correlation by Alignment Anchor', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color='gray', ls='-', lw=0.5)

    plt.tight_layout()
    out_path = save_dir / f'anchor_comparison_{sid}.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Per-session pipeline ──────────────────────────────────────────────────────

def run_analysis(session_id, group, trials_df, spikes_df, save_dir):
    """
    Run decoder pipeline for one session.
    USE_LOTO = False → pooled only (fast)
    USE_LOTO = True  → pooled + LOTO with comparison figures
    """
    cfg = {**GROUP_CONFIG[group], 'session_id': session_id, 'group': group}

    print(f"\n{'='*72}")
    print(f"  {session_id}  ({group})  [{'pooled + LOTO' if USE_LOTO else 'pooled only'}]")
    print(f"{'='*72}")

    t_sess = enrich_with_reward(trials_df, session_id)
    s_sess = spikes_df[spikes_df['session'] == session_id].copy()
    print(f"  Trials: {len(t_sess)} ({(~t_sess['miss_trial']).sum()} non-miss)   "
          f"Units: {s_sess['unit_id'].nunique()}")

    print("\nBuilding population matrices...")
    data = build_population_matrix(t_sess, s_sess, cfg['t_ref_col'])
    print(f"  {data['n_trials']} trials, {data['n_units']} units")

    # ── Pooled decoder (always) ───────────────────────────────────────────────
    print("\n[Pooled] Running decoder (train 80%, test all)...")
    pooled_dec = pooled_decode(data)
    print(f"  MAE: {pooled_dec['mae_per_trial'].mean():.3f} s   "
          f"R²: {np.nanmean(pooled_dec['r2_per_trial']):.3f}")

    # ── LOTO decoder (optional) ───────────────────────────────────────────────
    if USE_LOTO:
        print("\n[LOTO] Running leave-one-trial-out decoder...")
        loto_dec = loto_decode(data)
        print(f"  MAE: {loto_dec['mae_per_trial'].mean():.3f} s   "
              f"R²: {np.nanmean(loto_dec['r2_per_trial']):.3f}")

    # ── Shuffle (pooled null, always) ─────────────────────────────────────────
    print(f"\nShuffle control ({N_SHUFFLES} permutations)...")
    shuffle_results = shuffle_decode(data, use_loto=False)
    p_pooled = (shuffle_results['shuffle_maes'] <= pooled_dec['mae_per_trial'].mean()).mean()
    print(f"  Shuffle MAE: {shuffle_results['shuffle_maes'].mean():.3f} ± "
          f"{shuffle_results['shuffle_maes'].std():.3f}")
    print(f"  p (pooled): {p_pooled:.6f}")
    if USE_LOTO:
        p_loto = (shuffle_results['shuffle_maes'] <= loto_dec['mae_per_trial'].mean()).mean()
        print(f"  p (LOTO):   {p_loto:.6f}")

    # ── Clock speeds ──────────────────────────────────────────────────────────
    print("\nExtracting clock speeds...")
    pooled_speeds = extract_clock_speeds(pooled_dec)
    v = pooled_speeds['clock_speed'][~np.isnan(pooled_speeds['clock_speed'])]
    print(f"  Pooled: mean={v.mean():.4f} ± {v.std():.4f}")
    if USE_LOTO:
        loto_speeds = extract_clock_speeds(loto_dec)
        v2 = loto_speeds['clock_speed'][~np.isnan(loto_speeds['clock_speed'])]
        print(f"  LOTO:   mean={v2.mean():.4f} ± {v2.std():.4f}")

    # ── History effects ───────────────────────────────────────────────────────
    print("\nHistory effects...")
    pooled_hist       = analyze_history_effect(pooled_dec, pooled_speeds)
    pooled_time_match = time_matched_history_analysis(pooled_dec, pooled_speeds)
    print(f"  Pooled: after_rew={pooled_hist['after_reward_mean']:.4f}  "
          f"after_no_rew={pooled_hist['after_no_reward_mean']:.4f}  "
          f"p={pooled_hist['p_ttest']:.6f}  d={pooled_hist['cohens_d']:.3f}")
    if USE_LOTO:
        loto_hist       = analyze_history_effect(loto_dec, loto_speeds)
        loto_time_match = time_matched_history_analysis(loto_dec, loto_speeds)
        print(f"  LOTO:   after_rew={loto_hist['after_reward_mean']:.4f}  "
              f"after_no_rew={loto_hist['after_no_reward_mean']:.4f}  "
              f"p={loto_hist['p_ttest']:.6f}  d={loto_hist['cohens_d']:.3f}")

    # ── Save figures ──────────────────────────────────────────────────────────
    print("\nSaving figures...")
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_decoder_performance(pooled_dec, shuffle_results, data, cfg, save_dir, method='pooled')
    plot_clock_speed_analysis(pooled_dec, pooled_speeds, pooled_hist,
                               pooled_time_match, cfg, save_dir, method='pooled')
    plot_pca_trajectories(data, pooled_dec, cfg, save_dir)

    if USE_LOTO:
        plot_decoder_performance(loto_dec, shuffle_results, data, cfg, save_dir, method='loto')
        plot_clock_speed_analysis(loto_dec, loto_speeds, loto_hist,
                                   loto_time_match, cfg, save_dir, method='loto')
        plot_method_comparison(pooled_dec, loto_dec, pooled_speeds, loto_speeds,
                                pooled_hist, loto_hist, shuffle_results, cfg, save_dir)

    # ── Trial results CSV ─────────────────────────────────────────────────────
    p_idx = {tid: i for i, tid in enumerate(pooled_dec['trial_ids'])}
    trial_data = {
        'trial_id'          : pooled_dec['trial_ids'],
        'time_waited'       : pooled_dec['tw'],
        'rewarded'          : pooled_dec['rewarded'],
        'prev_rewarded'     : pooled_dec['prev_rewarded'],
        'bg_repeats'        : pooled_dec['bg_repeats'],
        'pooled_clock_speed': pooled_speeds['clock_speed'],
        'pooled_decoder_r2' : pooled_dec['r2_per_trial'],
        'pooled_decoder_mae': pooled_dec['mae_per_trial'],
    }
    if USE_LOTO:
        l_idx = {tid: i for i, tid in enumerate(loto_dec['trial_ids'])}
        trial_data['loto_clock_speed'] = [
            loto_speeds['clock_speed'][l_idx[t]] if t in l_idx else np.nan
            for t in pooled_dec['trial_ids']
        ]
        trial_data['loto_decoder_r2'] = [
            loto_dec['r2_per_trial'][l_idx[t]] if t in l_idx else np.nan
            for t in pooled_dec['trial_ids']
        ]
        trial_data['loto_decoder_mae'] = [
            loto_dec['mae_per_trial'][l_idx[t]] if t in l_idx else np.nan
            for t in pooled_dec['trial_ids']
        ]
    pd.DataFrame(trial_data).to_csv(save_dir / f'trial_results_{session_id}.csv', index=False)

    # ── Anchor comparison ─────────────────────────────────────────────────────
    print("\nRunning anchor comparison decoders (cue_on / cue_off / last_lick)...")
    anchor_results = run_anchor_decoders(t_sess, s_sess)
    for key, res in anchor_results.items():
        if res:
            print(f"  {ANCHOR_LABELS[key]:12s}: MAE={res['mae']:.3f}  r={res['r']:.3f}  "
                  f"(n={res['n_trials']} trials)")
        else:
            print(f"  {ANCHOR_LABELS[key]:12s}: skipped (missing column or too few trials)")

    # ── Raw decoder cache (pickle) ────────────────────────────────────────────
    _keep = ('y_true', 'y_pred', 'trial_ids', 'tw', 'rewarded', 'prev_rewarded',
             'bg_repeats', 'mae_per_trial', 'r2_per_trial', 'bias_per_trial')
    raw_cache = {
        'session_id'    : session_id,
        'group'         : group,
        'n_units'       : data['n_units'],
        'pooled'        : {k: pooled_dec[k] for k in _keep},
        'shuffle_maes'      : shuffle_results['shuffle_maes'],
        'pooled_speeds'     : pooled_speeds,
        'pooled_hist'       : pooled_hist,
        'pooled_time_match' : pooled_time_match,
        'anchor_results'    : anchor_results,
    }
    if USE_LOTO:
        raw_cache['loto']        = {k: loto_dec[k] for k in _keep}
        raw_cache['loto_speeds'] = loto_speeds
        raw_cache['loto_hist']   = loto_hist
    with open(save_dir / f'decoder_raw_{session_id}.pkl', 'wb') as fh:
        pickle.dump(raw_cache, fh)

    # ── Committee figures ─────────────────────────────────────────────────────
    plot_committee_point1(raw_cache, cfg, save_dir)
    plot_committee_point2(raw_cache, cfg, save_dir)
    plot_committee_point3_session(raw_cache, cfg, save_dir)
    plot_anchor_comparison_session(anchor_results, shuffle_results['shuffle_maes'].mean(),
                                   cfg, save_dir)

    print(f"  Saved to: {save_dir}")

    result = dict(session_id=session_id, group=group,
                  pooled_r2=np.nanmean(pooled_dec['r2_per_trial']),
                  pooled_mae=pooled_dec['mae_per_trial'].mean(),
                  pooled_shuffle_p=p_pooled,
                  pooled_history_p=pooled_hist['p_ttest'],
                  pooled_history_d=pooled_hist['cohens_d'])
    if USE_LOTO:
        result.update(loto_r2=np.nanmean(loto_dec['r2_per_trial']),
                      loto_mae=loto_dec['mae_per_trial'].mean(),
                      loto_shuffle_p=p_loto,
                      loto_history_p=loto_hist['p_ttest'],
                      loto_history_d=loto_hist['cohens_d'])
    return result


def plot_time_matched_summary(all_raw_caches, save_dir):
    """
    Cross-session summary of the time-matched (quartile) confound control.

    Panel A: Δ clock speed (after-reward − after-no-reward) per quartile,
             mean ± SEM across sessions with individual session dots overlaid.
             A consistent effect across all quartiles rules out the wait-time confound.

    Panel B: Per-session × per-quartile heatmap of Δ clock speed,
             with asterisks for p < 0.05. Shows which sessions / quartiles drive the effect.
    """
    if not all_raw_caches:
        return

    grp_color = {'Long BG': '#9080ff', 'Short BG': '#ffb400'}

    # ── Collect per-session quartile deltas ───────────────────────────────────
    # First pass: find how many quartiles (should always be 4, but be safe)
    n_q = max(len(rc['pooled_time_match']) for rc in all_raw_caches
              if 'pooled_time_match' in rc)

    sessions, groups = [], []
    # deltas[q][session] and pvals[q][session]
    deltas = [[] for _ in range(n_q)]
    pvals  = [[] for _ in range(n_q)]
    q_labels = None

    for rc in all_raw_caches:
        tm = rc.get('pooled_time_match')
        if not tm:
            continue
        sessions.append(rc['session_id'])
        groups.append(rc['group'])
        for qi, qr in enumerate(tm):
            d = (qr['after_rew_mean'] - qr['after_no_rew_mean']
                 if not (np.isnan(qr.get('after_rew_mean', np.nan)) or
                         np.isnan(qr.get('after_no_rew_mean', np.nan)))
                 else np.nan)
            deltas[qi].append(d)
            pvals[qi].append(qr.get('p_value', np.nan))
        if q_labels is None:
            q_labels = [f"Q{qr['quantile']}\n{qr['tw_range'][0]:.1f}–{qr['tw_range'][1]:.1f}s"
                        for qr in tm]

    if not sessions:
        print("  [warn] no pooled_time_match data in any raw cache — skipping summary")
        return

    n_sess = len(sessions)
    x_q    = np.arange(n_q)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Time-Matched Confound Control (All Sessions)', fontsize=13)

    # ── Panel A: mean Δ ± SEM per quartile, dots per session ─────────────────
    ax = axes[0]
    mean_d = np.array([np.nanmean(deltas[qi]) for qi in range(n_q)])
    sem_d  = np.array([np.nanstd(deltas[qi]) / np.sqrt(np.sum(~np.isnan(deltas[qi])))
                       for qi in range(n_q)])

    ax.bar(x_q, mean_d, color='#BBBBBB', alpha=0.7, edgecolor='none', width=0.5,
           yerr=sem_d, capsize=5, error_kw={'elinewidth': 1.5}, zorder=2)

    # Individual session dots, jittered and colored by group
    rng = np.random.default_rng(42)
    for si, (sid, grp) in enumerate(zip(sessions, groups)):
        clr = grp_color.get(grp, 'steelblue')
        for qi in range(n_q):
            d = deltas[qi][si]
            if not np.isnan(d):
                jitter = rng.uniform(-0.18, 0.18)
                ax.scatter(x_q[qi] + jitter, d, color=clr, s=40, alpha=0.8,
                           zorder=3, linewidths=0)

    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xticks(x_q)
    ax.set_xticklabels(q_labels, fontsize=10)
    ax.set_xlabel('Wait-time quartile', fontsize=11)
    ax.set_ylabel('Δ clock speed (after reward − no reward)', fontsize=11)
    ax.set_title('History Effect Within Wait-Time Quartiles\n(gray bars = mean ± SEM, dots = sessions)',
                 fontsize=10)

    # Group legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=c, markersize=8, label=g)
               for g, c in grp_color.items()]
    ax.legend(handles=handles, fontsize=9, loc='upper right')

    # ── Panel B: heatmap — sessions × quartiles ───────────────────────────────
    ax = axes[1]
    delta_mat = np.array([[deltas[qi][si] for qi in range(n_q)]
                           for si in range(n_sess)])   # (n_sess, n_q)
    pval_mat  = np.array([[pvals[qi][si]  for qi in range(n_q)]
                           for si in range(n_sess)])

    vmax = np.nanpercentile(np.abs(delta_mat), 95)
    im = ax.imshow(delta_mat, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Δ clock speed', shrink=0.85)

    # Asterisks for p < 0.05
    for si in range(n_sess):
        for qi in range(n_q):
            p = pval_mat[si, qi]
            if not np.isnan(p) and p < 0.05:
                ax.text(qi, si, '*', ha='center', va='center', fontsize=12,
                        color='white' if abs(delta_mat[si, qi]) > vmax * 0.5 else 'black')

    short_labels = [s.replace('_str', '').split('_')[0] for s in sessions]
    ax.set_yticks(range(n_sess))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xticks(x_q)
    ax.set_xticklabels([f'Q{qi+1}' for qi in range(n_q)], fontsize=10)
    ax.set_xlabel('Wait-time quartile', fontsize=11)
    ax.set_title('Δ Clock Speed per Session × Quartile\n(* = p < 0.05)', fontsize=10)

    plt.tight_layout()
    out_path = save_dir / 'time_matched_confound_summary.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Cross-session summary plots ───────────────────────────────────────────────

def run_summary_plots(results_dir=None):
    """
    Load saved CSVs and regenerate cross-session summary figures.
    Can be called independently (PLOT_ONLY = True) without re-running analysis.

    Reads from:
      results_dir/session_summary.csv      — one row per session
      results_dir/<session>/trial_results_<session>.csv  — one row per trial

    Saves to:
      results_dir/summary_decoder_performance.png
      results_dir/summary_clock_speed.png
      results_dir/summary_history_effect.png
    """
    if results_dir is None:
        results_dir = OUT_DIR / 'results'
    results_dir = results_dir if hasattr(results_dir, '__fspath__') else __import__('pathlib').Path(results_dir)

    summary_path = results_dir / 'session_summary.csv'
    if not summary_path.exists():
        print(f"No session_summary.csv found at {summary_path}. Run analysis first.")
        return

    summary = pd.read_csv(summary_path).reset_index(drop=True)
    print(f"Loaded {len(summary)} sessions from {summary_path}")

    # Load all trial-level CSVs
    trial_frames = []
    for _, row in summary.iterrows():
        sess     = row['session_id']
        csv_path = results_dir / sess / f'trial_results_{sess}.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['session_id'] = sess
            df['group']      = row['group']
            trial_frames.append(df)
        else:
            print(f"  [warn] missing trial CSV for {sess}")
    trials_all = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()

    # ── colour helpers ────────────────────────────────────────────────────────
    grp_color = {'Long BG': '#9080ff', 'Short BG': '#ffb400'}
    groups    = summary['group'].unique()

    # ── Fig 1: decoder performance across sessions ────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('Cross-Session Decoder Performance', fontsize=13)

    for grp in groups:
        s   = summary[summary['group'] == grp]
        clr = grp_color.get(grp, 'steelblue')
        x   = s.index.to_numpy()  # actual position in full session list

        # R²
        axes1[0].bar(x + (0.2 if grp == 'Short BG' else -0.2),
                     s['pooled_r2'], width=0.35, color=clr, alpha=0.8,
                     label=grp, edgecolor='none')
        # MAE
        axes1[1].bar(x + (0.2 if grp == 'Short BG' else -0.2),
                     s['pooled_mae'], width=0.35, color=clr, alpha=0.8,
                     label=grp, edgecolor='none')

    # Shuffle significance per session
    ax_p = axes1[2]
    for i, (_, row) in enumerate(summary.iterrows()):
        clr   = grp_color.get(row['group'], 'steelblue')
        sig   = row.get('pooled_shuffle_p', np.nan)
        alpha = 0.9 if (not np.isnan(sig) and sig < 0.05) else 0.35
        ax_p.scatter(i, -np.log10(max(sig, 1e-10)) if not np.isnan(sig) else np.nan,
                     color=clr, s=60, alpha=alpha, zorder=3)
    ax_p.axhline(-np.log10(0.05), color='gray', ls='--', lw=1, label='p=0.05')
    ax_p.set_xlabel('Session index')
    ax_p.set_ylabel('−log₁₀(shuffle p)')
    ax_p.set_title('Decoder Significance (pooled)')
    ax_p.legend()

    for ax, ylabel, title in zip(
            axes1[:2],
            ['Mean R²', 'Mean MAE (s)'],
            ['Per-Session R²', 'Per-Session MAE']):
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(summary)))
        ax.set_xticklabels(summary['session_id'], rotation=45, ha='right', fontsize=7)
        ax.legend()

    plt.tight_layout()
    out1 = results_dir / 'summary_decoder_performance.png'
    fig1.savefig(str(out1), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: {out1}")

    # ── Fig 2: clock speed across sessions ───────────────────────────────────
    if not trials_all.empty and 'pooled_clock_speed' in trials_all.columns:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        fig2.suptitle('Cross-Session Clock Speed', fontsize=13)

        # Panel A: per-session mean clock speed ± SD
        # Preserve session order from summary so x positions align with labels
        sess_cs = (trials_all.groupby(['session_id', 'group'])['pooled_clock_speed']
                   .agg(['mean', 'std']).reset_index())
        sess_order = {sid: i for i, sid in enumerate(summary['session_id'])}
        sess_cs['pos'] = sess_cs['session_id'].map(sess_order)
        sess_cs = sess_cs.sort_values('pos').reset_index(drop=True)

        for grp in groups:
            g   = sess_cs[sess_cs['group'] == grp]
            clr = grp_color.get(grp, 'steelblue')
            x   = g['pos'].to_numpy()
            axes2[0].bar(x + (0.2 if grp == 'Short BG' else -0.2),
                         g['mean'], width=0.35, color=clr, alpha=0.8,
                         label=grp, edgecolor='none', yerr=g['std'], capsize=3)
        axes2[0].axhline(1.0, color='k', ls='--', lw=1, label='Unity')
        axes2[0].set_ylabel('Mean clock speed')
        axes2[0].set_title('Mean Clock Speed per Session (±SD)')
        axes2[0].set_xticks(np.arange(len(summary)))
        axes2[0].set_xticklabels(summary['session_id'], rotation=45, ha='right', fontsize=7)
        axes2[0].legend()

        # Panel B: pooled distribution per group (all trials)
        for grp in groups:
            clr = grp_color.get(grp, 'steelblue')
            cs  = trials_all.loc[trials_all['group'] == grp, 'pooled_clock_speed'].dropna()
            axes2[1].hist(cs, bins=40, color=clr, alpha=0.55, label=f'{grp} (n={len(cs)})')
        axes2[1].axvline(1.0, color='k', ls='--', lw=1.5, label='Unity')
        axes2[1].set_xlabel('Clock speed')
        axes2[1].set_ylabel('Count')
        axes2[1].set_title('Clock Speed Distribution (all trials)')
        axes2[1].legend()

        # Panel C: clock speed vs time_waited scatter per group
        for grp in groups:
            clr  = grp_color.get(grp, 'steelblue')
            sub  = trials_all[trials_all['group'] == grp].dropna(
                      subset=['pooled_clock_speed', 'time_waited'])
            axes2[2].scatter(sub['time_waited'], sub['pooled_clock_speed'],
                             color=clr, alpha=0.2, s=8, label=grp)
            if len(sub) > 10:
                z   = np.polyfit(sub['time_waited'], sub['pooled_clock_speed'], 1)
                tw_rng = np.linspace(sub['time_waited'].min(), sub['time_waited'].max(), 100)
                axes2[2].plot(tw_rng, np.poly1d(z)(tw_rng), color=clr, lw=2)
        axes2[2].axhline(1.0, color='gray', ls='--', lw=1)
        axes2[2].set_xlabel('Time waited (s)')
        axes2[2].set_ylabel('Clock speed')
        axes2[2].set_title('Clock Speed vs. Time Waited')
        axes2[2].legend()

        plt.tight_layout()
        out2 = results_dir / 'summary_clock_speed.png'
        fig2.savefig(str(out2), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: {out2}")

    # ── Fig 3: history effect across sessions ─────────────────────────────────
    if not trials_all.empty and 'prev_rewarded' in trials_all.columns:
        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
        fig3.suptitle('Cross-Session History Effect (clock speed ~ prev reward)',
                      fontsize=13)

        # Panel A: per-session Cohen's d
        cohens_col = 'pooled_history_d' if 'pooled_history_d' in summary.columns else None
        if cohens_col:
            for grp in groups:
                s   = summary[summary['group'] == grp]
                clr = grp_color.get(grp, 'steelblue')
                x   = s.index.to_numpy()  # actual position in full session list
                axes3[0].bar(x + (0.2 if grp == 'Short BG' else -0.2),
                             s[cohens_col], width=0.35, color=clr, alpha=0.8,
                             label=grp, edgecolor='none')
            axes3[0].axhline(0, color='gray', ls='--', lw=1)
            axes3[0].set_ylabel("Cohen's d")
            axes3[0].set_title("History Effect Size per Session")
            axes3[0].set_xticks(np.arange(len(summary)))
            axes3[0].set_xticklabels(summary['session_id'], rotation=45, ha='right', fontsize=7)
            axes3[0].legend()

        # Panel B: after-reward vs after-no-reward clock speed, pooled across groups
        for ax_i, grp in enumerate(groups):
            ax = axes3[1] if len(groups) == 1 else axes3[1 + ax_i]
            sub  = trials_all[trials_all['group'] == grp].dropna(
                      subset=['pooled_clock_speed', 'prev_rewarded'])
            rew  = sub.loc[sub['prev_rewarded'] == 1, 'pooled_clock_speed']
            nrew = sub.loc[sub['prev_rewarded'] == 0, 'pooled_clock_speed']
            bp = ax.boxplot([nrew, rew], positions=[0, 1], widths=0.5, patch_artist=True)
            for patch, fc in zip(bp['boxes'], ['lightgray', 'lightgreen']):
                patch.set_facecolor(fc)
            ax.axhline(1.0, color='gray', ls='--', lw=1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['After\nNo Reward', 'After\nReward'])
            ax.set_ylabel('Clock speed')
            ax.set_title(f'{grp}\nAfter-Reward vs After-No-Reward')
            if len(groups) == 1:
                break

        # Panel C: history p-value per session
        p_col = 'pooled_history_p' if 'pooled_history_p' in summary.columns else None
        if p_col:
            for i, (_, row) in enumerate(summary.iterrows()):
                clr = grp_color.get(row['group'], 'steelblue')
                p   = row.get(p_col, np.nan)
                if not np.isnan(p):
                    axes3[2].scatter(i, -np.log10(max(p, 1e-10)),
                                     color=clr, s=60,
                                     alpha=0.9 if p < 0.05 else 0.4, zorder=3)
            axes3[2].axhline(-np.log10(0.05), color='gray', ls='--', lw=1, label='p=0.05')
            axes3[2].set_xlabel('Session index')
            axes3[2].set_ylabel('−log₁₀(history p)')
            axes3[2].set_title('History Effect Significance')
            axes3[2].legend()

        plt.tight_layout()
        out3 = results_dir / 'summary_history_effect.png'
        fig3.savefig(str(out3), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"  Saved: {out3}")

    # ── Committee summary: load raw pickles and generate cross-session figure ──
    all_raw_caches = []
    for _, row in summary.iterrows():
        sess     = row['session_id']
        pkl_path = results_dir / sess / f'decoder_raw_{sess}.pkl'
        if pkl_path.exists():
            with open(pkl_path, 'rb') as fh:
                all_raw_caches.append(pickle.load(fh))
        else:
            print(f"  [warn] no raw cache pickle for {sess} — skipping from committee summary")

    if all_raw_caches:
        plot_committee_point1_summary(all_raw_caches, results_dir)
        plot_committee_point3_summary(all_raw_caches, results_dir)
        plot_time_matched_summary(all_raw_caches, results_dir)

    print(f"\nSummary plots written to: {results_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if PLOT_ONLY:
        # ── Skip analysis, regenerate summary plots from saved CSVs ──────────
        print("PLOT_ONLY=True — loading saved results and regenerating summary plots...")
        run_summary_plots(OUT_DIR / 'results')
    else:
        # ── Full analysis ─────────────────────────────────────────────────────
        print("Loading data...")
        trials_df, spikes_df, units_df = load_decoder_data(
            group_dict=GROUP_DICT,
            msn_csv=MSN_CSV,
        )

        # Print full session table and get qualifying sessions
        qualifying = get_qualifying_sessions(trials_df, units_df)

        if qualifying.empty:
            print("\nNo sessions meet the threshold. Adjust MIN_UNITS / MIN_TRIALS.")
        else:
            all_results = []
            for _, row in qualifying.iterrows():
                sess     = row['session']
                group    = row['group']
                sess_dir = OUT_DIR / 'results' / sess
                try:
                    result = run_analysis(sess, group, trials_df, spikes_df, sess_dir)
                    all_results.append(result)
                except Exception as e:
                    print(f"\nERROR — {sess}: {e}")

            # Summary table across all sessions
            if all_results:
                summary_df   = pd.DataFrame(all_results)
                summary_path = OUT_DIR / 'results' / 'session_summary.csv'
                summary_df.to_csv(summary_path, index=False)
                print(f"\n{'='*72}")
                print("ALL SESSIONS SUMMARY")
                print(f"{'='*72}")
                cols = ['session_id', 'group',
                        'pooled_r2', 'pooled_mae', 'pooled_shuffle_p',
                        'pooled_history_p', 'pooled_history_d']
                if USE_LOTO:
                    cols += ['loto_r2', 'loto_mae', 'loto_shuffle_p',
                             'loto_history_p', 'loto_history_d']
                print(summary_df[cols].to_string(index=False))
                print(f"\nSummary saved to: {summary_path}")

                # Generate summary plots after analysis
                run_summary_plots(OUT_DIR / 'results')

        print(f"\nDone. All output in {OUT_DIR / 'results'}")
