"""
V1 → DMS Cross-Correlation Analysis
=====================================

Tests whether V1 activity leads DMS activity during the timing interval,
supporting the hypothesis that V1 provides timing input to striatum.

Analysis:
  1. Load simultaneous V1 + STR sessions from the recording log (Google Sheets).
     A session is simultaneous only if the recording log marks BOTH the V1 and
     STR insertions as simultaneous on the same date.
  2. Filter V1 units to visual-cortex neurons (RZ_v1_cortical.csv, waveform-verified).
     Filter STR units to Tier-2 MSNs (RZ_msn_waveform.csv, waveform-primary).
  3. Compute jitter-corrected spike cross-correlograms for all V1-MSN unit pairs.
  4. Test for V1-lead latency (~5-15 ms expected for monosynaptic corticostriatal).

Expected result if V1 → DMS:
  - CCG peak at positive lag (V1 spikes precede DMS spikes)
  - Peak latency ~5-15 ms (monosynaptic)
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

import paths as p

# ── Config ─────────────────────────────────────────────────────────────────────
CCG_WINDOW    = 0.100   # ±100 ms
CCG_BIN       = 0.001   # 1 ms bins
JITTER_WINDOW = 0.025   # 25 ms jitter for shuffle control
N_SHUFFLES    = 100

ANALYSIS_START = 0.5    # seconds after cue offset
ANALYSIS_END   = 4.0    # seconds after cue offset

MIN_FIRING_RATE = 0.5   # Hz
MIN_TRIALS      = 50

OUT_DIR = p.DATA_DIR / 'v1_dms_crosscorrelation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Unit metadata CSVs (outputs of 0g_cell_type_relabeling.py) ────────────────
V1_CSV  = p.LOGS_DIR / 'RZ_v1_cortical.csv'   # visual cortex, v1_cortical=True
MSN_CSV = p.LOGS_DIR / 'RZ_msn_waveform.csv'  # Tier-2 MSNs, STR probe


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def find_simultaneous_sessions():
    """
    Return list of dicts {mouse, date, v1_units, str_units} for sessions
    where both V1 and STR were recorded simultaneously (per recording log)
    AND both have units passing QC in the respective unit CSVs.

    Simultaneity is determined from the Google Sheets recording log: a session
    counts only if both the V1 insertion AND the STR insertion are marked 'y'.
    """
    # Recording log (local copy of Google Sheets)
    log = pd.read_csv(p.LOGS_DIR / 'recording_log.csv')
    log['region'] = log['region'].str.strip().str.lower()
    log['simultaneous'] = log['simultaneous'].str.strip().str.lower()

    sim_log = log[log['simultaneous'] == 'y']
    by_session = sim_log.groupby(['mouse', 'date'])['region'].apply(set)
    # Keep only sessions that have BOTH probes marked simultaneous
    both_probes = by_session[by_session.apply(lambda s: 'v1' in s and 'str' in s)]
    sim_sessions = both_probes.index.tolist()  # list of (mouse, date) tuples

    # Load unit metadata
    v1_df  = pd.read_csv(V1_CSV)
    msn_df = pd.read_csv(MSN_CSV)

    v1_df['date_str']  = pd.to_datetime(v1_df['datetime']).dt.strftime('%Y-%m-%d')
    msn_df['date_str'] = pd.to_datetime(msn_df['datetime']).dt.strftime('%Y-%m-%d')

    # v1_cortical is True for all rows in RZ_v1_cortical.csv (already filtered)
    # Build per-session unit ID sets
    v1_by_session  = v1_df.groupby(['mouse', 'date_str'])['id'].apply(set).to_dict()
    msn_by_session = msn_df.groupby(['mouse', 'date_str'])['id'].apply(set).to_dict()

    sessions = []
    for mouse, date in sim_sessions:
        v1_pkl  = p.PICKLE_DIR / f'{mouse}_{date}_v1.pkl'
        str_pkl = p.PICKLE_DIR / f'{mouse}_{date}_str.pkl'

        if not v1_pkl.exists() or not str_pkl.exists():
            continue

        v1_units  = v1_by_session.get((mouse, date), set())
        str_units = msn_by_session.get((mouse, date), set())

        if len(v1_units) == 0 or len(str_units) == 0:
            continue

        sessions.append({
            'mouse':     mouse,
            'date':      date,
            'v1_pkl':    v1_pkl,
            'str_pkl':   str_pkl,
            'v1_units':  v1_units,
            'str_units': str_units,
        })

    print(f"Found {len(sessions)} simultaneous sessions with V1 cortical + MSN units")
    return sessions


def load_session_pair(session):
    """
    Load a V1 + STR pickle pair. Returns (v1_data, str_data) dicts.
    """
    with open(session['v1_pkl'], 'rb') as f:
        v1_data = pickle.load(f)
    with open(session['str_pkl'], 'rb') as f:
        str_data = pickle.load(f)
    return v1_data, str_data


# ═══════════════════════════════════════════════════════════════════════════════
# SPIKE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_trial_spikes(units_dict, unit_ids, trial_ids, t_start, t_end):
    """
    Extract spike times (relative to cue offset) within the analysis window.

    Parameters
    ----------
    units_dict : {unit_id: DataFrame}  — pickle units dict, each df has
                 columns [trial_id, to_cue_off, ...]
    unit_ids   : set of int — unit IDs to include (QC-filtered)
    trial_ids  : list of trial IDs to include
    t_start, t_end : float — analysis window relative to cue offset (seconds)

    Returns
    -------
    {unit_id: {trial_id: np.ndarray of spike times relative to cue off}}
    """
    trial_set = set(trial_ids)
    result = {}

    for uid, spk_df in units_dict.items():
        if uid not in unit_ids:
            continue

        spk_df = spk_df[spk_df['trial_id'].isin(trial_set)]
        unit_spikes = {}

        for tid, grp in spk_df.groupby('trial_id'):
            times = grp['to_cue_off'].values
            mask  = (times >= t_start) & (times <= t_end)
            unit_spikes[tid] = times[mask]

        result[uid] = unit_spikes

    return result


def mean_firing_rate(spikes_dict, n_trials, window_dur):
    total = sum(len(sp) for sp in spikes_dict.values())
    return total / (n_trials * window_dur) if n_trials > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-CORRELOGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ccg(spikes1, spikes2, window=CCG_WINDOW, bin_size=CCG_BIN):
    """
    Spike cross-correlogram: spikes1 as reference, spikes2 as target.

    Returns (ccg counts, lag bin centers in seconds).
    Positive lag = spikes2 follows spikes1 (spikes1 leads).
    """
    n_bins = int(2 * window / bin_size)
    ccg = np.zeros(n_bins)

    for s1 in spikes1:
        diffs = spikes2 - s1
        valid = (diffs >= -window) & (diffs < window)
        indices = ((diffs[valid] + window) / bin_size).astype(int)
        indices = np.clip(indices, 0, n_bins - 1)
        np.add.at(ccg, indices, 1)

    lags = np.linspace(-window + bin_size / 2, window - bin_size / 2, n_bins)
    return ccg, lags


def compute_pairwise_ccg(v1_spikes, str_spikes, trial_ids):
    """Sum CCG across all trials for one V1-STR unit pair."""
    ccg_sum = None
    lags = None

    for tid in trial_ids:
        v1_sp  = v1_spikes.get(tid)
        str_sp = str_spikes.get(tid)

        if v1_sp is None or str_sp is None:
            continue
        if len(v1_sp) == 0 or len(str_sp) == 0:
            continue

        ccg, lags = compute_ccg(v1_sp, str_sp)
        ccg_sum = ccg if ccg_sum is None else ccg_sum + ccg

    return ccg_sum, lags


def jitter_spikes(spikes_dict):
    """Jitter spike times uniformly within ±JITTER_WINDOW."""
    return {
        tid: spikes + np.random.uniform(-JITTER_WINDOW, JITTER_WINDOW, len(spikes))
        for tid, spikes in spikes_dict.items()
    }


def compute_jitter_corrected_ccg(v1_spikes, str_spikes, trial_ids, n_shuffles=20):
    """
    Jitter-corrected CCG for one unit pair.

    Returns (raw_ccg, corrected_ccg, shuffle_std, lags).
    """
    raw_ccg, lags = compute_pairwise_ccg(v1_spikes, str_spikes, trial_ids)

    if raw_ccg is None:
        return None, None, None, None

    shuffles = []
    for _ in range(n_shuffles):
        shuf_ccg, _ = compute_pairwise_ccg(
            jitter_spikes(v1_spikes), jitter_spikes(str_spikes), trial_ids
        )
        if shuf_ccg is not None:
            shuffles.append(shuf_ccg)

    if not shuffles:
        return raw_ccg, raw_ccg, np.zeros_like(raw_ccg), lags

    shuffles   = np.array(shuffles)
    shuf_mean  = shuffles.mean(axis=0)
    shuf_std   = shuffles.std(axis=0)
    corrected  = raw_ccg - shuf_mean

    return raw_ccg, corrected, shuf_std, lags


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_session(session, v1_data, str_data):
    """
    Run V1→DMS cross-correlation analysis for one session.

    Returns results dict or None if session is skipped.
    """
    # Use STR trials table (both should be the same behavioural session)
    trials = str_data['trials']
    valid  = trials[trials['missed'] == False].copy()

    if len(valid) < MIN_TRIALS:
        print(f"  Skip: only {len(valid)} valid trials")
        return None

    trial_ids   = valid['trial_id'].tolist()
    window_dur  = ANALYSIS_END - ANALYSIS_START

    # Extract windowed spikes — V1 uses visual-cortex unit IDs, STR uses MSN IDs
    v1_spikes  = get_trial_spikes(v1_data['units'],  session['v1_units'],  trial_ids, ANALYSIS_START, ANALYSIS_END)
    str_spikes = get_trial_spikes(str_data['units'], session['str_units'], trial_ids, ANALYSIS_START, ANALYSIS_END)

    # Filter by minimum firing rate
    v1_good  = {uid: sp for uid, sp in v1_spikes.items()
                if mean_firing_rate(sp, len(trial_ids), window_dur) >= MIN_FIRING_RATE}
    str_good = {uid: sp for uid, sp in str_spikes.items()
                if mean_firing_rate(sp, len(trial_ids), window_dur) >= MIN_FIRING_RATE}

    print(f"  V1 cortical units: {len(v1_good)} / {len(v1_spikes)} (FR >= {MIN_FIRING_RATE} Hz)")
    print(f"  MSN units:         {len(str_good)} / {len(str_spikes)} (FR >= {MIN_FIRING_RATE} Hz)")

    if not v1_good or not str_good:
        print("  Skip: insufficient units after FR filter")
        return None

    # Pairwise CCGs
    n_pairs = len(v1_good) * len(str_good)
    print(f"  Computing {n_pairs} pairwise CCGs...")

    pair_results   = []
    population_ccg = None
    lags_out       = None

    for i, (v1_uid, v1_sp) in enumerate(v1_good.items()):
        for str_uid, str_sp in str_good.items():
            _, corr_ccg, shuf_std, lags = compute_jitter_corrected_ccg(
                v1_sp, str_sp, trial_ids
            )

            if corr_ccg is None:
                continue

            population_ccg = corr_ccg.copy() if population_ccg is None else population_ccg + corr_ccg
            lags_out = lags

            # Peak in positive-lag region (V1 leads STR)
            pos_mask = lags > 0.002
            if not pos_mask.any():
                continue

            pos_ccg  = corr_ccg[pos_mask]
            pos_lags = lags[pos_mask]
            pos_std  = shuf_std[pos_mask] if shuf_std is not None else None

            peak_idx = np.argmax(pos_ccg)
            peak_lag = pos_lags[peak_idx]
            peak_val = pos_ccg[peak_idx]
            z_score  = (peak_val / pos_std[peak_idx]
                        if pos_std is not None and pos_std[peak_idx] > 0 else np.nan)

            pair_results.append({
                'v1_unit':      v1_uid,
                'str_unit':     str_uid,
                'peak_lag_ms':  peak_lag * 1000,
                'peak_value':   peak_val,
                'z_score':      z_score,
            })

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(v1_good)} V1 units processed")

    if not pair_results:
        print("  No valid pairs")
        return None

    pair_df  = pd.DataFrame(pair_results)
    sig_pairs = pair_df[pair_df['z_score'] > 2]

    print(f"  Significant pairs: {len(sig_pairs)}/{len(pair_results)} (z > 2)")
    print(f"  Median peak lag: {pair_df['peak_lag_ms'].median():.2f} ms")

    return {
        'pair_df':        pair_df,
        'population_ccg': population_ccg,
        'lags':           lags_out,
        'n_v1_units':     len(v1_good),
        'n_str_units':    len(str_good),
        'n_pairs':        len(pair_results),
        'n_significant':  len(sig_pairs),
        'median_peak_lag': pair_df['peak_lag_ms'].median(),
        'mean_peak_lag':   pair_df['peak_lag_ms'].mean(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_session(results, session, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    lags_ms   = results['lags'] * 1000
    ccg_sm    = gaussian_filter1d(results['population_ccg'], sigma=2)

    # Population CCG
    ax = axes[0, 0]
    ax.bar(lags_ms, results['population_ccg'], width=1, color='gray', alpha=0.5)
    ax.plot(lags_ms, ccg_sm, 'b-', lw=2)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.axvline(results['median_peak_lag'], color='r', ls='--', lw=2,
               label=f"Median peak: {results['median_peak_lag']:.1f} ms")
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Jitter-corrected coincidences')
    ax.set_title('Population V1 → STR CCG')
    ax.set_xlim(-50, 50)
    ax.legend()

    # Peak lag distribution
    ax = axes[0, 1]
    lags = results['pair_df']['peak_lag_ms']
    ax.hist(lags, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.axvline(lags.median(), color='r', ls='--', lw=2, label=f'Median: {lags.median():.1f} ms')
    ax.axvline(10, color='g', ls=':', lw=2, label='Expected ~10 ms')
    ax.set_xlabel('Peak lag (ms)')
    ax.set_ylabel('Pair count')
    ax.set_title('Peak Lag Distribution')
    ax.legend()

    # Z-score distribution
    ax = axes[1, 0]
    z = results['pair_df']['z_score'].dropna()
    ax.hist(z, bins=30, color='coral', alpha=0.7, edgecolor='white')
    ax.axvline(2, color='r', ls='--', lw=2, label='z = 2')
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Pair count')
    ax.set_title(f'Significance: {results["n_significant"]}/{results["n_pairs"]} pairs (z > 2)')
    ax.legend()

    # Summary text
    ax = axes[1, 1]
    ax.axis('off')
    pct = 100 * results['n_significant'] / results['n_pairs']
    txt = (
        f"Session: {session['mouse']}  {session['date']}\n\n"
        f"V1 cortical units : {results['n_v1_units']}\n"
        f"MSN units         : {results['n_str_units']}\n"
        f"Pairs analyzed    : {results['n_pairs']}\n\n"
        f"Median peak lag   : {results['median_peak_lag']:.2f} ms\n"
        f"Mean peak lag     : {results['mean_peak_lag']:.2f} ms\n\n"
        f"Significant (z>2) : {results['n_significant']} ({pct:.1f}%)\n\n"
        f"+ lag = V1 leads STR\nExpected mono: 5-15 ms"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', va='top')

    plt.suptitle(f"V1 → DMS: {session['mouse']} {session['date']}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary(all_results, save_path):
    if not all_results:
        print("No results to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    all_lags       = []
    session_medians = []
    sig_fracs       = []

    for results in all_results.values():
        all_lags.extend(results['pair_df']['peak_lag_ms'].tolist())
        session_medians.append(results['median_peak_lag'])
        sig_fracs.append(results['n_significant'] / results['n_pairs'])

    # All peak lags
    ax = axes[0]
    ax.hist(all_lags, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.axvline(np.median(all_lags), color='r', ls='--', lw=2,
               label=f'Median: {np.median(all_lags):.1f} ms')
    ax.set_xlabel('Peak lag (ms)')
    ax.set_ylabel('Pair count')
    ax.set_title(f'All Pairs (n={len(all_lags)})')
    ax.legend()

    # Per-session medians
    ax = axes[1]
    ax.bar(range(len(session_medians)), session_medians, color='coral', alpha=0.7)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.axhline(10, color='g', ls=':', lw=2, label='Expected ~10 ms')
    ax.set_xticks(range(len(session_medians)))
    ax.set_xticklabels(list(all_results.keys()), rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Median peak lag (ms)')
    ax.set_title('Per-Session Median Lags')
    ax.legend()

    # Fraction significant per session
    ax = axes[2]
    ax.bar(range(len(sig_fracs)), sig_fracs, color='seagreen', alpha=0.7)
    ax.axhline(0.05, color='r', ls='--', lw=2, label='Chance (5%)')
    ax.set_xticks(range(len(sig_fracs)))
    ax.set_xticklabels(list(all_results.keys()), rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Fraction significant (z > 2)')
    ax.set_title('Significant Pairs per Session')
    ax.set_ylim(0, 1)
    ax.legend()

    plt.suptitle('V1 → DMS CCG: Summary Across Sessions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

sessions = find_simultaneous_sessions()

all_results = {}

for session in sessions:
    session_id = f"{session['mouse']}_{session['date']}"
    print(f"\n{'='*60}\nAnalyzing: {session_id}\n{'='*60}")

    try:
        v1_data, str_data = load_session_pair(session)
        results = analyze_session(session, v1_data, str_data)

        if results is not None:
            all_results[session_id] = results
            plot_session(results, session, OUT_DIR / f'ccg_{session_id}.png')

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

# Summary
if all_results:
    plot_summary(all_results, OUT_DIR / 'ccg_summary.png')

    summary_rows = []
    for sid, r in all_results.items():
        summary_rows.append({
            'session':            sid,
            'n_v1_units':         r['n_v1_units'],
            'n_str_units':        r['n_str_units'],
            'n_pairs':            r['n_pairs'],
            'n_significant':      r['n_significant'],
            'median_peak_lag_ms': r['median_peak_lag'],
            'mean_peak_lag_ms':   r['mean_peak_lag'],
        })

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'ccg_summary.csv', index=False)

    all_lags = [lag for r in all_results.values() for lag in r['pair_df']['peak_lag_ms']]

    from scipy.stats import ttest_1samp, wilcoxon
    t, p_t = ttest_1samp(all_lags, 0)
    _, p_w = wilcoxon(all_lags)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Sessions analyzed : {len(all_results)}")
    print(f"Total pairs       : {len(all_lags)}")
    print(f"Overall median lag: {np.median(all_lags):.2f} ms")
    print(f"Proportion positive: {100 * np.mean(np.array(all_lags) > 0):.1f}%")
    print(f"t-test vs 0       : t={t:.2f}, p={p_t:.2e}")
    print(f"Wilcoxon vs 0     : p={p_w:.2e}")
    print(f"\nOutput saved to: {OUT_DIR}")
else:
    print("\nNo sessions produced results.")
