"""
decoder_quick_test_v2.py
────────────────────────
Quick focused test of the most impactful decoder parameters.
Adapted for the population_decoder_v2 pipeline.

Tests:
  - Bin size: 100ms vs 200ms vs 300ms
  - Smoothing: 1.0 vs 2.0 (in bins)
  - PCA: None vs 10 components
  - bg_repeats filter: True vs False

Total: 24 combinations (vs full sweep).
"""

import pickle
import warnings
from itertools import product
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

from population_decoder_v2 import (
    enrich_with_reward,
    extract_clock_speeds,
    analyze_history_effect,
    GROUP_CONFIG,
)
from population_decoder import load_decoder_data

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# REDUCED PARAMETER GRID (24 combinations)
# ══════════════════════════════════════════════════════════════════════════════

PARAM_GRID = {
    'bin_size'         : [0.10, 0.20, 0.30],
    'smooth_sd'        : [1.0, 2.0],
    'n_pca'            : [None, 10],
    'filter_bg_repeats': [False, True],
}

# Fixed
MAX_DECODE  = 10.0
MIN_WAIT    = 0.3
MIN_BINS    = 5
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
N_SHUFFLES  = 10   # reduced for speed


# ── Spike binning ──────────────────────────────────────────────────────────────

def bin_spikes(spike_times, t_ref, t_end, bin_size, smooth_sd):
    duration = min(t_end - t_ref, MAX_DECODE)
    if duration <= 0:
        return np.array([]), np.array([])
    edges      = np.arange(0, duration + bin_size, bin_size)
    rel_spikes = np.asarray(spike_times) - t_ref
    counts, _  = np.histogram(rel_spikes, bins=edges)
    rates      = gaussian_filter1d(counts.astype(float) / bin_size, sigma=smooth_sd)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    return rates, bin_centers


# ── Population matrix builder (parametric, v2-compatible output) ───────────────

def build_data(trials_df, spikes_df, t_ref_col, bin_size, smooth_sd, filter_bg):
    """Build population matrices; returns a v2-compatible data dict."""
    if filter_bg:
        good = trials_df[(~trials_df['miss_trial']) & (trials_df['bg_repeats'] == 0)].copy()
    else:
        good = trials_df[~trials_df['miss_trial']].copy()

    unit_ids = sorted(spikes_df['unit_id'].unique())
    spike_lookup = {
        (r['unit_id'], r['trial_id']): np.asarray(r['spike_times'])
        for _, r in spikes_df.iterrows()
    }

    X_all, y_all = [], []
    trial_ids, tw_all, rewarded_all, prev_rewarded_all, bg_repeats_all = [], [], [], [], []

    for _, row in good.iterrows():
        tid, t_ref, t_end = row['trial_id'], row[t_ref_col], row['decision_time']
        if pd.isna(t_ref) or pd.isna(t_end) or t_end <= t_ref:
            continue

        trial_rates, n_bins, bins = [], None, None
        for uid in unit_ids:
            spks = spike_lookup.get((uid, tid), np.array([]))
            rates, bc = bin_spikes(spks, t_ref, t_end, bin_size, smooth_sd)
            if len(rates) == 0:
                continue
            if n_bins is None:
                n_bins, bins = len(rates), bc
            else:
                rates = rates[:n_bins]
                if len(rates) < n_bins:
                    rates = np.pad(rates, (0, n_bins - len(rates)))
            trial_rates.append(rates)

        if not trial_rates or n_bins is None or n_bins < 2:
            continue

        pop = np.stack(trial_rates, axis=1)
        valid = bins >= MIN_WAIT
        if valid.sum() < MIN_BINS:
            continue

        X_all.append(pop[valid])
        y_all.append(bins[valid])
        trial_ids.append(tid)
        tw_all.append(row.get('time_waited', np.nan))
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


# ── PCA ────────────────────────────────────────────────────────────────────────

def apply_pca(data, n_comp):
    if n_comp is None:
        return data
    X_stack = np.vstack(data['X'])
    n_comp = min(n_comp, X_stack.shape[1], X_stack.shape[0] - 1)
    if n_comp < 2:
        return data
    pca   = PCA(n_components=n_comp).fit(X_stack)
    X_pca = [pca.transform(x) for x in data['X']]
    return {**data, 'X': X_pca, 'n_units': n_comp}


# ── Decoder (v2-compatible keys, quiet for sweep) ─────────────────────────────

def loto_decode(data):
    """LOTO decoder — returns v2-compatible result dict."""
    X, y = data['X'], data['y']
    n    = len(X)

    y_true_all, y_pred_all = [], []
    mae_list, r2_list, bias_list = [], [], []

    for i in range(n):
        X_train = np.vstack([X[j] for j in range(n) if j != i])
        y_train = np.concatenate([y[j] for j in range(n) if j != i])
        model   = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True).fit(X_train, y_train)
        y_hat   = model.predict(X[i])
        y_test  = y[i]
        err     = y_hat - y_test

        y_true_all.append(y_test)
        y_pred_all.append(y_hat)
        mae_list.append(np.mean(np.abs(err)))
        bias_list.append(np.mean(err))
        ss_res = np.sum(err**2)
        ss_tot = np.sum((y_test - y_test.mean())**2)
        r2_list.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)

    return dict(
        y_true        = y_true_all,
        y_pred        = y_pred_all,
        mae_per_trial = np.array(mae_list),
        r2_per_trial  = np.array(r2_list),
        bias_per_trial= np.array(bias_list),
        trial_ids     = data['trial_ids'],
        tw            = data['tw'],
        rewarded      = data['rewarded'],
        prev_rewarded = data['prev_rewarded'],
        bg_repeats    = data['bg_repeats'],
    )


# ── Shuffle control ────────────────────────────────────────────────────────────

def quick_shuffle(data, n=N_SHUFFLES):
    rng  = np.random.default_rng(42)
    maes = []
    for _ in range(n):
        y_shuf  = [rng.permutation(yi) for yi in data['y']]
        X_stack = np.vstack(data['X'])
        y_stack = np.concatenate(y_shuf)
        model   = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True).fit(X_stack, y_stack)
        maes.append(np.mean([np.mean(np.abs(model.predict(x) - y))
                             for x, y in zip(data['X'], y_shuf)]))
    return np.array(maes)


# ── Parameter sweep ────────────────────────────────────────────────────────────

def run_quick_sweep(trials_df, spikes_df, t_ref_col, session_name='session'):
    """Run reduced parameter sweep using the v2 pipeline."""
    keys   = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))

    print(f"\n{'='*60}")
    print(f"QUICK PARAMETER SWEEP (v2): {session_name}")
    print(f"{'='*60}")
    print(f"Testing {len(combos)} combinations...\n")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        print(f"[{i+1:2d}/{len(combos)}] bin={params['bin_size']:.2f} "
              f"σ={params['smooth_sd']:.1f} pca={str(params['n_pca']):>4} "
              f"filt={params['filter_bg_repeats']}", end='')

        try:
            data = build_data(trials_df, spikes_df, t_ref_col,
                              params['bin_size'], params['smooth_sd'],
                              params['filter_bg_repeats'])

            if data['n_trials'] < 30:
                print(f"  → skip (n={data['n_trials']})")
                continue

            data = apply_pca(data, params['n_pca'])
            dec  = loto_decode(data)

            # Pooled correlation
            yt_pool = np.concatenate(dec['y_true'])
            yp_pool = np.concatenate(dec['y_pred'])
            r, _    = pearsonr(yt_pool, yp_pool)

            # Shuffle p-value
            shuf_maes = quick_shuffle(data)
            real_mae  = dec['mae_per_trial'].mean()
            p_shuf    = (shuf_maes <= real_mae).mean()

            # History effect via v2 helpers
            clock = extract_clock_speeds(dec)
            try:
                hist = analyze_history_effect(dec, clock)
                hist_p = hist['p_ttest']
                hist_d = hist['cohens_d']
            except Exception:
                hist_p, hist_d = np.nan, np.nan

            results.append({
                **params,
                'n_trials'  : data['n_trials'],
                'pooled_r2' : r**2,
                'pooled_r'  : r,
                'mae'       : real_mae,
                'median_r2' : np.nanmedian(dec['r2_per_trial']),
                'pct_r2_pos': (dec['r2_per_trial'] > 0).mean() * 100,
                'p_shuffle' : p_shuf,
                'history_p' : hist_p,
                'history_d' : hist_d,
            })

            print(f"  → R²={r**2:.3f} MAE={real_mae:.3f} d={hist_d:.2f}")

        except Exception as e:
            print(f"  → ERROR: {e}")

    df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*60}")
    print("TOP 5 CONFIGURATIONS (by Pooled R²)")
    print(f"{'='*60}")

    if len(df) > 0:
        top5 = df.nlargest(5, 'pooled_r2')
        print(f"{'bin':>5} {'σ':>4} {'PCA':>4} {'filt':>5} | "
              f"{'R²':>6} {'MAE':>6} {'med_r2':>7} {'%r2>0':>6} | {'d':>5}")
        print("-" * 60)
        for _, row in top5.iterrows():
            pca_str = str(int(row['n_pca'])) if pd.notna(row['n_pca']) else '-'
            filt    = 'Y' if row['filter_bg_repeats'] else 'N'
            print(f"{row['bin_size']:>5.2f} {row['smooth_sd']:>4.1f} {pca_str:>4} {filt:>5} | "
                  f"{row['pooled_r2']:>6.3f} {row['mae']:>6.3f} {row['median_r2']:>7.3f} "
                  f"{row['pct_r2_pos']:>6.1f} | {row['history_d']:>5.2f}")

        # Baseline vs best
        baseline = df[(df['bin_size'] == 0.10) & (df['smooth_sd'] == 1.0) &
                      (df['n_pca'].isna()) & (df['filter_bg_repeats'] == False)]
        best = df.loc[df['pooled_r2'].idxmax()]

        if len(baseline) > 0:
            bl = baseline.iloc[0]
            print(f"\n{'='*60}")
            print("BASELINE vs BEST")
            print(f"{'='*60}")
            print(f"Baseline (bin=0.10, σ=1.0, no PCA, no filter):")
            print(f"  R²={bl['pooled_r2']:.3f}, MAE={bl['mae']:.3f}, "
                  f"median_trial_r²={bl['median_r2']:.3f}")
            print(f"\nBest (bin={best['bin_size']}, σ={best['smooth_sd']}, "
                  f"PCA={best['n_pca']}, filter={best['filter_bg_repeats']}):")
            print(f"  R²={best['pooled_r2']:.3f}, MAE={best['mae']:.3f}, "
                  f"median_trial_r²={best['median_r2']:.3f}")
            print(f"\nImprovement: R² +{best['pooled_r2'] - bl['pooled_r2']:.3f}, "
                  f"MAE {best['mae'] - bl['mae']:+.3f}")

    return df


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_quick_results(df, save_path='quick_sweep_results_v2.png'):
    """Compact visualisation — same panels as v1."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Bin size × smoothing heatmap (no PCA, no filter)
    ax = axes[0]
    subset = df[(df['n_pca'].isna()) & (df['filter_bg_repeats'] == False)]
    if len(subset) > 0:
        pivot = subset.pivot(index='smooth_sd', columns='bin_size', values='pooled_r2')
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{y:.1f}' for y in pivot.index])
        ax.set_xlabel('Bin size (s)')
        ax.set_ylabel('Smoothing σ (bins)')
        ax.set_title('Pooled R² (no PCA, all trials)')
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f'{pivot.values[i,j]:.2f}', ha='center', va='center')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. PCA effect
    ax = axes[1]
    no_pca   = df[df['n_pca'].isna()].groupby(['bin_size', 'smooth_sd'])['pooled_r2'].mean()
    with_pca = df[df['n_pca'] == 10].groupby(['bin_size', 'smooth_sd'])['pooled_r2'].mean()
    common   = no_pca.index.intersection(with_pca.index)
    if len(common) > 0:
        ax.scatter(no_pca[common], with_pca[common], s=80, alpha=0.7)
        lim = max(no_pca[common].max(), with_pca[common].max()) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
        ax.set_xlabel('R² without PCA')
        ax.set_ylabel('R² with PCA(10)')
        ax.set_title('Effect of PCA')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    # 3. bg_repeats filter effect
    ax = axes[2]
    no_filt   = df[df['filter_bg_repeats'] == False].groupby(['bin_size', 'smooth_sd', 'n_pca'])['pooled_r2'].mean()
    with_filt = df[df['filter_bg_repeats'] == True].groupby(['bin_size', 'smooth_sd', 'n_pca'])['pooled_r2'].mean()
    common    = no_filt.index.intersection(with_filt.index)
    if len(common) > 0:
        ax.scatter(no_filt[common], with_filt[common], s=80, alpha=0.7)
        lim = max(no_filt[common].max(), with_filt[common].max()) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
        ax.set_xlabel('R² (all trials)')
        ax.set_ylabel('R² (bg_repeats=0)')
        ax.set_title('Effect of bg_repeats Filter')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {save_path}")
    plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── Configure your session here ───────────────────────────────────────────
    SESSION   = 'RZ050_2024-11-20_str'
    GROUP     = 'Short BG'   # 'Short BG' → t_ref = last_lick_time
                              # 'Long BG'  → t_ref = cue_on
    # ─────────────────────────────────────────────────────────────────────────

    print(f"Loading data for {SESSION} ({GROUP})...")

    trials_df, spikes_df, _ = load_decoder_data()

    t_ref_col = GROUP_CONFIG[GROUP]['t_ref_col']
    t = enrich_with_reward(trials_df, SESSION)
    s = spikes_df[spikes_df['session'] == SESSION].copy()

    print(f"  {len(t)} trials, {s['unit_id'].nunique()} units, t_ref='{t_ref_col}'")

    results = run_quick_sweep(t, s, t_ref_col, SESSION)
    plot_quick_results(results, save_path=f'quick_sweep_{SESSION}.png')
    results.to_csv(f'param_sweep_results_{SESSION}.csv', index=False)
    print(f"\nResults saved to param_sweep_results_{SESSION}.csv")
