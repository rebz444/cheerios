"""
decoder_quick_test.py
─────────────────────
Quick focused test of the most impactful decoder parameters.
Reduced grid for fast iteration (~5-10 min per session).

Tests:
  - Bin size: 100ms vs 200ms vs 300ms
  - Smoothing: 1.0 vs 2.0 (in bins)  
  - PCA: None vs 10 components
  - bg_repeats filter: True vs False

Total: 24 combinations (vs 200 in full sweep)
"""

import pickle
import warnings
from itertools import product
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# REDUCED PARAMETER GRID (24 combinations)
# ══════════════════════════════════════════════════════════════════════════════

PARAM_GRID = {
    'bin_size': [0.10, 0.20, 0.30],
    'smooth_sd': [1.0, 2.0],
    'n_pca': [None, 10],
    'filter_bg_repeats': [False, True],
}

# Fixed
MAX_DECODE = 10.0
MIN_WAIT = 0.3
MIN_BINS = 5
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
N_SHUFFLES = 10  # Very reduced for speed


def bin_spikes(spike_times, t_ref, t_end, bin_size, smooth_sd):
    duration = min(t_end - t_ref, MAX_DECODE)
    if duration <= 0:
        return np.array([]), np.array([])
    edges = np.arange(0, duration + bin_size, bin_size)
    rel_spikes = np.asarray(spike_times) - t_ref
    counts, _ = np.histogram(rel_spikes, bins=edges)
    rates = gaussian_filter1d(counts.astype(float) / bin_size, sigma=smooth_sd)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    return rates, bin_centers


def build_data(trials_df, spikes_df, t_ref_col, bin_size, smooth_sd, filter_bg):
    """Build population matrices."""
    if filter_bg:
        good = trials_df[(~trials_df['miss_trial']) & (trials_df['bg_repeats'] == 0)].copy()
    else:
        good = trials_df[~trials_df['miss_trial']].copy()
    
    unit_ids = sorted(spikes_df['unit_id'].unique())
    spike_lookup = {
        (r['unit_id'], r['trial_id']): np.asarray(r['spike_times'])
        for _, r in spikes_df.iterrows()
    }
    
    X_all, y_all, prev_rew = [], [], []
    
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
        prev_rew.append(row.get('prev_rewarded', np.nan))
    
    return {'X': X_all, 'y': y_all, 'prev_rewarded': np.array(prev_rew),
            'n_trials': len(X_all), 'n_units': len(unit_ids)}


def apply_pca(data, n_comp):
    if n_comp is None:
        return data
    X_stack = np.vstack(data['X'])
    n_comp = min(n_comp, X_stack.shape[1], X_stack.shape[0] - 1)
    if n_comp < 2:
        return data
    pca = PCA(n_components=n_comp).fit(X_stack)
    X_pca = [pca.transform(x) for x in data['X']]
    return {**data, 'X': X_pca, 'n_units': n_comp}


def loto_decode(data):
    """LOTO decoder - returns key metrics."""
    X, y = data['X'], data['y']
    n = len(X)
    
    y_true_all, y_pred_all, mae_list, r2_list = [], [], [], []
    
    for i in range(n):
        X_train = np.vstack([X[j] for j in range(n) if j != i])
        y_train = np.concatenate([y[j] for j in range(n) if j != i])
        
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_train, y_train)
        y_hat = model.predict(X[i])
        y_test = y[i]
        err = y_hat - y_test
        
        y_true_all.append(y_test)
        y_pred_all.append(y_hat)
        mae_list.append(np.mean(np.abs(err)))
        
        ss_res, ss_tot = np.sum(err**2), np.sum((y_test - y_test.mean())**2)
        r2_list.append(1 - ss_res/ss_tot if ss_tot > 0 else np.nan)
    
    return {'y_true': y_true_all, 'y_pred': y_pred_all, 
            'mae': np.array(mae_list), 'r2': np.array(r2_list),
            'prev_rewarded': data['prev_rewarded']}


def quick_shuffle(data, n=N_SHUFFLES):
    """Quick shuffle MAE estimate."""
    rng = np.random.default_rng(42)
    maes = []
    for _ in range(n):
        y_shuf = [rng.permutation(yi) for yi in data['y']]
        X_all, y_all = np.vstack(data['X']), np.concatenate(y_shuf)
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_all, y_all)
        trial_maes = [np.mean(np.abs(model.predict(x) - y)) 
                      for x, y in zip(data['X'], y_shuf)]
        maes.append(np.mean(trial_maes))
    return np.array(maes)


def history_effect(results):
    """Compute clock speed history effect."""
    speeds = []
    for yt, yp in zip(results['y_true'], results['y_pred']):
        if len(yt) < MIN_BINS:
            speeds.append(np.nan)
        else:
            slope, _ = np.polyfit(yt, yp, 1)
            speeds.append(slope)
    speeds = np.array(speeds)
    
    pr = results['prev_rewarded']
    valid = ~np.isnan(pr) & ~np.isnan(speeds)
    
    if valid.sum() < 20:
        return np.nan, np.nan
    
    after_rew = speeds[valid & (pr == 1)]
    after_no = speeds[valid & (pr == 0)]
    
    if len(after_rew) < 5 or len(after_no) < 5:
        return np.nan, np.nan
    
    _, p = ttest_ind(after_rew, after_no)
    pooled_std = np.sqrt((after_rew.std()**2 + after_no.std()**2) / 2)
    d = (after_rew.mean() - after_no.mean()) / pooled_std if pooled_std > 0 else np.nan
    return p, d


def run_quick_sweep(trials_df, spikes_df, t_ref_col, session_name='session'):
    """Run reduced parameter sweep."""
    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    
    print(f"\n{'='*60}")
    print(f"QUICK PARAMETER SWEEP: {session_name}")
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
            dec = loto_decode(data)
            
            # Metrics
            yt_pool = np.concatenate(dec['y_true'])
            yp_pool = np.concatenate(dec['y_pred'])
            r, _ = pearsonr(yt_pool, yp_pool)
            
            shuf_maes = quick_shuffle(data)
            real_mae = dec['mae'].mean()
            p_shuf = (shuf_maes <= real_mae).mean()
            
            hist_p, hist_d = history_effect(dec)
            
            results.append({
                **params,
                'n_trials': data['n_trials'],
                'pooled_r2': r**2,
                'pooled_r': r,
                'mae': real_mae,
                'median_r2': np.nanmedian(dec['r2']),
                'pct_r2_pos': (dec['r2'] > 0).mean() * 100,
                'p_shuffle': p_shuf,
                'history_p': hist_p,
                'history_d': hist_d,
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
        for _, r in top5.iterrows():
            pca_str = str(int(r['n_pca'])) if pd.notna(r['n_pca']) else '-'
            filt = 'Y' if r['filter_bg_repeats'] else 'N'
            print(f"{r['bin_size']:>5.2f} {r['smooth_sd']:>4.1f} {pca_str:>4} {filt:>5} | "
                  f"{r['pooled_r2']:>6.3f} {r['mae']:>6.3f} {r['median_r2']:>7.3f} "
                  f"{r['pct_r2_pos']:>6.1f} | {r['history_d']:>5.2f}")
        
        # Best vs baseline comparison
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


def plot_quick_results(df, save_path='quick_sweep_results.png'):
    """Compact visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 1. Bin size x smoothing heatmap (no PCA, no filter)
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
    no_pca = df[df['n_pca'].isna()].groupby(['bin_size', 'smooth_sd'])['pooled_r2'].mean()
    with_pca = df[df['n_pca'] == 10].groupby(['bin_size', 'smooth_sd'])['pooled_r2'].mean()
    common = no_pca.index.intersection(with_pca.index)
    if len(common) > 0:
        ax.scatter(no_pca[common], with_pca[common], s=80, alpha=0.7)
        lim = max(no_pca[common].max(), with_pca[common].max()) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
        ax.set_xlabel('R² without PCA')
        ax.set_ylabel('R² with PCA(10)')
        ax.set_title('Effect of PCA')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    
    # 3. Filter effect
    ax = axes[2]
    no_filt = df[df['filter_bg_repeats'] == False].groupby(['bin_size', 'smooth_sd', 'n_pca'])['pooled_r2'].mean()
    with_filt = df[df['filter_bg_repeats'] == True].groupby(['bin_size', 'smooth_sd', 'n_pca'])['pooled_r2'].mean()
    common = no_filt.index.intersection(with_filt.index)
    if len(common) > 0:
        ax.scatter(no_filt[common], with_filt[common], s=80, alpha=0.7)
        lim = max(no_filt[common].max(), with_filt[common].max()) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
        ax.set_xlabel('R² (all trials)')
        ax.set_ylabel('R² (bg_repeats=0)')
        ax.set_title('Effect of bg_repeats Filter')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {save_path}")
    plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("""
QUICK DECODER PARAMETER SWEEP
─────────────────────────────

To run:
    
    # Load your data
    from population_decoder import load_decoder_data
    trials_df, spikes_df, _ = load_decoder_data(...)
    
    # Filter to one session
    session = 'RZ050_2024-11-20_str'
    t = trials_df[trials_df['session'] == session].copy()
    s = spikes_df[spikes_df['session'] == session].copy()
    
    # Add reward info (from your enrich_with_reward function)
    # ... 
    
    # Run sweep
    from decoder_quick_test import run_quick_sweep, plot_quick_results
    results = run_quick_sweep(t, s, 'last_lick_time', session)
    plot_quick_results(results)
    results.to_csv('param_sweep_results.csv', index=False)
    
Expected runtime: ~5-10 minutes per session.
""")
