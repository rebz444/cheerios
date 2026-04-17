import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind
from collections import defaultdict

import paths as p
import constants as c
from population_decoder import load_decoder_data, build_session_matrix

def plot_all_trial_trajectories(session_data, color_by='time_waited', 
                                 max_trials=None, ax=None):
    """
    Plot every trial's trajectory in PC space, colored by a behavioral variable.
    
    Parameters
    ----------
    session_data : dict from build_session_matrix
    color_by : str
        'time_waited' - color by total wait duration
        'early_slope' - color by PC1 slope in first 2s (proxy for "ramping")
        'trial_index' - color by trial order (to check for drift)
    max_trials : int, optional
        Subsample if too many trials for clarity
    """
    X_all = session_data['X']
    y_all = session_data['y']
    tw_all = session_data['tw']
    n_trials = len(X_all)
    
    # Fit PCA on all data
    X_stacked = np.vstack(X_all)
    pca = PCA(n_components=3)
    pca.fit(X_stacked)
    
    # Project each trial
    trial_pcs = []
    cumsum = 0
    for X_trial in X_all:
        n_bins = X_trial.shape[0]
        trial_pcs.append(pca.transform(X_trial))
        cumsum += n_bins
    
    # Compute coloring variable
    if color_by == 'time_waited':
        color_vals = tw_all
        cmap = 'coolwarm'
        clabel = 'Time waited (s)'
    elif color_by == 'early_slope':
        # Slope of PC1 in first 2s — positive = ramping up
        slopes = []
        for pc_traj, y_trial in zip(trial_pcs, y_all):
            early_mask = y_trial <= 2.0
            if early_mask.sum() >= 3:
                pc1_early = pc_traj[early_mask, 0]
                t_early = y_trial[early_mask]
                slope = np.polyfit(t_early, pc1_early, 1)[0]
            else:
                slope = 0
            slopes.append(slope)
        color_vals = np.array(slopes)
        cmap = 'coolwarm'
        clabel = 'Early PC1 slope (AU/s)'
    elif color_by == 'trial_index':
        color_vals = np.arange(n_trials)
        cmap = 'viridis'
        clabel = 'Trial index'
    else:
        raise ValueError(f"Unknown color_by: {color_by}")
    
    # Subsample if needed
    if max_trials and n_trials > max_trials:
        idx = np.random.choice(n_trials, max_trials, replace=False)
    else:
        idx = np.arange(n_trials)
    
    # Create figure
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    else:
        fig = ax[0].figure
        axes = ax
    
    norm = Normalize(vmin=np.percentile(color_vals, 5), 
                     vmax=np.percentile(color_vals, 95))
    cmap_obj = plt.cm.get_cmap(cmap)
    
    # Panel 1: PC1-PC2 trajectories
    ax1 = axes[0]
    for i in idx:
        pc_traj = trial_pcs[i]
        color = cmap_obj(norm(color_vals[i]))
        ax1.plot(pc_traj[:, 0], pc_traj[:, 1], '-', color=color, 
                alpha=0.4, lw=0.8)
        ax1.scatter(pc_traj[0, 0], pc_traj[0, 1], c=[color], s=15, 
                   marker='o', edgecolor='none', alpha=0.6)
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title(f'All trajectories (n={len(idx)})\nColored by {color_by}')
    
    # Panel 2: PC1 vs elapsed time (the key plot)
    ax2 = axes[1]
    for i in idx:
        pc_traj = trial_pcs[i]
        y_trial = y_all[i]
        color = cmap_obj(norm(color_vals[i]))
        ax2.plot(y_trial, pc_traj[:, 0], '-', color=color, alpha=0.4, lw=0.8)
    
    ax2.set_xlabel('Elapsed time (s)')
    ax2.set_ylabel('PC1')
    ax2.set_title('PC1 vs. time\n(each line = one trial)')
    
    # Panel 3: Distribution of coloring variable + trajectory metric
    ax3 = axes[2]
    
    # Compute trajectory "linearity" — R² of PC1 vs time
    linearities = []
    for pc_traj, y_trial in zip(trial_pcs, y_all):
        if len(y_trial) >= 3:
            corr = np.corrcoef(y_trial, pc_traj[:, 0])[0, 1]
            linearities.append(corr ** 2)
        else:
            linearities.append(np.nan)
    linearities = np.array(linearities)
    
    ax3.scatter(color_vals[idx], linearities[idx], c=color_vals[idx], 
               cmap=cmap, norm=norm, s=20, alpha=0.6)
    ax3.set_xlabel(clabel)
    ax3.set_ylabel('Trajectory linearity (R²)')
    ax3.set_title('Does wait time predict\ntrajectory quality?')
    
    # Add correlation
    valid = ~np.isnan(linearities)
    if valid.sum() > 10:
        r = np.corrcoef(color_vals[valid], linearities[valid])[0, 1]
        ax3.text(0.05, 0.95, f'r = {r:.2f}', transform=ax3.transAxes, 
                fontsize=10, va='top')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(clabel)
    
    plt.tight_layout()
    return fig, pca, {'linearities': linearities, 'color_vals': color_vals,
                      'trial_pcs': trial_pcs, 'y_all': y_all}


def plot_trajectory_clusters(session_data, n_clusters=2):
    """
    Cluster trials by trajectory shape and visualize.
    
    Uses trajectory features: early slope, late slope, overall linearity, PC1 range
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    X_all = session_data['X']
    y_all = session_data['y']
    tw_all = session_data['tw']
    
    # Fit PCA
    X_stacked = np.vstack(X_all)
    pca = PCA(n_components=3)
    pca.fit(X_stacked)
    
    trial_pcs = [pca.transform(X) for X in X_all]
    
    # Extract trajectory features per trial
    features = []
    for pc_traj, y_trial in zip(trial_pcs, y_all):
        if len(y_trial) < 5:
            features.append([np.nan] * 5)
            continue
        
        pc1 = pc_traj[:, 0]
        
        # Feature 1: Overall slope
        slope_all = np.polyfit(y_trial, pc1, 1)[0]
        
        # Feature 2: Early slope (first 1.5s)
        early = y_trial <= 1.5
        if early.sum() >= 3:
            slope_early = np.polyfit(y_trial[early], pc1[early], 1)[0]
        else:
            slope_early = slope_all
        
        # Feature 3: Late slope (after 2s)
        late = y_trial >= 2.0
        if late.sum() >= 3:
            slope_late = np.polyfit(y_trial[late], pc1[late], 1)[0]
        else:
            slope_late = slope_all
        
        # Feature 4: Linearity (R²)
        corr = np.corrcoef(y_trial, pc1)[0, 1]
        linearity = corr ** 2 if not np.isnan(corr) else 0
        
        # Feature 5: PC1 range
        pc1_range = pc1.max() - pc1.min()
        
        features.append([slope_all, slope_early, slope_late, linearity, pc1_range])
    
    features = np.array(features)
    
    # Remove trials with NaN features
    valid = ~np.any(np.isnan(features), axis=1)
    features_valid = features[valid]
    tw_valid = tw_all[valid]
    trial_pcs_valid = [trial_pcs[i] for i in range(len(trial_pcs)) if valid[i]]
    y_all_valid = [y_all[i] for i in range(len(y_all)) if valid[i]]
    
    # Normalize and cluster
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_valid)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = ['#2166AC', '#D6604D']  # Match your group colors
    cluster_names = ['Cluster 0', 'Cluster 1']
    
    # Panel 1: PC1 vs time by cluster
    ax1 = axes[0, 0]
    for i, (pc_traj, y_trial, label) in enumerate(zip(trial_pcs_valid, y_all_valid, labels)):
        ax1.plot(y_trial, pc_traj[:, 0], '-', color=colors[label], alpha=0.3, lw=0.6)
    
    # Add cluster means
    for c in range(n_clusters):
        mask = labels == c
        # Interpolate to common time base for averaging
        t_common = np.arange(0.3, 4.0, 0.1)
        pc1_interp = []
        for pc_traj, y_trial in zip(
            [trial_pcs_valid[i] for i in range(len(labels)) if labels[i] == c],
            [y_all_valid[i] for i in range(len(labels)) if labels[i] == c]
        ):
            if y_trial.max() >= 3.0:
                pc1_interp.append(np.interp(t_common, y_trial, pc_traj[:, 0]))
        
        if pc1_interp:
            mean_pc1 = np.mean(pc1_interp, axis=0)
            ax1.plot(t_common, mean_pc1, '-', color=colors[c], lw=3, 
                    label=f'{cluster_names[c]} (n={mask.sum()})')
    
    ax1.set_xlabel('Elapsed time (s)')
    ax1.set_ylabel('PC1')
    ax1.set_title('Trajectories by cluster')
    ax1.legend()
    
    # Panel 2: Time waited distribution by cluster
    ax2 = axes[0, 1]
    for c in range(n_clusters):
        tw_c = tw_valid[labels == c]
        ax2.hist(tw_c, bins=20, alpha=0.6, color=colors[c], 
                label=f'{cluster_names[c]}: μ={tw_c.mean():.2f}s')
    ax2.set_xlabel('Time waited (s)')
    ax2.set_ylabel('Count')
    ax2.set_title('Wait time by cluster')
    ax2.legend()
    
    # Panel 3: Feature space (slope_early vs linearity)
    ax3 = axes[0, 2]
    for c in range(n_clusters):
        mask = labels == c
        ax3.scatter(features_valid[mask, 1], features_valid[mask, 3], 
                   c=colors[c], alpha=0.5, s=30, label=cluster_names[c])
    ax3.set_xlabel('Early slope (PC1/s)')
    ax3.set_ylabel('Linearity (R²)')
    ax3.set_title('Trajectory features')
    ax3.legend()
    
    # Panel 4: PC1-PC2 trajectories by cluster
    ax4 = axes[1, 0]
    for pc_traj, label in zip(trial_pcs_valid, labels):
        ax4.plot(pc_traj[:, 0], pc_traj[:, 1], '-', color=colors[label], alpha=0.3, lw=0.6)
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title('PC space trajectories')
    
    # Panel 5: Cluster assignment vs time_waited
    ax5 = axes[1, 1]
    ax5.scatter(tw_valid, labels + np.random.randn(len(labels))*0.05, 
               c=[colors[l] for l in labels], alpha=0.5, s=20)
    ax5.set_xlabel('Time waited (s)')
    ax5.set_ylabel('Cluster (jittered)')
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(cluster_names)
    ax5.set_title('Cluster vs. wait time')
    
    # Panel 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = "Cluster Summary\n" + "="*40 + "\n\n"
    for c in range(n_clusters):
        mask = labels == c
        tw_c = tw_valid[mask]
        lin_c = features_valid[mask, 3]
        slope_c = features_valid[mask, 1]
        summary_text += f"{cluster_names[c]} (n={mask.sum()}):\n"
        summary_text += f"  Time waited: {tw_c.mean():.2f} ± {tw_c.std():.2f} s\n"
        summary_text += f"  Linearity:   {lin_c.mean():.2f} ± {lin_c.std():.2f}\n"
        summary_text += f"  Early slope: {slope_c.mean():.2f} ± {slope_c.std():.2f}\n\n"
    
    # Statistical test
    from scipy.stats import ttest_ind
    tw_0 = tw_valid[labels == 0]
    tw_1 = tw_valid[labels == 1]
    t_stat, p_val = ttest_ind(tw_0, tw_1)
    summary_text += f"Wait time difference:\n  t={t_stat:.2f}, p={p_val:.3f}"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Trajectory-based clustering', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    return fig, {'labels': labels, 'features': features_valid, 'tw': tw_valid,
                 'valid_mask': valid, 'kmeans': kmeans}


def extract_trajectory_features(session_data):
    """
    Extract trajectory features for clustering.
    Returns features array and metadata.
    """
    X_all = session_data['X']
    y_all = session_data['y']
    tw_all = session_data['tw']

    # Fit PCA
    X_stacked = np.vstack(X_all)
    pca = PCA(n_components=3)
    pca.fit(X_stacked)

    trial_pcs = [pca.transform(X) for X in X_all]

    features = []
    valid_idx = []

    for i, (pc_traj, y_trial) in enumerate(zip(trial_pcs, y_all)):
        if len(y_trial) < 5:
            continue

        pc1 = pc_traj[:, 0]

        # Feature 1: Overall slope
        slope_all = np.polyfit(y_trial, pc1, 1)[0]

        # Feature 2: Early slope (first 1.5s)
        early = y_trial <= 1.5
        if early.sum() >= 3:
            slope_early = np.polyfit(y_trial[early], pc1[early], 1)[0]
        else:
            slope_early = slope_all

        # Feature 3: Late slope (after 2s)
        late = y_trial >= 2.0
        if late.sum() >= 3:
            slope_late = np.polyfit(y_trial[late], pc1[late], 1)[0]
        else:
            slope_late = slope_all

        # Feature 4: Linearity (R²)
        corr = np.corrcoef(y_trial, pc1)[0, 1]
        linearity = corr ** 2 if not np.isnan(corr) else 0

        # Feature 5: PC1 range
        pc1_range = pc1.max() - pc1.min()

        features.append([slope_all, slope_early, slope_late, linearity, pc1_range])
        valid_idx.append(i)

    features = np.array(features)
    tw_valid = tw_all[valid_idx]

    return features, tw_valid, valid_idx, pca, trial_pcs


def cluster_session(session_data, k_range=[2, 3, 4]):
    """
    Cluster one session with multiple k values.
    Returns results for each k.
    """
    features, tw_valid, valid_idx, pca, trial_pcs = extract_trajectory_features(session_data)

    if len(features) < 20:
        return None

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    results = {}

    for k in k_range:
        if k > len(features) // 5:  # Need at least 5 samples per cluster
            continue

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        # Silhouette score (higher = better separation)
        if k > 1:
            sil = silhouette_score(features_scaled, labels)
        else:
            sil = 0

        # Behavioral separation: compute pairwise t-tests between clusters
        cluster_tw = {c: tw_valid[labels == c] for c in range(k)}

        # For k=2, simple t-test
        if k == 2:
            t_stat, p_val = ttest_ind(cluster_tw[0], cluster_tw[1])
            tw_diff = abs(cluster_tw[0].mean() - cluster_tw[1].mean())
        else:
            # For k>2, use F-statistic or max pairwise difference
            from scipy.stats import f_oneway
            groups = [cluster_tw[c] for c in range(k) if len(cluster_tw[c]) > 1]
            if len(groups) >= 2:
                f_stat, p_val = f_oneway(*groups)
                t_stat = f_stat
            else:
                t_stat, p_val = 0, 1
            # Max difference between any two cluster means
            means = [cluster_tw[c].mean() for c in range(k)]
            tw_diff = max(means) - min(means)

        results[k] = {
            'labels': labels,
            'silhouette': sil,
            't_stat': t_stat,
            'p_val': p_val,
            'tw_diff': tw_diff,
            'cluster_sizes': [np.sum(labels == c) for c in range(k)],
            'cluster_tw_means': [cluster_tw[c].mean() for c in range(k)],
            'cluster_tw_stds': [cluster_tw[c].std() for c in range(k)],
            'cluster_linearity': [features[labels == c, 3].mean() for c in range(k)],
            'cluster_early_slope': [features[labels == c, 1].mean() for c in range(k)],
            'features': features,
            'tw_valid': tw_valid,
        }

    return results


def run_clustering_all_sessions(trials_df, spikes_df, units_df, GROUP_DICT,
                                 group='Short BG', k_range=[2, 3, 4]):
    """
    Run clustering on all sessions for one group.
    """
    mice = GROUP_DICT[group]
    group_sessions = trials_df[trials_df['mouse_id'].isin(mice)]['session'].unique()

    all_results = {}

    for sess in group_sessions:
        data = build_session_matrix(sess, group, trials_df, spikes_df, units_df, mode='all')
        if data is None:
            continue

        results = cluster_session(data, k_range=k_range)
        if results is None:
            continue

        all_results[sess] = results

        # Print summary for this session
        if 2 in results:
            r = results[2]
            print(f"{sess}: n={sum(r['cluster_sizes'])}, "
                  f"sil={r['silhouette']:.3f}, "
                  f"Δtw={r['tw_diff']:.2f}s, p={r['p_val']:.3f}")

    return all_results


def aggregate_clustering_results(all_results, k_range=[2, 3, 4]):
    """
    Aggregate results across sessions.
    """
    summary = {k: defaultdict(list) for k in k_range}

    for sess, results in all_results.items():
        for k in k_range:
            if k not in results:
                continue
            r = results[k]
            summary[k]['session'].append(sess)
            summary[k]['silhouette'].append(r['silhouette'])
            summary[k]['tw_diff'].append(r['tw_diff'])
            summary[k]['p_val'].append(r['p_val'])
            summary[k]['n_trials'].append(sum(r['cluster_sizes']))

            # For k=2, track which cluster is "ramping" vs "flat"
            if k == 2:
                # Ramping cluster = higher linearity
                if r['cluster_linearity'][0] > r['cluster_linearity'][1]:
                    ramp_idx, flat_idx = 0, 1
                else:
                    ramp_idx, flat_idx = 1, 0

                summary[k]['ramp_tw'].append(r['cluster_tw_means'][ramp_idx])
                summary[k]['flat_tw'].append(r['cluster_tw_means'][flat_idx])
                summary[k]['ramp_linearity'].append(r['cluster_linearity'][ramp_idx])
                summary[k]['flat_linearity'].append(r['cluster_linearity'][flat_idx])
                summary[k]['ramp_n'].append(r['cluster_sizes'][ramp_idx])
                summary[k]['flat_n'].append(r['cluster_sizes'][flat_idx])

    return summary


def plot_aggregated_results(summary, k_range=[2, 3, 4]):
    """
    Plot aggregated results across sessions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Panel 1: Silhouette score by k
    ax1 = axes[0, 0]
    sil_means = [np.mean(summary[k]['silhouette']) for k in k_range]
    sil_sems = [np.std(summary[k]['silhouette']) / np.sqrt(len(summary[k]['silhouette']))
                for k in k_range]
    ax1.bar(k_range, sil_means, yerr=sil_sems, capsize=5, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Silhouette score')
    ax1.set_title('Cluster quality by k\n(higher = better separation)')
    ax1.set_xticks(k_range)

    # Panel 2: Wait time difference by k
    ax2 = axes[0, 1]
    tw_means = [np.mean(summary[k]['tw_diff']) for k in k_range]
    tw_sems = [np.std(summary[k]['tw_diff']) / np.sqrt(len(summary[k]['tw_diff']))
               for k in k_range]
    ax2.bar(k_range, tw_means, yerr=tw_sems, capsize=5, color='coral', alpha=0.7)
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Max Δ time_waited (s)')
    ax2.set_title('Behavioral separation by k')
    ax2.set_xticks(k_range)

    # Panel 3: Proportion of sessions with p < 0.05
    ax3 = axes[0, 2]
    prop_sig = [np.mean(np.array(summary[k]['p_val']) < 0.05) for k in k_range]
    ax3.bar(k_range, prop_sig, color='forestgreen', alpha=0.7)
    ax3.set_xlabel('Number of clusters (k)')
    ax3.set_ylabel('Proportion sessions p < 0.05')
    ax3.set_title('Behavioral significance by k')
    ax3.set_xticks(k_range)
    ax3.set_ylim(0, 1)
    ax3.axhline(0.05, color='gray', ls='--', lw=1)  # chance level

    # Panel 4: Per-session silhouette (k=2)
    ax4 = axes[1, 0]
    sessions = summary[2]['session']
    sil_vals = summary[2]['silhouette']
    ax4.barh(range(len(sessions)), sil_vals, color='steelblue', alpha=0.7)
    ax4.set_yticks(range(len(sessions)))
    ax4.set_yticklabels([s.split('_')[0] for s in sessions], fontsize=8)
    ax4.set_xlabel('Silhouette score (k=2)')
    ax4.set_title('Per-session cluster quality')
    ax4.axvline(0, color='gray', ls='-', lw=0.5)

    # Panel 5: Ramping vs Flat cluster wait times (k=2)
    ax5 = axes[1, 1]
    ramp_tw = summary[2]['ramp_tw']
    flat_tw = summary[2]['flat_tw']

    ax5.scatter(flat_tw, ramp_tw, s=60, alpha=0.7, c='steelblue', edgecolor='k')

    # Identity line
    lims = [min(min(flat_tw), min(ramp_tw)) - 0.5,
            max(max(flat_tw), max(ramp_tw)) + 0.5]
    ax5.plot(lims, lims, 'k--', lw=1, alpha=0.5)

    ax5.set_xlabel('Flat cluster: mean time_waited (s)')
    ax5.set_ylabel('Ramping cluster: mean time_waited (s)')
    ax5.set_title('Cluster wait times (k=2)\nEach dot = 1 session')

    # Count sessions where ramping > flat
    n_correct = sum(r > f for r, f in zip(ramp_tw, flat_tw))
    ax5.text(0.05, 0.95, f'{n_correct}/{len(ramp_tw)} sessions:\nramping > flat',
             transform=ax5.transAxes, va='top', fontsize=10)

    # Panel 6: Summary statistics table
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Compute aggregate stats for k=2
    ramp_tw_all = np.array(summary[2]['ramp_tw'])
    flat_tw_all = np.array(summary[2]['flat_tw'])

    t_agg, p_agg = ttest_ind(ramp_tw_all, flat_tw_all)

    text = f"""Aggregate Summary (k=2)
{'='*40}

Sessions analyzed: {len(summary[2]['session'])}

Ramping cluster (high linearity):
  Mean time_waited: {np.mean(ramp_tw_all):.2f} ± {np.std(ramp_tw_all):.2f} s
  Mean linearity:   {np.mean(summary[2]['ramp_linearity']):.2f}
  Mean cluster size: {np.mean(summary[2]['ramp_n']):.0f} trials

Flat cluster (low linearity):
  Mean time_waited: {np.mean(flat_tw_all):.2f} ± {np.std(flat_tw_all):.2f} s
  Mean linearity:   {np.mean(summary[2]['flat_linearity']):.2f}
  Mean cluster size: {np.mean(summary[2]['flat_n']):.0f} trials

Session-level t-test on cluster means:
  t = {t_agg:.2f}, p = {p_agg:.4f}

Sessions with ramping > flat wait time:
  {n_correct}/{len(ramp_tw_all)} ({100*n_correct/len(ramp_tw_all):.0f}%)
"""

    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Short BG: Trajectory Clustering Across Sessions',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


if __name__ == '__main__':
    OUT_DIR = p.DATA_DIR / 'timing_trajectory'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    GROUP_DICT = {
        'Long BG' : c.GROUP_DICT['l'],
        'Short BG': c.GROUP_DICT['s'],
    }

    TIER2_CSV = p.LOGS_DIR / 'RZ_msn_waveform.csv'

    print("Loading data...")
    trials_df, spikes_df, units_df = load_decoder_data(
        group_dict=GROUP_DICT,
        msn_tier_csv=TIER2_CSV,
    )

    for group, mice in GROUP_DICT.items():
        group_sessions = trials_df[trials_df['mouse_id'].isin(mice)]['session'].unique()
        group_tag = group.replace(' ', '_').lower()

        # Pick the session with the most usable trials
        best_sess, best_data = None, None
        for sess in group_sessions:
            d = build_session_matrix(sess, group, trials_df, spikes_df, units_df, mode='all')
            if d is not None and (best_data is None or d['n_trials'] > best_data['n_trials']):
                best_sess, best_data = sess, d

        if best_data is None:
            print(f"{group}: no usable sessions, skipping.")
            continue

        print(f"\n{group} — {best_sess} ({best_data['n_trials']} trials, {best_data['n_units']} units)")

        for color_by in ('time_waited', 'early_slope', 'trial_index'):
            fig_traj, _, _ = plot_all_trial_trajectories(best_data, color_by=color_by,
                                                         max_trials=200)
            fig_traj.suptitle(f'{group} — {best_sess} | color={color_by}',
                              fontsize=11, fontweight='bold')
            fig_traj.savefig(OUT_DIR / f'trajectories_{group_tag}_{color_by}.pdf',
                             bbox_inches='tight')
            plt.close(fig_traj)

        fig_clust, _ = plot_trajectory_clusters(best_data, n_clusters=2)
        fig_clust.suptitle(f'{group} — {best_sess}', fontsize=11, fontweight='bold')
        fig_clust.savefig(OUT_DIR / f'trajectory_clusters_{group_tag}.pdf',
                          bbox_inches='tight')
        plt.close(fig_clust)

    print(f"\nFigures saved to {OUT_DIR}")

    # ── Aggregate clustering across all Short BG sessions ─────────────────
    print("\n" + "=" * 60)
    print("Running trajectory clustering across all Short BG sessions...")
    print("=" * 60)

    all_results = run_clustering_all_sessions(
        trials_df, spikes_df, units_df,
        GROUP_DICT=GROUP_DICT,
        group='Short BG',
        k_range=[2, 3, 4],
    )

    print(f"\nAggregating results ({len(all_results)} sessions)...")
    summary = aggregate_clustering_results(all_results, k_range=[2, 3, 4])

    fig_agg = plot_aggregated_results(summary, k_range=[2, 3, 4])
    fig_agg.savefig(OUT_DIR / 'trajectory_clustering_aggregate_short_bg.pdf',
                    bbox_inches='tight')
    plt.close(fig_agg)

    print(f"\nAll figures saved to {OUT_DIR}")
    plt.show()