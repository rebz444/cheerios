import numpy as np
from scipy.ndimage import gaussian_filter1d

def generate_time_frame(trials, time_step, trial_count_mask=1):
    """
    Generate time bins and active trial counts.
    """
    # Calculate bounds (with protection against empty data)
    if len(trials) == 0:
        return np.array([]), np.array([]), np.array([])
        
    bounds = (
        np.round(trials.aligned_start_time.min(), decimals=1),
        np.round(trials.aligned_end_time.max(), decimals=1)
    )
    
    # Create bins
    bin_edges = np.arange(
        bounds[0] - time_step,
        bounds[1] + 2*time_step,
        time_step
    )
    bin_centers = bin_edges[:-1] + time_step/2
    
    # Calculate active trials
    active_trials = np.zeros(len(bin_edges) - 1, dtype=int)
    for _, trial in trials.iterrows():
        occupied = (bin_edges[:-1] < trial['aligned_end_time']) & \
                  (bin_edges[1:] > trial['aligned_start_time'])
        active_trials[occupied] += 1

    # Apply mask with safety checks
    if trial_count_mask > 0:
        valid_mask = active_trials >= trial_count_mask
        if not np.any(valid_mask):  # No valid bins
            return np.array([]), np.array([]), np.array([])
            
        bin_edges = np.append(
            bin_edges[:-1][valid_mask],
            bin_edges[:-1][valid_mask][-1] + time_step
        )
        bin_centers = bin_centers[valid_mask]
        active_trials = active_trials[valid_mask]
        
    return bin_edges, bin_centers, active_trials

def calculate_firing_rates(trials, spikes, anchor, time_step, trial_count_mask, sigma=None, normalize_by=None):
    bin_edges, bin_centers, active_trials = generate_time_frame(trials, time_step, trial_count_mask)
    
    # Handle case where no bins meet criteria
    if len(bin_edges) == 0:
        return np.array([]), np.array([]), np.array([])

    # Bin spikes for each trial, shape is [trial, bin]
    counts = np.array([
        np.histogram(trial[anchor], bins=bin_edges)[0]
        for _, trial in spikes.groupby('trial_id')
    ])
    rates = counts / time_step

    mean_fr = np.nansum(rates, axis=0) / active_trials
    sem_fr = np.std(rates, axis=0) / np.sqrt(active_trials)

    if sigma and sigma > 0:
        mean_fr = gaussian_filter1d(mean_fr, sigma=sigma)
        sem_fr = gaussian_filter1d(sem_fr, sigma=sigma)

    if normalize_by:
        mean_fr = mean_fr/normalize_by
        sem_fr = sem_fr/normalize_by

    return bin_centers, mean_fr, sem_fr