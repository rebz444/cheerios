"""
Crop-and-resort — edge-cell confound check for the Q2→Q3 rescaling signal.

Background
----------
d_tau_stratification.py showed the headline rescaling r is carried by units whose
peak sits in the edge τ-bins (τ_Q2 < 0.2 or > 0.8). This script runs step 2 of
the Open Questions.md resolution path: crop-and-resort.

For each unit, crop the first/last X% of its normalized-time PSTH, re-find the
peak inside the cropped window [X, 1-X], and re-run the rescaling correlation.

    - A unit that fires at the edges *and* the middle keeps a genuine mid
      peak once the edges are removed → it still tracks Q2↔Q3.
    - A pure onset / offset transient has no real peak inside the window; its
      re-found "peak" lands on the crop boundary (the PSTH is still monotone
      there) or in noise → it stops tracking Q2↔Q3 and gets exposed.

No cache regenerate is needed: each results_cache.pkl already stores
``q_norm_spikes`` (per-trial normalized spike times), so the cropped PSTH is
rebuilt — re-binned, re-smoothed, re-z-scored — directly from the cache. At
crop=0 this reproduces the cached τ_Q2 / τ_Q3 exactly (verified to ~1e-16).

Two readouts per (cell set × group × anchor × crop):

    r_all       peak-time correlation over every unit with a valid cropped
                PSTH. Note: a transient that decays monotonically into the
                window still pins its argmax to the crop boundary in *both*
                quartiles, so it sits on the (X,X) diagonal and can keep
                r_all high even though it is not a real timer.

    r_interior  the clean readout — correlation restricted to units whose
                re-found peak is strictly interior (not boundary-pinned) in
                *both* Q2 and Q3. ``frac_interior`` reports how many units
                survive as genuine interval-tilers.

If r_interior holds up as X grows, the rescaling signal is real. If it
collapses while r_all stays high, the headline r was edge-cell contamination.

Outputs (data/rescaling/crop_resort/)
-------------------------------------
    crop_resort_table.csv     one row per cell_set × group × anchor × crop.
    crop_resort_curves_{cue_on,cue_off,last_lick}.png
                              r vs crop fraction, per cell set — one figure
                              per anchor, short and long BG at the same anchor.
    crop_resort_heatmap.png   region × crop heatmap of r_interior (n_interior
                              annotated), faceted by anchor × group.
    crop_<XX>%/<sort>_sort/<anchor>/normsort_<set_label>.png
                              per (cell set × anchor × crop × sort method) —
                              short vs long BG compared at the SAME anchor.
                              2 rows × 4 cols (QA abs, QB abs, QA norm
                              cropped, QB norm cropped); SORT_METHODS picks
                              the row order: 'qa' = sort by QA cropped peak
                              only (the original asymmetric layout); 'mean' =
                              sort by mean of QA+QB cropped peaks (symmetric
                              consensus). Default renders both side by side
                              under each crop folder.

Usage
-----
    python e_crop_resort_rescaling.py
"""
from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from a_rescaling import (
    DIR_BASE, paths_for, compute_psth,
    N_BINS_NORM, SIGMA, MIN_TRIALS, N_SHUFFLE,
    TIME_STEP as ABS_TIME_STEP, T_MAX_SHORT, T_MAX_LONG,
    COLOR_SHORT, COLOR_LONG, QUARTILE_PAIR,
)
from b_cross_region_rescaling import (
    PANEL, GROUPS, ANCHORS, BEHAVIOR_ANCHOR, REGION_COLORS,
)


# ── Config ──────────────────────────────────────────────────────────────────

# Quartile pair compared (project-wide default from rescaling).
QA, QB = QUARTILE_PAIR

# Fraction cropped from EACH end of the normalized interval. 0.0 = uncropped
# (reproduces the cached metrics); 0.15 keeps the middle 70%.
CROP_FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
HEADLINE_CROP  = 0.15            # crop used for the console digest
# Crop fractions to render normsort population heatmaps at — one
# crop_<XX>%/ subfolder per entry.
NORMSORT_CROPS = [0.15, 0.25]
# Sort methods rendered side by side under each crop folder.
#   'qa'       — sort by the QA cropped-norm peak only (asymmetric: QA
#                panel diagonal by construction, QB diagonality = rescaling
#                test). The original layout.
#   'mean'     — sort by the mean of QA and QB cropped-norm peak bins
#                (symmetric consensus; neither panel favored).
#   'spectral' — spectral seriation on the joint [QA|QB] cropped-norm PSTH
#                matrix. Uses the whole PSTH shape (not just argmax), so
#                units with similar joint dynamics sit adjacent. Tends to
#                give the visually cleanest diagonal across both panels.
#                Caveat: row order is shape-similarity, not strictly
#                peak-time; orientation auto-aligned so low row = early peak.
SORT_METHODS = ['qa', 'mean', 'spectral']
ANCHOR_ORDER   = ['cue_on', 'cue_off', 'last_lick']

TIME_STEP = 1.0 / N_BINS_NORM    # 0.01 — matches the cached normalized PSTH

# A re-found peak is "boundary-pinned" if its argmax lands within GUARD_BINS of
# either end of the cropped window — i.e. the true peak is outside the window.
GUARD_BINS = 1

# Minimum units before r / slope are estimated for a stratum.
MIN_UNITS = 5

# frac_rescaling — matches a_rescaling.compute_metrics.
TAU_FLOOR = 0.05
RESCALE_TOL  = 0.2

OUT_DIR = DIR_BASE / 'crop_resort'


# ── Cropped peak detection ──────────────────────────────────────────────────

def cropped_peak(spike_lists, crop):
    """Re-find a unit's normalized peak inside the cropped window [crop, 1-crop].

    Rebuilds the PSTH from per-trial normalized spike times exactly as
    rescaling does (re-bin, gaussian-smooth, z-score), then takes
    the argmax. Returns (tau, pinned) where tau is the peak position in the
    original [0, 1] normalized frame and pinned is True if the argmax sits on
    the crop boundary. Returns None if the unit has too few trials or a flat
    (all-zero) PSTH inside the window.
    """
    if len(spike_lists) < MIN_TRIALS:
        return None
    t_min, t_max = crop, 1.0 - crop
    bin_centers, norm = compute_psth(spike_lists, t_min, t_max, TIME_STEP, SIGMA)
    if norm is None:
        return None
    peak_idx = int(np.argmax(norm))
    n_bins   = len(norm)
    tau      = t_min + peak_idx * TIME_STEP
    pinned   = (peak_idx <= GUARD_BINS - 1) or (peak_idx >= n_bins - GUARD_BINS)
    return tau, pinned


def cropped_taus(q_norm_spikes, sort_uids, crop):
    """Cropped τ_Q2 / τ_Q3 for every unit valid in both quartiles.

    Returns (tau_a, tau_b, interior) arrays aligned across the kept units,
    plus n_dropped (units with no valid cropped PSTH in one quartile).
    ``interior`` is True for units strictly interior in BOTH quartiles.
    """
    tau_a, tau_b, interior = [], [], []
    n_dropped = 0
    for uid in sort_uids:
        p3 = cropped_peak(q_norm_spikes[QA].get(uid, []), crop)
        p4 = cropped_peak(q_norm_spikes[QB].get(uid, []), crop)
        if p3 is None or p4 is None:
            n_dropped += 1
            continue
        tau_a.append(p3[0])
        tau_b.append(p4[0])
        interior.append((not p3[1]) and (not p4[1]))
    return (np.array(tau_a), np.array(tau_b),
            np.array(interior, dtype=bool), n_dropped)


# ── Rescaling metrics ───────────────────────────────────────────────────────

def rescaling_metrics(tau_a, tau_b, rng):
    """r (+ permutation p), OLS slope, frac_rescaling for one set of units."""
    n = len(tau_a)
    out = {'n': n, 'r': np.nan, 'p_shuffle': np.nan,
           'slope': np.nan, 'frac_rescaling': np.nan}
    if n < MIN_UNITS or np.std(tau_a) == 0 or np.std(tau_b) == 0:
        # frac_rescaling can still be defined for a tiny / degenerate stratum.
        valid = tau_b > TAU_FLOOR
        if valid.any():
            sf = tau_a[valid] / tau_b[valid]
            out['frac_rescaling'] = float(np.mean(np.abs(sf - 1.0) < RESCALE_TOL))
        return out

    r = float(np.corrcoef(tau_a, tau_b)[0, 1])
    r_shuffle = np.array([
        np.corrcoef(tau_a, rng.permutation(tau_b))[0, 1]
        for _ in range(N_SHUFFLE)
    ])
    out['r']         = r
    out['p_shuffle'] = float(np.mean(r_shuffle >= r))
    out['slope']     = float(stats.linregress(tau_a, tau_b).slope)

    valid = tau_b > TAU_FLOOR
    if valid.any():
        sf = tau_a[valid] / tau_b[valid]
        out['frac_rescaling'] = float(np.mean(np.abs(sf - 1.0) < RESCALE_TOL))
    return out


# ── Collect ─────────────────────────────────────────────────────────────────

def collect():
    """Walk the PANEL caches, sweeping crop fractions. Returns the long table."""
    rows = []
    for display, label, region in PANEL:
        _, _, _, cache_file = paths_for(label)
        if not cache_file.exists():
            print(f"  [skip] {label}: no cache at {cache_file}")
            continue
        with open(cache_file, 'rb') as f:
            data_cache = pickle.load(f)['data_cache']

        for group in GROUPS:
            for anchor in ANCHORS:
                key = (group, f'{anchor}_time')
                if key not in data_cache:
                    continue
                entry     = data_cache[key]
                sort_uids = entry['metrics'].get('sort_uids', [])
                q_norm    = entry['q_norm_spikes']
                if not sort_uids:
                    continue

                for crop in CROP_FRACTIONS:
                    rng = np.random.default_rng(42)
                    tau3, tau4, interior, n_dropped = cropped_taus(
                        q_norm, sort_uids, crop)
                    n_units = len(tau3)
                    if n_units == 0:
                        continue
                    m_all = rescaling_metrics(tau3, tau4, rng)
                    m_int = rescaling_metrics(tau3[interior], tau4[interior],
                                              np.random.default_rng(43))
                    rows.append({
                        'display': display, 'cell_set': label, 'region': region,
                        'group': group, 'anchor': anchor,
                        'crop_frac': crop,
                        'n_units': n_units, 'n_dropped': n_dropped,
                        'n_interior': int(interior.sum()),
                        'frac_interior': float(interior.mean()),
                        'r_all': m_all['r'],
                        'p_shuffle_all': m_all['p_shuffle'],
                        'slope_all': m_all['slope'],
                        'frac_rescaling_all': m_all['frac_rescaling'],
                        'r_interior': m_int['r'],
                        'p_shuffle_interior': m_int['p_shuffle'],
                        'slope_interior': m_int['slope'],
                        'frac_rescaling_interior': m_int['frac_rescaling'],
                    })
        print(f"  done {label}")

    return pd.DataFrame(rows).sort_values(
        ['region', 'cell_set', 'group', 'anchor', 'crop_frac']
    ).reset_index(drop=True)


# ── Figures ─────────────────────────────────────────────────────────────────

def figure_curves(df, anchor):
    """r vs crop fraction, one panel per cell set, for a single anchor —
    short and long BG compared like-for-like (no cross-group anchor mixing)."""
    displays = [d for d, _, _ in PANEL if d in df['display'].unique()]
    ncol = 4
    nrow = int(np.ceil(len(displays) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False)

    for idx, disp in enumerate(displays):
        ax = axes[idx // ncol][idx % ncol]
        region = next(r for d, _, r in PANEL if d == disp)
        for group, color in [('short_BG', COLOR_SHORT), ('long_BG', COLOR_LONG)]:
            sub = df[(df['display'] == disp) & (df['group'] == group) &
                     (df['anchor'] == anchor)].sort_values('crop_frac')
            if sub.empty:
                continue
            xv = sub['crop_frac'].values * 100
            ax.plot(xv, sub['r_all'], '-o', color=color, ms=4, lw=1.8,
                    label=f'{group.replace("_BG", "")}  all')
            ax.plot(xv, sub['r_interior'], '--s', color=color, ms=4, lw=1.6,
                    alpha=0.85, mfc='white',
                    label=f'{group.replace("_BG", "")}  interior')
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_title(disp, fontsize=10, color=REGION_COLORS.get(region, 'black'),
                     fontweight='bold')
        ax.set_ylim(-0.35, 1.05)
        ax.set_xlabel('% cropped per end')
        ax.set_ylabel('Q2→Q3 peak-time r')
        ax.legend(fontsize=6.5, loc='lower left', ncol=2)
        ax.grid(alpha=0.25)

    for idx in range(len(displays), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis('off')

    fig.suptitle(f'Crop-and-resort — rescaling r vs interval cropping  ·  '
                 f'anchor: {anchor}  (short and long BG at the SAME anchor)\n'
                 'solid = all units · dashed = interior-peak units only',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'crop_resort_curves_{anchor}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


def figure_heatmap(df):
    """region × crop heatmap of r_interior, faceted by anchor × group."""
    displays = [d for d, _, _ in PANEL if d in df['display'].unique()]
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.85')
    crops = CROP_FRACTIONS

    fig, axes = plt.subplots(
        len(ANCHORS), len(GROUPS),
        figsize=(6.4 * len(GROUPS), 0.46 * len(displays) * len(ANCHORS) + 1.5),
        squeeze=False,
        constrained_layout=True,
    )
    im = None
    for ai, anchor in enumerate(ANCHORS):
        for gi, group in enumerate(GROUPS):
            ax = axes[ai][gi]
            vals = np.full((len(displays), len(crops)), np.nan)
            ns   = np.zeros((len(displays), len(crops)), dtype=int)
            for i, disp in enumerate(displays):
                sub = df[(df['display'] == disp) & (df['group'] == group) &
                         (df['anchor'] == anchor)]
                for _, row in sub.iterrows():
                    j = crops.index(row['crop_frac'])
                    vals[i, j] = row['r_interior']
                    ns[i, j]   = int(row['n_interior'])
            im = ax.imshow(np.ma.masked_invalid(vals), aspect='auto',
                           cmap=cmap, vmin=-1.0, vmax=1.0)
            for i in range(len(displays)):
                for j in range(len(crops)):
                    v, n = vals[i, j], ns[i, j]
                    if np.isnan(v):
                        ax.text(j, i, f'n={n}', ha='center', va='center',
                                fontsize=6, color='0.4')
                    else:
                        tc = 'white' if abs(v) > 0.55 else 'black'
                        ax.text(j, i, f'{v:.2f}\nn={n}', ha='center',
                                va='center', fontsize=6, color=tc)
            ax.set_xticks(range(len(crops)))
            ax.set_xticklabels([f'{int(c * 100)}%' for c in crops], fontsize=7)
            ax.set_yticks(range(len(displays)))
            ax.set_yticklabels(displays, fontsize=8)
            tag = ' ★' if anchor == BEHAVIOR_ANCHOR[group] else ''
            ax.set_title(f'{group.replace("_", " ")}  ·  {anchor}{tag}',
                         fontsize=10)
            if ai == len(ANCHORS) - 1:
                ax.set_xlabel('% cropped per end', fontsize=8)

    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cb.set_label('r_interior (interior-peak units only)')
    fig.suptitle('Crop-and-resort — interior-peak rescaling r across crop '
                 'fractions   (★ = behavioral anchor; n = interior units)',
                 fontsize=12, fontweight='bold')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'crop_resort_heatmap.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


# ── Per-cell-set committee figures (fig1b normsort, cropped+resorted) ───────

# Per-quartile percentile of pooled per-trial last-spike-time, used as a
# conservative "by now ~every trial of this quartile has ended" cutoff for
# the absolute heatmap. Higher = wider x-axis with more grey on the right.
TRIAL_END_PERCENTILE = 98


def _per_quartile_t_cap(q_spikes_quartile, percentile=TRIAL_END_PERCENTILE):
    """Estimate when ~all trials of this quartile have ended, from the pooled
    per-trial last-spike-time distribution. A lower bound on the true trial
    duration (a trial's last spike can come before its decision), so the cap
    is conservative — typically a touch shorter than the true tail. Returns
    None if no spikes anywhere."""
    last = [spk[-1] for trial_list in q_spikes_quartile.values()
            for spk in trial_list if len(spk) > 0]
    return float(np.percentile(last, percentile)) if last else None


def _build_normsort_panels(q_spk, q_norm, uids_candidate, crop, t_max_abs,
                            sort_method='mean'):
    """Build the 4 matrices for the fig1b normsort layout on cropped+resorted
    data, restricted to units valid in EVERY panel.

    sort_method controls the row order applied to all four panels:
      'qa'   — sort by QA cropped-norm peak only (the original layout: QA
               panel is a diagonal by construction, QB diagonality is the
               rescaling test).
      'mean' — sort by the mean of QA and QB cropped-norm peak bins
               (symmetric consensus order; neither panel is favored, both
               read as the same shared sequence).

    Returns a dict with mat_a_abs, mat_b_abs, mat_a_norm, mat_b_norm,
    n_units, and the per-quartile absolute-time caps. Each absolute PSTH is
    built on [0, cap_<quartile>] so the x-axis tracks where trials actually
    end (no stretched x-axis, no post-trial zero region biasing z-scoring).
    """
    t_min, t_max_norm = crop, 1.0 - crop

    # Per-quartile trial-end caps (fall back to the group's t_max_abs if a
    # quartile has no spikes — shouldn't happen in practice).
    cap_a = _per_quartile_t_cap(q_spk[QA]) or t_max_abs
    cap_b = _per_quartile_t_cap(q_spk[QB]) or t_max_abs

    valid = []
    n_a, n_b, a_a, a_b = {}, {}, {}, {}
    for uid in uids_candidate:
        sl_an = q_norm[QA].get(uid, [])
        sl_bn = q_norm[QB].get(uid, [])
        sl_aa = q_spk[QA].get(uid, [])
        sl_ba = q_spk[QB].get(uid, [])
        if (len(sl_an) < MIN_TRIALS or len(sl_bn) < MIN_TRIALS
                or len(sl_aa) < MIN_TRIALS or len(sl_ba) < MIN_TRIALS):
            continue
        _, na = compute_psth(sl_an, t_min, t_max_norm, TIME_STEP, SIGMA)
        _, nb = compute_psth(sl_bn, t_min, t_max_norm, TIME_STEP, SIGMA)
        _, aa = compute_psth(sl_aa, 0.0, cap_a, ABS_TIME_STEP, SIGMA)
        _, ab = compute_psth(sl_ba, 0.0, cap_b, ABS_TIME_STEP, SIGMA)
        if na is None or nb is None or aa is None or ab is None:
            continue
        valid.append(uid)
        n_a[uid], n_b[uid] = na, nb
        a_a[uid], a_b[uid] = aa, ab
    if not valid:
        return None
    mat_an_unsorted = np.array([n_a[u] for u in valid])
    mat_bn_unsorted = np.array([n_b[u] for u in valid])
    peak_a = np.argmax(mat_an_unsorted, axis=1).astype(float)
    peak_b = np.argmax(mat_bn_unsorted, axis=1).astype(float)
    if sort_method == 'qa':
        order = np.argsort(peak_a)
    elif sort_method == 'mean':
        order = np.argsort((peak_a + peak_b) / 2.0)
    elif sort_method == 'spectral':
        # Spectral seriation on the joint [QA | QB] cropped-norm PSTH matrix.
        # Each unit is a 2*n_bins feature vector; rows are ordered by the
        # Fiedler vector (2nd-smallest-eigenvalue eigenvector) of the
        # row-similarity graph Laplacian. Places units with similar joint
        # dynamics adjacent — typically gives the visually cleanest diagonal.
        if mat_an_unsorted.shape[0] < 3:
            order = np.argsort((peak_a + peak_b) / 2.0)
        else:
            X = np.hstack([mat_an_unsorted, mat_bn_unsorted])
            # Pearson r between rows shifted to [0, 1] for a non-negative
            # similarity; NaN rows (degenerate PSTHs) become zero weight.
            W = np.nan_to_num((np.corrcoef(X) + 1.0) / 2.0, nan=0.0)
            np.fill_diagonal(W, 0.0)
            L = np.diag(W.sum(axis=1)) - W
            _evals, evecs = np.linalg.eigh(L)
            fiedler = evecs[:, 1]              # 2nd smallest eigenvalue
            order = np.argsort(fiedler)
            # Fiedler sign is arbitrary — orient so low row = early peak.
            mean_peaks_ordered = ((peak_a + peak_b) / 2.0)[order]
            if mean_peaks_ordered.std() > 0:
                trend = np.corrcoef(np.arange(len(order)),
                                    mean_peaks_ordered)[0, 1]
                if trend < 0:
                    order = order[::-1]
    else:
        raise ValueError(f"Unknown sort_method: {sort_method!r}")
    sorted_uids = [valid[i] for i in order]

    return {
        'mat_a_abs':  np.array([a_a[u] for u in sorted_uids]),
        'mat_b_abs':  np.array([a_b[u] for u in sorted_uids]),
        'mat_a_norm': np.array([n_a[u] for u in sorted_uids]),
        'mat_b_norm': np.array([n_b[u] for u in sorted_uids]),
        'n_units':    len(sorted_uids),
        'cap_a':      cap_a,
        'cap_b':      cap_b,
    }


def make_normsort_figures(df, crops=NORMSORT_CROPS, sort_methods=SORT_METHODS):
    """Per (cell set × anchor × crop × sort method): fig1b-style population
    heatmaps (normsort) on cropped+resorted data, short vs long BG compared
    at the SAME anchor.

    Layout per figure: 2 rows (short BG, long BG) × 4 cols (QA absolute,
    QB absolute, QA normalized [cropped], QB normalized [cropped]). All four
    panels in a row share the same row order; ``sort_methods`` controls how
    that order is computed (see _build_normsort_panels).

    Output is grouped by crop fraction × sort method × anchor so the two
    sort views sit side by side under each crop folder:

        data/rescaling/crop_resort/crop_<XX>%/<sort>_sort/<anchor>/normsort_<set>.png
    """
    for display, label, _ in PANEL:
        _, _, _, cache_file = paths_for(label)
        if not cache_file.exists():
            print(f"  [skip] {label}: no cache")
            continue
        with open(cache_file, 'rb') as f:
            data_cache = pickle.load(f)['data_cache']

        for crop in crops:
            for sort_method in sort_methods:
                sort_desc = {
                    'qa':       f"each group's cropped {QA} peak",
                    'mean':     f"mean of each group's cropped {QA}+{QB} peak",
                    'spectral': (f"spectral seriation on each group's joint "
                                 f"{QA}|{QB} cropped PSTH"),
                }[sort_method]
                for anchor in ANCHORS:
                    # Skip (cell set, anchor) entirely if neither group has data.
                    keys_ok = {g: (g, f'{anchor}_time') in data_cache
                               for g in GROUPS}
                    if not any(keys_ok.values()):
                        continue

                    fig = plt.figure(figsize=(16, 8))
                    gs  = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.25)
                    drew_any = False

                    for row, group in enumerate(GROUPS):
                        if not keys_ok[group]:
                            continue
                        key       = (group, f'{anchor}_time')
                        entry     = data_cache[key]
                        r_cached  = entry['metrics'].get('r', np.nan)
                        sort_uids = entry['metrics'].get('sort_uids', [])
                        t_max_abs = (T_MAX_SHORT if group == 'short_BG'
                                     else T_MAX_LONG)
                        q_med     = entry['q_medians']
                        counts    = entry['counts']

                        panels = _build_normsort_panels(
                            entry['q_spikes'], entry['q_norm_spikes'],
                            sort_uids, crop, t_max_abs,
                            sort_method=sort_method)
                        if panels is None:
                            continue
                        n_u = panels['n_units']
                        drew_any = True

                        rrow = df[(df['display'] == display)
                                  & (df['group'] == group)
                                  & (df['anchor'] == anchor)
                                  & (df['crop_frac'] == crop)]
                        r_int = (rrow['r_interior'].iloc[0]
                                 if len(rrow) else np.nan)
                        r_int_txt    = ('nan' if pd.isna(r_int)
                                        else f'{r_int:.2f}')
                        r_cached_txt = ('nan' if not np.isfinite(r_cached)
                                        else f'{r_cached:.2f}')

                        cap_a_p, cap_b_p = panels['cap_a'], panels['cap_b']
                        matrices = [panels['mat_a_abs'], panels['mat_b_abs'],
                                    panels['mat_a_norm'], panels['mat_b_norm']]
                        titles = [
                            f"{QA} absolute\n(median {q_med.get(QA, 0):.1f}s)",
                            f"{QB} absolute\n(median {q_med.get(QB, 0):.1f}s)",
                            f"{QA} normalized (cropped {int(crop * 100)}%)",
                            f"{QB} normalized (cropped {int(crop * 100)}%)",
                        ]
                        extents = [
                            [0, cap_a_p, 0, n_u], [0, cap_b_p, 0, n_u],
                            [crop, 1.0 - crop, 0, n_u],
                            [crop, 1.0 - crop, 0, n_u],
                        ]
                        xlabels = ['Time (s)', 'Time (s)',
                                   'Normalized time (cropped)',
                                   'Normalized time (cropped)']
                        vlines = [q_med.get(QA, 0), q_med.get(QB, 0),
                                  None, None]
                        group_label = ('Short BG' if group == 'short_BG'
                                       else 'Long BG')

                        for col in range(4):
                            ax = fig.add_subplot(gs[row, col])
                            mat = matrices[col]
                            if mat is not None and mat.size:
                                vmax = float(np.percentile(np.abs(mat), 95))
                                if not np.isfinite(vmax) or vmax == 0:
                                    vmax = 1.0
                                ax.imshow(mat, aspect='auto', cmap='viridis',
                                          vmin=-vmax, vmax=vmax, origin='lower',
                                          extent=extents[col])
                                if vlines[col] is not None:
                                    ax.axvline(vlines[col], color='white',
                                               lw=2, ls='--', alpha=0.9)
                            ax.set_xlabel(xlabels[col], fontsize=8)
                            ax.set_title(titles[col], fontsize=8)
                            if col == 0:
                                ax.set_ylabel(
                                    f"{group_label}  ({QA}→{QB}  "
                                    f"r={r_cached_txt}, "
                                    f"r_int@{int(crop * 100)}%={r_int_txt})\n"
                                    f"n={counts['n_mice']} mice · "
                                    f"{counts['n_sessions']} sess · "
                                    f"{n_u} units · "
                                    f"{counts['n_trials']:,} trials\n\nUnit #",
                                    fontsize=9)
                            else:
                                ax.set_yticklabels([])

                    if not drew_any:
                        plt.close(fig)
                        continue

                    fig.suptitle(
                        f'[{label}]  Crop-and-resort population heatmaps  '
                        f'(normsort, {int(crop * 100)}% cropped each end)\n'
                        f'anchor: {anchor.replace("_", " ")}  ·  rows sorted '
                        f'by {sort_desc}  ·  short vs long compared',
                        fontsize=13, fontweight='bold', y=0.97)

                    anchor_dir = (OUT_DIR / f'crop_{int(crop * 100)}%'
                                  / f'{sort_method}_sort' / anchor)
                    anchor_dir.mkdir(parents=True, exist_ok=True)
                    out_path = anchor_dir / f'normsort_{label}.png'
                    fig.savefig(out_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Saved figure → {out_path}")


# ── Console digest ──────────────────────────────────────────────────────────

def print_digest(df):
    """Uncropped vs HEADLINE_CROP — all three anchors, both groups at each."""
    def fmt(x):
        return ' nan' if pd.isna(x) else f'{x:.2f}'

    print()
    print("=" * 84)
    print(f"  Crop-and-resort digest — uncropped vs {int(HEADLINE_CROP*100)}% "
          f"crop, all anchors group-matched")
    print("  r_interior collapsing while r_all holds  ->  headline r was "
          "edge-cell driven")
    print("=" * 84)
    hdr = (f"  {'cell set':<13}{'group':<10}"
           f"{'r0':>7}{'r_all':>8}{'r_int':>8}{'n_int':>7}{'%interior':>11}")
    for anchor in ANCHOR_ORDER:
        sub = df[df['anchor'] == anchor]
        print(f"\n--- {anchor} ---")
        print(hdr)
        for (region, disp, group), g in sub.groupby(
                ['region', 'display', 'group']):
            g = g.set_index('crop_frac')
            if HEADLINE_CROP not in g.index:
                continue
            r0 = g.loc[0.0, 'r_all'] if 0.0 in g.index else np.nan
            hc = g.loc[HEADLINE_CROP]
            print(f"  {hc['cell_set']:<13}{group:<10}"
                  f"{fmt(r0):>7}{fmt(hc['r_all']):>8}{fmt(hc['r_interior']):>8}"
                  f"{int(hc['n_interior']):>7}{hc['frac_interior']*100:>10.0f}%")
    print()


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 86)
    print("  Crop-and-resort rescaling analysis")
    print("=" * 86)

    df = collect()
    n_sets = df['cell_set'].nunique()
    print(f"\n  Collected {n_sets} cell sets, {len(df)} "
          f"cell_set × group × anchor × crop rows")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / 'crop_resort_table.csv'
    df.to_csv(table_path, index=False)
    print(f"  Saved table  → {table_path}")

    for anchor in ANCHOR_ORDER:
        figure_curves(df, anchor)
    figure_heatmap(df)
    make_normsort_figures(df)
    print_digest(df)
    print("Done.")
