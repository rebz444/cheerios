"""
Peak-aligned width test — the decisive rerun for norm-vs-absolute.

The 2x2 peak-aligned panels in f_psth_similarity.py compared Q3 vs Q4 (stretch
factor only ~1.3-2x) and the normalized-minus-absolute contrast came out flat.
But adjacent quartiles are exactly where a width-rescaling test has the least
power. This script gives that contrast a fair test:

1. Maximal duration contrast. Run the peak-aligned normalized-vs-absolute
   contrast on Q1 vs Q4 (and Q3 vs Q4 for reference), not adjacent quartiles.
   Q1 vs Q4 reaches 5-11x stretch at the cue_off anchor — where, if response
   width rescales, the normalized profiles must match far better than the
   absolute ones, and if width is fixed the absolute profiles must win.

2. Fraction-positive vs a shuffle null. The median is the wrong statistic for
   a mixed population; the right one is the subpopulation skew. Per region we
   report the fraction of units with a positive contrast (norm > abs => the
   unit's width rescales) and test it against a within-unit Qa/Qb-label
   shuffle null: the unit's Qa+Qb trials are pooled and randomly relabelled,
   destroying the genuine duration difference. The null frac-positive captures
   the metric's own bias, so a region's real frac-positive either clears it or
   does not.

3. Per-region duration ratio is logged (median Qb wait / median Qa wait, from
   the cached q_medians) so the test's sensitivity is known per cell, not
   assumed. NOTE: the long-BG behavioral anchor (cue_on) reaches only ~2.3x
   even at Q1 vs Q4 — its durations are dominated by the background period —
   so cue_off is the genuine high-power anchor. All three anchors are run.

Verdict logic: if, at the high-stretch cells (cue_off, Q1 vs Q4), the real
frac-positive does not clear the shuffle null, the contrast is flat where it
should have power -> fixed-width is the real answer, reportable with confidence.

Cropped PSTHs only (CROP_FRAC per end): a peak-aligned width test must drop
the anchor transients before re-finding the peak. The uncropped peak-aligned
numbers are in f_psth_similarity.py.

No cache regenerate: q_spikes / q_norm_spikes hold the per-trial spike times.
Speed: each unit's per-trial spike histograms are precomputed once, so every
shuffle iteration is a sum + smooth + z-score, not a re-histogram.

Outputs (data/rescaling/peak_width_test/)
-----------------------------------------
    peak_width_test_per_unit.csv   one row per unit x anchor x Q-pair.
    peak_width_test_region.csv     per cell_set x group x anchor x Q-pair x
                                   metric — frac-positive, shuffle null, p,
                                   duration ratio, verdict.
    peak_width_test_stretch.png    skew (real - null frac-positive) vs
                                   duration ratio — does the contrast emerge
                                   as stretch grows?
    peak_width_test_maxstretch.png caterpillar of frac-positive vs null at the
                                   highest-stretch cells (cue_off, Q1 vs Q4).

Usage
-----
    python rescaling/g_peak_width_test.py
"""
from __future__ import annotations

import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from a_rescaling import (
    DIR_BASE, paths_for, SIGMA, MIN_TRIALS, TIME_STEP,
    T_MAX_SHORT, T_MAX_LONG, N_BINS_NORM,
)
from b_cross_region_rescaling import PANEL, GROUPS, ANCHORS, REGION_COLORS
from f_psth_similarity import xcorr, procsim, peak_align, CROP_FRAC


# ── Config ──────────────────────────────────────────────────────────────────

Q_PAIRS   = [('Q3', 'Q4'), ('Q1', 'Q4')]   # reference stretch, then maximal
N_SHUFFLE = 200                             # within-unit Qa/Qb-label shuffles
RNG_SEED  = 20260521
METRICS   = ['xcorr', 'procsim']

TS_NORM    = 1.0 / N_BINS_NORM
EDGES_NORM = np.arange(0.0, 1.0 + TS_NORM, TS_NORM)        # matches compute_psth
T_MAX_ABS  = {'short_BG': T_MAX_SHORT, 'long_BG': T_MAX_LONG}
C0 = int(round(CROP_FRAC * N_BINS_NORM))                   # cropped-norm columns
C1 = N_BINS_NORM - C0

OUT_DIR = DIR_BASE / 'peak_width_test'

assert len(EDGES_NORM) - 1 == N_BINS_NORM, "normalized bin grid mismatch"


# ── PSTH from precomputed per-trial histograms ──────────────────────────────

def psth_from_counts(counts):
    """z-scored smoothed PSTH from a summed spike-count vector.

    Equivalent to a_rescaling.compute_psth: the rate scaling it
    applies (/ n_trials / bin_width) is removed by the z-score, and smoothing
    is linear, so smoothing+z-scoring the raw counts gives the identical PSTH.
    """
    sm = gaussian_filter1d(np.asarray(counts, float), SIGMA)
    sd = sm.std()
    if sd == 0:
        return None
    return (sm - sm.mean()) / sd


def precompute_unit(abs_lists, norm_lists, edges_abs):
    """Per-trial spike-count histograms for one unit's pooled Qa+Qb trials.

    Returns (H_norm, H_abs_crop): H_norm is the normalized-time histogram over
    [0, 1] (a column slice gives the cropped version); H_abs_crop is the
    absolute-time histogram of only the spikes whose normalized position lies
    in [CROP_FRAC, 1-CROP_FRAC] (the cropped absolute PSTH)."""
    n = len(abs_lists)
    H_norm     = np.zeros((n, N_BINS_NORM))
    H_abs_crop = np.zeros((n, len(edges_abs) - 1))
    for i, (a, nrm) in enumerate(zip(abs_lists, norm_lists)):
        a = np.asarray(a, float)
        nrm = np.asarray(nrm, float)
        H_norm[i] = np.histogram(nrm, bins=EDGES_NORM)[0]
        keep = (nrm >= CROP_FRAC) & (nrm <= 1.0 - CROP_FRAC)
        H_abs_crop[i] = np.histogram(a[keep], bins=edges_abs)[0]
    return H_norm, H_abs_crop


def unit_contrast(H_norm, H_abs_crop, idx_a, idx_b):
    """Peak-aligned normalized − absolute similarity for one Qa/Qb split."""
    n_a = psth_from_counts(H_norm[idx_a][:, C0:C1].sum(0))
    n_b = psth_from_counts(H_norm[idx_b][:, C0:C1].sum(0))
    a_a = psth_from_counts(H_abs_crop[idx_a].sum(0))
    a_b = psth_from_counts(H_abs_crop[idx_b].sum(0))
    seg_norm = (peak_align(n_a, n_b)
                if n_a is not None and n_b is not None else None)
    seg_abs  = (peak_align(a_a, a_b)
                if a_a is not None and a_b is not None else None)
    out = {}
    for name, metric in (('xcorr', xcorr), ('procsim', procsim)):
        nv = metric(*seg_norm) if seg_norm is not None else np.nan
        av = metric(*seg_abs) if seg_abs is not None else np.nan
        out[name]            = (nv - av if not (np.isnan(nv) or np.isnan(av))
                                else np.nan)
        out[f'{name}_norm']  = nv
        out[f'{name}_abs']   = av
    return out


# ── Collect ─────────────────────────────────────────────────────────────────

def collect():
    """Run the real + shuffle contrasts for every unit, anchor and Q-pair."""
    rng = np.random.default_rng(RNG_SEED)
    per_unit = []
    # cells[(display,label,region,group,anchor,qa,qb)] = dict(dur_ratio, real, null)
    cells = {}

    for display, label, region in PANEL:
        _, _, _, cache_file = paths_for(label)
        if not cache_file.exists():
            print(f"  [skip] {label}: no cache")
            continue
        with open(cache_file, 'rb') as f:
            data_cache = pickle.load(f)['data_cache']

        for group in GROUPS:
            edges_abs = np.arange(0.0, T_MAX_ABS[group] + TIME_STEP, TIME_STEP)
            for anchor in ANCHORS:
                key = (group, f'{anchor}_time')
                if key not in data_cache:
                    continue
                entry     = data_cache[key]
                sort_uids = entry['metrics'].get('sort_uids', [])
                q_abs, q_norm = entry['q_spikes'], entry['q_norm_spikes']
                q_med     = entry['q_medians']

                for qa, qb in Q_PAIRS:
                    dur_ratio = (float(q_med[qb] / q_med[qa])
                                 if qa in q_med.index and qb in q_med.index
                                 and q_med[qa] > 0 else np.nan)
                    cell_key = (display, label, region, group, anchor, qa, qb)
                    cell = {'dur_ratio': dur_ratio,
                            'real': {m: [] for m in METRICS},
                            'null': {m: [] for m in METRICS}}

                    for uid in sort_uids:
                        abs_a, norm_a = q_abs[qa].get(uid, []), q_norm[qa].get(uid, [])
                        abs_b, norm_b = q_abs[qb].get(uid, []), q_norm[qb].get(uid, [])
                        n_a, n_b = len(abs_a), len(abs_b)
                        if n_a < MIN_TRIALS or n_b < MIN_TRIALS:
                            continue
                        H_norm, H_abs_crop = precompute_unit(
                            list(abs_a) + list(abs_b),
                            list(norm_a) + list(norm_b), edges_abs)
                        pool = n_a + n_b
                        idx_a0 = np.arange(n_a)
                        idx_b0 = np.arange(n_a, pool)

                        real = unit_contrast(H_norm, H_abs_crop, idx_a0, idx_b0)
                        null = {m: np.empty(N_SHUFFLE) for m in METRICS}
                        for s in range(N_SHUFFLE):
                            perm = rng.permutation(pool)
                            c = unit_contrast(H_norm, H_abs_crop,
                                              perm[:n_a], perm[n_a:])
                            for m in METRICS:
                                null[m][s] = c[m]

                        for m in METRICS:
                            cell['real'][m].append(real[m])
                            cell['null'][m].append(null[m])

                        row = {'display': display, 'cell_set': label,
                               'region': region, 'group': group,
                               'anchor': anchor, 'unit_gid': uid,
                               'qa': qa, 'qb': qb, 'dur_ratio': dur_ratio}
                        for m in METRICS:
                            row[f'{m}_contrast']    = real[m]
                            row[f'{m}_norm']        = real[f'{m}_norm']
                            row[f'{m}_abs']         = real[f'{m}_abs']
                            row[f'null_{m}_mean']   = float(np.nanmean(null[m]))
                            row[f'null_{m}_std']    = float(np.nanstd(null[m]))
                        per_unit.append(row)

                    cells[cell_key] = cell
        print(f"  done {label}")

    return pd.DataFrame(per_unit), cells


# ── Region-level frac-positive vs shuffle null ──────────────────────────────

def region_table(cells):
    """Aggregate each cell to frac-positive, shuffle null and a verdict."""
    rows = []
    for (display, label, region, group, anchor, qa, qb), cell in cells.items():
        for metric in METRICS:
            real = np.array(cell['real'][metric], float)
            if real.size == 0:
                continue
            null = np.vstack(cell['null'][metric])          # (n_units, N_SHUFFLE)
            keep = ~np.isnan(real)
            n = int(keep.sum())
            if n == 0:
                continue
            real_fp = float(np.mean(real[keep] > 0))
            # Null frac-positive per shuffle, over the same units.
            sub = null[keep]
            null_fp = np.array([
                np.mean(col[~np.isnan(col)] > 0) if np.any(~np.isnan(col))
                else np.nan
                for col in sub.T])
            null_fp = null_fp[~np.isnan(null_fp)]
            if null_fp.size == 0:
                continue
            nm, ns = float(null_fp.mean()), float(null_fp.std())
            lo, hi = np.percentile(null_fp, [2.5, 97.5])
            # Two-sided p with add-one smoothing.
            ge = (1 + np.sum(null_fp >= real_fp)) / (null_fp.size + 1)
            le = (1 + np.sum(null_fp <= real_fp)) / (null_fp.size + 1)
            p = float(min(1.0, 2 * min(ge, le)))
            z = (real_fp - nm) / ns if ns > 0 else np.nan
            if p < 0.05 and real_fp > nm:
                verdict = 'width rescales'
            elif p < 0.05 and real_fp < nm:
                verdict = 'fixed width'
            else:
                verdict = 'flat (n.s.)'
            rows.append({
                'display': display, 'cell_set': label, 'region': region,
                'group': group, 'anchor': anchor, 'qa': qa, 'qb': qb,
                'metric': metric, 'dur_ratio': cell['dur_ratio'],
                'n_units': n,
                'real_frac_pos': real_fp,
                'real_contrast_median': float(np.nanmedian(real[keep])),
                'null_fp_mean': nm, 'null_fp_std': ns,
                'null_fp_lo': float(lo), 'null_fp_hi': float(hi),
                'z': z, 'p_two_sided': p, 'verdict': verdict,
            })
    return pd.DataFrame(rows).sort_values(
        ['region', 'cell_set', 'group', 'qa', 'anchor', 'metric']
    ).reset_index(drop=True)


# ── Figures ─────────────────────────────────────────────────────────────────

GROUP_MARKER = {'short_BG': 'o', 'long_BG': 's'}


def figure_stretch(rt):
    """Skew (real − null frac-positive) vs duration ratio, per metric."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), squeeze=False)
    for ax, metric in zip(axes[0], METRICS):
        sub = rt[(rt['metric'] == metric) & rt['dur_ratio'].notna()]
        for _, r in sub.iterrows():
            skew = r['real_frac_pos'] - r['null_fp_mean']
            err = r['null_fp_hi'] - r['null_fp_mean']
            ax.errorbar(r['dur_ratio'], skew, yerr=err, fmt=GROUP_MARKER[r['group']],
                        ms=6, mfc=REGION_COLORS.get(r['region'], 'gray'),
                        mec='black', mew=0.5, ecolor='0.7', elinewidth=1,
                        capsize=2, alpha=0.9)
        ax.axhline(0, color='red', ls='--', lw=1)
        ax.set_xscale('log')
        ax.set_xticks([1.3, 2, 3, 5, 8, 11])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel('Q-pair duration ratio  (median Qb wait / median Qa wait)')
        ax.set_ylabel('frac-positive skew  (real − shuffle-null)')
        ax.set_title(f'{metric}   — >0: width rescales · <0: fixed width\n'
                     'error bar = null 95% band; outside it = clears the null',
                     fontsize=10)
        ax.grid(alpha=0.25)
    region_handles = [plt.Line2D([], [], marker='o', ls='', mfc=c, mec='black',
                                 label=reg)
                      for reg, c in REGION_COLORS.items()]
    group_handles = [plt.Line2D([], [], marker=m, ls='', mfc='0.6', mec='black',
                                label=g.replace('_', ' '))
                     for g, m in GROUP_MARKER.items()]
    axes[0][0].legend(handles=region_handles + group_handles, fontsize=8,
                      loc='upper left')
    fig.suptitle('Peak-aligned width test — does the normalized−absolute '
                 'contrast emerge as stretch grows?  (cropped, all anchors)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / 'peak_width_test_stretch.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {path}")


def figure_maxstretch(rt):
    """Caterpillar of frac-positive vs null at the highest-stretch cells
    (cue_off, Q1 vs Q4)."""
    sub = rt[(rt['anchor'] == 'cue_off') & (rt['qa'] == 'Q1') &
             (rt['qb'] == 'Q4')].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), squeeze=False)
    for ax, metric in zip(axes[0], METRICS):
        d = sub[sub['metric'] == metric].sort_values(
            ['region', 'cell_set', 'group']).reset_index(drop=True)
        for i, r in d.iterrows():
            ax.plot([i, i], [r['null_fp_lo'], r['null_fp_hi']],
                    color='0.6', lw=4, alpha=0.6, solid_capstyle='round')
            ax.plot(i, r['null_fp_mean'], '_', color='0.3', ms=10)
            col = ('#2ca02c' if r['verdict'] == 'width rescales'
                   else '#d62728' if r['verdict'] == 'fixed width' else '0.4')
            ax.plot(i, r['real_frac_pos'], GROUP_MARKER[r['group']],
                    ms=9, mfc=col, mec='black', mew=0.7)
        ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.6)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(
            [f"{r['cell_set']}/{r['group'].replace('_BG','')}\n×{r['dur_ratio']:.1f}"
             for _, r in d.iterrows()], rotation=55, ha='right', fontsize=7)
        ax.set_ylabel('frac-positive (norm > abs)')
        ax.set_ylim(0, 1)
        ax.set_title(f'{metric}', fontsize=10)
        ax.grid(axis='y', alpha=0.25)
    legend = [
        plt.Line2D([], [], marker='o', ls='', mfc='#2ca02c', mec='black',
                   label='width rescales (p<.05)'),
        plt.Line2D([], [], marker='o', ls='', mfc='#d62728', mec='black',
                   label='fixed width (p<.05)'),
        plt.Line2D([], [], marker='o', ls='', mfc='0.4', mec='black',
                   label='flat (n.s.)'),
        plt.Line2D([], [], lw=4, color='0.6', label='shuffle-null 95% band'),
    ]
    axes[0][0].legend(handles=legend, fontsize=8, loc='upper right')
    fig.suptitle('Peak-aligned width test at maximal stretch — '
                 'cue_off anchor, Q1 vs Q4  (×N = duration ratio)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / 'peak_width_test_maxstretch.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {path}")


# ── Console digest ──────────────────────────────────────────────────────────

def print_digest(rt):
    """Highest-stretch cells (cue_off, Q1 vs Q4), xcorr — the decisive panel."""
    print()
    print("=" * 92)
    print("  Peak-aligned width test — DECISIVE cells: cue_off anchor, "
          "Q1 vs Q4, xcorr")
    print("  real frac-positive vs within-unit shuffle null  "
          "(verdict from p<0.05 two-sided)")
    print("=" * 92)
    hdr = (f"  {'cell set':<13}{'group':<10}{'×stretch':>9}{'n':>5}"
           f"{'real_fp':>9}{'null_fp':>9}{'p':>8}  verdict")
    key = rt[(rt['anchor'] == 'cue_off') & (rt['qa'] == 'Q1') &
             (rt['qb'] == 'Q4') & (rt['metric'] == 'xcorr')]
    for region, reg in key.groupby('region'):
        print(f"\n  [{region}]")
        print(hdr)
        for _, r in reg.iterrows():
            print(f"  {r['cell_set']:<13}{r['group']:<10}"
                  f"{r['dur_ratio']:>8.1f}×{r['n_units']:>5}"
                  f"{r['real_frac_pos']:>9.2f}{r['null_fp_mean']:>9.2f}"
                  f"{r['p_two_sided']:>8.3f}  {r['verdict']}")
    n_resc = int((key['verdict'] == 'width rescales').sum())
    n_fix  = int((key['verdict'] == 'fixed width').sum())
    n_flat = int((key['verdict'] == 'flat (n.s.)').sum())
    print(f"\n  At maximal stretch: {n_resc} width-rescales, {n_fix} "
          f"fixed-width, {n_flat} flat (of {len(key)} cells).")
    print()


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 92)
    print("  Peak-aligned width test — maximal-stretch rerun of the "
          "norm-vs-absolute contrast")
    print(f"  Q-pairs={Q_PAIRS}  N_SHUFFLE={N_SHUFFLE}  cropped {int(CROP_FRAC*100)}%")
    print("=" * 92)

    per_unit, cells = collect()
    rt = region_table(cells)
    print(f"\n  {len(per_unit)} unit×anchor×pair rows, {len(rt)} region cells")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_unit.to_csv(OUT_DIR / 'peak_width_test_per_unit.csv', index=False)
    rt.to_csv(OUT_DIR / 'peak_width_test_region.csv', index=False)
    print(f"  Saved per-unit → {OUT_DIR / 'peak_width_test_per_unit.csv'}")
    print(f"  Saved region   → {OUT_DIR / 'peak_width_test_region.csv'}")

    figure_stretch(rt)
    figure_maxstretch(rt)
    print_digest(rt)
    print("Done.")
