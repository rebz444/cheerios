"""
Full-PSTH similarity — Q2 vs Q3, the population-level rescaling test.

Background
----------
d_tau_stratification.py and e_crop_resort_rescaling.py compared Q2→Q3 *peak
positions*. This script runs step 3 of the Open Questions.md resolution path:
compare the *whole* Q2 and Q3 PSTH of each unit, not just its peak.

Two similarity metrics per unit (Q2 PSTH vs Q3 PSTH):

    xcorr    zero-lag normalized cross-correlation — Pearson r of the two
             PSTH vectors. Higher = more similar.
    procsim  1 - Procrustes disparity. Each PSTH is treated as a 2D curve of
             (time, rate) points; scipy.spatial.procrustes standardizes both
             and finds the optimal translation+scale+rotation, returning a
             disparity in [0, 1]. procsim = 1 - disparity, so higher = more
             similar, like xcorr. A genuine curve-shape metric, distinct from
             the cross-correlation.

Run as a 2x2 of alignment x cropping:

  Alignment
    anchor-aligned  PSTHs kept on the [anchor -> decision] axis. In normalized
                    time a *fixed-latency* unit's Q2/Q3 PSTHs already diverge
                    (the fixed event lands at a smaller normalized fraction in
                    the longer Q3), so a high Q2<->Q3 similarity = genuine
                    rescaling. THIS is the population rescaling test. Uncropped
                    it is inflated by edge transients (a cue-locked peak sits
                    at normalized position ~0 in both quartiles); cropped is
                    the key panel.
    peak-aligned    Each PSTH is shifted to put its peak at a common reference,
                    removing peak position so similarity reflects only the
                    response shape/width. Peak-alignment is not itself a
                    rescaling test — the informative readout is the
                    normalized-time vs absolute-time match: if the response
                    WIDTH rescales, the normalized-time peak-aligned profiles
                    match better than the absolute-time ones. Reported as the
                    per-unit contrast  procsim/xcorr(normalized) - (absolute).

  Cropping
    uncropped       full PSTH.
    cropped         first/last CROP_FRAC of the normalized interval removed
                    (CROP_FRAC=0.15, matching crop_resort_rescaling's headline).
                    For peak-aligned the peak is re-found inside the crop.

No cache regenerate is needed: every results_cache.pkl stores both q_spikes
(absolute) and q_norm_spikes (normalized) per-trial spike times.

Outputs (data/rescaling/psth_similarity/)
-----------------------------------------
    psth_similarity_per_unit.csv   one row per unit x anchor — every panel
                                   metric (the per-unit distributions).
    psth_similarity_summary.csv    per cell_set x group x anchor — median+IQR.
    psth_similarity_{xcorr,procrustes}_{cue_on,cue_off,last_lick}.png
                                   per-anchor 2x2 panels of per-region violins
                                   (short and long BG at the same anchor).

Usage
-----
    python rescaling/f_psth_similarity.py
"""
from __future__ import annotations

import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import procrustes

from a_rescaling import (
    DIR_BASE, paths_for, compute_psth,
    N_BINS_NORM, SIGMA, MIN_TRIALS, TIME_STEP,
    T_MAX_SHORT, T_MAX_LONG, QUARTILE_PAIR,
)
from b_cross_region_rescaling import PANEL, GROUPS, ANCHORS, REGION_COLORS


# ── Config ──────────────────────────────────────────────────────────────────

# Quartile pair compared (project-wide default from rescaling).
QA, QB = QUARTILE_PAIR
ANCHOR_ORDER = ['cue_on', 'cue_off', 'last_lick']

CROP_FRAC = 0.15                 # fraction cropped per end (normalized time)
TS_NORM   = 1.0 / N_BINS_NORM    # 0.01 — normalized-time PSTH bin width
T_MAX_ABS = {'short_BG': T_MAX_SHORT, 'long_BG': T_MAX_LONG}

# Peak-aligned overlap must be at least this fraction of the PSTH length,
# else the comparison is dropped (peaks too far apart).
MIN_OVERLAP_FRAC = 0.40
# Minimum segment length for a similarity metric to be computed.
MIN_SEG = 10

OUT_DIR = DIR_BASE / 'psth_similarity'

# Panel metric columns, in CSV order.
METRIC_COLS = [
    'anchor_uncrop_xcorr',       'anchor_uncrop_procsim',
    'anchor_crop_xcorr',         'anchor_crop_procsim',
    'peak_uncrop_norm_xcorr',    'peak_uncrop_norm_procsim',
    'peak_uncrop_abs_xcorr',     'peak_uncrop_abs_procsim',
    'peak_uncrop_xcorr_contrast', 'peak_uncrop_procsim_contrast',
    'peak_crop_norm_xcorr',      'peak_crop_norm_procsim',
    'peak_crop_abs_xcorr',       'peak_crop_abs_procsim',
    'peak_crop_xcorr_contrast',  'peak_crop_procsim_contrast',
]


# ── Similarity metrics ──────────────────────────────────────────────────────

def xcorr(a, b):
    """Zero-lag normalized cross-correlation (Pearson r) of two PSTH vectors."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < MIN_SEG or len(a) != len(b) or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def procsim(a, b):
    """1 - Procrustes disparity between two PSTHs as (time, rate) 2D curves.

    Both the time and the rate column are standardized to unit variance before
    stacking, so the 2D curve is non-degenerate and the procrustes rotation is
    a genuine shape comparison rather than collapsing onto the rate axis.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < MIN_SEG or len(a) != len(b) or a.std() == 0 or b.std() == 0:
        return np.nan
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    t = np.arange(len(a), dtype=float)
    t = (t - t.mean()) / t.std()
    m1 = np.column_stack([t, a])
    m2 = np.column_stack([t, b])
    try:
        _, _, disparity = procrustes(m1, m2)
    except ValueError:
        return np.nan
    return float(1.0 - disparity)


def peak_align(psth_a, psth_b):
    """Shift psth_b so its peak sits under psth_a's peak; return the
    overlapping segments (equal length), or None if the overlap is too small."""
    n = len(psth_a)
    if n != len(psth_b):
        return None
    offset = int(np.argmax(psth_a)) - int(np.argmax(psth_b))
    lo = max(0, offset)
    hi = min(n, n + offset)
    if hi - lo < MIN_OVERLAP_FRAC * n:
        return None
    return psth_a[lo:hi], psth_b[lo - offset:hi - offset]


# ── PSTH construction ───────────────────────────────────────────────────────

def make_psth(spike_lists, t_min, t_max, step):
    """z-scored PSTH over [t_min, t_max], or None (too few trials / flat)."""
    if len(spike_lists) < MIN_TRIALS:
        return None
    _, norm = compute_psth(spike_lists, t_min, t_max, step, SIGMA)
    return norm


def crop_abs_by_norm(abs_list, norm_list, crop):
    """Keep, per trial, only the absolute-time spikes whose normalized position
    falls inside [crop, 1-crop]. abs_list and norm_list are parallel (same
    spikes, two coordinate systems)."""
    out = []
    for abs_spk, norm_spk in zip(abs_list, norm_list):
        abs_spk = np.asarray(abs_spk, float)
        norm_spk = np.asarray(norm_spk, float)
        keep = (norm_spk >= crop) & (norm_spk <= 1.0 - crop)
        out.append(abs_spk[keep])
    return out


# ── Per-unit metrics ────────────────────────────────────────────────────────

def unit_metrics(abs_q3, norm_q3, abs_q4, norm_q4, t_max_abs):
    """All 16 panel metrics for one unit."""
    out = {c: np.nan for c in METRIC_COLS}

    # Normalized-time PSTHs (anchor on the [0, 1] axis).
    nq3 = make_psth(norm_q3, 0.0, 1.0, TS_NORM)
    nq4 = make_psth(norm_q4, 0.0, 1.0, TS_NORM)
    nq3_c = make_psth(norm_q3, CROP_FRAC, 1.0 - CROP_FRAC, TS_NORM)
    nq4_c = make_psth(norm_q4, CROP_FRAC, 1.0 - CROP_FRAC, TS_NORM)

    # Absolute-time PSTHs (anchor on the [0, t_max] seconds axis).
    aq3 = make_psth(abs_q3, 0.0, t_max_abs, TIME_STEP)
    aq4 = make_psth(abs_q4, 0.0, t_max_abs, TIME_STEP)
    aq3_c = make_psth(crop_abs_by_norm(abs_q3, norm_q3, CROP_FRAC),
                      0.0, t_max_abs, TIME_STEP)
    aq4_c = make_psth(crop_abs_by_norm(abs_q4, norm_q4, CROP_FRAC),
                      0.0, t_max_abs, TIME_STEP)

    # ── Anchor-aligned: normalized-time Q2<->Q3 similarity (rescaling test) ──
    if nq3 is not None and nq4 is not None:
        out['anchor_uncrop_xcorr']   = xcorr(nq3, nq4)
        out['anchor_uncrop_procsim'] = procsim(nq3, nq4)
    if nq3_c is not None and nq4_c is not None:
        out['anchor_crop_xcorr']   = xcorr(nq3_c, nq4_c)
        out['anchor_crop_procsim'] = procsim(nq3_c, nq4_c)

    # ── Peak-aligned: normalized vs absolute profile match ──────────────────
    for tag, q3n, q4n, q3a, q4a in [
        ('uncrop', nq3,   nq4,   aq3,   aq4),
        ('crop',   nq3_c, nq4_c, aq3_c, aq4_c),
    ]:
        if q3n is not None and q4n is not None:
            seg = peak_align(q3n, q4n)
            if seg is not None:
                out[f'peak_{tag}_norm_xcorr']   = xcorr(*seg)
                out[f'peak_{tag}_norm_procsim'] = procsim(*seg)
        if q3a is not None and q4a is not None:
            seg = peak_align(q3a, q4a)
            if seg is not None:
                out[f'peak_{tag}_abs_xcorr']   = xcorr(*seg)
                out[f'peak_{tag}_abs_procsim'] = procsim(*seg)
        for met in ('xcorr', 'procsim'):
            nv = out[f'peak_{tag}_norm_{met}']
            av = out[f'peak_{tag}_abs_{met}']
            if not (np.isnan(nv) or np.isnan(av)):
                out[f'peak_{tag}_{met}_contrast'] = nv - av
    return out


# ── Collect ─────────────────────────────────────────────────────────────────

def collect():
    """Walk the PANEL caches; return the per-unit table (all anchors)."""
    rows = []
    for display, label, region in PANEL:
        _, _, _, cache_file = paths_for(label)
        if not cache_file.exists():
            print(f"  [skip] {label}: no cache at {cache_file}")
            continue
        with open(cache_file, 'rb') as f:
            data_cache = pickle.load(f)['data_cache']

        for group in GROUPS:
            t_max_abs = T_MAX_ABS[group]
            for anchor in ANCHORS:
                key = (group, f'{anchor}_time')
                if key not in data_cache:
                    continue
                entry     = data_cache[key]
                sort_uids = entry['metrics'].get('sort_uids', [])
                q_abs     = entry['q_spikes']
                q_norm    = entry['q_norm_spikes']
                for uid in sort_uids:
                    m = unit_metrics(
                        q_abs[QA].get(uid, []),  q_norm[QA].get(uid, []),
                        q_abs[QB].get(uid, []),  q_norm[QB].get(uid, []),
                        t_max_abs,
                    )
                    rows.append({
                        'display': display, 'cell_set': label, 'region': region,
                        'group': group, 'anchor': anchor, 'unit_gid': uid, **m,
                    })
        print(f"  done {label}")

    return pd.DataFrame(rows)


def summarize(df):
    """Per cell_set x group x anchor: n_units + median/IQR of each metric."""
    rows = []
    keys = ['display', 'cell_set', 'region', 'group', 'anchor']
    for key_vals, sub in df.groupby(keys):
        row = dict(zip(keys, key_vals))
        row['n_units'] = len(sub)
        for col in METRIC_COLS:
            v = sub[col].dropna()
            row[f'{col}_median'] = v.median() if len(v) else np.nan
            row[f'{col}_iqr'] = (v.quantile(0.75) - v.quantile(0.25)
                                 if len(v) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


# ── Figures ─────────────────────────────────────────────────────────────────

# short_BG (solid) and long_BG (hatched) are drawn as paired violins per
# cell set — they sit at different behavioral anchors and must not be pooled.
GROUP_STYLE = {
    'short_BG': dict(offset=-0.21, hatch='',    alpha=0.72),
    'long_BG':  dict(offset=+0.21, hatch='///', alpha=0.52),
}


def _violin_panel(ax, df, col, displays, title, ref):
    """Per-region, per-group paired violins of a per-unit metric column."""
    counts = {}
    for group, st in GROUP_STYLE.items():
        data, positions, colors = [], [], []
        for i, disp in enumerate(displays):
            vals = df[(df['display'] == disp) &
                      (df['group'] == group)][col].dropna().values
            counts[(disp, group)] = len(vals)
            if len(vals) >= 3:
                region = next(r for d, _, r in PANEL if d == disp)
                data.append(vals)
                positions.append(i + st['offset'])
                colors.append(REGION_COLORS.get(region, 'gray'))
        if not data:
            continue
        vp = ax.violinplot(data, positions=positions, widths=0.36,
                           showmedians=True, showextrema=False)
        for body, c in zip(vp['bodies'], colors):
            body.set_facecolor(c)
            body.set_alpha(st['alpha'])
            body.set_edgecolor('black')
            body.set_linewidth(0.6)
            if st['hatch']:
                body.set_hatch(st['hatch'])
        if 'cmedians' in vp:
            vp['cmedians'].set_color('black')
            vp['cmedians'].set_linewidth(1.3)
    ax.axhline(ref, color='red', ls='--', lw=1.0, alpha=0.6)
    ax.set_xticks(range(len(displays)))
    ax.set_xticklabels(displays, rotation=35, ha='right', fontsize=7)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlim(-0.7, len(displays) - 0.3)
    y0 = ax.get_ylim()[0]
    for i, disp in enumerate(displays):
        ax.text(i, y0, f"{counts.get((disp, 'short_BG'), 0)}/"
                       f"{counts.get((disp, 'long_BG'), 0)}",
                ha='center', va='bottom', fontsize=5, color='0.45')


def figure_2x2(df, metric, anchor):
    """2x2 alignment x cropping figure for one metric at one anchor —
    short and long BG compared at the SAME anchor (no cross-group mixing).

    Anchor rows show the raw normalized-time Q2<->Q3 similarity (high =
    rescales). Peak rows show the normalized - absolute contrast (>0 = the
    response width rescales)."""
    sub = df[df['anchor'] == anchor]
    displays = [d for d, _, _ in PANEL if d in sub['display'].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(7.6 * 2, 5.4 * 2), squeeze=False,
                             constrained_layout=True)
    panels = [
        (0, 0, f'anchor_uncrop_{metric}', 0.0,
         'Anchor-aligned · uncropped — Q2↔Q3 normalized-time similarity'),
        (0, 1, f'anchor_crop_{metric}', 0.0,
         f'Anchor-aligned · cropped {int(CROP_FRAC*100)}% — '
         'Q2↔Q3 normalized-time similarity'),
        (1, 0, f'peak_uncrop_{metric}_contrast', 0.0,
         'Peak-aligned · uncropped — normalized − absolute similarity'),
        (1, 1, f'peak_crop_{metric}_contrast', 0.0,
         f'Peak-aligned · cropped {int(CROP_FRAC*100)}% — '
         'normalized − absolute similarity'),
    ]
    for r, c, col, ref, title in panels:
        _violin_panel(axes[r][c], sub, col, displays, title, ref)
        ylab = ('Q2↔Q3 similarity' if r == 0
                else 'norm − abs similarity (>0 rescales)')
        axes[r][c].set_ylabel(f'{ylab}\n[{metric}]', fontsize=8)

    legend_handles = [
        mpatches.Patch(facecolor='0.6', edgecolor='black', alpha=0.72,
                       label='short BG'),
        mpatches.Patch(facecolor='0.6', edgecolor='black', alpha=0.52,
                       hatch='///', label='long BG'),
    ]
    axes[0][1].legend(handles=legend_handles, fontsize=8, loc='lower left')

    metric_name = ('zero-lag cross-correlation' if metric == 'xcorr'
                   else '1 − Procrustes disparity')
    fig.suptitle(
        f'Full-PSTH similarity — Q2 vs Q3  ({metric_name})  ·  anchor: {anchor}\n'
        'short and long BG at the SAME anchor · per-unit distribution per '
        'region · n shown at base of each violin',
        fontsize=12, fontweight='bold')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mname = 'xcorr' if metric == 'xcorr' else 'procrustes'
    out_path = OUT_DIR / f'psth_similarity_{mname}_{anchor}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


CROP_DROP_MARK = {'short_BG': 'o', 'long_BG': 's'}
CROP_DROP_OFF  = {'short_BG': -0.18, 'long_BG': +0.18}

# Short interpretive tag per anchor for multi-anchor panel titles.
ANCHOR_TAG = {
    'cue_off':   'clean wait window — the headline',
    'cue_on':    'includes background period (inflated)',
    'last_lick': 'motor-dominated window (inflated)',
}


def _median_ci(vals, rng, nboot=1000):
    """Median of vals with a bootstrap 95% CI. Returns (median, lo, hi)."""
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(vals))
    if len(vals) < 3:
        return med, med, med
    boot = np.array([np.median(rng.choice(vals, len(vals), replace=True))
                     for _ in range(nboot)])
    return med, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _crop_drop_records(sub, displays, metric, rng):
    """Per cell-set × group medians (uncropped, cropped + CI) for one anchor.

    ``displays`` is the shared (display, region) ordering so x positions are
    consistent across anchors; a cell set absent at this anchor is just skipped.
    """
    unc_col = f'anchor_uncrop_{metric}'
    crp_col = f'anchor_crop_{metric}'
    recs = []
    for i, (disp, region) in enumerate(displays):
        for group in GROUPS:
            g = sub[(sub['display'] == disp) & (sub['group'] == group)]
            if g.empty:
                continue
            y_unc, _, _   = _median_ci(g[unc_col].values, rng)
            y_crp, lo, hi = _median_ci(g[crp_col].values, rng)
            if np.isnan(y_unc) or np.isnan(y_crp):
                continue
            recs.append(dict(i=i, region=region, group=group, n=len(g),
                             y_unc=y_unc, y_crp=y_crp, lo=lo, hi=hi))
    return recs


def _draw_crop_drop_panel(ax, recs, displays, label_y, *,
                          region_labels=True, xtick_labels=True):
    """Render one dumbbell panel onto ``ax``. Open marker = uncropped, filled =
    cropped 15%; the connector is the drop; whisker = bootstrap CI of the
    cropped median; the dashed line at 0 is the persistence reference."""
    # Alternating region bands (+ optional per-block region label).
    blocks, start = [], 0
    for i in range(1, len(displays) + 1):
        if i == len(displays) or displays[i][1] != displays[start][1]:
            blocks.append((start, i - 1, displays[start][1]))
            start = i
    for k, (lo_i, hi_i, region) in enumerate(blocks):
        if k % 2 == 1:
            ax.axvspan(lo_i - 0.5, hi_i + 0.5, color='0.95', zorder=0)
        if region_labels:
            ax.text((lo_i + hi_i) / 2, label_y, region, ha='center',
                    va='bottom', fontsize=9, fontweight='bold',
                    color=REGION_COLORS.get(region, 'black'))

    for r in recs:
        color = REGION_COLORS.get(r['region'], 'gray')
        x = r['i'] + CROP_DROP_OFF[r['group']]
        mark = CROP_DROP_MARK[r['group']]
        ax.plot([x, x], [r['y_unc'], r['y_crp']], color=color, lw=1.6,
                alpha=0.75, zorder=2)                          # drop connector
        ax.plot([x, x], [r['lo'], r['hi']], color=color, lw=3.0, alpha=0.30,
                zorder=1)                                      # CI on cropped
        ax.plot(x, r['y_unc'], marker=mark, mfc='white', mec=color, mew=1.6,
                ms=8, zorder=3)                                # uncropped (open)
        ax.plot(x, r['y_crp'], marker=mark, mfc=color, mec=color, ms=8,
                zorder=3)                                      # cropped (filled)

    ax.axhline(0, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.set_xticks(range(len(displays)))
    if xtick_labels:
        ax.set_xticklabels([d for d, _ in displays], rotation=35, ha='right',
                           fontsize=8)
        for tick, (_, region) in zip(ax.get_xticklabels(), displays):
            tick.set_color(REGION_COLORS.get(region, 'black'))
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.7, len(displays) - 0.3)


def _crop_drop_legend_handles():
    return [
        plt.Line2D([], [], marker='o', mfc='white', mec='0.3', mew=1.6, ls='',
                   ms=8, label='uncropped'),
        plt.Line2D([], [], marker='o', mfc='0.3', mec='0.3', ls='', ms=8,
                   label='cropped 15%'),
        plt.Line2D([], [], marker='o', mfc='0.5', mec='0.5', ls='', ms=8,
                   label='short BG'),
        plt.Line2D([], [], marker='s', mfc='0.5', mec='0.5', ls='', ms=8,
                   label='long BG'),
    ]


def figure_crop_drop(df, metric, anchor):
    """Single-anchor dumbbell summary of anchor-aligned Q2↔Q3 similarity:
    median uncropped (open) connected to median cropped-15% (filled), showing
    the drop while staying above 0. See _draw_crop_drop_panel for encoding."""
    sub = df[df['anchor'] == anchor]
    displays = [(d, r) for d, _, r in PANEL if d in sub['display'].unique()]
    rng = np.random.default_rng(0)
    recs = _crop_drop_records(sub, displays, metric, rng)

    yvals = [v for r in recs for v in (r['y_unc'], r['y_crp'], r['lo'], r['hi'])]
    dmin, dmax = min(yvals + [0.0]), max(yvals + [0.0])
    span = dmax - dmin

    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    _draw_crop_drop_panel(ax, recs, displays, dmax + 0.10 * span)
    ax.set_ylim(dmin - 0.05 * span, dmax + 0.18 * span)
    ax.set_ylabel(f'median Q2↔Q3 similarity  [{metric}]', fontsize=10)
    ax.legend(handles=_crop_drop_legend_handles(), fontsize=8,
              loc='lower left', ncol=2, framealpha=0.9)

    metric_name = ('zero-lag cross-correlation' if metric == 'xcorr'
                   else '1 − Procrustes disparity')
    fig.suptitle(
        f'Full-PSTH similarity — Q2 vs Q3 ({metric_name})  ·  anchor: {anchor}\n'
        'edge cropping lowers per-unit similarity in every region, but it stays '
        'above 0 — similarity persists  (whisker = bootstrap 95% CI of median)',
        fontsize=12, fontweight='bold')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mname = 'xcorr' if metric == 'xcorr' else 'procrustes'
    out_path = OUT_DIR / f'psth_crop_drop_{mname}_{anchor}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


def figure_crop_drop_anchors(df, metric, anchors=('cue_off', 'cue_on',
                                                  'last_lick')):
    """All three anchors stacked in one figure for a single metric — same
    dumbbell encoding per panel, shared x and y so anchors are comparable.
    cue_off (the clean window) is on top; the other anchors are inflated by
    in-window motor/background activity and are shown for context."""
    displays = [(d, r) for d, _, r in PANEL if d in df['display'].unique()]
    rng = np.random.default_rng(0)

    recs_by_anchor = {a: _crop_drop_records(df[df['anchor'] == a], displays,
                                            metric, rng) for a in anchors}
    yvals = [v for recs in recs_by_anchor.values() for r in recs
             for v in (r['y_unc'], r['y_crp'], r['lo'], r['hi'])]
    dmin, dmax = min(yvals + [0.0]), max(yvals + [0.0])
    span = dmax - dmin
    label_y = dmax + 0.06 * span

    fig, axes = plt.subplots(len(anchors), 1, figsize=(15, 4.2 * len(anchors)),
                             sharex=True, sharey=True, constrained_layout=True)
    for ai, anchor in enumerate(anchors):
        ax = axes[ai]
        _draw_crop_drop_panel(ax, recs_by_anchor[anchor], displays, label_y,
                              region_labels=(ai == 0),
                              xtick_labels=(ai == len(anchors) - 1))
        ax.set_ylim(dmin - 0.05 * span, dmax + 0.12 * span)
        ax.set_ylabel(f'median similarity\n[{metric}]', fontsize=9)
        tag = ANCHOR_TAG.get(anchor, '')
        ax.set_title(f'anchor: {anchor}   —   {tag}', loc='left', fontsize=10,
                     fontweight='bold')
        if ai == 0:
            ax.legend(handles=_crop_drop_legend_handles(), fontsize=8,
                      loc='lower left', ncol=2, framealpha=0.9)

    metric_name = ('zero-lag cross-correlation' if metric == 'xcorr'
                   else '1 − Procrustes disparity')
    fig.suptitle(
        f'Full-PSTH similarity — Q2 vs Q3 ({metric_name})  ·  all anchors\n'
        'edge cropping lowers per-unit similarity in every region, but it stays '
        'above 0 — similarity persists  (whisker = bootstrap 95% CI of median)',
        fontsize=12, fontweight='bold')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mname = 'xcorr' if metric == 'xcorr' else 'procrustes'
    out_path = OUT_DIR / f'psth_crop_drop_{mname}_all_anchors.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


# ── Console digest ──────────────────────────────────────────────────────────

def print_digest(df):
    """Anchor-aligned uncropped vs cropped (the key panel) — all three
    anchors, both groups reported at each anchor."""
    print()
    print("=" * 86)
    print("  Full-PSTH similarity digest — anchor-aligned Q2↔Q3, all anchors")
    print("  median normalized-time similarity; cropped is the key rescaling "
          "panel")
    print("=" * 86)
    hdr = (f"  {'cell set':<13}{'group':<10}{'n':>6}"
           f"{'xcorr_unc':>11}{'xcorr_crop':>12}"
           f"{'proc_unc':>11}{'proc_crop':>11}")
    for anchor in ANCHOR_ORDER:
        sub = df[df['anchor'] == anchor]
        print(f"\n--- {anchor} ---")
        print(hdr)
        for (region, disp, group), g in sub.groupby(
                ['region', 'display', 'group']):
            def med(c):
                v = g[c].dropna()
                return ' nan' if not len(v) else f'{v.median():.2f}'
            print(f"  {g['cell_set'].iloc[0]:<13}{group:<10}{len(g):>6}"
                  f"{med('anchor_uncrop_xcorr'):>11}"
                  f"{med('anchor_crop_xcorr'):>12}"
                  f"{med('anchor_uncrop_procsim'):>11}"
                  f"{med('anchor_crop_procsim'):>11}")
    print()


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 88)
    print("  Full-PSTH similarity — Q2 vs Q3 (Open Questions rescaling step 3)")
    print("=" * 88)

    df = collect()
    print(f"\n  Collected {df['cell_set'].nunique()} cell sets, "
          f"{len(df)} unit × anchor rows")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_unit_path = OUT_DIR / 'psth_similarity_per_unit.csv'
    summary_path  = OUT_DIR / 'psth_similarity_summary.csv'
    df.to_csv(per_unit_path, index=False)
    summarize(df).to_csv(summary_path, index=False)
    print(f"  Saved per-unit → {per_unit_path}")
    print(f"  Saved summary  → {summary_path}")

    for metric in ('xcorr', 'procsim'):
        for anchor in ANCHOR_ORDER:
            figure_2x2(df, metric, anchor)
            figure_crop_drop(df, metric, anchor)
        figure_crop_drop_anchors(df, metric)
    print_digest(df)
    print("Done.")
