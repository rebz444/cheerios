"""
Cross-region decoder overview — three claims × three regions × three anchors.

For each of three regions (Motor cortex, Striatum MSN, Thalamus aggregate)
and each of three anchors (cue_on, cue_off, last_lick), test three claims:

  Claim 1  Region encodes elapsed time              → per-trial Pearson r
  Claim 2  Per-trial clock speed is extractable     → per-trial decoded-slope
  Claim 3  Clock speed is modulated by reward       → Cohen's d (rew vs no-rew)

Consumes the per-session ``decoder_raw_*.pkl`` files produced by
``population_decoder.py`` after the ``run_anchor_decoders`` extension adds
per-anchor ``speeds`` and ``history_stats`` sub-dicts.

Each region uses a region-appropriate MIN_UNITS threshold (the lowest
threshold where per-session decoding is still usable).

Usage:
    python cross_region_decoder.py
"""
from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from population_decoder import DIR_BASE, paths_for, ANCHOR_LABELS


# Display label → cell-set label → MIN_UNITS → region class. Column order in
# figures follows PANEL order.
PANEL = [
    # (display,          set_label,    min_units, region_class)
    ('Motor cortex',     'mc_l5l6_rs', 15,        'Cortex'),
    ('Striatum (MSN)',   'msn',         5,        'Striatum'),
    ('Thalamus',         'thal',       None,      'Thalamus'),
]

ANCHORS         = ['cue_on', 'cue_off', 'last_lick']
GROUPS          = ['Short BG', 'Long BG']
BEHAVIOR_ANCHOR = {'Short BG': 'last_lick', 'Long BG': 'cue_on'}

REGION_COLORS = {
    'Cortex':    '#D6604D',
    'Striatum':  '#2166AC',
    'Thalamus':  '#E08C24',
}
GROUP_COLORS = {
    'Short BG': '#ffb400',
    'Long BG':  '#9080ff',
}

OUT_DIR = DIR_BASE / 'cross_region'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_per_session_records():
    """Walk each cell-set's per-session pkls and return a long-format DataFrame.

    Columns: display, set_label, region_class, group, session_id, mouse,
             anchor, mae, r, r2, n_trials_anchor, n_trials_speeds,
             median_clock_speed, frac_slope_gt_0p2,
             cohens_d, p_ttest, after_reward_mean, after_no_reward_mean,
             n_units, is_behavioral_anchor
    """
    rows = []
    for display, set_label, mu, region_class in PANEL:
        _, results_dir, _ = paths_for(set_label, min_units=mu)
        if not results_dir.exists():
            print(f"  [skip] {set_label}: results dir not found at {results_dir}")
            continue
        sess_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
        for sd in sess_dirs:
            sid = sd.name
            pkl = sd / f'decoder_raw_{sid}.pkl'
            if not pkl.exists():
                continue
            with open(pkl, 'rb') as f:
                cache = pickle.load(f)
            group   = cache.get('group')
            n_units = cache.get('n_units')
            mouse   = sid.split('_')[0]
            anchor_results = cache.get('anchor_results') or {}
            for anchor in ANCHORS:
                ar = anchor_results.get(anchor)
                if ar is None:
                    continue
                speeds = ar.get('speeds') or {}
                hist   = ar.get('history_stats') or {}
                slopes = np.asarray(speeds.get('clock_speed', []))
                slopes_finite = slopes[np.isfinite(slopes)]
                med_slope = (float(np.median(slopes_finite))
                             if len(slopes_finite) else np.nan)
                frac_gt = (float(np.mean(slopes_finite > 0.2))
                           if len(slopes_finite) else np.nan)
                rows.append({
                    'display':              display,
                    'set_label':            set_label,
                    'region_class':         region_class,
                    'group':                group,
                    'session_id':           sid,
                    'mouse':                mouse,
                    'anchor':               anchor,
                    'mae':                  ar.get('mae'),
                    'r':                    ar.get('r'),
                    'r2':                   ar.get('r2'),
                    'n_trials_anchor':      ar.get('n_trials'),
                    'n_trials_speeds':      len(slopes_finite),
                    'median_clock_speed':   med_slope,
                    'frac_slope_gt_0p2':    frac_gt,
                    'cohens_d':             hist.get('cohens_d'),
                    'p_ttest':              hist.get('p_ttest'),
                    'after_reward_mean':    hist.get('after_reward_mean'),
                    'after_no_reward_mean': hist.get('after_no_reward_mean'),
                    'n_units':              n_units,
                    'is_behavioral_anchor': anchor == BEHAVIOR_ANCHOR.get(group),
                })
    return pd.DataFrame(rows)


# ── Per-trial slopes (cached as numpy arrays, fetched on demand) ──────────────

def collect_slopes_per_cell(df_records):
    """Return a dict keyed by (display, group, anchor) → concatenated array of
    per-trial clock_speed slopes (pooled across all sessions in that cell)."""
    out = {}
    for (display, set_label, mu, _) in PANEL:
        _, results_dir, _ = paths_for(set_label, min_units=mu)
        if not results_dir.exists():
            continue
        for sd in sorted(d for d in results_dir.iterdir() if d.is_dir()):
            sid = sd.name
            pkl = sd / f'decoder_raw_{sid}.pkl'
            if not pkl.exists():
                continue
            with open(pkl, 'rb') as f:
                cache = pickle.load(f)
            group = cache.get('group')
            ar    = cache.get('anchor_results') or {}
            for anchor in ANCHORS:
                ares = ar.get(anchor)
                if ares is None:
                    continue
                speeds = ares.get('speeds') or {}
                slopes = np.asarray(speeds.get('clock_speed', []))
                slopes = slopes[np.isfinite(slopes)]
                if len(slopes) == 0:
                    continue
                key = (display, group, anchor)
                out.setdefault(key, []).append(slopes)
    return {k: np.concatenate(v) for k, v in out.items()}


# ── Figure helpers ────────────────────────────────────────────────────────────

def _annotate_behavioral_border(ax, group_color='black', lw=2.5):
    for spine in ax.spines.values():
        spine.set_edgecolor(group_color)
        spine.set_linewidth(lw)


def _row_y_label(group_name, n_short, n_long, single_mouse_short, single_mouse_long):
    parts = []
    if 'Short' in group_name or group_name == 'Short BG':
        sfx = ' *1 mouse' if single_mouse_short else ''
        parts.append(f"Short BG (n={n_short}){sfx}")
    if 'Long' in group_name or group_name == 'Long BG':
        sfx = ' *1 mouse' if single_mouse_long else ''
        parts.append(f"Long BG (n={n_long}){sfx}")
    return '\n'.join(parts)


# ── Figure A: Claim 1 (encoding) — one figure per group ──────────────────────

def figure_claim1_encoding(df, group):
    """3 rows (regions) × 3 cols (anchors). Each cell: per-session Pearson r
    boxplot + per-session scatter points. ★ marks the behavioral anchor for
    this group."""
    sub = df[df['group'] == group]
    n_rows = len(PANEL)
    n_cols = len(ANCHORS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.0 * n_rows),
                             sharey=True)
    fig.suptitle(
        f'Claim 1 — Each region encodes elapsed time   [{group}]\n'
        '(per-session Pearson r between decoded and true time; '
        '★ = behavioral anchor for this group)',
        fontsize=13, fontweight='bold', y=0.995,
    )

    for i, (display, _, _, _) in enumerate(PANEL):
        for j, anchor in enumerate(ANCHORS):
            ax = axes[i, j]
            cell = sub[(sub['display'] == display) & (sub['anchor'] == anchor)]
            vals = cell['r'].dropna().values
            if len(vals) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                ax.set_xticks([])
                continue

            bp = ax.boxplot([vals], positions=[0], widths=0.6,
                            patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(GROUP_COLORS[group])
            bp['boxes'][0].set_alpha(0.65)
            bp['medians'][0].set_color('black'); bp['medians'][0].set_linewidth(1.5)
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(jitter, vals, s=28, color='black', alpha=0.55, zorder=3)

            ax.axhline(0, color='gray', ls='--', alpha=0.4, lw=1)
            ax.set_xticks([0])
            ax.set_xticklabels([f"n={len(vals)}"], fontsize=9)
            ax.set_xlim([-0.6, 0.6])
            if j == 0:
                ax.set_ylabel(f'{display}\nPearson r', fontsize=10)
            if i == 0:
                title = ANCHOR_LABELS.get(anchor, anchor)
                if BEHAVIOR_ANCHOR.get(group) == anchor:
                    title = '★ ' + title
                ax.set_title(title, fontsize=11)

            # Median annotation
            med = float(np.median(vals))
            ax.text(0.97, 0.03, f'med={med:+.2f}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = group.replace(' ', '_')
    out = OUT_DIR / f'figA_claim1_encoding__{tag}.png'
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Figure B: Claim 2 (clock speed extractable) — one figure per group ───────

def figure_claim2_clock_speed(slopes_by_cell, group):
    """3 rows × 3 cols. Per-trial clock_speed slope distribution (pooled
    across this group's sessions only). Reference lines at 0 and 1."""
    n_rows = len(PANEL)
    n_cols = len(ANCHORS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.0 * n_rows),
                             sharey=True)
    fig.suptitle(
        f'Claim 2 — Per-trial clock speed is extractable   [{group}]\n'
        '(per-trial decoded-time slope; red dashed = no signal (0), '
        'green dashed = perfect (1); ★ = behavioral anchor)',
        fontsize=13, fontweight='bold', y=0.995,
    )

    for i, (display, _, _, _) in enumerate(PANEL):
        for j, anchor in enumerate(ANCHORS):
            ax = axes[i, j]
            slopes = slopes_by_cell.get((display, group, anchor), np.array([]))
            slopes = slopes[np.isfinite(slopes)]
            if len(slopes) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                ax.set_xticks([])
                continue

            clipped = np.clip(slopes, -2, 2)
            bp = ax.boxplot([clipped], positions=[0], widths=0.5,
                            patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(GROUP_COLORS[group])
            bp['boxes'][0].set_alpha(0.65)
            bp['medians'][0].set_color('black'); bp['medians'][0].set_linewidth(1.5)

            ax.axhline(0, color='red',   ls='--', alpha=0.5, lw=1)
            ax.axhline(1, color='green', ls='--', alpha=0.5, lw=1)

            med = float(np.median(clipped))
            frac_pos = float(np.mean(slopes > 0.2))
            ax.text(0.97, 0.03,
                    f'med={med:+.2f}\nfrac>0.2={frac_pos:.0%}\nn_tr={len(slopes)}',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

            ax.set_xticks([0])
            ax.set_xticklabels([''])
            ax.set_xlim([-0.6, 0.6])
            if j == 0:
                ax.set_ylabel(f'{display}\nper-trial slope', fontsize=10)
            if i == 0:
                title = ANCHOR_LABELS.get(anchor, anchor)
                if BEHAVIOR_ANCHOR.get(group) == anchor:
                    title = '★ ' + title
                ax.set_title(title, fontsize=11)
            ax.set_ylim([-1.5, 2.0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = group.replace(' ', '_')
    out = OUT_DIR / f'figB_claim2_clock_speed__{tag}.png'
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Figure C: Claim 3 (reward modulation) — one figure per group ─────────────

def figure_claim3_history(df, group):
    """3 rows × 3 cols. Per-session Cohen's d bars within a single group.
    Reference line at d=0; ★ marks p<.001 sessions."""
    sub = df[df['group'] == group]
    n_rows = len(PANEL)
    n_cols = len(ANCHORS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.0 * n_rows),
                             sharey=True)
    fig.suptitle(
        f"Claim 3 — Reward history modulates clock speed   [{group}]\n"
        "(per-session Cohen's d: speed-after-rew vs speed-after-no-rew; "
        '★ = p<0.001)',
        fontsize=13, fontweight='bold', y=0.995,
    )

    for i, (display, _, _, _) in enumerate(PANEL):
        for j, anchor in enumerate(ANCHORS):
            ax = axes[i, j]
            cell = sub[(sub['display'] == display) & (sub['anchor'] == anchor)].copy()
            if cell.empty:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                ax.set_xticks([])
                continue

            cell = cell.sort_values('cohens_d')
            xs   = np.arange(len(cell))
            ds   = cell['cohens_d'].values
            ps   = cell['p_ttest'].values
            ax.bar(xs, ds, color=GROUP_COLORS[group],
                   edgecolor='black', linewidth=0.6)

            for x, d, p in zip(xs, ds, ps):
                if pd.notna(p) and p < 0.001:
                    yoff = 0.05 if d >= 0 else -0.20
                    ax.text(x, d + yoff, '★', ha='center', va='bottom',
                            fontsize=11, color='red')

            ax.axhline(0, color='gray', lw=1)
            ax.set_xticks(xs)
            ax.set_xticklabels(
                [s.replace('_str', '').replace('_v1', '') for s in cell['session_id']],
                rotation=70, ha='right', fontsize=7,
            )
            if j == 0:
                ax.set_ylabel(f"{display}\nCohen's d", fontsize=10)
            if i == 0:
                title = ANCHOR_LABELS.get(anchor, anchor)
                if BEHAVIOR_ANCHOR.get(group) == anchor:
                    title = '★ ' + title
                ax.set_title(title, fontsize=11)

            med_d    = float(np.nanmedian(ds))
            sig_frac = (float(np.nanmean(ps < 0.001))
                        if len(ps) and np.any(np.isfinite(ps)) else np.nan)
            ax.text(0.02, 0.98,
                    f'med d = {med_d:+.2f}\nsig (p<.001): {sig_frac:.0%}',
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = group.replace(' ', '_')
    out = OUT_DIR / f'figC_claim3_history__{tag}.png'
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


# ── CSV companion ─────────────────────────────────────────────────────────────

def write_companion_csv(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / 'cross_region_decoder_table.csv'
    cols = ['display', 'set_label', 'region_class', 'group', 'session_id',
            'mouse', 'anchor', 'is_behavioral_anchor',
            'r', 'r2', 'mae', 'n_trials_anchor', 'n_trials_speeds',
            'median_clock_speed', 'frac_slope_gt_0p2',
            'cohens_d', 'p_ttest',
            'after_reward_mean', 'after_no_reward_mean',
            'n_units']
    df[cols].sort_values(['display', 'group', 'session_id', 'anchor']) \
            .to_csv(out, index=False)
    print(f"  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 72)
    print("  Cross-region decoder overview (3 claims × 3 regions × 3 anchors)")
    print("=" * 72)

    df = load_per_session_records()
    if df.empty:
        raise FileNotFoundError(
            "No per-session pkls found — run population_decoder.py "
            "for mc_l5l6_rs / msn / thal first."
        )
    print(f"  Loaded {len(df)} (session × anchor) rows from "
          f"{df['set_label'].nunique()} cell sets.")
    for display in df['display'].unique():
        sub = df[df['display'] == display]
        nss = sub['session_id'].nunique()
        nmice = sub['mouse'].nunique()
        n_short = sub[sub['group'] == 'Short BG']['session_id'].nunique()
        n_long  = sub[sub['group'] == 'Long BG']['session_id'].nunique()
        print(f"    {display:<18}  sessions={nss:>2}  mice={nmice:>2}  "
              f"(short={n_short}, long={n_long})")

    print("\nCollecting per-trial clock-speed slopes from pkls...")
    slopes_by_cell = collect_slopes_per_cell(df)
    print(f"  Pooled slopes available for {len(slopes_by_cell)} "
          f"(region × group × anchor) cells.")

    for group in GROUPS:
        print(f"\n── {group} ──")
        print("  Figure A (Claim 1 — encoding)")
        figure_claim1_encoding(df, group)
        print("  Figure B (Claim 2 — clock speed extractable)")
        figure_claim2_clock_speed(slopes_by_cell, group)
        print("  Figure C (Claim 3 — history modulation)")
        figure_claim3_history(df, group)

    print("\nCSV companion")
    write_companion_csv(df)

    print("\nDone.")
