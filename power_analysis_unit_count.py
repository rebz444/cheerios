"""
power_analysis_unit_count.py
────────────────────────────
Subsample MSN units from selected sessions and measure how Ridge decoder
performance (R², MAE) scales with unit count. Used to decide whether
MIN_UNITS=15 is too strict for true MSN populations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

import paths as p
import population_decoder as pd_mod

# (session_id, group, t_ref column in trials_df, plot color)
SESSIONS = [
    ('RZ065_2025-02-22_str', 'Long BG',  'cue_on',         '#9080ff'),
    ('RZ063_2025-03-07_str', 'Short BG', 'last_lick_time', '#ffb400'),
]

UNIT_COUNTS    = [2, 4, 6, 8, 10, 12, 14, 16]
N_SUBSAMPLES   = 20                        # random unit draws per N
TRAIN_SEED     = 42                        # fixed train/test split
TEST_FRAC      = 0.2

OUT_DIR = p.DATA_DIR / 'population_decoding' / 'msn'
OUT_PNG = OUT_DIR / 'power_analysis_unit_count.png'
OUT_CSV = OUT_DIR / 'power_analysis_unit_count.csv'


def pooled_decode_subset(X_all, y_all, unit_idx, seed=TRAIN_SEED):
    """Pooled Ridge with a column subset of the population matrix."""
    X_sub = [X[:, unit_idx] for X in X_all]
    n     = len(X_sub)
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(n)
    n_test    = max(1, int(n * TEST_FRAC))
    train_idx = perm[n_test:]
    test_idx  = perm[:n_test]

    X_train = np.vstack([X_sub[i] for i in train_idx])
    y_train = np.concatenate([y_all[i] for i in train_idx])
    model   = RidgeCV(alphas=pd_mod.RIDGE_ALPHA, fit_intercept=True)
    model.fit(X_train, y_train)

    mae_per_trial, r2_per_trial = [], []
    for i in range(n):
        y_hat  = model.predict(X_sub[i])
        y_test = y_all[i]
        err    = y_hat - y_test
        mae_per_trial.append(np.mean(np.abs(err)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2_per_trial.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)

    # All-trial means — matches what session_summary.csv reports
    # (line 1742: pooled_r2 = nanmean(r2_per_trial))
    return float(np.nanmean(r2_per_trial)), float(np.mean(mae_per_trial))


def run_session(session_id, t_ref_col, trials_df, spikes_df):
    t_sess = trials_df[trials_df['session'] == session_id].copy()
    s_sess = spikes_df[spikes_df['session'] == session_id].copy()
    n_units_total = s_sess['unit_id'].nunique()
    print(f"\n{session_id}: {n_units_total} MSN units, {len(t_sess)} trials")

    data = pd_mod.build_population_matrix(t_sess, s_sess, t_ref_col)
    X_all, y_all  = data['X'], data['y']
    n_units_used  = data['n_units']
    n_trials_used = data['n_trials']
    print(f"  matrix: {n_trials_used} trials × {n_units_used} units")

    rng = np.random.default_rng(0)
    rows = []
    for N in UNIT_COUNTS:
        if N > n_units_used:
            continue
        r2s, maes = [], []
        for _ in range(N_SUBSAMPLES):
            unit_idx = rng.choice(n_units_used, size=N, replace=False)
            r2, mae  = pooled_decode_subset(X_all, y_all, unit_idx)
            r2s.append(r2)
            maes.append(mae)
        rows.append({
            'session' : session_id,
            'n_units' : N,
            'r2_mean' : float(np.mean(r2s)),
            'r2_std'  : float(np.std(r2s)),
            'r2_min'  : float(np.min(r2s)),
            'r2_max'  : float(np.max(r2s)),
            'mae_mean': float(np.mean(maes)),
            'mae_std' : float(np.std(maes)),
        })
        print(f"  N={N:2d} | R²={np.mean(r2s):+.3f} ± {np.std(r2s):.3f}  "
              f"(min {np.min(r2s):+.3f}, max {np.max(r2s):+.3f})  "
              f"MAE={np.mean(maes):.3f} ± {np.std(maes):.3f}")
    return rows


def main():
    print("Loading all sessions (is_str_msn ∩ qc_pass_all)…")
    trials_df, spikes_df, _ = pd_mod.load_decoder_data(
        str_units_csv=p.LOGS_DIR / 'unit_properties_final.csv',
        filter_col='is_str_msn',
    )

    all_rows = []
    for session_id, _group, t_ref_col, _color in SESSIONS:
        all_rows.extend(run_session(session_id, t_ref_col, trials_df, spikes_df))

    summary = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for session_id, group, _t_ref, color in SESSIONS:
        sdf = summary[summary['session'] == session_id]
        if sdf.empty:
            continue
        label = f"{session_id.replace('_str', '')} ({group})"

        axes[0].errorbar(sdf['n_units'], sdf['r2_mean'], yerr=sdf['r2_std'],
                         marker='o', color=color, capsize=3, lw=1.5, label=label)
        axes[0].fill_between(sdf['n_units'], sdf['r2_min'], sdf['r2_max'],
                             alpha=0.15, color=color)

        axes[1].errorbar(sdf['n_units'], sdf['mae_mean'], yerr=sdf['mae_std'],
                         marker='o', color=color, capsize=3, lw=1.5, label=label)

    axes[0].axhline(0, ls=':', color='gray', lw=0.8)
    axes[0].set_xlabel('# MSN units')
    axes[0].set_ylabel('R² (mean over all trials)')
    axes[0].set_title('R² vs unit count')
    axes[0].set_xticks(UNIT_COUNTS)
    axes[0].legend(fontsize=8, loc='lower right')

    axes[1].set_xlabel('# MSN units')
    axes[1].set_ylabel('MAE (s)')
    axes[1].set_title('MAE vs unit count')
    axes[1].set_xticks(UNIT_COUNTS)
    axes[1].legend(fontsize=8, loc='upper right')

    fig.suptitle(f'Power analysis — {N_SUBSAMPLES} random unit draws per N, '
                 f'pooled Ridge (seed={TRAIN_SEED})', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
    print(f"Saved {OUT_PNG}")


if __name__ == '__main__':
    main()
