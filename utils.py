import os
import pickle

import numpy as np
import pandas as pd

import paths as p
from constants import FS

_WAVEFORM_JOIN_KEYS = ["mouse", "datetime", "insertion_number", "paramset_idx", "id"]


def load_waveform_metrics(df, join_how="inner"):
    """
    Load waveform templates, merge with df, and compute waveform shape metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Unit properties with columns: mouse, datetime, insertion_number, paramset_idx, id.
    join_how : str
        "inner" (default) — return only matched units; waveforms[i] == out_df.iloc[i].
        "left"            — return all rows; unmatched rows get NaN metrics.
                            Use out_df["template_idx"].notna() to identify matched rows.

    Returns
    -------
    out_df         : pd.DataFrame  — df with pt_duration_ms, trough_amp_uv, peak_amp_uv, pt_ratio
    waveforms      : np.ndarray (n_matched, T)  — raw templates for matched units
    waveforms_norm : np.ndarray (n_matched, T)  — trough-normalised waveforms
    t_ms           : np.ndarray (T,)            — time axis in milliseconds
    """
    raw       = np.load(p.LOGS_DIR / "RZ_unit_templates.npz", allow_pickle=True)
    templates = raw["templates"]
    cols      = [str(c) for c in raw["columns"]]
    uid_df = (
        pd.DataFrame(raw["unit_ids"], columns=cols)
        .rename(columns={"subject": "mouse", "session_datetime": "datetime", "unit": "id"})
    )
    uid_df["datetime"]         = pd.to_datetime(uid_df["datetime"])
    uid_df["insertion_number"] = uid_df["insertion_number"].astype(int)
    uid_df["id"]               = uid_df["id"].astype(int)
    uid_df["paramset_idx"]     = uid_df["paramset_idx"].astype(int)
    uid_df["template_idx"]     = np.arange(len(uid_df))
    print(f"  {len(uid_df):,} waveform templates loaded")

    out_df  = df.merge(uid_df[_WAVEFORM_JOIN_KEYS + ["template_idx"]],
                       on=_WAVEFORM_JOIN_KEYS, how=join_how)
    matched = out_df["template_idx"].notna()
    waveforms = templates[out_df.loc[matched, "template_idx"].astype(int).values]
    print(f"  {matched.sum():,} / {len(out_df):,} units matched to waveforms")

    trough_idx = np.argmin(waveforms, axis=1)
    peak_idx   = np.argmax(waveforms, axis=1)
    trough_amp = waveforms.min(axis=1)
    peak_amp   = waveforms.max(axis=1)
    pt_ms = np.array([
        (peak_idx[i] - trough_idx[i]) / FS * 1000
        if peak_idx[i] > trough_idx[i] else np.nan
        for i in range(len(waveforms))
    ])

    out_df["pt_duration_ms"] = np.nan
    out_df["trough_amp_uv"]  = np.nan
    out_df["peak_amp_uv"]    = np.nan
    out_df["pt_ratio"]       = np.nan
    out_df.loc[matched, "pt_duration_ms"] = pt_ms
    out_df.loc[matched, "trough_amp_uv"]  = trough_amp
    out_df.loc[matched, "peak_amp_uv"]    = peak_amp
    out_df.loc[matched, "pt_ratio"]       = np.abs(peak_amp / trough_amp)

    trough_vals    = waveforms.min(axis=1, keepdims=True)
    waveforms_norm = waveforms / np.abs(trough_vals)
    t_ms           = (np.arange(waveforms.shape[1]) / FS) * 1000

    return out_df, waveforms, waveforms_norm, t_ms

def get_session_data(session_id):
    pickle_name = session_id + ".pkl"
    pickle_path = os.path.join(p.PICKLE_DIR, pickle_name)
    with open(pickle_path, 'rb') as f:
        session_data = pickle.load(f)
    
    events = session_data['events']
    trials = session_data['trials']
    units = session_data['units']
    return events, trials, units

def get_data_for_debugging(units_vetted, session_id='RZ051_2024-11-19_str', unit_id=20):
    units_by_session = units_vetted.groupby("session_id")
    test_session = units_by_session.get_group(session_id)
    events, trials, units = get_session_data(session_id)
    spikes = units[unit_id]
    return events, trials, spikes