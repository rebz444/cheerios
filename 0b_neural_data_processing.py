"""
0b_neural_data_processing.py
=============================
Loads the curated spike-sorting pickle, merges with the recording log, and
processes events / trials / spikes per session. Saves one pickle per session
to p.PICKLE_DIR plus a session log CSV.

Inputs:
  p.DATA_DIR / 'neural_data_0519.pkl'   - curated sessions list (from spike sorting)
  Google Sheet recording log            - session metadata (region, etc.)

Outputs:
  p.PICKLE_DIR / '{session_id}.pkl'         - per-session processed data
  p.LOGS_DIR / 'region_cell_count.csv'      - cells per mouse x region
  p.LOGS_DIR / 'sessions_official_raw.csv'  - master session log (no events/units)

Set REGENERATE = True to overwrite existing per-session pickles.
"""

import os
import pickle

import numpy as np
import pandas as pd

import constants as k
import paths as p


RAW_PICKLE_NAME = "neural_data_0519.pkl"
REGENERATE = False

_RECORDING_LOG_SHEET_ID = "1xpb52EO7BII-_5zIcB2HaOpZqepFcYTvBYgxA4zLyNo"
RECORDING_LOG_URL = (
    f"https://docs.google.com/spreadsheets/d/{_RECORDING_LOG_SHEET_ID}"
    "/export?format=csv&gid=0"
)


# ── 1. Load curated sessions ──────────────────────────────────────────────────

def generate_sessions_sorted(sorted_sessions_list):
    rows = []
    for session in sorted_sessions_list:
        rows.append({
            "mouse": session["subject"],
            "datetime": session["session_datetime"],
            "date": session["session_datetime"].strftime("%Y-%m-%d"),
            "insertion_number": session["insertion_number"],
            "paramset_idx": session["paramset_idx"],
            "num_units": len(session["unit_spikes"]),
            "events": session["events"],
            "unit_ids": session["unit_ids"],
            "units": session["unit_spikes"],
        })
    return pd.DataFrame(rows)


# ── 2. Events processing ──────────────────────────────────────────────────────

def process_raw_events(events_df):
    trial_starts = events_df.loc[events_df["event_type"] == "trial"]
    trial_starts = trial_starts.set_index("trial_id")["event_start_time"]
    events_df["trial_start_time"] = events_df["trial_id"].map(trial_starts)
    events_df["event_start_trial_time"] = (
        events_df["event_start_time"] - events_df["trial_start_time"]
    )
    events_df["event_end_trial_time"] = (
        events_df["event_end_time"] - events_df["trial_start_time"]
    )
    events_df = events_df.drop(columns=["trial_start_time"])
    return events_df


# ── 3. Trials processing ──────────────────────────────────────────────────────

def get_trial_data(trial):
    visual_events = trial[trial["event_type"] == "visual"]
    cue_on_time = visual_events["event_start_trial_time"].iloc[0]
    cue_off_time = visual_events["event_end_trial_time"].iloc[0]
    bg_repeat = trial[trial["event_type"] == "lick_bg"]
    num_bg_repeat = len(bg_repeat)

    trial_data = {
        "missed": True,
        "rewarded": False,
        "cue_on_time": cue_on_time,
        "cue_off_time": cue_off_time,
        "decision_time": np.nan,
        "background_length": cue_off_time - cue_on_time,
        "wait_length": 60,
        "num_bg_repeat": num_bg_repeat,
    }

    if "reward" in trial["event_type"].values:
        reward_time = trial.loc[trial["event_type"] == "reward", "event_start_trial_time"].iloc[0]
        trial_data.update({"missed": False, "wait_length": reward_time - cue_off_time})
        for cons_type in ["cons_reward", "cons_no_reward"]:
            if cons_type in trial["event_type"].values:
                trial_data.update({
                    "decision_time": trial.loc[
                        trial["event_type"] == cons_type, "event_start_trial_time"
                    ].iloc[0],
                    "rewarded": (cons_type == "cons_reward"),
                })

    trial_data["good"] = (trial_data["num_bg_repeat"] == 0) and (trial_data["missed"] is False)
    return trial_data


def compute_last_lick_times(trials, events):
    """For each trial, find the last lick onset before cue_off_time (session-wide,
    so the lick can come from a prior trial e.g. consumption). Returns last_lick_time
    in trial-relative time (negative if from a prior trial). NaN if no prior lick."""
    licks = events[events["event_type"].str.contains("lick", na=False)].copy()
    last_lick_map = {}
    for _, row in trials.iterrows():
        tid = row["trial_id"]
        cue_off_rel = row["cue_off_time"]
        if pd.isna(cue_off_rel):
            last_lick_map[tid] = np.nan
            continue
        trial_start_abs = row["event_start_time"]
        cue_off_abs = trial_start_abs + cue_off_rel
        before = licks[licks["event_start_time"] < cue_off_abs]
        if before.empty:
            last_lick_map[tid] = np.nan
        else:
            last_lick_map[tid] = before["event_start_time"].max() - trial_start_abs
    trials = trials.copy()
    trials["last_lick_time"] = trials["trial_id"].map(last_lick_map)
    return trials


def generate_trials(events):
    trials = events.loc[events["event_type"] == "trial"].copy()
    trial_data_list = []
    for t, trial in events.groupby("trial_id"):
        trial_data = {"trial_id": t} | get_trial_data(trial)
        trial_data_list.append(trial_data)
    trial_data_df = pd.DataFrame(trial_data_list)
    trials = pd.merge(trials, trial_data_df, on="trial_id")
    trials["consumption_length"] = trials["event_end_trial_time"] - trials["decision_time"]
    trials = trials.rename(columns={"event_end_trial_time": "trial_length"})
    trials = compute_last_lick_times(trials, events)
    trials = trials.drop(columns=["event_start_trial_time", "event_type"])
    return trials


# ── 4. Spikes processing ──────────────────────────────────────────────────────

def add_trial_time_to_spikes(spikes, trials):
    for _, trial_basics in trials.iterrows():
        trial_start_time = trial_basics["event_start_time"]
        trial_end_time = trial_basics["event_end_time"]
        spikes.loc[
            spikes["spike_time"].between(trial_start_time, trial_end_time), "trial_id"
        ] = trial_basics["trial_id"]
        spikes.loc[
            spikes["spike_time"].between(trial_start_time, trial_end_time), "trial_time"
        ] = (spikes["spike_time"] - trial_start_time)
    time_columns = ["cue_on_time", "cue_off_time", "last_lick_time", "decision_time"]
    trials_to_merge = trials[["trial_id"] + time_columns].copy()
    spikes = trials_to_merge.merge(spikes, on="trial_id", how="inner")
    return spikes


def add_lick_lookback_spikes(spikes_raw, spikes_assigned, trials):
    """For trials where the last lick was before trial start (last_lick_time < 0),
    duplicate spikes from [last_lick_abs, trial_start) into that trial's group with
    negative trial_time so they appear correctly in the TO_LAST_LICK raster."""
    time_columns = ["cue_on_time", "cue_off_time", "last_lick_time", "decision_time"]
    extra_rows = []
    for _, trial in trials.iterrows():
        last_lick_rel = trial["last_lick_time"]
        if pd.isna(last_lick_rel) or last_lick_rel >= 0:
            continue
        trial_start_abs = trial["event_start_time"]
        last_lick_abs = trial_start_abs + last_lick_rel
        lookback = spikes_raw[
            (spikes_raw["spike_time"] >= last_lick_abs)
            & (spikes_raw["spike_time"] < trial_start_abs)
        ].copy()
        if lookback.empty:
            continue
        lookback["trial_id"] = trial["trial_id"]
        lookback["trial_time"] = lookback["spike_time"] - trial_start_abs
        for col in time_columns:
            lookback[col] = trial[col]
        extra_rows.append(lookback)
    if extra_rows:
        return pd.concat([spikes_assigned] + extra_rows, ignore_index=True)
    return spikes_assigned


def align_spike_time_to_anchors(spikes):
    spikes[k.TO_CUE_ON] = spikes["trial_time"] - spikes["cue_on_time"]
    spikes[k.TO_CUE_OFF] = spikes["trial_time"] - spikes["cue_off_time"]
    spikes[k.TO_LAST_LICK] = spikes["trial_time"] - spikes["last_lick_time"]
    spikes[k.TO_DECISION] = spikes["trial_time"] - spikes["decision_time"]
    return spikes


def add_period_to_spikes(row):
    if row["cue_on_time"] <= row["trial_time"] < row["cue_off_time"]:
        return k.BACKGROUND
    if pd.isna(row["decision_time"]):
        if row["cue_off_time"] <= row["trial_time"]:
            return k.WAIT
    else:
        if row["cue_off_time"] <= row["trial_time"] < row["decision_time"]:
            return k.WAIT
        elif row["decision_time"] <= row["trial_time"]:
            return k.CONSUMPTION


def add_lick_period_to_spikes(row):
    """Secondary period column for the TO_LAST_LICK anchor.
    LICK_TO_CUE: from last_lick up to cue_off (can include negative trial_time
    for lookback spikes from previous trials).
    WAIT: from cue_off to decision. None otherwise."""
    if pd.isna(row["last_lick_time"]):
        return None
    if row["last_lick_time"] <= row["trial_time"] < row["cue_off_time"]:
        return k.LICK_TO_CUE
    if not pd.isna(row["decision_time"]):
        if row["cue_off_time"] <= row["trial_time"] < row["decision_time"]:
            return k.WAIT
    return None


def process_spikes(spikes, trials):
    spikes_raw = pd.DataFrame(spikes, columns=["spike_time"])
    spikes_assigned = add_trial_time_to_spikes(spikes_raw.copy(), trials)
    spikes_assigned = add_lick_lookback_spikes(spikes_raw, spikes_assigned, trials)
    spikes_assigned = align_spike_time_to_anchors(spikes_assigned)
    spikes_assigned["period"] = spikes_assigned.apply(add_period_to_spikes, axis=1)
    spikes_assigned["lick_period"] = spikes_assigned.apply(add_lick_period_to_spikes, axis=1)
    return spikes_assigned


# ── 5. Align events to anchor times ───────────────────────────────────────────

def align_events(events, trials):
    time_columns = ["cue_on_time", "cue_off_time", "last_lick_time", "decision_time"]
    trials_to_merge = trials[["trial_id"] + time_columns].copy()
    events = trials_to_merge.merge(events, on="trial_id", how="inner")
    events[k.TO_CUE_ON] = events["event_start_trial_time"] - events["cue_on_time"]
    events[k.TO_CUE_OFF] = events["event_start_trial_time"] - events["cue_off_time"]
    events[k.TO_LAST_LICK] = events["event_start_trial_time"] - events["last_lick_time"]
    events[k.TO_DECISION] = events["event_start_trial_time"] - events["decision_time"]
    return events


# ── 6. Per-session pipeline ───────────────────────────────────────────────────

def process_session(session):
    events = process_raw_events(session["events"])
    trials = generate_trials(events)
    events_aligned = align_events(events, trials)
    units_aligned = {
        uid: process_spikes(unit, trials)
        for uid, unit in zip(session["unit_ids"], session["units"])
    }
    return events_aligned, trials, units_aligned


def process_and_save_all_sessions(curated_sessions, pickle_dir, regenerate):
    pickle_dir.mkdir(parents=True, exist_ok=True)
    for _, session in curated_sessions.iterrows():
        output_path = os.path.join(pickle_dir, f"{session['id']}.pkl")
        if os.path.exists(output_path) and not regenerate:
            print(f"Session {session['id']} already exists - skipping")
            continue

        events, trials, units = process_session(session)

        session_data = {
            "id": session["id"],
            "mouse": session["mouse"],
            "date": session["date"],
            "region": session["region"],
            "events": events,
            "trials": trials,
            "unit_ids": session["unit_ids"],
            "units": units,
        }

        with open(output_path, "wb") as f:
            pickle.dump(session_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved session {session['id']} to {output_path}")


# ── Run ───────────────────────────────────────────────────────────────────────

print("Loading curated sessions pickle...")
with open(p.DATA_DIR / RAW_PICKLE_NAME, "rb") as f:
    curated_sessions_list = pickle.load(f)

curated_sessions_all = generate_sessions_sorted(curated_sessions_list)
curated_sessions_with_units = curated_sessions_all.loc[curated_sessions_all["num_units"] > 0]

print(f"  Total sessions: {len(curated_sessions_all)}")
print(f"  Sessions with units: {len(curated_sessions_with_units)}")
print(f"  Total cells: {sum(curated_sessions_all.num_units)}")

print("\nLoading recording log from Google Sheets...")
recording_log = pd.read_csv(RECORDING_LOG_URL)
recording_log.columns = recording_log.columns.str.lower().str.strip().str.replace(" ", "_")
recording_log["date"] = pd.to_datetime(recording_log["date"]).dt.date.astype(str)
recording_log["mouse"] = recording_log["mouse"].astype(str).str.strip()

print("Merging recording log into curated sessions...")
curated_sessions = pd.merge(
    recording_log,
    curated_sessions_with_units,
    on=["mouse", "date", "insertion_number"],
    how="inner",
)
curated_sessions["id"] = curated_sessions[["mouse", "date", "region"]].agg("_".join, axis=1)
curated_sessions = curated_sessions.sort_values(by="mouse")

region_cell_count = (
    curated_sessions.groupby(["mouse", "region"])["num_units"].sum().unstack(fill_value=0)
)
region_cell_count["total"] = region_cell_count.sum(axis=1)
region_cell_count.to_csv(p.LOGS_DIR / "region_cell_count.csv")

print("\nProcessing and saving per-session pickles...")
process_and_save_all_sessions(curated_sessions, p.PICKLE_DIR, REGENERATE)

print("\nWriting session log...")
sorted_sessions = curated_sessions.drop(columns=["events", "units", "unit_ids"])
sorted_sessions.to_csv(p.LOGS_DIR / "sessions_official_raw.csv")

print("\nDone.")
