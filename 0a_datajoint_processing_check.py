"""
0a_datajoint_processing_check.py
=================================
Cross-checks the DataJoint pipeline progress log against the recording log
to track which sessions are stuck at which pipeline stage.

Inputs:
  p.LOGS_DIR / 'RZ_dj_progress.csv'  - downloaded by progress_checker.py
  Google Sheet recording log         - fetched at runtime

Outputs (p.LOGS_DIR):
  sessions_cross_checked.csv  - merged log with First_X_Column pipeline stage
  location_summary.csv        - per-mouse recording location summary
"""

import pandas as pd

import paths as p


PROGRESS_FILE_NAME = "RZ_dj_progress.csv"

_RECORDING_LOG_SHEET_ID = "1xpb52EO7BII-_5zIcB2HaOpZqepFcYTvBYgxA4zLyNo"
RECORDING_LOG_URL = (
    f"https://docs.google.com/spreadsheets/d/{_RECORDING_LOG_SHEET_ID}"
    "/export?format=csv&gid=0"
)


def load_dj_progress():
    df = pd.read_csv(p.LOGS_DIR / PROGRESS_FILE_NAME)
    df["date"] = pd.to_datetime(df["session_datetime"]).dt.date.astype(str)
    df["mouse"] = df["subject"].astype(str).str.strip()
    df["insertion_number"] = df["insertion_number"].astype(int)
    return df


def load_recording_log():
    df = pd.read_csv(RECORDING_LOG_URL)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df["mouse"] = df["mouse"].astype(str).str.strip()
    df["insertion_number"] = df["insertion_number"].astype(int)
    return df


def get_stage_sessions(cross_checked, dj_df, stage):
    """Return dj_progress info for sessions stuck at a given pipeline stage."""
    rows = cross_checked[cross_checked["First_X_Column"] == stage]
    result = []
    for _, row in rows.iterrows():
        match = dj_df[
            (dj_df["mouse"] == row["mouse"])
            & (dj_df["date"] == row["date"])
            & (dj_df["insertion_number"] == row["insertion_number"])
        ]
        if not match.empty:
            info = match.iloc[0][["subject", "session_datetime", "insertion_number"]].to_dict()
            result.append(info)
    return result


def build_location_summary(cross_checked):
    location_order = ["l_str", "r_str", "l_v1", "r_v1"]
    exp2 = cross_checked.loc[
        ~cross_checked["sorting_notes"].isin(["exp3", "tester"])
    ].copy()
    exp2["location"] = exp2["hemisphere"] + "_" + exp2["region"]

    def sort_locations(locs):
        return sorted(
            locs,
            key=lambda x: location_order.index(x) if x in location_order else len(location_order),
        )

    summary = (
        exp2.groupby("mouse")["location"]
        .apply(lambda x: sort_locations(x.unique()))
        .reset_index()
        .rename(columns={"location": "unique_locations"})
    )

    session_counts = (
        exp2.groupby(["mouse", "location"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[c for c in location_order if c in exp2["location"].unique()])
    )

    return summary.set_index("mouse").join(session_counts).reset_index()


print("Loading DataJoint progress log...")
dj_progress = load_dj_progress()

print("Loading recording log from Google Sheets...")
recording_log = load_recording_log()

print("Merging logs...")
cross_checked = recording_log.merge(
    dj_progress[["mouse", "date", "insertion_number", "First_X_Column"]],
    on=["mouse", "date", "insertion_number"],
    how="left",
)
cross_checked["First_X_Column"] = cross_checked["First_X_Column"].fillna("not_uploaded")
cross_checked.to_csv(p.LOGS_DIR / "sessions_cross_checked.csv")

print("\nPipeline stage overview:")
print(cross_checked["First_X_Column"].value_counts().to_string())

print("\nSessions stuck per stage:")
for stage in ["SIClustering", "SIExport", "ManualCuration"]:
    sessions = get_stage_sessions(cross_checked, dj_progress, stage)
    print(f"\n=== {stage} ({len(sessions)} sessions) ===")
    for s in sessions:
        print(s)

print("\nBuilding location summary...")
location_summary = build_location_summary(cross_checked)
location_summary.to_csv(p.LOGS_DIR / "location_summary.csv")
print(location_summary.to_string(index=False))

print("\nDone.")
