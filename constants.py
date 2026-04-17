# periods
BACKGROUND = "background"
WAIT = "wait"
CONSUMPTION = "consumption"
LICK_TO_CUE = "lick_to_cue"
PERIODS = [BACKGROUND, WAIT, CONSUMPTION]

# anchors
TO_CUE_ON = "to_cue_on"
TO_CUE_OFF = "to_cue_off"
TO_LAST_LICK = "to_last_lick"
TO_DECISION = "to_decision"
ANCHORS = [TO_CUE_ON, TO_CUE_OFF, TO_LAST_LICK, TO_DECISION]

ANCHORED_PERIODS = {
    TO_CUE_ON:    [BACKGROUND],
    TO_CUE_OFF:   [BACKGROUND, WAIT],
    TO_LAST_LICK: [LICK_TO_CUE, WAIT],
    TO_DECISION:  [WAIT, CONSUMPTION],
}

# which period column each anchor uses for filtering
ANCHOR_PERIOD_COL = {
    TO_CUE_ON:    "period",
    TO_CUE_OFF:   "period",
    TO_LAST_LICK: "lick_period",
    TO_DECISION:  "period",
}

# sorters
TRIAL_NUM = "trial_id"
BACKGROUND_LENGTH = "background_length"
WAIT_LENGTH = "wait_length"
BACKGROUND_WAIT_LENGTH = "background_wait_length"
REWARD_WAIT_LENGTH = ["missed", "rewarded", "wait_length"]

# quality filters
MIN_UNITS = 1                       # sessions must have > this many units
MIN_SESSION_LENGTH = 1500           # seconds; sessions shorter than this are excluded
MIN_PERCENT_TRIALS_WITH_SPIKES = 0.8  # units must fire in at least this fraction of trials

# spike-sorting QC thresholds: {metric: (operator_str, threshold)}
# These are used to compute qc_pass_all. NaN values are treated as not-computable (not a failure).
# Note: firing_rate threshold for STR is overridden to 0.05 Hz in the QC script (MSN floor).
QC_THRESHOLDS = {
    "firing_rate":        (">",  0.1),   # STR override: > 0.05 Hz (see QC script)
    "isi_violation":      ("<",  1.0),
    "amplitude_cutoff":   ("<",  0.1),   # NaN → not failed, tracked separately
    "presence_ratio":     (">",  0.8),
    "contamination_rate": ("<",  0.2),
}

QC_THRESHOLDS_SECONDARY = {
    "isolation_distance": (">",  20.0),  # informational only; NaN on NP1.0 (not computed)
    "l_ratio":            ("<",   0.1),
    "d_prime":            (">",   2.0),
    "silhouette_score":   (">",   0.5),
    "max_drift":          ("<",  50.0),
    "cumulative_drift":   ("<", 1000.0),
}

# animal groups
GROUP_DICT = {
    "s": ["RZ034", "RZ035", "RZ036",
          "RZ047", "RZ048", "RZ049", "RZ050", "RZ051",
          "RZ061", "RZ062", "RZ063", "RZ064", "RZ069", "RZ070", "RZ071", "RZ072"],
    "l": ["RZ037", "RZ038", "RZ039",
          "RZ052", "RZ053", "RZ054", "RZ055", "RZ056",
          "RZ057", "RZ058", "RZ059", "RZ060", "RZ065", "RZ066", "RZ067", "RZ068"],
}

COHORT_DICT = {
    "5": ["RZ034", "RZ036", "RZ037", "RZ038", "RZ039"],
    "6": ["RZ047", "RZ049", "RZ050", "RZ051", "RZ052", "RZ053", "RZ055"],
    "7": ["RZ057", "RZ058", "RZ059", "RZ060", "RZ061", "RZ062", "RZ063", "RZ064",
          "RZ065", "RZ066", "RZ067", "RZ068", "RZ069", "RZ070", "RZ071", "RZ072"],
}

# ── Electrophysiology ──────────────────────────────────────────────────────────
FS = 30_000   # Neuropixels sampling rate (Hz)

# ── Histology cohorts (animals with completed clearing + registration) ─────────
EARLY_COHORT = ["RZ034", "RZ036", "RZ037", "RZ038", "RZ039"]
LATER_COHORT = ["RZ047", "RZ049", "RZ050", "RZ051", "RZ053"]
ALL_ANIMALS  = EARLY_COHORT + LATER_COHORT

# ── Brain region color palette (colorblind-friendly, used across location scripts)
REGION_COLORS = {
    "CP":      "#2166AC",
    "STR":     "#1a4f8a",
    "GPe":     "#4DAC26",
    "GPi":     "#276113",
    "MOp5":    "#D6604D",
    "MOp2/3":  "#F4A582",
    "MOp1":    "#FDDBC7",
    "MOp6a":   "#e08070",
    "MOs5":    "#c45040",
    "MOs2/3":  "#e8b090",
    "MOs1":    "#f5c5b0",
    "VISp5":   "#762A83",
    "VISp2/3": "#C2A5CF",
    "VISp4":   "#9970AB",
    "VISp6a":  "#5c1a6e",
    "VISp6b":  "#7a3a8e",
    "VISp1":   "#DDB8E8",
    "VISl1":   "#b899c5",
    "VISl2/3": "#9a7aad",
    "VAL":     "#E08214",
    "AV":      "#FDB863",
    "CL":      "#B35806",
    "LD":      "#f5c060",
    "LP":      "#e09030",
    "RT":      "#878787",
    "CA1":     "#01665E",
    "SUB":     "#5AB4AC",
    "PAL":     "#8c6d31",
}
