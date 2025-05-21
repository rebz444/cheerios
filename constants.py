# periods
BACKGROUND = "background"
WAIT = "wait"
CONSUMPTION = "consumption"
PERIODS = [BACKGROUND, WAIT, CONSUMPTION]

# anchors
TO_CUE_ON = "to_cue_on"
TO_CUE_OFF = "to_cue_off"
TO_DECISION = "to_decision"
ANCHORS = [TO_CUE_ON, TO_CUE_OFF, TO_DECISION]

ANCHORED_PERIODS = {
    TO_CUE_ON: [BACKGROUND],
    TO_CUE_OFF: [BACKGROUND, WAIT],
    TO_DECISION: [WAIT, CONSUMPTION]
}

# sorters
TRIAL_NUM = "trial_num"
BACKGROUND_LENGTH = "background_length"
WAIT_LENGTH = "wait_length"
REWARD_WAIT_LENGTH = ["missed", "rewarded", "wait_length"]