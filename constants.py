# periods
BACKGROUND = "background"
WAIT = "wait"
CONSUMPTION = "consumption"

# anchors
TO_CUE_ON = "to_cue_on"
TO_CUE_OFF = "to_cue_off"
TO_DECISION = "to_decision"

anchored_periods = {
    TO_CUE_ON: [BACKGROUND],
    TO_CUE_OFF: [BACKGROUND, WAIT],
    TO_DECISION: [WAIT, CONSUMPTION]
}

periods = [BACKGROUND, WAIT, CONSUMPTION]

# sorters
REWARD_WAIT_LENGTH = ["missed", "rewarded", "wait_length"]
BACKGROUND_LENGTH = "background_length"
WAIT_LENGTH = "wait_length"
TRIAL_NUM = "trial_num"
WAIT_LENGTH = 'wait_length'