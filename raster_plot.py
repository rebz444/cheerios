import matplotlib.pyplot as plt

import constants as k

event_colors = {
        'visual': 'g',
        'wait': 'orange', 
        'cons_reward': 'b', 
        'cons_no_reward': 'r'
    }

def prepare_data_for_raster(events, trials, spikes, sorter):
    sorted_trial_id = trials.sort_values(by=sorter).trial_id.tolist()
    events_raster = events.groupby('trial_id')
    spikes_raster = spikes.groupby('trial_id')
    return events_raster, spikes_raster, sorted_trial_id

def plot_raster(ax, events, trials, spikes, anchor, sorter, full_trial=False, show_legend=True):
    events_raster, spikes_raster, sorted_trial_id = prepare_data_for_raster(events, trials, spikes, sorter)

    ax.axvline(0, color='tab:gray', linestyle='--', alpha=0.5, label=anchor)

    for trial_offset, t in enumerate(sorted_trial_id):
        # Plot spikes first
        if t in spikes_raster.groups:
            trial_spikes = spikes_raster.get_group(t)
            relevant_periods = k.ANCHORED_PERIODS[anchor]
            relevant_spike_times = trial_spikes.loc[trial_spikes.period.isin(relevant_periods), anchor]
            ax.eventplot(
                relevant_spike_times, 
                lineoffsets=trial_offset, 
                color='k', 
                linelengths=0.8, 
                linewidths=0.4
            )
            irrelevant_spike_times = trial_spikes.loc[~trial_spikes.period.isin(relevant_periods), anchor]
            ax.eventplot(
                irrelevant_spike_times, 
                lineoffsets=trial_offset, 
                # color='darkgrey', 
                color='k',
                linelengths=0.8, 
                linewidths=0.4
            )

        # Plot trial events
        if t in events_raster.groups:
            trial_events = events_raster.get_group(t)
            for event_type, color in event_colors.items():
                event_time = trial_events.loc[trial_events['event_type'] == event_type, anchor]
                ax.eventplot(
                    event_time, lineoffsets=trial_offset, color=color, 
                    linelengths=1.0, linewidths=0.8, alpha=1
                )

    # Set y and x limits
    ax.set_ylim(-0.5, len(sorted_trial_id) - 0.5)
    # if not full_trial:
    #     if anchor == k.TO_CUE_ON:
    #         ax.set_xlim(-0.2, 10)
    #     elif anchor == k.TO_CUE_OFF:
    #         ax.set_xlim(-8, 6)
    #     elif anchor == k.TO_DECISION:
    #         ax.set_xlim(-10, 3)

    # Create legend only for the last raster plot
    if show_legend:
        handles = [plt.Line2D([0], [0], color=c, lw=2, label=label) 
                   for label, c in event_colors.items()]
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.92))