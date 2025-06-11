import os
import pickle

import paths as p

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