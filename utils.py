import os
import pickle

def get_session_data(session_id, pickle_dir):
    pickle_name = session_id + ".pkl"
    pickle_path = os.path.join(pickle_dir, pickle_name)
    with open(pickle_path, 'rb') as f:
        session_data = pickle.load(f)
    
    events = session_data['events']
    trials = session_data['trials']
    units = session_data['units']
    return events, trials, units