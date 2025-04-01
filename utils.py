import os
import pickle

def get_session_data(file_name, pickle_dir):
    pickle_path = os.path.join(pickle_dir, file_name)
    with open(pickle_path, 'rb') as f:
        session_data = pickle.load(f)
    
    events = session_data['events']
    trials = session_data['trials']
    units = session_data['units']
    idx = session_data['id']
    return events, trials, units, idx