import os
import pandas as pd
import numpy as np
import json

def diff(time1, time2):
    # compute time2-time1
    # return difference in hours
    a = np.datetime64(time1)
    b = np.datetime64(time2)
    return (b-a).astype('timedelta64[h]').astype(int)

class TextReader():
    def __init__(self, dbpath):
        self.dbpath = dbpath
        self.all_files = set(os.listdir(dbpath))

    def get_name_from_filename(self, fname):
        # '24610_episode1_timeseries.csv'
        tokens = fname.split('_')
        pid = tokens[0]
        episode_id = tokens[1].replace('episode', '').strip()
        return pid, episode_id

    def read_text_concat(self, patient_id):
        filepath = os.path.join(self.dbpath, str(patient_id))
        data = pd.read_csv(filepath, header=None, sep='\t')
        return data[1].str.cat(sep=' ')

    def read_text_events(self, patient_id):
        filepath = os.path.join(self.dbpath, str(patient_id))
        data = pd.read_csv(filepath, header=None, sep='\t')
        data.columns = ['A', 'B']
        data.sort_values(by='A', inplace=True)
        times = data['A'].values
        texts = data['B'].values
        return times, texts

    def read_all_text_concat(self, names):
        texts = {}
        for patient_id in names:
            pid, _ = self.get_name_from_filename(patient_id)
            if pid in self.all_files:
                texts[patient_id] = self.read_text_concat(pid)
        return texts

    def read_all_text_events(self, names):
        texts = {}
        for name in names:
            pid, eid = self.get_name_from_filename(name)
            text_file_name = pid + "_" + eid
            if text_file_name in self.all_files:
                texts[name] = self.read_text_event_json(text_file_name)
        # for each filename (which contains pateintid, eid) and can be used to merge.
        # it will store a list with timestep and text at that time step.
        return texts

    def read_all_text_concat_json(self, names, period_length=48.0):
        texts = {}
        for patient_id in names:
            pid, eid = self.get_name_from_filename(patient_id)
            text_file_name = pid + "_" + eid
            if text_file_name in self.all_files:
                time, texts = self.read_text_event_json(text_file_name)
                final_concatenated_text = ""
                for (t, txt) in zip(time, texts):
                    if diff(t, time[0]) <= period_length:
                        final_concatenated_text = final_concatenated_text + " "+txt
                    else:
                        break
            texts[patient_id] = final_concatenated_text
        return texts

    def read_text_event_json(self, text_file_name):
        filepath = os.path.join(self.dbpath, str(text_file_name))
        with open(filepath, 'r') as f:
            d = json.load(f)
        time = sorted(d.keys())
        text = []
        for t in time:
            text.append(" ".join(d[t]))
        assert len(time) == len(text)
        return time, text


def merge_text_raw(textdict, raw):
    mask = []
    text = []
    names = []
    suceed = 0
    missing = 0
    for item in raw['names']:
        if item in textdict:
            mask.append(True)
            text.append(textdict[item])
            names.append(item)
            suceed += 1
        else:
            mask.append(False)
            missing += 1

    print("Suceed Merging: ", suceed)
    print("Missing Merging: ", missing)

    data = [[], [], [], []]
    data[0] = raw['data'][0][mask]
    data[1] = np.array(raw['data'][1])[mask]
    data[2] = text
    data[3] = names
    # X,y,T,names
    return data
