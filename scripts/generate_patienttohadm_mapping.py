"""
Preprocess PubMed abstracts or MIMIC-III reports
"""
import re
import pandas as pd
import os

df = pd.read_csv('../mimic3/NOTEEVENTS.csv')
df.CHARTDATE = pd.to_datetime(df.CHARTDATE)
df.CHARTTIME = pd.to_datetime(df.CHARTTIME)
df.STORETIME = pd.to_datetime(df.STORETIME)

df2 = df[df.SUBJECT_ID.notnull()]
df2 = df2[df2.HADM_ID.notnull()]
df2 = df2[df2.CHARTTIME.notnull()]
df2 = df2[df2.TEXT.notnull()]

df2 = df2[['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']]

del df

dataset_path = '../mimic3-benchmarks/data/root2/train/'
all_files = os.listdir(dataset_path)
all_folders = list(filter(lambda x: x.isdigit(), all_files))

output_folder = '../mimic3-benchmarks/data/root2/text2/'

suceed = 0
failed = 0
failed_exception = 0

all_folders = all_folders

patient2hadmid = {}

for folder in all_folders:
    try:
        patient_id = int(folder)
        sliced = df2[df2.SUBJECT_ID == patient_id]
        if sliced.shape[0] == 0:
            print("No notes for PATIENT_ID : %d" % patient_id)
            failed += 1
            continue
        sliced.sort_values(by='CHARTTIME')
        # take only the first admission
        first_hadm_id = sliced.iloc[0].HADM_ID
        patient2hadmid[patient_id] = first_hadm_id
        suceed += 1
    except:
        print("Failed with Exception FOR Patient ID: %s", folder)
        failed_exception += 1

print(len(patient2hadmid), failed, suceed)

import pickle
with open('../mimic3-benchmarks/data/root2/patient2hadmid.pkl', 'wb') as f:
    pickle.dump(patient2hadmid, f, pickle.HIGHEST_PROTOCOL)
