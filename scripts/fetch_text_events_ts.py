import sys
sys.path.insert(0, '..')
from config import Config
from utility import TextReader, merge_text_raw
import pickle

args = Config()

with open(args.evalpicklepath, 'rb') as f:
    data = pickle.load(f)

treader = TextReader(args.textdata)
train_text = treader.read_all_text_events(data[3])
#val_raw = treader.read_all_text_concat(val_raw[''])
print(len(train_text))

events = []

for fname in data[3]:
    events.append(train_text[fname])

data.append(events)

with open(args.evalpicklepath_new, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
