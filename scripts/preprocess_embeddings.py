from gensim.models import KeyedVectors
import pickle
import os
from config import Config

args = Config()

filename = '~/embeds/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

print(model.vectors.shape)
print(len(model.index2word))
print(model.index2word[:20])


set_index2word = set(model.index2word)
oov = dict()
freq = dict()

with open(os.path.join(args.picklepath, 'train.pkl'), 'rb') as f:
    data = pickle.load(f)

for text in data[2]:
    tokens = text.split(' ')
    for t in tokens:
        if t not in set_index2word:
            if t in oov:
                oov[t] += 1
            else:
                oov[t] = 1
        else:
            if t in freq:
                freq[t] += 1
            else:
                freq[t] = 1

for k, v in oov.items():
    print("%s,%d" % (k, v))

with open('oov.txt', 'w') as f:
    f.write(str(oov))

new_model = {"index2word": [], "vectors": []}
mask = []
for item in model.index2word:
    if item in freq:
        new_model["index2word"].append(item)
        mask.append(True)
    else:
        mask.append(False)
import numpy as np
new_vectors = model.vectors[mask]
new_model['index2word'].append('<pad>')
new_vectors = np.append(
    new_vectors, np.random.randn(2, 200), axis=0)  # for oov, pad
new_model['vectors'] = new_vectors

wv = {"model": new_model, "freq": freq}

import pickle
with open('wv.pkl', 'wb') as f:
    pickle.dump(wv, f, pickle.HIGHEST_PROTOCOL)
