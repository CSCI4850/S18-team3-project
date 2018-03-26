from nltk import pos_tag, word_tokenize
from nltk.data import load as load_nltk_data
import numpy as np
from nltk.data import load
import h5py
from pos_tagging import pos_tag_alt
from word2vec import embedplusword, dictionary 

#wordPlusPos : {'word': pos[128]} 
#embedplusword : {'word': emb[128]}
#embedPlusPos : [ (emb[128] , pos[128]) ]
#embedPlusPos is an list of tuples of numpy arrays

wordPlusPos = {}
for keys in dictionary.keys():
    wordPlusPos[keys] = pos_tag_alt(keys)


embedPlusPos = []
embedder = []
finalDict = {}
for key, vals in embedplusword.items():
    try:
        embedPlusPos = zip(vals, wordPlusPos[key])
        x,y = zip(*embedPlusPos)
        embedder.append((x, y))
        finalDict[key] = (x, y)

    except ValueError:
        pass




#opens up hdf5 file to write our [([embeddings], [pos])] to the file
h5f = h5py.File('embedPlusPos.h5', 'w')
h5f.create_dataset('dataset1', data = str(finalDict))
h5f.close()

#used to retrieve the data stored in our hdf5 file
#stored in h5fData
h5fread = h5py.File('embedPlusPos.h5', 'r')
item = h5fread['dataset1']
h5fData = item.value
#print(h5fData)
h5fread.close()




