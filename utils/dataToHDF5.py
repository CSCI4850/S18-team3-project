from nltk import pos_tag, word_tokenize
from nltk.data import load as load_nltk_data
import numpy as np
from nltk.data import load
import h5py
# from utils.pos_tagging import pos_tag_alt
#from utils.word2vec import embedplusword, dictionary 

#wordPlusPos : {'word': pos[128]} 
#embedplusword : {'word': emb[128]}
#embedPlusPos : [ (emb[128] , pos[128]) ]
#embedPlusPos is an list of tuples of numpy arrays

def create_mapping():
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
    return finalDict

def save_mapping(d):
    #opens up hdf5 file to write our [([embeddings], [pos])] to the file
    h5f = h5py.File('embedPlusPos.h5', 'w')
    h5f.create_dataset('dataset1', data = str(d))
    h5f.close()

def recall_mapping(filepath):
    #used to retrieve the data stored in our hdf5 file
    #stored in h5fData
    h5fread = h5py.File(filepath, 'r')
    item = h5fread['dataset1']
    h5fData = dict(item.value
    #print(h5fData)
    h5fread.close()
    return h5fData


if __name__ == "__main__":
    d = create_mapping()
    save_mapping(d)
    recall_mapping()
