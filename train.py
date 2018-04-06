'''
Trains the model
'''

import numpy as np
import os
from models.rnn import rnn
from keras.callbacks import ModelCheckpoint
from utils.word2vec import recall_mapping


########## SET DIRECTORIES ##########
DATA_DIR = os.path.join("data", "train")
EMBEDDING_FILE = os.path.join("utils", "embedPlusPos.h5")
corpus = os.path.join(DATA_DIR, "south_park.txt") # to be fixed later

########## IMPORT DATA ##########
embeddings = recall_mapping(EMBEDDING_FILE)
print("**** Data Loaded ****")

########## PROCESS DATA ##########
with open(corpus, 'r') as f:
    corpus_data = f.read().split()

data = []
for word in corpus_data:
    if word in embeddings.keys():
        data.append(np.array(embeddings[word]).T)
        #TODO: pos_tag corpus_data, append to eatch data element

data = np.array(data)
print(data[0])
print(data[0].shape)
#print(embeddings[:5])
print(data[:5])

ground_truth = data.copy()
ground_truth = ground_truth[1:-1]

########## LOAD MODEL ##########

model = rnn(128)
print(model.summary())

########## CALLBACKS ##########


########## TRAIN ##########

#TODO: specify what Y should be
#TODO: determine proper shapes for output
model.fit(data[:-2], ground_truth[:,:,0],
        batch_size=128,
        epochs=1,
        callbacks=[ModelCheckpoint('weights.hdf5', monitor='acc', verbose=0)])

test = np.random.rand(1,128,1)
print(test)
print(model.predict(test))
