'''
Trains the model
'''

import numpy as np
import sys
import os
from tqdm import tqdm
from models.rnn import encoder_decoder as rnn 
from keras.callbacks import ModelCheckpoint
from utils.word2vec import recall_mapping

def dist(x,y):
    s = 0
    z = x-y

    for element in z:
        s += element**2
    return np.sqrt(s)

def closest(dictionary, vec):
    min_dist = 1000000000000
    for key,val in tqdm(dictionary.items()):


        v = np.array(val)[0]

        d = dist(v, vec)
        if d < min_dist:
            min_dist = d
            closest = key
    return key

if __name__ == '__main__':

    ########## SET DIRECTORIES ##########
    DATA_DIR = os.path.join("data", "train")
    EMBEDDING_FILE = os.path.join("utils", "embedPlusPos.h5")
    corpus = os.path.join(DATA_DIR, "south_park.txt")  # to be fixed later

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
            # TODO: pos_tag corpus_data, append to eatch data element


    # TODO: split data into separate sentences
    data = np.array(data)
    ground_truth = data.copy()
    pre_ground_truth = ground_truth [:,0:ground_truth.shape[1]-1,:]
    post_ground_truth = ground_truth [:,1:ground_truth.shape[1],:]


    ########## LOAD MODEL ##########

    learning_rate = 1e-4

    '''
    model = rnn(128, data.shape[1], learning_rate)

    '''
    model, encoder_model, decoder_model = rnn(embedding_size=128,
                                              single_timestep_elements=data[0].shape[-1],
                                              single_timestep_gt=ground_truth[0].shape[-1],
                                              learning_rate=learning_rate)
    print(model.summary())

    ########## CALLBACKS ##########

    ########## TRAIN ##########

    model.fit([data, pre_ground_truth], post_ground_truth,
              batch_size=128,
              epochs=1,
              callbacks=[ModelCheckpoint('weights.hdf5', monitor='acc', verbose=0)])
    '''
    print("Data shape: {}".format(data.shape))
    model.fit(data, ground_truth,
              batch_size=128,
              epochs=1,
              callbacks=[ModelCheckpoint('weights.hdf5', monitor='acc', verbose=0)])
    '''


    # TODO actually generate sentences
    start_token = np.random.rand(1, 128, 1)
    context = np.random.rand(1, 128, 1) # first predict on start token, store that here
    pred = model.predict([start_token, context])
    words = []

    word = closest(embeddings, pred[0,:,0])
    prev_word = closest(embeddings, start_token[0,:,0])

    print(prev_word, word)


'''
    for _ in range(20):
        start_token = pred
        pred, h, c = decoder_model.predict([start_token, context])
        context = [h, c]

        words.append(closest(embeddings, pred[0,:,0]))
        


    for word in words:
        print(word)
'''
