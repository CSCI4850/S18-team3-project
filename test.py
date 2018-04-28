'''
Tests the model
'''

import numpy as np
import sys
import os
import json
from tqdm import tqdm
from keras.models import load_model
from keras.models import model_from_json 
from keras import backend as K
import pickle

def dist(x,y):
    s = 0
    z = x-y

    for element in z:
        s += element**2
    return np.sqrt(s)

def closest(dictionary, vec):
    min_dist = 1000000000000
    for key,val in dictionary.items():
        v = np.array(val)[0]

        d = dist(v, vec)

        if d < min_dist:
            min_dist = d
            closest = key
            closest_vec = val
    return closest, np.array(closest_vec)

if __name__ == '__main__':

    ########## SET DIRECTORIES ##########
    DATA_DIR = os.path.join("data", "train", "cleaned")
    MAPPING_FILE = os.path.join("utils", "mapping.pkl")
    RNN_MODEL = os.path.join("models", "rnn_model.hdf5")

    INCLUDE_POS = True 
    NUM_POS_TAGS = 47

    ########## IMPORT DATA ##########
    with open(MAPPING_FILE, 'rb') as f:
        mapping = pickle.load(f)
    print("**** Data Loaded ****")

    ########## LOAD MODEL ##########

    model = load_model(RNN_MODEL)

    print("**** Models Loaded ****")

    ########## GENERATE ##########
    print("**** Generating Sentences ****")

    # set up start token
    token = mapping['ST']
    token = np.array(token)
    token = np.reshape(token, (1,) + token.shape)

    if INCLUDE_POS:
        final_shape = token.shape[-1] + NUM_POS_TAGS 
    else:
        final_shape = token.shape[-1]

    tmp = np.zeros(shape=(1,1,final_shape))
    tmp[0,0,:len(token[0,0])] = token[0,0,:]
    token = tmp
    noise = np.random.rand(token.shape[0], token.shape[1], token.shape[2])
    noise /= 10 #small amount of noise

    print(token.shape)
    print(noise.shape)

    en_count = 0

    words = []
    words.append('ST')

    # generate words
    while en_count <= 50:
        out = model.predict([token, noise])

        # snap the network's prediction to the closest real word, and also
        # snap the network's prediction to the closest vector in our space
        # so that it predicts with real words as previous values
        closest_word, closest_vec = closest(mapping, out[0,0,:])
        token = np.zeros(shape=out.shape)
        token[0,0,:] = closest_vec

        # fix shapes
        tmp = np.zeros(shape=(1,1,final_shape))
        tmp[0,0,:len(out[0,0])] = out[0,0,:]
        out = tmp

        tmp = np.zeros(shape=(1,1,final_shape))
        tmp[0,0,:len(token[0,0])] = token[0,0,:]
        token = tmp

        noise = np.random.rand(token.shape[0], token.shape[1], token.shape[2])
        noise /= 10

        words.append(closest_word)
        

        if closest_word == "EN":
            en_count += 1
            print(closest_word)
        else:
            print(closest_word, end=' ')

    # write the output to a file
    if INCLUDE_POS:
        filename = "pos_output.txt"
    else:
        filename = "no_pos_output.txt"
    with open(filename, 'w', encoding='utf8') as f:
        for word in words:
            f.write(word)
            if word == "EN":
                f.write('\n')
            else:
                f.write(" ")

    print("**** Generating Sentences Complete ****")

    K.clear_session()
