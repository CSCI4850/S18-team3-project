'''
Trains the model
'''

import numpy as np
import sys
import os
import json
from tqdm import tqdm
from keras.models import load_model
from keras.models import model_from_json 
from keras import backend as K
from utils.word2vec import recall_mapping

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
    EMBEDDING_FILE = os.path.join("utils", "embedPlusPos.pkl")
    ENCODER_MODEL = os.path.join("models", "encoder_model.hdf5")
    DECODER_MODEL = os.path.join("models", "decoder_model.hdf5")
    RNN_MODEL = os.path.join("models", "rnn_model.hdf5")

    teacher_forcing = True 

    ########## IMPORT DATA ##########
    embeddings = recall_mapping(EMBEDDING_FILE)
    print("**** Data Loaded ****")

    ########## LOAD MODEL ##########

    if teacher_forcing:

        loss = 'cosine'
        encoder_model = load_model(ENCODER_MODEL)
        encoder_model.compile(optimizer='Adam', loss=loss)

        decoder_model = load_model(DECODER_MODEL)
        decoder_model.compile(optimizer='Adam', loss=loss)

    else:
        model = load_model(RNN_MODEL)

    print("**** Models Loaded ****")

    ########## GENERATE ##########
    print("**** Generating Sentences ****")

    # set up start token
    token = embeddings['ST']
    token = np.array(token)
    token = np.reshape(token, token.shape + (1,))

    print("Start token min: {:.4f}".format(np.min(token[0,:,0])))
    print("Start token med: {:.4f}".format(np.median(token[0,:,0])))
    print("Start token max: {:.4f}".format(np.max(token[0,:,0])))

    if teacher_forcing:
        context = encoder_model.predict(token)
        words = []
        words.append('ST')

        # generate words
        num_words = 10
        for _ in tqdm(range(num_words)):
            out, h, c = decoder_model.predict([token]+context)
            context = [h, c]


            # snap the network's prediction to the closest real word, and also
            # snap the network's prediction to the closest vector in our space
            # so that it predicts with real words as previous values
            closest_word, closest_vec = closest(embeddings, out[0,:,0])
            token = np.zeros(shape=out.shape)
            token[0,:,0] = closest_vec

            print("Prediction min: {:.4f}".format(np.min(token)))
            print("Start token med: {:.4f}".format(np.median(token)))
            print("Prediction max: {:.4f}".format(np.max(token)))

            words.append(closest_word)

            # update context with new token
            context = encoder_model.predict(token)

    else:
        words = []
        words.append('ST')

        # generate words
        num_words = 10
        for _ in tqdm(range(num_words)):
            out = model.predict(token)


            # snap the network's prediction to the closest real word, and also
            # snap the network's prediction to the closest vector in our space
            # so that it predicts with real words as previous values
            closest_word, closest_vec = closest(embeddings, out[0,:,0])
            token = np.zeros(shape=out.shape)
            token[0,:,0] = closest_vec

            print("Prediction min: {:.4f}".format(np.min(token)))
            print("Start token med: {:.4f}".format(np.median(token)))
            print("Prediction max: {:.4f}".format(np.max(token)))

            words.append(closest_word)

    # write the output to a file
    with open("RNN_output.txt", 'w', encoding='utf8') as f:
        for word in words:
            f.write(word + ' ')
    # print to console
    #for word in words:
    #    print(word)

    K.clear_session()
