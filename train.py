'''
Trains the model
'''

import numpy as np
import sys
import os
import json
from keras_tqdm import TQDMCallback
from tqdm import tqdm
from keras import backend as K
from models.rnn import encoder_decoder, rnn
from utils.word2vec import recall_mapping

if __name__ == '__main__':

    ########## SET DIRECTORIES ##########
    DATA_DIR = os.path.join("data", "train", "cleaned")
    EMBEDDING_FILE = os.path.join("utils", "embedPlusPos.pkl")
    ENCODER_MODEL = os.path.join("models", "encoder_model.hdf5")
    DECODER_MODEL = os.path.join("models", "decoder_model.hdf5")
    RNN_MODEL = os.path.join("models", "rnn_model.hdf5")
    #corpus = os.path.join(DATA_DIR, "rick_and_morty.txt")
    corpus = os.path.join(DATA_DIR, "simple.txt")

    teacher_forcing = False 

    ########## IMPORT DATA ##########
    embeddings = recall_mapping(EMBEDDING_FILE)
    print("**** Data Loaded ****")

    ########## PROCESS DATA ##########
    with open(corpus, 'r', encoding='utf8') as f:
        corpus_data = f.read().split()

    data = []
    for word in corpus_data:
        if word in embeddings.keys():
            data.append(np.array(embeddings[word]).T)
            # TODO: pos_tag corpus_data, append to eatch data element


    # TODO: split data into separate sentences?
    data = np.array(data)
    ground_truth = data.copy()

    if teacher_forcing:
        pre_ground_truth = ground_truth [:,0:ground_truth.shape[1]-1,:]
        post_ground_truth = ground_truth [:,1:ground_truth.shape[1],:]
    else:
        post_ground_truth = np.append(ground_truth [:,1:,:], 
                np.reshape(ground_truth[:,0,:], ground_truth[:,0,:].shape + (1,)), axis=1)




    ########## LOAD MODEL ##########


    learning_rate = 1e-4


    if teacher_forcing:
        model, encoder_model, decoder_model = encoder_decoder(embedding_size=512,
                                                              recurrent_dropout=0,
                                                              single_timestep_elements=data[0].shape[-1],
                                                              single_timestep_gt=ground_truth[0].shape[-1],
                                                              learning_rate=learning_rate,
                                                              loss='cosine')
    else:
        model = rnn(embedding_size=256,
                  recurrent_dropout=0,
                  single_timestep_elements=data[0].shape[-1],
                  single_timestep_gt=ground_truth[0].shape[-1],
                  learning_rate=learning_rate)

    print(model.summary())

    ########## CALLBACKS ##########

    ########## TRAIN ##########


    NUM_EPOCHS = 5000
    try:
        if teacher_forcing:
            BATCH_SIZE = 2**8
            model.fit([data, pre_ground_truth], post_ground_truth,
                      batch_size=BATCH_SIZE,
                      epochs=NUM_EPOCHS,
                      verbose=0,
                      callbacks=[TQDMCallback(leave_inner=False, show_inner=False)])
            encoder_model.save(ENCODER_MODEL)
            decoder_model.save(DECODER_MODEL)
        else:
            BATCH_SIZE = 2**10
            model.fit(data, post_ground_truth,
                      batch_size=BATCH_SIZE,
                      epochs=NUM_EPOCHS,
                      verbose=0,
                      callbacks=[TQDMCallback(leave_inner=False, show_inner=False)])
            model.save(RNN_MODEL)


        K.clear_session()

    except KeyboardInterrupt:
        if teacher_forcing:
            encoder_model.save(ENCODER_MODEL)
            decoder_model.save(DECODER_MODEL)
        else:
            model.save(RNN_MODEL)
        K.clear_session()
