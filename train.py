'''
Trains the model
'''

import numpy as np
import sys
import os
import json
from tqdm import tqdm
from models.rnn import encoder_decoder as rnn
from utils.word2vec import recall_mapping

if __name__ == '__main__':

    ########## SET DIRECTORIES ##########
    DATA_DIR = os.path.join("data", "train", "cleaned")
    EMBEDDING_FILE = os.path.join("utils", "embedPlusPos.pkl")
    ENCODER_MODEL = os.path.join("models", "encoder_model.hdf5")
    DECODER_MODEL = os.path.join("models", "decoder_model.hdf5")
    corpus = os.path.join(DATA_DIR, "rick_and_morty.txt")

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
    pre_ground_truth = ground_truth [:,0:ground_truth.shape[1]-1,:]
    post_ground_truth = ground_truth [:,1:ground_truth.shape[1],:]


    ########## LOAD MODEL ##########

    learning_rate = 1e-5

    model, encoder_model, decoder_model = rnn(embedding_size=64,
                                              recurrent_dropout=0,
                                              single_timestep_elements=data[0].shape[-1],
                                              single_timestep_gt=ground_truth[0].shape[-1],
                                              learning_rate=learning_rate)
    print(model.summary())

    ########## CALLBACKS ##########

    ########## TRAIN ##########

    BATCH_SIZE = 2**8
    NUM_EPOCHS = 200

    try:
        model.fit([data, pre_ground_truth], post_ground_truth,
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,)
        encoder_model.save(ENCODER_MODEL)
        decoder_model.save(DECODER_MODEL)

    except KeyboardInterrupt:
        encoder_model.save(ENCODER_MODEL)
        decoder_model.save(DECODER_MODEL)
