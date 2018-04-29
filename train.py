'''
Trains the model
'''

import numpy as np
import sys
import os
import json
from keras_tqdm import TQDMCallback
import pickle
from tqdm import tqdm
from keras import backend as K
from models.rnn import rnn
from utils.pos_tagging import pos_tag_alt
from keras.utils import to_categorical
import matplotlib.pyplot as plt

if __name__ == '__main__':

    ########## SET DIRECTORIES ##########
    DATA_DIR = os.path.join("data", "train", "cleaned")
    MAPPING_FILE = os.path.join("utils", "mapping.pkl")
    RNN_MODEL = os.path.join("models", "rnn_model.hdf5")
    corpus = os.path.join(DATA_DIR, "simple.txt")
	
    INCLUDE_POS = True 

    ########## IMPORT DATA ##########
    with open(corpus, 'r', encoding='utf8') as f:
        all_corpus_data = f.read()
        corpus_data = all_corpus_data.split()

    print("**** Data Loaded ****")

    ########## PROCESS DATA ##########
    
    # create mapping
    mapping = {}
    unique_words = list(set(corpus_data))
    for i, word in enumerate(unique_words):
        mapping[word] = to_categorical(i, num_classes=len(unique_words))
        mapping[word] = np.reshape(mapping[word], (1,) + mapping[word].shape)

    # save mapping
    with open(MAPPING_FILE, 'wb') as f:
        pickle.dump(mapping, f, pickle.HIGHEST_PROTOCOL)
	
    data = []
    ground_truth = []

    if INCLUDE_POS:
        pos_pairs = pos_tag_alt(all_corpus_data, len(unique_words))
        print("Pos length:", len(pos_pairs[0][1]))

        for i, word in enumerate(corpus_data):
            data.append(np.concatenate([mapping[word][0],pos_pairs[i][1]]))
            ground_truth.append(mapping[word])
    else:
        for word in corpus_data:
            data.append(mapping[word][0])
            ground_truth.append(mapping[word])

    data = np.array(data)
    ground_truth = np.array(ground_truth)

    num_words = 10
    new_data = np.zeros(shape=(data.shape[0], num_words, data.shape[1]))

    for i in range(len(data)):
        vec = np.zeros(shape=(num_words, data.shape[1]))
        for j in range(num_words):
            if (i+j+1) < len(data):
                vec[j] = data[i+j+1]
            else:
                k = 0
                vec[j] = data[k+j]
        #print(vec)
        new_data[i] = vec

    data = new_data.copy()

    noise = np.random.rand(data.shape[0],data.shape[1],  data.shape[2])
    noise /= 10 # small amount of noise

    post_ground_truth = np.append(ground_truth [:,1:,:], 
                    ground_truth[:,0:1,:],
                    axis=1)


    ########## LOAD MODEL ##########

    learning_rate = 1e-4

    model = rnn(embedding_size=256,
                      recurrent_dropout=0,
                      single_timestep_elements=data[0].shape[-1],
                      single_timestep_gt=post_ground_truth[0].shape[-1],
                      learning_rate=learning_rate,
                      loss='categorical_crossentropy')

    print(model.summary())

    ########## CALLBACKS ##########

    callbacks = []
    tqdm_callback = TQDMCallback(leave_inner=False,
                                 show_inner=False,
                                 outer_description="Training")

    callbacks.append(tqdm_callback)

    ########## TRAIN ##########

    NUM_EPOCHS = 10000
    try:
        BATCH_SIZE = 2**10
        history = model.fit([data, noise], post_ground_truth,
                          batch_size=BATCH_SIZE,
                          epochs=NUM_EPOCHS,
                          verbose=0,
                          callbacks=callbacks)
        model.save(RNN_MODEL)

        print("\nModel saved successfully")

        ########## VISUALIZE TRAINING CURVES ##########

        if INCLUDE_POS:
            plot_filename = "training_curves_pos.png"
            plot_title = "Model Accuracy with Part of Speech"
            plot_title_2 = "Model Loss with Part of Speech"
        else:
            plot_filename = "training_curves_no_pos.png"
            plot_title = "Model Accuracy without Part of Speech"
            plot_title_2 = "Model Loss without Part of Speech"

        plt.figure(1)
        plt.subplot(211)
        plt.plot(history.history['acc'])
        plt.title(plot_title)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.title(plot_title_2)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.tight_layout()
        plt.savefig(plot_filename)

        K.clear_session()

    except KeyboardInterrupt:
        model.save(RNN_MODEL)
        print("\nModel saved successfully")
        K.clear_session()
