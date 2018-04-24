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
from utils.word2vec import save_mapping, recall_mapping
from utils.pos_tagging import pos_tag_alt
from keras.utils import to_categorical
import matplotlib.pyplot as plt

if __name__ == '__main__':

    ########## SET DIRECTORIES ##########
    DATA_DIR = os.path.join("data", "train", "cleaned")
    EMBEDDING_FILE = os.path.join("utils", "embedPlusPos.pkl")
    MAPPING_FILE = os.path.join("utils", "mapping.pkl")
    ENCODER_MODEL = os.path.join("models", "encoder_model.hdf5")
    DECODER_MODEL = os.path.join("models", "decoder_model.hdf5")
    RNN_MODEL = os.path.join("models", "rnn_model.hdf5")
    corpus = os.path.join(DATA_DIR, "simple.txt")

    teacher_forcing = False 

    ########## IMPORT DATA ##########
    embeddings = recall_mapping(EMBEDDING_FILE)
    print("**** Data Loaded ****")

    ########## PROCESS DATA ##########
    with open(corpus, 'r', encoding='utf8') as f:
        all_corpus_data = f.read()
        corpus_data = all_corpus_data.split()
    
    # create mapping
    mapping = {}
    unique_words = list(set(corpus_data))
    for i, word in enumerate(unique_words):
        mapping[word] = to_categorical(i, num_classes=len(unique_words))
        mapping[word] = np.reshape(mapping[word], (1,) + mapping[word].shape)

    save_mapping(MAPPING_FILE, mapping)
        
 #   pos_pairs = pos_tag_alt(all_corpus_data, len(unique_words))

 #   print("Pos length:", len(pos_pairs[0][1]))

  #  print(len(mapping['ST'][0]))
  #  print(len(pos_pairs[0][1]))
 #   print(mapping['ST'][0])
 #   print(pos_pairs[0][1])

    # TODO: figure out how to concatenate part of speech with word embedding
    data = []
    ground_truth = []
    for word in corpus_data:
        data.append(mapping[word][0])
        ground_truth.append(mapping[word])



 #   for i, word in enumerate(corpus_data):
        #if word in mapping.keys():
 #       print(len(mapping[word][0]))
 #       print(len(pos_pairs[i][1]))
        #print(pos_pairs[i])
        #print(mapping[word])
        #print(mapping[word][0])
        #mapping[word] is an array of the array of one hot
        #mapping[wprd][0] grabs just that array and that solves this problem
    #    data.append(mapping[word][0])
     #   data.append(np.concatenate([mapping[word][0],pos_pairs[i][1]]))
    #    ground_truth.append(mapping[word])
    
    '''
    data = []
    for word in corpus_data:
        if word in embeddings.keys():
            data.append(np.array(embeddings[word]))
    '''


    data = np.array(data)
    #ground_truth = data.copy()
    ground_truth = np.array(ground_truth)

    num_words = 10
  #  print(data)
    print(data.shape[0])
    print(data.shape[1])
    new_data = np.zeros(shape=(data.shape[0], num_words, data.shape[1]))
    print("new data")
    print(new_data.shape)

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

    print(data.shape)
    print("neewest data")
    print(new_data.shape)
    print(ground_truth.shape)

    data = new_data.copy()
    #ground_truth = new_data.copy()
    #ground_truth = [x[0][:len(pos_pairs[0][1])] for x in ground_truth]
    #ground_truth = np.array(ground_truth)

    noise = np.random.rand(data.shape[0],data.shape[1],  data.shape[2])
    noise /= 10 # small amount of noise
    print(noise.shape)

    '''
    data = []
    for word in corpus_data:
        if word in embeddings.keys():
            data.append(np.array(embeddings[word]))
            # TODO: pos_tag corpus_data, append to eatch data element


    # TODO: split data into separate sentences?
    data = np.array(data)
    ground_truth = data.copy()

    num_words = 10
    new_data = np.zeros(shape=(data.shape[0], num_words, data.shape[2]))

    #print(data.shape)
    #print(new_data.shape)

    for i in range(len(data)):
        vec = np.zeros(shape=(num_words, data.shape[2]))
        for j in range(num_words):
            if (i+j+1) < len(data):
                vec[j] = data[i+j+1]
            else:
                k = 0
                vec[j] = data[k+j]
        #print(vec)
        new_data[i] = vec

    ground_truth = new_data.copy()
    '''


    if teacher_forcing:
        pre_ground_truth = ground_truth [:,0:ground_truth.shape[1]-1,:]
        post_ground_truth = ground_truth [:,1:ground_truth.shape[1],:]
    else:
        post_ground_truth = np.append(ground_truth [:,1:,:], 
                ground_truth[:,0:1,:],
                axis=1)
    print(data.shape)
    print(post_ground_truth.shape)
    print(noise.shape)


    ########## LOAD MODEL ##########


    learning_rate = 1e-4


    if teacher_forcing:
        model, encoder_model, decoder_model = encoder_decoder(embedding_size=128,
                                                              recurrent_dropout=0,
                                                              single_timestep_elements=data[0].shape[-1],
                                                              single_timestep_gt=post_ground_truth[0].shape[-1],
                                                              learning_rate=learning_rate,
                                                              loss='categorical_crossentropy')
    else:
        print(data[0].shape[-1])
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
        if teacher_forcing:
            BATCH_SIZE = 2**8
            model.fit([data[0:1], pre_ground_truth[0:1]], post_ground_truth[0:1],
                      batch_size=BATCH_SIZE,
                      epochs=NUM_EPOCHS,
                      verbose=0,
                      callbacks=callbacks)
            encoder_model.save(ENCODER_MODEL)
            decoder_model.save(DECODER_MODEL)
        else:
            BATCH_SIZE = 2**10
            history = model.fit([data, noise], post_ground_truth,
                      batch_size=BATCH_SIZE,
                      epochs=NUM_EPOCHS,
                      verbose=0,
                      callbacks=callbacks)
            model.save(RNN_MODEL)
            
            plt.figure(1)
            plt.subplot(211)
            plt.plot(history.history['acc'])
            plt.title('Model Accuracy without Part of Speech')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.subplot(212)
            plt.plot(history.history['loss'])
            plt.title('Model Loss with Part of Speech')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.tight_layout()
            plt.savefig('curves_without_pos.png')

        K.clear_session()

    except KeyboardInterrupt:
        if teacher_forcing:
            encoder_model.save(ENCODER_MODEL)
            decoder_model.save(DECODER_MODEL)
        else:
            model.save(RNN_MODEL)
        K.clear_session()
