from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation, concatenate, Flatten,\
        Embedding, Lambda, Dropout, Reshape, LeakyReLU, add
from keras.optimizers import Adam
import keras.backend as K
from keras import losses 
from tensorflow import float16
import numpy as np

def rnn(embedding_size, single_timestep_elements, single_timestep_gt, recurrent_dropout=0, learning_rate=1e-4, loss='mean_squared_error'):
    inputs = Input(shape=(None, single_timestep_elements))

    # add noise
    noise = Input(shape=(None, single_timestep_elements))
    x = add([inputs, noise])

    x = Dense(embedding_size//4)(x)
    x = LeakyReLU()(x)
    x = LSTM(embedding_size, return_sequences=True, recurrent_dropout=recurrent_dropout, name='a')(x)
    x = LSTM(embedding_size//2, return_sequences=True, recurrent_dropout=recurrent_dropout, name='b')(x)
    x = Dense(embedding_size//2)(x)
    x = Dense(single_timestep_gt, activation='softmax')(x)

    model = Model([inputs, noise], x)
    model.compile(loss=loss,
                optimizer=Adam(lr=learning_rate),
                metrics = ['accuracy'])

    return model
