from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation, concatenate, Flatten,\
        Embedding, Lambda, Dropout, Reshape, LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
from keras import losses 
from tensorflow import float16
import numpy as np

def rnn(embedding_size, single_timestep_elements, single_timestep_gt, recurrent_dropout=0, learning_rate=1e-4):
    inputs = Input(shape=(None, single_timestep_elements))

    x = LSTM(embedding_size, return_sequences=True, name='a')(inputs)
    #x = LSTM(150, return_sequences=True, stateful=False, name='b')(x)
    #x = Dropout(0.2)(x)
    #x = LSTM(200, return_sequences=True, stateful=False, name='c')(x)
    #x = Dropout(0.2)(x)
    #x = LSTM(32, return_sequences=True, stateful=False, name='d')(x)
    x = Dense(single_timestep_gt, activation='tanh')(x)
    x = Dense(single_timestep_gt, activation='linear')(x)

    model = Model(inputs, x)
    model.compile(loss='mean_squared_error',
                optimizer=Adam(lr=learning_rate),)

    return model

def encoder_decoder(embedding_size, single_timestep_elements, single_timestep_gt, recurrent_dropout=0, learning_rate=1e-4, loss='cosine'):

    encoder_input = Input(shape=(None, single_timestep_elements))
    #encoder_hidden_dense = Dense(256)(encoder_input)
    #encoder_hidden_dense = LeakyReLU()(encoder_hidden_dense)
    encoder_output, encoder_state_h, encoder_state_c = LSTM(
        embedding_size, recurrent_dropout=recurrent_dropout, return_state=True)(encoder_input)

    decoder_input = Input(shape=(None, single_timestep_gt))

    #decoder_hidden_dense = Dense(256)(decoder_input)
    #decoder_hidden_dense = LeakyReLU()(decoder_hidden_dense)
    decoder_output, decoder_state_h, decoder_state_c = LSTM(embedding_size,
                                                            recurrent_dropout=recurrent_dropout, 
                                                            return_sequences=True,
                                                            return_state=True)(decoder_input, initial_state=[encoder_state_h, encoder_state_c])
    #decoder_output = LeakyReLU()(decoder_output)
    #decoder_output = Dense(256,activation='tanh')(decoder_output)
    #decoder_output = Dense(256,activation='tanh')(decoder_output)
    decoder_output = Dense(128,activation='tanh')(decoder_output)
    decoder_output = Dense(single_timestep_gt,activation='linear')(decoder_output)


    model = Model([encoder_input, decoder_input], decoder_output)

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=loss)

    encoder_model = Model(encoder_input, [encoder_state_h, encoder_state_c])

    decoder_state_input_h = Input(shape=(embedding_size,))
    decoder_state_input_c = Input(shape=(embedding_size,))

    #decoder_hidden_dense = Dense(256)(decoder_input)
    #decoder_hidden_dense = LeakyReLU()(decoder_hidden_dense)
    decoder_output, decoder_state_h, decoder_state_c = LSTM(embedding_size,
                                                            recurrent_dropout=recurrent_dropout, 
                                                            return_state=True, 
                                                            return_sequences=True)(
                                decoder_input, 
                                initial_state=[decoder_state_input_h, decoder_state_input_c])
    decoder_output = LeakyReLU()(decoder_output)
    #decoder_output = Dense(256,activation='tanh')(decoder_output)
    #decoder_output = Dense(256,activation='tanh')(decoder_output)
    decoder_output = Dense(128,activation='linear')(decoder_output)
    decoder_output = Dense(single_timestep_gt, activation='tanh')(decoder_output)

    decoder_model = Model([decoder_input] + [decoder_state_input_h, decoder_state_input_c],
                          [decoder_output] + [decoder_state_h, decoder_state_c])

    return model, encoder_model, decoder_model


def encoder_decoder_other(embedding_size, single_timestep_elements, single_timestep_gt, recurrent_dropout=0, learning_rate=1e-4):

    ########## COMBINED ENCODER-DECOER ##########
    encoder_input = Input(shape=(None, single_timestep_elements))
    encoder_output, encoder_state_h, encoder_state_c = LSTM(
        embedding_size, recurrent_dropout=recurrent_dropout, return_state=True)(encoder_input)

    decoder_input = Input(shape=(None, single_timestep_gt))
    decoder_output, decoder_state_h, decoder_state_c = LSTM(embedding_size,
                                                            #activation='linear',
                                                            recurrent_dropout=recurrent_dropout, 
                                                            return_sequences=True,
                                                            return_state=True)(decoder_input, initial_state=[encoder_state_h, encoder_state_c])

    decoder_output = Dense(128, activation='relu')(decoder_output)
    decoder_output = Dense(128, activation='relu')(decoder_output)
    decoder_output = Dense(128, activation='relu')(decoder_output)
    decoder_output = Dense(single_timestep_gt, activation='tanh')(decoder_output)

    #decoder_output = Lambda((lambda x: x*10))(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_output)

    model.compile(optimizer=Adam(lr=learning_rate),
                  #metrics=['accuracy'],
                  loss='cosine')

    ########## ENCODER ##########
    encoder_model = Model(encoder_input, [encoder_state_h, encoder_state_c])

    ########## DECODER ##########
    decoder_state_input_h = Input(shape=(embedding_size,))
    decoder_state_input_c = Input(shape=(embedding_size,))

    decoder_output, decoder_state_h, decoder_state_c = LSTM(embedding_size,
                                                            #activation='linear',
                                                            recurrent_dropout=recurrent_dropout, 
                                                            return_state=True, 
                                                            return_sequences=True)(
                                decoder_input, 
                                initial_state=[decoder_state_input_h, decoder_state_input_c])
    decoder_output = Dense(single_timestep_gt, activation='tanh')(decoder_output)

    #decoder_output = Lambda((lambda x: x*10))(decoder_output)
    
    decoder_model = Model([decoder_input] + [decoder_state_input_h, decoder_state_input_c],
                          [decoder_output] + [decoder_state_h, decoder_state_c])

    return model, encoder_model, decoder_model
