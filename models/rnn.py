from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation, concatenate, Flatten,\
    Embedding, Lambda, Dropout, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras import losses 
from tensorflow import float16
import numpy as np

def rnn(embedding_size, single_timestep_elements, single_timestep_gt, recurrent_dropout=0, learning_rate=1e-4):
    inputs = Input(shape=(None, single_timestep_elements))

    x = LSTM(50, name='a')(inputs)
    x = LSTM(150, return_sequences=False, stateful=False, name='b')(x)
    x = Dropout(0.2)(x)
    x = LSTM(200, return_sequences=False, stateful=False, name='c')(x)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=False, stateful=False, name='d')(x)
    x = Dense(single_timestep_gt, activation='sigmoid')(x)

    model = Model(inputs, x)
    model.compile(loss='mean_squared_error',
                optimizer=Adam(lr=learning_rate),
                metrics=['accuracy'])

    return model

def encoder_decoder(embedding_size, single_timestep_elements, single_timestep_gt, recurrent_dropout=0, learning_rate=1e-4):

    encoder_input = Input(shape=(None, single_timestep_elements))
    encoder_output, encoder_state_h, encoder_state_c = LSTM(
        embedding_size, recurrent_dropout=recurrent_dropout, return_state=True)(encoder_input)

    decoder_input = Input(shape=(None, single_timestep_gt))
    decoder_output, decoder_state_h, decoder_state_c = LSTM(embedding_size,
                                                            #activation='linear',
                                                            recurrent_dropout=recurrent_dropout, 
                                                            return_sequences=True,
                                                            return_state=True)(decoder_input, initial_state=[encoder_state_h, encoder_state_c])
    decoder_output = Dense(
        single_timestep_gt, activation='tanh')(decoder_output)

    #decoder_output = Lambda((lambda x: x*10))(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_output)

    model.compile(optimizer=Adam(lr=learning_rate),
                  #metrics=['accuracy'],
                  loss='mean_absolute_error')

    encoder_model = Model(encoder_input, [encoder_state_h, encoder_state_c])

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
