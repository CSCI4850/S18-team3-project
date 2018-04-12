from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation, concatenate, Flatten,\
    Embedding, Lambda
from keras.optimizers import Adam


def rnn(embedding_size, single_timestep_elements, single_timestep_gt, recurrent_dropout=0, learning_rate=1e-4):

    encoder_input = Input(shape=(None, single_timestep_elements))
    encoder_output, encoder_state_h, encoder_state_c = LSTM(
        embedding_size, recurrent_dropout=recurrent_dropout, return_state=True)(encoder_input)

    decoder_input = Input(shape=(None, single_timestep_gt))
    decoder_output, decoder_state_h, decoder_state_c = LSTM(embedding_size,
                                                            recurrent_dropout=recurrent_dropout, 
                                                            return_sequences=True,
                                                            return_state=True)(decoder_input, initial_state=[encoder_state_h, encoder_state_c])
    decoder_output = Dense(
        single_timestep_gt, activation='tanh')(decoder_output)

    decoder_output = Lambda((lambda x: x*2))(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_output)

    model.compile(optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'],
                  loss='mean_squared_error')

    encoder_model = Model(encoder_input, [encoder_state_h, encoder_state_c])

    decoder_state_input_h = Input(shape=(embedding_size,))
    decoder_state_input_c = Input(shape=(embedding_size,))

    decoder_output, decoder_state_h, decoder_state_c = LSTM(embedding_size,
                                                            recurrent_dropout=recurrent_dropout, 
                                                            return_state=True, 
                                                            return_sequences=True)(
                                decoder_input, 
                                initial_state=[decoder_state_input_h, decoder_state_input_c])
    decoder_output = Dense(single_timestep_gt, activation='tanh')(decoder_output)

    decoder_output = Lambda((lambda x: x*2))(decoder_output)
    
    decoder_model = Model([decoder_input] + [decoder_state_input_h, decoder_state_input_c],
                          [decoder_output] + [decoder_state_h, decoder_state_c])

    return model, encoder_model, decoder_model
