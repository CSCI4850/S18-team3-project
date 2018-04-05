from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation, concatenate, Flatten,\
                        Embedding
from keras.optimizers import RMSprop 

def rnn(embedding_size):
    #TODO: auxiliary input will be the part of speech embedding, handled separately
    inputs = Input(shape=(embedding_size,1))

    x = LSTM(128)(inputs)
    
    word_embedding = Dense(embedding_size, activation='sigmoid')(x)
    part_of_speech = Dense(embedding_size, activation='sigmoid')(x)

    out = concatenate([word_embedding, part_of_speech])
    out = Dense(128)(out)
    #TODO: remove this final Dense layer when multidimensional vectors are implemented

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=RMSprop(), 
                monitor=['accuracy'],
                loss='binary_crossentropy')

    return model
