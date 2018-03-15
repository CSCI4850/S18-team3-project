# Specifications

## Data Cleaning & Overall Data Flow
1. script which cleans all TV transcripts by setting all cases to lowercase, prepends all
sentences with a START token and appends all sentences with an END token.  The TV transcripts
should all be in a single file in this format (ie: South Park, R&M, Simpsons, etc. all into
a single, cleaned file)

2. Get word embeddings for the compiled, clean file from 1. using word2vec.  The mappings of 
word:embedding need to be stored somehow so we can discover the proper embeddings once,
and not retrain the word2vec model every time we want to start the script up.  Can we store
this mapping as an .hdf5 file or JSON instead of holding it in memory?

3. Get parts of speech for the compiled, clean file from 1. using NLTK.  Each individual word
must be associated with its respective part of speech.  This step should occur inline with step 4.

4. Read text into main training file.  If we read some number of batches at a time, then for each
batch we load into the script, we must feed into the RNN that word's respective embedding and
one-hot part of speech.  Two options here:

    a. zero-pad the one-hot vector to be the same length as the word embedding

    b. feed two inputs into the RNN and feed the one-hot part of speech into an embedding layer

The latter should theoretically give a better end result, but this will have to be iterated on.

## Model
1. Need to design a simple RNN

2. A model file `lstm.py` should exist in the `model/` directory.  It should have a single function
named `lstm()` which takes appropriate arguments and returns the model argument at the end. 

## Metrics
1. Need to contact Dr. Carroll about BLAST (Mike has experience with this?)

2. Redesign word probability comparison to have some useful number.
