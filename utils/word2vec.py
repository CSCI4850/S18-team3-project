from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from tqdm import tqdm
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import nltk
import pickle
import h5py
import itertools
from nltk.data import load
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize

# After tunning Sam's script to get all of the data from whatever URL, this retrieves
# it and creates a list full of words split just by spaces,
def read_data(filename):
    with open(filename, 'r', encoding='utf8') as f:

        nontok_data = []
        tok_data = []

        lines = map(lambda line: 
                    word_tokenize(line.rstrip()), tqdm(f.readlines()))

        data = list(itertools.chain.from_iterable(lines))

    return data


def create_mapping(dictionary, embedplusword):
    from pos_tagging import pos_tag_alt
    wordPlusPos = {}
    for keys in dictionary.keys():
        wordPlusPos[keys] = pos_tag_alt(keys)

    embedPlusPos = []
    embedder = []
    finalDict = {}
    for key, vals in embedplusword.items():
        embedPlusPos = zip(vals, wordPlusPos[key])
        x, y = zip(*embedPlusPos)
        embedder.append((x,y))
        finalDict[key] = (x,y)

    return finalDict

def save_mapping(filename, dictionary):
    #h5f = h5py.File(filename, 'w')
    #h5f.create_dataset('dataset1', data=str(dictionary))
    #h5f.close()
    #dd.io.save(filename, dictionary)
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

def recall_mapping(filepath):
    '''
    import ast

    h5read = h5py.File(filepath, 'r')
    item = h5read['dataset1']
    mapping = ast.literal_eval(item)
    h5fData = item.value
    h5read.close()

    return mapping
    '''
    #return dd.io.load(filepath)
    with open(filepath, 'rb') as f:
        return pickle.load(f)

#Prints out awesome data, the dots with similar words are closer together!
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    # set plot size in inches
    plt.figure(figsize=(18, 18))
    # loop through all labels
    for i, label in enumerate(labels):
        # get the embedding vectors
        x, y = low_dim_embs[i, :]
        # plot them in a scatterplot
        plt.scatter(x, y)
        # annotations
        plt.annotate(label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')

    # save the figure out
    plt.savefig(filename)

# generate batch data for mini-batch gradient descent
# here, the gradient is averaged over a small number of samples,
# as opposed to just one sample or ALL of the data
data_index = 0

def generate_batch(batch_size, num_skips, skip_window, data):
  
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

#Builds a dictionary that stores the UNIQUE words.
def build_dataset(words, n_words):
    #Process raw inputs into a dataset.
    count = [['UNK', -1]]  #unk is for unknown
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
    #revdictionary allows you to look up the word via it's number
    #dictionary allows you to look up the number via it's word


def main(path, output_dictionary_file):

    print("*** Adding vocab from {} ***".format(path))

    vocab = read_data((path))

    vocabsize = len(vocab)


    uniqueVocab = open("uniquevocab.txt", "w", encoding="utf8")
    uniqueVocab.write("Total number of words in corpus: ")
    uniqueVocab.write(str(vocabsize) + '\n')

    uniqueSet = set(vocab)
    uniqueVocab.write("Total number of unique words: ")
    uniqueVocab.write(str(len(uniqueSet)) + '\n' + '\n')

    print("Creating unique vocab")
    for un in tqdm(uniqueSet):
            uniqueVocab.write(un + '\n')
    uniqueVocab.close()


    data, count, dictionary, revdictionary = build_dataset(vocab, len(vocab));
    vocabsize = len(revdictionary)
    vocabulary_size = len(revdictionary)



    del vocab  # Hint to reduce memory.
    #print('Most common words (+UNK)', count[:5])
    #print('Sample data', data[:10], [revdictionary[i] for i in data[:10]])

    #data_index = 0



    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, data=data)
    batch_size = 128
    embedding_size = 16 # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 64  # Number of negative examples to sample.



    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)


    graph = tf.Graph()

    print("*** Creating embeddings ***")

    with graph.as_default():
        # input data
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # operations and variables
        # look up embeddings for inputs
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                         labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))

        # construct the SGD optimizer using a learning rate of 1.0
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # compute the cosine similarity between minibatch examples and all embeddings
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        # add variable initializer
        init = tf.global_variables_initializer()



    num_steps = 100001

    with tf.Session(graph=graph) as session:
        # we must initialize all variables before using them
        init.run()
     #   print('initialized.')

        # loop through all training steps and keep track of loss
        average_loss = 0
        for step in xrange(num_steps):
            # generate a minibatch of training data
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data=data)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # we perform a single update step by evaluating the optimizer operation (including it
            # in the list of returned values of session.run())
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            # print average loss every 2,000 steps
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # the average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            # computing cosine similarity (expensive!)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    # get a single validation sample
                    valid_word = revdictionary[valid_examples[i]]
                    # number of nearest neighbors
                    top_k = 8
                    # computing nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = revdictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    #print(log_str)
        final_embeddings = normalized_embeddings.eval() #Saves embeddings for use in other tensors
        #np.savetxt('final_embedding_dic.txt', final_embeddings)

        #A dictionary in the format of {word: [embeddings]}
        embedplusword = {}

        for word in dictionary:
            embedplusword[word] = [final_embeddings[dictionary[word]]]

    # final_dict = create_mapping(dictionary, embedplusword)
    save_mapping(output_dictionary_file, embedplusword)

    '''
    try:
        # import t-SNE and matplotlib.pyplot
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        # create the t-SNE object with 2 components, PCA initialization, and 5000 iterations
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        # plot only so many words in the embedding
        plot_only = 500
        # fit the TSNE dimensionality reduction technique to the word vector embedding
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        # get the words associated with the points
        labels = [revdictionary[i] for i in xrange(plot_only)]
        # call the plotting function
        plot_with_labels(low_dim_embs, labels)

    except ImportError:
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
    '''

if __name__ == '__main__':
    path = os.path.join("..", "data", "train", "cleaned","all_data.txt")
    if os.path.exists(path):
        main(path, "embedPlusPos.hdf5")
    else:
        print("First download all the data; please run preprocess.py instead of this file")
