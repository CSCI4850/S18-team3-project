import os
import collections
import math
import random
import numpy as np
import nltk
from nltk.data import load
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize


def read_data(filename):
    with open(filename, 'r') as f:
        nontok_data = [word for line in f for word in line.split()]
        tokdata = [word_tokenize(i) for i in nontok_data]
        data = []
	# tokdata holds the tokenized data
	# using nltk's word_tokenize() function returns list of word(s), so can't will
        # become ['can', 't'] -- so the lines below simply run through tokdata, which is
        # a list of lists, and appends all words into one big list, instead of having
        # lists inside of lists
        for wordList in tokdata:
            data += wordList
    return data



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


#all files in ../data/train/cleaned will be looped through and their stats will be found in stat.txt
path = os.path.join("..", "data", "train", "cleaned")
vocab = []

uniqueVocab = open("stat.txt", "w")
#writes statistics to stat.txt
#Stats: total words in corpus, total unique words, and count of outlier words
for filename in os.listdir(path):
    indVocab = []
    indVocab = read_data(os.path.join(path, filename))
    vocab += read_data(os.path.join(path, filename))
    vocabsize = len(indVocab)
    data, count, dictionary, revdictionary = build_dataset(indVocab, vocabsize);
    
    #if word occurs less than 5 times, count it as an outlier
    outlierCount = 0
    for item in count:
        if item[1] > 5:
            outlierCount += 1

    uniqueVocab.write(filename + '\n')
    uniqueVocab.write("Total number of words in corpus: ")
    uniqueVocab.write(str(vocabsize) + '\n')
    uniqueSet = set(indVocab)
    uniqueVocab.write("Total number of unique words: ")
    uniqueVocab.write(str(len(uniqueSet)) + '\n')
    uniqueVocab.write("Outlier count: ")
    uniqueVocab.write(str(outlierCount) + '\n' + '\n')
    #for un in uniqueSet:
    #    uniqueVocab.write(un + '\n')
uniqueVocab.close()
