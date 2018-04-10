import numpy as np
import os
import nltk
from nltk import load
from nltk.tokenize import word_tokenize, sent_tokenize

def read_data(filename):
    with open(filename, 'r') as f:
        nontok_data = [word for line in f for word in line.split()]
        tokdata = [word_tokenize(i) for i in nontok_data]
        data = []
        for wordList in tokdata:
            data += wordList
    return data

text = []
path = os.path.join("..", "data", "train", "cleaned")
for filename in os.listdir(path):
    text += read_data(os.path.join(path, filename))

def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])
        
pairs = make_pairs(text)

model = {}
for word_1, word_2 in pairs:
    if word_1 in model.keys():
        model[word_1].append(word_2)
    else:
        model[word_1] = [word_2]

#Generates 10 sentences that start with 'ST' and end in 'EN'
for i in range(10):
    count = 0
    stcount = 0
    first_word = np.random.choice(text)
    newSentence = ['ST']
    newSentence.append(first_word)
    numWords = 30
    #Sentences are less than 30 words
    for i in range(numWords):
        newWord = np.random.choice(model[newSentence[-1]])
        if (newWord == 'EN'):
            newSentence.append(newWord)
            output = " ".join(newSentence)
            print(output)
            break          
        else:
            newSentence.append(newWord)

