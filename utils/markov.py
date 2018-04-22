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
#path = os.path.join("..", "data", "train", "cleaned")
#for filename in os.listdir(path):
text += read_data("../data/train/cleaned/simple.txt")

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


writeFile = open("markovSentences.txt", "w")

count = 0
first_word = np.random.choice(text)
newSentence = ['ST']
newSentence.append(first_word)
#Generates 1000 sequential sentences and writes them to markovSentences.txt
while(count < 1000):
    if (newSentence):
        newWord = np.random.choice(model[newSentence[-1]])
    if (newWord == 'EN'):
        count += 1
        newSentence.append(newWord)
        output = " ".join(newSentence)
        writeFile.write(output + '\n')
        print(output + '\n')
        newWord = np.random.choice(model[newSentence[-1]])
        newSentence = [newWord]
    else:
        newSentence.append(newWord)

writeFile.close()

