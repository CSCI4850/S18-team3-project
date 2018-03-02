import random
import nltk
from nltk.tokenize import word_tokenize

#After tunning Sam's script to get all of the data from whatever URL, this retrieves it and creates a list full of words split just by spaces
def read_data(filename):
    with open(filename, 'r') as f:
        data = [word for line in f for word in line.split()]
    return data

#Entire, non-unique vocabulary
vocab = read_data('../data/train/south_park.txt')


#This little loop is not being used anywhere else, it is just an example of how nltk.word_tokenize can be used with our data set.
#This is just showing the first 100 words
#Main issue with this - figuring out what to do with contractions - "dont" turns into 
#["do", 'n't'] but I think I can write a simple regex that could fix this 
#Simply converting n't to not and 's to is, possibly checking most common cases and scrapping the rest of the cases or something?
for i in range(100):
        tokens = nltk.word_tokenize(vocab[i])
        print(tokens)



