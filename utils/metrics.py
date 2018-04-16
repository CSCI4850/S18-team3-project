import textdistance
import nltk as nltk
from nltk import pos_tag
from nltk.data import load
from nltk.tokenize import word_tokenize, sent_tokenize
from pos_tagging import pos_tagging


#print(textdistance.needleman_wunsch('text is hard', 'test is hard'))


data = ["I am a cat.", "I am a pickle!", "I am not a hamburger.", "I go over the store.", "Yes, I am not a cat.", "No, I am not a dog.", "Firetruck goes fast."]

def binary_distance(first, second):
    #1.0 is labels are identical, 0.0 is they are different

    if first == second:
        return 1.0
    else:
        return 0.0

count = 0
prevWord = ''
for item in data:
    count = 0
    for word in item.split():
        count+=1
        #print(prevWord, word)
        #print(binary_distance(prevWord, word))
        prevWord = word

#print(binary_distance('NN', 'PN'))
#print(binary_distance('PN','PN'))

#for item in data:
#    print(sent_tokenize(item))




if __name__ == "__main__":
    sentOne = []
    sentTwo = []
    item2 = ''
    for item in data:
        print(item + ' --vs-- ' + item2)
        d = pos_tagging(item)
        for couple in d:
            sentOne = []
            for word in couple:
         #       print(str(word[1]))
                sentOne.append(str(word[1]))
            
            print(str(sentOne), str(sentTwo))
            #print(textdistance.gotoh(str(sentOne), str(sentTwo)))
            print("hamming: ")
            print(textdistance.hamming(str(sentOne), str(sentTwo)))
            print("cosine:  ")
            print(textdistance.cosine(str(sentOne), str(sentTwo)))
            print("gotoh:  ")
            print(textdistance.gotoh(str(sentOne), str(sentTwo)))
            print("levenshtein: ")
            print(textdistance.levenshtein(str(sentOne), str(sentTwo)))
            print('\n')
            sentTwo = sentOne
        item2 = item  
