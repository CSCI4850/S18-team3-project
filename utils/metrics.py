import textdistance
import nltk as nltk
from nltk import pos_tag
from nltk.data import load
from nltk.tokenize import word_tokenize, sent_tokenize
from pos_tagging import pos_tagging


if __name__ == "__main__":
    sentOne = []
    sentTwo = []
    contentOne = []
    contentTwo = []
    with open("markovSentences.txt", "r") as f:
        contentOne = f.readlines()
    contentOne = [x.strip() for x in contentOne]
    with open("../data/train/cleaned/cleanRM.txt", "r") as f:
        contentTwo = f.readlines()
    contentTwo = [x.strip() for x in contentTwo]

    
    totalHamming = 0
    totalCosine = 0
    totalGotoh = 0
    totalLev = 0
    count = 0
    for item in contentOne:
        print(item + ' --vs-- ' + contentTwo[count])
        posOne = pos_tagging(item)
        posTwo = pos_tagging(contentTwo[count])
        for x,y in zip(posOne, posTwo):
            sentOne = []
            sentTwo = []
            for pair in x:
                sentOne.append(str(pair[1]))
            for pair in y:
                sentTwo.append(str(pair[1]))
            if (sentOne == ['NN']):
                continue

            print(str(sentOne), str(sentTwo))
            print("hamming: ")
            ham = textdistance.hamming(str(sentOne), str(sentTwo))
            totalHamming += ham
            print(ham)
            print("cosine:  ")
            cos = textdistance.cosine(str(sentOne), str(sentTwo))
            totalCosine += cos
            print(cos)
            print("gotoh:  ")
            got = textdistance.gotoh(str(sentOne), str(sentTwo))
            totalGotoh += got
            print(got)
            print("levenshtein: ")
            lev = textdistance.levenshtein(str(sentOne), str(sentTwo))
            totalLev += lev
            print(lev)
            print('\n')
            sentTwo = sentOne
        count += 1

f = open("metricStats.txt", "w")
f.write("Average Hamming:  " + str(totalHamming/ 1000) + '\n')
f.write("Average Cosine:  " + str(totalCosine/1000) + '\n')
f.write("Average Gotoh:  " + str(totalGotoh/1000) + '\n')
f.write("Average Levenshtein:  " + str(totalLev/1000) + '\n')

