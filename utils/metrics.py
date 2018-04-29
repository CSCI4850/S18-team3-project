import textdistance
import nltk as nltk
from nltk import pos_tag
from nltk.data import load
from nltk.tokenize import word_tokenize, sent_tokenize
from pos_tagging import pos_tagging
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File names for comparing")
    parser.add_argument('--file1', type=argparse.FileType('r', encoding='UTF-8'), required = True, action = 'store', dest = 'firstFile', help ="Name of first file")
    parser.add_argument('--file2', type=argparse.FileType('r', encoding='UTF-8'), required = True, action = 'store', dest = 'secondFile', help ="Name of second file")

    #reading lines from the two files passed in as arguments
    args = parser.parse_args()
    firstFile = args.firstFile
    secondFile = args.secondFile
    sentOne = []
    sentTwo = []
    contentOne = []
    contentTwo = []
    contentOne = firstFile.readlines()
    contentOne = [x.strip() for x in contentOne]
    contentTwo = secondFile.readlines()
    contentTwo = [x.strip() for x in contentTwo]

    firstFile.close()
    secondFile.close()
    
    numSentences = len(contentOne)
    
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
f.write("Average Hamming:  " + str(totalHamming/numSentences) + '\n')
f.write("Average Cosine:  " + str(totalCosine/numSentences) + '\n')
f.write("Average Gotoh:  " + str(totalGotoh/numSentences) + '\n')
f.write("Average Levenshtein:  " + str(totalLev/numSentences) + '\n')

