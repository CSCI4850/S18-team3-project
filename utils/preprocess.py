'''
Run this file from the root directory!!!
'''


import os
import cleaner
import word2vec
import obtain_data

if __name__ == "__main__":
    ########## DIRECTORIES ##########
    DATA_DIR = os.path.join("data", "train")
    CLEAN_DIR = os.path.join("data", "train", "cleaned")

    for d in [DATA_DIR, CLEAN_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    OUTPUT_DICTIONARY_FILE = os.path.join("utils", "embedPlusPos.pkl")

    # first download all files into data
    print("***** DOWNLOADING *****")
    THRESHOLD = 1 # TODO: this is temporary
    if len(os.listdir(DATA_DIR)) <= THRESHOLD:
        obtain_data.get_south_park()
        print("Done South Park")
        obtain_data.get_rick_and_morty()
        print("Done Rick and Morty")
        obtain_data.get_simpsons()
        print("Done Simpsons")

    # clean into the clean folder
    print("***** CLEANING *****")
    filenames = [os.path.join(DATA_DIR, x) for x in os.listdir(DATA_DIR)
                if not os.path.isdir(os.path.join(DATA_DIR,x))]

    print(filenames)

    for filename in filenames:
        src = filename
        dst = os.path.join(CLEAN_DIR, os.path.basename(filename))

        args = ['tmp', src, dst]

        if len(os.listdir(CLEAN_DIR)) < 3:
            cleaner.main(args)

    # aggregate into a single file
    final_data_file = os.path.join(CLEAN_DIR, "final_data.txt")

    # create word embeddings
    print("***** EMBEDDING *****")
    if not os.path.exists(OUTPUT_DICTIONARY_FILE):
        word2vec.main(CLEAN_DIR, OUTPUT_DICTIONARY_FILE)

    print("***** DONE *****")

