'''
Run this file from the root directory!!!
'''


import os
import cleaner
from tqdm import tqdm
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
    THRESHOLD = 4 
    if len(os.listdir(DATA_DIR)) < THRESHOLD:
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

    cleaned_file_paths = []

    for filename in filenames:
        src = filename
        dst = os.path.join(CLEAN_DIR, os.path.basename(filename))
        cleaned_file_paths.append(dst)

        args = ['tmp', src, dst]

        if len(os.listdir(CLEAN_DIR)) < 3:
            cleaner.main(args)

    # aggregate into a single file
    print("***** AGGREGATING *****")
    #final_data_file = os.path.join(CLEAN_DIR, "all_data.txt")
    final_data_file = os.path.join(CLEAN_DIR, "simple.txt")

    if not os.path.exists(final_data_file):
        with open(final_data_file, 'w', encoding='utf8') as all_data_file:
            for clean_file in tqdm(cleaned_file_paths):
                with open(clean_file, 'r', encoding='utf8') as individual_data_file:
                    lines = individual_data_file.readlines()
                    all_data_file.writelines(lines)


    # create word embeddings
    print("***** EMBEDDING *****")
    if not os.path.exists(OUTPUT_DICTIONARY_FILE):
        word2vec.main(final_data_file, OUTPUT_DICTIONARY_FILE)

    print("***** DONE *****")

