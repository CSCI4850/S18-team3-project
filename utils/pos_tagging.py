from nltk import pos_tag, word_tokenize
from nltk.data import load as load_nltk_data
import os
import nltk
import numpy as np
from collections import OrderedDict
from nltk.data import load
from keras.utils import to_categorical

def enumerate_tags():
    ''' 
    Enumerating part of speech tags
    Returns:
        tag_dict: dict<String, ndarray>, mapping part of speech to categorical one-hot
    '''
    # read tagset data
    tagset_help = os.path.join("help", "tagsets", "upenn_tagset.pickle")
    load_nltk_data('nltk:'+tagset_help)
    # get tags
    tag_dict = load_nltk_data(tagset_help)
    tags = list(tag_dict.keys())
    tags.sort()
    tags.append('<start>')
    tags.append('<end>')
    tag_dict = OrderedDict(zip(tags,
                               to_categorical(range(0, len(tags)),
                                              num_classes=len(tags))
                               ))
    return tag_dict


def pos_tagging(paragraph):

    tok_sentences = []
    tag_sentences = []

    tokenizer = load_nltk_data('nltk:tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)

    for word in sentences:
        tok_sentences.append(nltk.word_tokenize(word))

    for word in tok_sentences:
        tag_sentences.append(nltk.pos_tag(word))
    # returns a tuple of (string: original_word, string: part_of_speech_code)
    return tag_sentences

def pos_tag_alt(text):
    '''
    Calculates the part of speech for each word in text.
    Params:
        text: string, the text to get pos for
    Returns:
        categorical_token_list: list<(string, ndarray)>, words and one-hot pos tags
    '''
    tokenized = pos_tag(word_tokenize(text))
    categorical_token_list = []
    try:    
        for tup in tokenized:
            parsed_pos_pair = list(tup)
            if (parsed_pos_pair[0] == 'ST'):
                parsed_pos_pair[1] = '<start>'
            elif (parsed_pos_pair[0] == 'EN'):
                parsed_pos_pair[1] = '<end>'
            parsed_pos_pair[1] = enumerate_tags()[parsed_pos_pair[1]]
            parsed_pos_pair[1] = np.pad(parsed_pos_pair[1], (0, 81), 'constant')
            categorical_token_list.append(parsed_pos_pair[1])
    except KeyError:
        pass
    return categorical_token_list


if __name__ == '__main__':
    ''' Testing part of speech tagger '''
