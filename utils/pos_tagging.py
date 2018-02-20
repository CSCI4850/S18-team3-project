import nltk

def pos_tagging(paragraph):

    tok_sentences = []
    tag_sentences = []

    tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)

    for word in sentences:
        tok_sentences.append(nltk.word_tokenize(word))

    for word in tok_sentences:
        tag_sentences.append(nltk.pos_tag(word))

    return tag_sentences
