import spacy
nlp = spacy.load('en_core_web_sm')
import numpy as np

def tokenize(sentence):
    '''Tokenize the sentence'''
    return nlp(sentence)

def lemmatize(tokens):
    '''Lemmatize the tokens to their base form'''
    return [token.lemma_ for token in tokens]

def bag_of_words(tokenized_sentence, words):
    '''Create a bag of words from the tokens'''
    bag = np.zeros(len(words), dtype=np.float32)
    lemmatized_sentence = lemmatize(tokenized_sentence)
    for idx, word in enumerate(words):
        if word in lemmatized_sentence:
            bag[idx] = 1.0
    return bag

# print(lemmatize(tokenize("What is better than getting the best of both worlds?")))
# ['what', 'be', 'well', 'than', 'get', 'the', 'good', 'of', 'both', 'world', '?']
# print(bag_of_words(tokenize("how is it done for being better at everything at the world."), ['what', 'be', 'well', 'than', 'get', 'the', 'good', 'of', 'both', 'world', '?']))
# [0. 1. 1. 0. 0. 1. 0. 0. 0. 1. 0.]