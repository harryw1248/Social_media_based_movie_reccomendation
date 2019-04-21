'''
Contains functions for preprocessing text, including removing stopwords and stemming
'''

import re
import os
import sys
import collections
from PorterStemmer import PorterStemmer

'''
List of stopwords given on Canvas
'''
stopwords = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been',
             'but', 'by', 'few', 'from', 'for', 'have', 'he', 'her', 'here', 'him', 'his',
             'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on',
             'or', 'our', 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that',
             'this', 'to', 'us', 'was', 'what', 'when', 'where', 'which', 'who', 'why',
             'will', 'with', 'you', 'your']


This function removes the stopwords according to the stopwords list provided from Assignment 1
'''
def removeStopWords(list_of_tokens):
    number_of_tokens = len(list_of_tokens)
    token_index = 0

    # While loop iterates through all the tokens to remove stopwords by seeing if the
    # token is in the list of stopwords, in which case the word is removed
    while token_index < number_of_tokens:
        if list_of_tokens[token_index] in stopwords:
            list_of_tokens.pop(token_index)
            number_of_tokens = number_of_tokens - 1
        else:
            token_index = token_index + 1

    return list_of_tokens   # Returns tokens without stopwords

'''
Performs stemming of words using the PorterStemmer class code
'''
def stemWords(list_of_tokens):
    stemmer = PorterStemmer()   # Declares the stemmer object
    for token_index, token in enumerate(list_of_tokens):
        list_of_tokens[token_index] = stemmer.stem(token, 0, len(token) - 1)    # Stems the word using the function

    return list_of_tokens   # Returns the "post-stem" list of tokens
