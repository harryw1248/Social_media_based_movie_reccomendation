# Name: Vinayak Ahluwalia
# uniqname: vahluw

import re
import os
import sys
import collections
from porterStemmer import PorterStemmer

# List of stopwords given on Canvas
stopwords = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been',
             'but', 'by', 'few', 'from', 'for', 'have', 'he', 'her', 'here', 'him', 'his',
             'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on',
             'or', 'our', 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that',
             'this', 'to', 'us', 'was', 'what', 'when', 'where', 'which', 'who', 'why',
             'will', 'with', 'you', 'your']

# Dictionary of contractions that allow for expansion when apostrophe encountered in appropriate situations
contractions = {"we're": ("we", "are"), "can't": "cannot", "i'm": ("i", "am"), "you're": ("you", "are"),
                "i'd": ("i", "would"), "it's": ("it", "is"), "isn't": ("is", "not"), "won't": ("will", "not"),
                "shouldn't": ("should", "not"), "wouldn't": ("would", "not"), "didn't": ("did", "not"),
                "couldn't": ("could", "not"), "don't": ("do", "not"), "haven't": ("have", "not"),
                "hasn't": ("has", "not"), "wasn't": ("was", "not"), "weren't": ("were", "not"),
                "they're": ("they", "are"), "let's": ("let", "us"), "i'll": ("i", "will")}

# List of commmon abbreviations preventing period from being removed in tokenizeText()
abbreviations = ["mr.", "dr.", "mrs.", "ms.", "sr.", "jr.", "d.c.", "dept.", "j.d.", "oz.", "lbs.", "m.d.", "u.s.a.",
                 "co.", "ae.", "scs."]

# List of months allowing dates to be tokenized
months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december"]

# List of month abbreviations also allowing date tokenization
months_abbreviations = ["jan.", "feb.", "mar.", "apr.", "may", "jun.", "jul.", "aug.", "sept.", "oct.", "nov.", "dec."]

# List of possible dates in a month allowing date tokenization
# Includes multiple forms of how a date can be presented
days_in_month = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                 "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
                 "1,", "2,", "3,", "4,", "5,", "6,", "7,", "8,", "9,", "10,", "11,", "12,", "13,", "14,", "15,", "16,",
                 "17,", "18,", "19,", "20,", "21,", "22,", "23,", "24,", "25,", "26,", "27,", "28,", "29,", "30,", "31,",
                 "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", "13th", "14th",
                 "15th", "16th", "17th", "18th", "19th", "20th", "21st", "22nd", "23rd", "24th", "25th", "26th", "27th",
                 "28th", "29th", "30th", "31st", "1st,", "2nd,", "3rd,", "4th,", "5th,", "6th,", "7th,", "8th,", "9th,",
                 "10th,", "11th,", "12th,", "13th,", "14th,", "15th,", "16th,", "17th,", "18th,", "19th,", "20th,",
                 "21st,", "22nd,", "23rd,", "24th,", "25th,", "26th,", "27th,", "28th,", "29th,", "30th,", "31st,"]

# Possible digits, used to determine if a year exists as part of a date so it can all be tokenized
digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

# Common two-letter words that can commonly end the sentence and will not be confused for other abbreviations
common_two_letter_words = ["it", "be", "ab", "do", "ex", "go", "he", "hi", "is", "on", "ox", "so", "to", "up"]


# Function removes SGML tags from the documents
def removeSGML(input_doc):
    location_of_first_symbol = input_doc.find('<')      # Locates the defining "<" symbol, if it exists

    while location_of_first_symbol != -1:               # As long as the beginning carrot symbol exists
        location_of_end_carrot = input_doc.find('>')    # Try to see if there is a corresponding end carrot
        if location_of_end_carrot != -1:                # If there is a corresponding end carrot, delete the whole thing
            input_doc = input_doc[:location_of_first_symbol] + input_doc[location_of_end_carrot + 1:]
            location_of_first_symbol = input_doc.find('<')
        else:                                           # If no end carrot exists, SGML processing is done
            break

    return input_doc    # Returns the string without the SGML tags


def tokenizeText(input_string):
    # Since the regex only splits by whitespace here, words hyphenated together remain as one token
    tokens = re.split("\s", input_string)
    length = len(tokens)

    # Removes stand-alone empty characters from list of tokens because they don't count
    # Also removes parenthesis characters from tokens
    index = 0
    while index < length:
        item = tokens[index]
        if item == '':
            tokens.pop(index)
        else:
            if '(' in item:
                tokens[index] = tokens[index].replace('(', '')
            elif ')' in item:
                tokens[index] = tokens[index].replace(')', '')
            index = index + 1

        length = len(tokens)

    index = 0

    # Tokenizes dates with two different cases:
    # Case #1: Month and day only; conditions are:
    #   1. First token is a month (either full name or abbreviated)
    #   2. If there is only one more token after this token, this token is a day of the month,
    #      then only a month/day token will be formed
    # Case #2: Month and year only; conditions are:
    #   1. First token is a month (either full name or abbreviated)
    #   2. Second token is not a typical day, but contains a digit, so it's most likely a year
    # Case #3: Month, day, and year; conditions are:
    #   1. First token is a month (either full name or abbreviated)
    #   2. If there is at least one more token after this token, this token is a day of the month
    #   3. If there ANOTHER token in the list and this third token contains at least one digit, then
    #      all three are concatenated to form one specific date token
    index = 0
    while index < length:
        item = tokens[index]
        if (item in months or item in months_abbreviations) and index <= length - 2:
            if tokens[index + 1] in days_in_month:
                if index + 1 == length - 1:         # Month and day only
                    tokenized_date_month_day_only = tokens[index] + " " + tokens[index + 1]
                    tokens.pop(index)
                    tokens.pop(index)
                    tokens.append(tokenized_date_month_day_only)
                elif (index + 2 <= length - 1) and tokens[index + 2][0] in digits:  # Month, day, and year
                    tokenized_date_month_day_year = tokens[index] + " " + tokens[index + 1] + " " + tokens[index + 2]
                    tokens.pop(index)
                    tokens.pop(index)
                    tokens.pop(index)
                    tokens.append(tokenized_date_month_day_year)
            elif tokens[index + 1][0] in digits:    # Month and year only
                tokenized_month_year = tokens[index] + " " + tokens[index + 1]
                tokens.pop(index)
                tokens[index] = tokenized_month_year
            else:                             # Probably word that means month and other meaning (i.e., march or may)
                index = index + 1

        else:
            index = index + 1

        length = len(tokens)

    index = 0   # Reset the index for the next loop

    # Tokenizes apostrophes using a variety of cases
    # Case #1: Item directly found in hard-coded dictionary of contractions, then its corresponding expanded tokens
    #          replace the contraction
    # Case #2: Item not found in list of contractions, but it contains "'re", in which case its root and the word "are"
    #          replace the contraction; also this substring cannot be located at start of word like "O'reilly"
    # Case #3: Item not found in list of contractions, but it contains "'ll", in which case its root and the word "will"
    #          replace the contraction; also this substring cannot be located at start of word like "O'll"
    # Case #4: Item not found in list of contractions, but it contains "'s"
    #       Case #4a: The item is "he's", which is directly replaced by "he" and "is"
    #       Case #4b: The item is "she's", which is directly replaced by "she" and "is"
    #       Case #4c: The item is not "he's" or "she's" which makes it possessive; its replaced by its root and "'s"
    while index < length:
        item = tokens[index]
        if item in contractions:
            expanded = contractions[item]
            tokens.pop(index)
            for expanded_word in expanded:
                tokens.append(expanded_word)
        elif "'re" in item and item.find("'re", 0, len(item)) >= 2:
            end_of_root = item.find("'re", 0, len(item))
            root = item[0:end_of_root]
            tokens.pop(index)
            tokens.append("are")
            tokens.append(root)
        elif "'ll" in item and item.find("'ll", 0, len(item)) >= 2:
            end_of_root = item.find("'ll", 0, len(item))
            root = item[0:end_of_root]
            tokens.pop(index)
            tokens.append("will")
            tokens.append(root)
        elif "'s" in item and item != "'s":
            tokens.pop(index)
            if item == "he's":
                tokens.append("he")
                tokens.append("is")
            elif item == "she's":
                tokens.append("she")
                tokens.append("is")
            else:
                end_of_root = item.find("'s", 0, len(item))
                root = item[0:end_of_root]
                tokens.append("'s")
                tokens.append(root)
        else:
            index = index + 1

        length = len(tokens)

    # This loop systematically tokenizes commas when necessary
    # Conditions necessary to tokenize a comma from the end of a token:
    #   1. Contains a comma at the very end of token
    #   Prevents comma from within a number from being removed
    #   ex: 16,300 will not have its comma removed BUT
    #   "His annual income is $16,300, although..." will remove the end comma from the number at the end of the clause
    # However, if the entire token is a comma, it will not add another unnecessary comma to the list of tokens

    index = 0
    while index < length:
        item = tokens[index]
        if ',' in item and item.find(',', 0, len(item)) == len(item) - 1 and item != ',':
            tokens[index] = item[0: len(item) - 1]                       # Replaces token with the "comma-less" version
            tokens.append(',')                                           # Adds individual comma token to the list
        else:
            index = index + 1

        length = len(tokens)

    # This loop systematically tokenizes periods when necessary by separating and adding period to token list
    # Conditions necessary to tokenize a period from the end of a token:
    #   1. Be a common two-letter word that can be used to end a sentence
    #   2. Not be a word that can be used as an abbreviation commonly
    #   OR
    #   1. Word not in list of abbreviations or month abbreviations
    #   2. There is only one period in the string and it is at the very end of the string (if there was more than one
    #       period, it would also indicate some sort of acronym)

    index = 0
    while index < length:
        item = tokens[index]
        if item[0:len(item) - 1] in common_two_letter_words and item[len(item) - 1] == "." and item != '.':
            tokens[index] = item[0:len(item) - 1]
            tokens.append('.')
        elif (len(item) > 3) and (item not in abbreviations) and (item not in months_abbreviations) and \
             (item.find('.', 0, len(item)) == len(item) - 1) and (index <= length - 2) and item != '.' and \
                (',' not in item):
            tokens[index] = item[0:len(item) - 1]
            tokens.append('.')
        else:
            index = index + 1

        length = len(tokens)

    return tokens   # Return final tokenized list


# This function removes the stopwords according to the stopwords list provided
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


# Performs stemming of words using the PorterStemmer class code
def stemWords(list_of_tokens):
    stemmer = PorterStemmer()   # Declares the stemmer object
    for token_index, token in enumerate(list_of_tokens):
        list_of_tokens[token_index] = stemmer.stem(token, 0, len(token) - 1)    # Stems the word using the function

    return list_of_tokens   # Returns the "post-stem" list of tokens
