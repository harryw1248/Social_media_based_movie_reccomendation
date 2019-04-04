from preprocess import removeStopWords, stemWords
import math
import collections
import pickle
import string
import time
import nltk
import getpass
import calendar
import os
import platform
import sys
import urllib.request

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# Class that holds a posting list, length of posting list, and max term frequency for any term in the inverted index
class PostingList():

    def __init__(self):
        self.posting_list = list()
        self.length = 1
        self.max_tf = 0.0


# For each element in a posting list, the document ID and the corresponding term frequency are stored
class PostingData():

    def __init__(self, movieID_in, tf_in):
        self.movieID = movieID_in
        self.tf = tf_in

# When the dictionary of document weights is created, each document maps to a vector of weights and its document length
class SimilarityData:
    def __init__(self, vocab_size):
        self.doc_length = 0
        self.weights = [0] * vocab_size


# Global variable that I use to store document term weightings so I could obey function signatures
# Holds mapping of movieID to weights vector as well as length of document
doc_term_weightings = dict()

# Computes the document vector weights using weighting scheme #1: TF-IDF
def computeDocWeightsTFIDF(inverted_index, num_files):
    # Intializes global variable holding document term weightings to SimilarityData() objects
    for movieID in range(0, num_files + 1):
        doc_term_weightings[movieID] = SimilarityData(len(inverted_index))

    index = 0
    for term in inverted_index: # Iterates through each term in the inverted index
        for posting_data in inverted_index[term].posting_list:  # Iterates through each document in each posting list
            tf = posting_data.tf                        # tf is frequency of term in a document
            num_postings = inverted_index[term].length  # num_postings is document frequency
            movieID = posting_data.movieID
            if tf > 0.0:    # Only spend computational effort if term frequency is greater than zero
                # tf-idf = tf * idf
                idf = math.log10(num_files / num_postings)
                doc_term_weightings[movieID].weights[index] = tf * idf
                doc_term_weightings[movieID].doc_length += ((tf * idf) * (tf * idf))  # Update running total for length
            else:
                continue
        index += 1  # Increment index to move to next term

    for movieID in range(1, num_files + 1):   # Finish calculating doc lengths
        doc_term_weightings[movieID].doc_length = math.sqrt(doc_term_weightings[movieID].doc_length)

# Computes the document vector weights using weighting scheme #2: Best-Weighting Probabilistic Weight
def computeDocWeightsBWPW(inverted_index, num_files):
    # Intializes global variable holding document term weightings to SimilarityData() objects
    for movieID in range(1, num_files + 1):
        doc_term_weightings[movieID] = SimilarityData(len(inverted_index))

    index = 0
    for term in inverted_index:  # Iterates through each term in the inverted index
        max_tf = inverted_index[term].max_tf + 0.0  # Finds the maximum term frequency for each term
        for posting_data in inverted_index[term].posting_list:  # Iterates through each document in each posting list
            tf = posting_data.tf + 0.0
            movieID = posting_data.movieID
            weight = 0.5 + 0.5 * (tf/max_tf)    # Weight is calculated as (0.5 + 0.5 * (tf/max_tf))
            doc_term_weightings[movieID].weights[index] = weight
            doc_term_weightings[movieID].doc_length += weight * weight    # Running total of document length updated
        index += 1  # Increment index to move to next term

    for movieID in range(1, num_files + 1):   # Finally, each document length is finally calculated by taking square root
        doc_term_weightings[movieID].doc_length = math.sqrt(doc_term_weightings[movieID].doc_length)


# Function that calculates the cosine similarity of document and query vectors
def calculateCosineSimilarity(doc_weights, query_weights, doc_length, query_length):
    dot_product = 0.0

    # Iteratively computes dot product by multiplying respective elements in query and document vectors and summing
    for index in range(0, len(doc_weights)):
        dot_product += doc_weights[index] * query_weights[index]

    # Final value is the dot product divided by the product of document and query lengths
    return dot_product/(doc_length * query_length)


# Creates the inverted index by initializing each term in the corpus
def createInvertedIndex(inverted_index, movieID, tokens):
    for token in tokens:    # Iterate through each token in the document
        if token in inverted_index: # If the token is already in the inverted index
            current_posting_list_length = inverted_index[token].length  # Calculate length of posting list so far
            # If the last entry in the posting list is the same document, just increment the term frequency
            if inverted_index[token].posting_list[current_posting_list_length - 1].movieID == movieID:
                inverted_index[token].posting_list[current_posting_list_length - 1].tf += 1
                new_tf = inverted_index[token].posting_list[current_posting_list_length - 1].tf
                # Update the max term frequency as needed
                if inverted_index[token].max_tf < new_tf:
                    inverted_index[token].max_tf = new_tf
            # If movieID not yet part of this posting list, add it to the end and increase the length of list by one
            else:
                inverted_index[token].posting_list.append(PostingData(movieID, 1))
                inverted_index[token].length += 1
        # However, if term not yet in the index, create a whole new posting list and make this movieID the first entry
        else:
            inverted_index[token] = PostingList()
            inverted_index[token].posting_list.append(PostingData(movieID, 1))
            inverted_index[token].max_tf = 1


# Calculates weights for query vector using tf-idf scheme
def calculateQueryDataTFIDF(query_weights, query_appearances, query_length, inverted_index):
    num_files = len(doc_term_weightings) + 0.0
    l = list(inverted_index.keys())

    # Iterate through each term in the query vector and assign nonzero weight if the term appears in inverted index
    for query_term in query_appearances:
        if query_term in inverted_index:
            index_of_word = l.index(query_term)     # Since ordered dict, calculate index of term
            num_postings = inverted_index[query_term].length + 0.0      # Document frequency
            idf = math.log10(num_files / num_postings)                  # Inverse document frequency
            tf = query_appearances[query_term]                          # Term frequency
            query_weights[index_of_word] = tf * idf                     # Query weight
            query_length += (tf * idf) * (tf * idf)                     # Update running total for query length

    query_length = math.sqrt(query_length)                              # Calculate final query length
    return query_length


def calculateQueryDataBWPW( query_weights, query_appearances, query_length, inverted_index):
    num_files = len(doc_term_weightings) + 0.0

    # Iterate through each term in the query vector and assign nonzero weight if the term appears in inverted index
    # weight = log ((N - n)/n), where N = number of documents and n = document frequency
    for query_term in query_appearances:
        if query_term in inverted_index:
            index_of_word = inverted_index.keys().index(query_term)     # Since ordered dict, calculate index of term
            num_postings = inverted_index[query_term].length + 0.0      # Document frequency
            query_weights[index_of_word] = math.log10((num_files - num_postings)/num_postings)
            query_length += query_weights[index_of_word] * query_weights[index_of_word]

    query_length = math.sqrt(query_length)                              # Calculate final query length
    return query_length


# Returns ordered ranking of retrieved documents
def calculateDocumentSimilarity(query_appearances, inverted_index, query_weights, query_length):
    docs_with_at_least_one_matching_query_term = set()
    docs_with_scores = dict()

    # Use a set to hold every movieID in which at least one query term appears
    for query_term in query_appearances:
        if query_term in inverted_index:
            for posting_data in inverted_index[query_term].posting_list:
                docs_with_at_least_one_matching_query_term.add(posting_data.movieID)

    # For each movieID in the set, calculate the cosine similarity and store in a map of movieID to similarity value
    for movieID in docs_with_at_least_one_matching_query_term:
        docs_with_scores[movieID] = calculateCosineSimilarity(doc_term_weightings[movieID].weights, query_weights,
                                                            doc_term_weightings[movieID].doc_length, query_length)

    return docs_with_scores


def indexDocument(document, doc_weighting_scheme, inverted_index, movieID, use_kaggle = False):
    tokens = nltk.word_tokenize(document)

    if use_kaggle:
        pickle_in = open("kaggle.pickle", "rb")
        kaggle_dict = pickle.load(pickle_in)

        for (movieID, title) in index_to_movies:
            if title in kaggle_dict.keys():
                for keyword in kaggle_dict[title]:
                    tokens.append(keyword)

    tokens = [x for x in tokens if x not in string.punctuation]
    tokens = removeStopWords(tokens)  # Remove the stopwords
    tokens = stemWords(tokens)      # PorterStemmer

    for i in range(0, len(tokens)):
        tokens[i] = tokens[i].lower()

    createInvertedIndex(inverted_index, movieID, tokens)   # Create the inverted index


def retrieveDocuments(query, inverted_index, doc_weighting_scheme, query_weighting_scheme):
    tokens = nltk.word_tokenize(query)
    tokens = [x for x in tokens if x not in string.punctuation]
    query_tokens = removeStopWords(tokens)  # Remove the stopwords
    query_tokens = stemWords(query_tokens)

    out_file_query = open("query_tokens.txt", 'w')

    for i in range(0, len(query_tokens)):
        query_tokens[i] = query_tokens[i].lower()
        out_file_query.write(query_tokens[i] + '\n')

    query_weights = [0] * len(inverted_index)   # Initialize vector to hold query weights
    query_appearances = collections.Counter()   # Initialize counter to hold appearances of each query term
    for query_token in query_tokens:
        query_appearances[query_token] += 1
    query_length = 0.0

    # Use query weighting scheme to appropriately calculate query term weights
    if query_weighting_scheme == "tfidf":
        query_length = calculateQueryDataTFIDF(query_weights, query_appearances, query_length, inverted_index)
    elif query_weighting_scheme == "bwpw":
        query_length = calculateQueryDataBWPW(query_weights, query_appearances, query_length, inverted_index)

    pickle_out = open("query_vector.pickle", "wb")
    pickle_out.write(query_weights)
    pickle_out.close()

    # After calculating query weights and length, returns ranked list of documents by calculating similarity
    return calculateDocumentSimilarity(query_appearances, inverted_index, query_weights, query_length)


def create_data(doc_weighting_scheme, inverted_index, num_files, use_kaggle=False):
    current_index = 0

    for filename in os.listdir(os.getcwd() + "/" + doc_folder):  # Iterates through each doc in passed-in folder
        file = open(os.getcwd() + "/" + doc_folder + filename, 'r')  # Open the file

        if filename == ".DS_Store":
            continue

        index1 = 7
        index2 = filename.find(".")
        movie_title = filename[index1:index2]

        if "_" in movie_title:
            movie_title = movie_title.replace("_", ":")

        if ", The" in movie_title:
            index = movie_title.find(", The")
            movie_title = "The " + movie_title[0: index]

        print(movie_title + ", Index: " + str(current_index))

        index_to_movies[current_index] = movie_title
        line = file.read()
        movieID = current_index
        indexDocument(line, doc_weighting_scheme, inverted_index, movieID, use_kaggle)  # Update the inverted index
        file.close()
        num_files += 1
        current_index = current_index + 1

    # Once inverted index is complete, compute document weights using the appropriate weighting scheme
    if doc_weighting_scheme == "tfidf":
        computeDocWeightsTFIDF(inverted_index, num_files)
    elif doc_weighting_scheme == "bwpw":
        computeDocWeightsBWPW(inverted_index, num_files)

    pickle_out1 = open("inverted_index.pickle", "wb")
    pickle.dump(inverted_index, pickle_out1)
    pickle_out1.close()
    pickle_out2 = open("doc_term_weightings.pickle", "wb")
    pickle.dump(doc_term_weightings, pickle_out2)
    pickle_out2.close()
    pickle_out3 = open("index_to_movies.pickle", "wb")
    pickle.dump(index_to_movies, pickle_out3)
    pickle_out3.close()


def get_metadata(synopsis_info, index_to_movies):
    """ Logging into our own profile """

    # try:
    global driver

    options = Options()

    chromedriver = "/Users/Vinchenzo4335/PycharmProjects/EECS486/Final_Project/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver
    # chrome_options.add_argument("--headless")
    options.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

    #  Code to disable notifications pop up of Chrome Browser
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-infobars")
    options.add_argument("--mute-audio")
    # options.add_argument("headless")

    driver = webdriver.Chrome(executable_path=chromedriver, options=options)
    num_found = 0
    movie_num = 1

    for movieID in index_to_movies:
        movie_title = index_to_movies[movieID]

        if "_" in movie_title:
            movie_title = movie_title.replace("_", ":")

        if ", The" in movie_title:
            index = movie_title.find(", The")
            movie_title = "The " + movie_title[0: index]

        index_to_movies[movieID] = movie_title

        movie_title = movie_title.lower()
        movie_title = movie_title.replace(" ", "_")

        if ":" in movie_title:
            movie_title = movie_title.replace(":", "")

        if "'" in movie_title:
            movie_title = movie_title.replace("'", "")

        if "-" in movie_title:
            movie_title = movie_title.replace("-", "")

        if "&" in movie_title:
            movie_title = movie_title.replace("&", "and")

        try:
            driver.get("https://www.rottentomatoes.com/m/" + movie_title)
            synopsis = driver.find_element_by_id('movieSynopsis').text
            #print(str(movie_num) + " " + movie_title + " FOUND")
            synopsis_info[movieID] = synopsis
            num_found += 1
        except:
            try:
                if movie_title[0:4] == "the_":
                    movie_title = movie_title[4: len(movie_title)]
                driver.get("https://www.rottentomatoes.com/m/" + movie_title)
                synopsis = driver.find_element_by_id('movieSynopsis').text
                synopsis_info[movieID] = synopsis
                num_found += 1
            except:
                synopsis_info[movieID] = "No synopsis found."
                print(str(movie_num) + " " + movie_title + " NOT FOUND")
                movie_num += 1
                continue

            synopsis_info[movieID] = "No synopsis found."
            print(str(movie_num) + " " +movie_title + " NOT FOUND")

        movie_num += 1

    print(num_found)

    pickle_out1 = open("synopsis_info.pickle", "wb")
    pickle_out1.write(synopsis_info)
    pickle_out1.close()

    pickle_out2 = open("index_to_movies.pickle", "wb")
    pickle_out2.write(index_to_movies)
    pickle_out2.close()

if __name__ == '__main__':

    queries = "Posts.txt"
    create_index = False
    read_in_synopsis_info = True
    use_kaggle = False

    index_to_movies = dict()
    synopsis_info = dict()
    doc_weighting_scheme = "tfidf"
    inverted_index = collections.OrderedDict()  # Inverted index is ordered dictionary to allow for consistent indexing
    doc_folder = "Testing/"
    num_files = 0

    if create_index:
        create_data(doc_weighting_scheme, inverted_index, num_files, use_kaggle)

    else:
        pickle_in = open("inverted_index.pickle", "rb")
        inverted_index = pickle.load(pickle_in)
        pickle_in = open("doc_term_weightings.pickle", "rb")
        doc_term_weightings = pickle.load(pickle_in)
        pickle_in = open("index_to_movies.pickle", "rb")
        index_to_movies = pickle.load(pickle_in)

    if read_in_synopsis_info:
        get_metadata(synopsis_info, index_to_movies)

    else:
        pickle_in = open("synopsis_info.pickle", "rb")
        synopsis_info = pickle.load(pickle_in)


    t0 = time.time()
    query_weighting_scheme = "tfidf"

    query_doc = open(os.getcwd() + "/" + queries, 'r')  # Open the file
    out_file = open(os.getcwd() + "/" + "recommendations.txt", 'w')
    line = query_doc.read()
    query_num = 1

    docs_with_scores = retrieveDocuments(line, inverted_index, doc_weighting_scheme, query_weighting_scheme)
    ordered_list = sorted(docs_with_scores.items(), key=lambda x: x[1])  # Order the list


    rank = 1
    print("\nTotal time to make recommendation:" + str(time.time() - t0) + " seconds")    # Print computation time
    print("Your Top 10 Movie Recommendations:\n")
    out_file.write("Your Top 10 Movie Recommendations:\n")

    for (movieID, score) in reversed(ordered_list):  # Print each ranking member to the output file
        movie_title = index_to_movies[movieID]

        out_file.write(str(rank) + ". " + movie_title + " " + str(score) + '\n')
        print(str(rank) + ". " + movie_title+ " " + str(score) + '\n')
        print(synopsis_info[movie_title] + '\n')
        out_file.write(synopsis_info[movie_title] + '\n')

        rank += 1
        if rank == 11:
            break

    out_file.close()
    query_doc.close()


