# Vinayak Ahluwalia
# uniqname: vahluw
import nltk
from nltk import word_tokenize,sent_tokenize
from preprocess import*
import os
import math
import collections
import re
import pickle
import string

index_to_movies = dict()
index_to_movies[1] = "The Avengers"
index_to_movies[2] = "Venom"

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
    for movieID in range(1, num_files + 1):
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

    # Iterate through each term in the query vector and assign nonzero weight if the term appears in inverted index
    for query_term in query_appearances:
        if query_term in inverted_index:
            index_of_word = inverted_index.keys().index(query_term)     # Since ordered dict, calculate index of term
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


def indexDocument(document, doc_weighting_scheme, inverted_index, movieID):
    #tokens = nltk.word_tokenize(document)
    tokens = tokenizeText(document)
    tokens = [x for x in tokens if x not in string.punctuation]
    tokens = removeStopWords(tokens)  # Remove the stopwords
    tokens = stemWords(tokens)      # PorterStemmer

    createInvertedIndex(inverted_index, movieID, tokens)   # Create the inverted index


def retrieveDocuments(query, inverted_index, doc_weighting_scheme, query_weighting_scheme):
    #tokens = nltk.word_tokenize(query)
    tokens = tokenizeText(query)
    tokens = [x for x in tokens if x not in string.punctuation]
    query_tokens = removeStopWords(tokens)  # Remove the stopwords
    query_tokens = stemWords(query_tokens)

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

    # After calculating query weights and length, returns ranked list of documents by calculating similarity
    return calculateDocumentSimilarity(query_appearances, inverted_index, query_weights, query_length)


if __name__ == '__main__':
    inverted_index = collections.OrderedDict()  # Inverted index is ordered dictionary to allow for consistent indexing
    num_files = 0
    doc_folder = "movies/"
    doc_weighting_scheme = "tfidf"

    for filename in os.listdir(os.getcwd() + "/" + doc_folder):         # Iterates through each doc in passed-in folder
        file = open(os.getcwd() + "/" + doc_folder + filename, 'r')     # Open the file

        if filename == ".DS_Store":
            continue

        line = file.read()                                              # Read the file
        movieID = int(''.join(ch for ch in filename if ch.isdigit()))
        indexDocument(line, doc_weighting_scheme, inverted_index, movieID)    # Update the inverted index
        file.close()
        num_files += 1

    # Once inverted index is complete, compute document weights using the appropriate weighting scheme
    if doc_weighting_scheme == "tfidf":
        computeDocWeightsTFIDF(inverted_index, num_files)
    elif doc_weighting_scheme == "bwpw":
        computeDocWeightsBWPW(inverted_index, num_files)

    print(inverted_index)
    pickle_out = open("harry_is_dumb.pickle", "wb")
    pickle.dump(inverted_index, pickle_out)
    pickle_out.close()
    pickle_in = open("harry_is_dumb.pickle", "rb")
    example = pickle.load(pickle_in)
    print(example)
    print(example["face"].posting_list[0].movieID)

    '''
    # Dictionary to store relevance judgments for each queries from cranfield.reljudge
    relevance_judgments = dict()
    reljudge = open('cranfield.reljudge', 'r')
    input_string = reljudge.readline()

    # Create mapping of each query number to relevant documents
    while input_string:
        tokens = re.split("\s", input_string)
        query_num = int(tokens[0])
        movieID = int(tokens[1])

        if query_num not in relevance_judgments:
            relevance_judgments[query_num] = list()
        relevance_judgments[query_num].append(movieID)
        input_string = reljudge.readline()

    reljudge.close()

    out_file = open('cranfield.' + doc_weighting_scheme + '.' + query_weighting_scheme + '.' + 'output', 'w+')
    query_doc = open(os.getcwd() + "/" + queries, 'r')  # Open the file

    line = query_doc.readline()
    query_num = 1

    # Variables that will hold different metric values
    precision_total_before_macro_average_10 = 0.0
    recall_total_before_macro_average_10 = 0.0
    precision_total_before_macro_average_50 = 0.0
    recall_total_before_macro_average_50 = 0.0
    precision_total_before_macro_average_100 = 0.0
    recall_total_before_macro_average_100 = 0.0
    precision_total_before_macro_average_500 = 0.0
    recall_total_before_macro_average_500 = 0.0
    num_retrieved = [10, 50, 100, 500]  # Different document ranking quantities for each metric

    while line:     # Keep calculating similarities as long as queries are being passed in
        # Retreive document ranking
        docs_with_scores = retrieveDocuments(line, inverted_index, doc_weighting_scheme, query_weighting_scheme)
        ordered_list = sorted(docs_with_scores.items(), key=lambda x:x[1])  # Order the list

        num_relevant = len(relevance_judgments[query_num])

        for (movieID, score) in reversed(ordered_list):       # Print each ranking member to the output file
            out_file.write(str(query_num) + " " + str(movieID) + " " + str(score) + '\n')

        for max_retrieved in num_retrieved:
            num = 0
            num_relevant_retrieved = 0

            for (movieID, score) in reversed(ordered_list):  # Calculate relevant docs retrieved for each quantity
                if movieID in relevance_judgments[query_num] and num < max_retrieved:
                    num_relevant_retrieved += 1
                num += 1

            # Update running totals for appropriate metrics
            # Since macro averaging is used, the precision/recall values for each query are added to a running total
            # that will be divided by the number of queries to compute the final macro average
            if max_retrieved == 10:
                precision_total_before_macro_average_10 += num_relevant_retrieved/float(max_retrieved)
                recall_total_before_macro_average_10 += num_relevant_retrieved / float(num_relevant)
            elif max_retrieved == 50:
                precision_total_before_macro_average_50 += num_relevant_retrieved / float(max_retrieved)
                recall_total_before_macro_average_50 += num_relevant_retrieved / float(num_relevant)
            elif max_retrieved == 100:
                precision_total_before_macro_average_100 += num_relevant_retrieved / float(max_retrieved)
                recall_total_before_macro_average_100 += num_relevant_retrieved / float(num_relevant)
            elif max_retrieved == 500:
                precision_total_before_macro_average_500 += num_relevant_retrieved / float(max_retrieved)
                recall_total_before_macro_average_500 += num_relevant_retrieved / float(num_relevant)

        line = query_doc.readline()
        query_num = query_num + 1

    out_file.close()
    query_doc.close()

    # Calculate final metrics by dividing by number of queries and print results to console
    final_precision_10 = precision_total_before_macro_average_10/float(query_num)
    final_recall_10 = recall_total_before_macro_average_10/float(query_num)
    final_precision_50 = precision_total_before_macro_average_50/float(query_num)
    final_recall_50 = recall_total_before_macro_average_50/float(query_num)
    final_precision_100 = precision_total_before_macro_average_100/float(query_num)
    final_recall_100 = recall_total_before_macro_average_100/float(query_num)
    final_precision_500 = precision_total_before_macro_average_500/float(query_num)
    final_recall_500 = recall_total_before_macro_average_500/float(query_num)

    print ("Precision for Top 10 Documents: " + str(final_precision_10) + "\n")
    print ("Recall for Top 10 Documents: " + str(final_recall_10) + "\n")

    print ("Precision for Top 50 Documents: " + str(final_precision_50) + "\n")
    print ("Recall for Top 50 Documents: " + str(final_recall_50) + "\n")

    print ("Precision for Top 100 Documents: " + str(final_precision_100) + "\n")
    print ("Recall for Top 100 Documents: " + str(final_recall_100) + "\n")

    print ("Precision for Top 500 Documents: " + str(final_precision_500) + "\n")
    print ("Recall for Top 500 Documents: " + str(final_recall_500) + "\n")
    '''