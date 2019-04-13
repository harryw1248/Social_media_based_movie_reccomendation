import pickle
import math
import collections
import time
import os
import sys
from inverted_index import *

# Class that holds a posting list, length of posting list, and max term frequency for any term in the inverted index
class PostingList:

    def __init__(self):
        self.posting_list = list()
        self.length = 1
        self.max_tf = 0.0


# For each element in a posting list, the document ID and the corresponding term frequency are stored
class PostingData:

    def __init__(self, movieID_in, tf_in):
        self.movieID = movieID_in
        self.tf = tf_in


# When the dictionary of document weights is created, each document maps to a vector of weights and its document length
class SimilarityData:
    def __init__(self, vocab_size):
        self.doc_length = 0
        self.weights = [0] * vocab_size


# Function that calculates the cosine similarity of document and query vectors
def calculateCosineSimilarity(doc_weights, query_weights, doc_length, query_length):
    dot_product = 0.0

    # Iteratively computes dot product by multiplying respective elements in query and document vectors and summing
    for index in range(0, len(doc_weights)):
        dot_product += doc_weights[index] * query_weights[index]

    # Final value is the dot product divided by the product of document and query lengths
    return dot_product / (doc_length * query_length)


# Returns ordered ranking of retrieved documents
def calculateDocumentSimilarity(doc_term_weightings, query_appearances, inverted_index, query_weights, query_length):
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


def sumVector(v1, v2):
    finalResult = [0.0] * len(v1)
    for elt in range(0, len(v1)):
        finalResult[elt] = v1[elt] + v2[elt]

    return finalResult


def ide_regular(alpha, beta, gamma, queryVec, doc_term_weightings, Dr, notDr):
    sumAllDj = [0.0] * len(doc_term_weightings[0].weights)

    for Dj in Dr:
        sumAllDj = sumVector(doc_term_weightings[Dj].weights, sumAllDj)

    leftSide = [0.0 for x in sumAllDj]
    if len(Dr) != 0:
        leftSide = [x * (beta) for x in sumAllDj]

    sumNotRelevant = [0.0] * len(doc_term_weightings[0].weights)

    for Dj in notDr:
        sumNotRelevant = sumVector(doc_term_weightings[Dj].weights, sumNotRelevant)

    rightSide = [0.0 for x in sumNotRelevant]
    if len(notDr) != 0:
        rightSide = [x * (gamma) for x in sumNotRelevant]

    finalVec = [0.0] * len(doc_term_weightings[0].weights)

    queryVec = [x * (alpha) for x in queryVec]

    for x in range(len(leftSide)):
        finalVec[x] = queryVec[x] + leftSide[x] - rightSide[x]

    return finalVec


def ide_dec_hi(alpha, beta, gamma, queryVec, doc_term_weightings, Dr, notDr):
    sumAllDj = [0.0] * len(doc_term_weightings[0].weights)

    for Dj in Dr:
        sumAllDj = sumVector(doc_term_weightings[Dj].weights, sumAllDj)

    leftSide = [x * (beta) for x in sumAllDj]
    sumNotRelevant = [0.0] * len(doc_term_weightings[0].weights)

    sumNotRelevant = notDr[0]

    rightSide = [x * (gamma) for x in sumNotRelevant]
    finalVec = [0.0] * len(doc_term_weightings[0].weights)

    queryVec = [x * (alpha) for x in queryVec]

    for x in range(len(leftSide)):
        finalVec[x] = queryVec[x] + leftSide[x] - rightSide[x]

    return finalVec


def rocchioModel(alpha, beta, gamma, queryVec, doc_term_weightings, Dr, notDr):
    sumAllDj = [0.0] * len(doc_term_weightings[0].weights)

    for Dj in Dr:
        sumAllDj = sumVector(doc_term_weightings[Dj].weights, sumAllDj)

    leftSide = [0.0 for x in sumAllDj]
    if len(Dr) != 0:
        leftSide = [x * (beta) / len(Dr) for x in sumAllDj]

    sumNotRelevant = [0.0] * len(doc_term_weightings[0].weights)

    for Dj in notDr:
        sumNotRelevant = sumVector(doc_term_weightings[Dj].weights, sumNotRelevant)

    rightSide = [0.0 for x in sumNotRelevant]
    if len(notDr) != 0:
        rightSide = [x * (gamma) / len(notDr) for x in sumNotRelevant]

    finalVec = [0.0] * len(doc_term_weightings[0].weights)

    queryVec = [x * (alpha) for x in queryVec]

    for x in range(len(leftSide)):
        finalVec[x] = queryVec[x] + leftSide[x] - rightSide[x]

    return finalVec


def kendallTau(vectorOne, vectorTwo, profile):
    pickle_in_kendall_tau = open("kendall_tau_data.pickle", "rb")
    kendall_tau_data = pickle.load(pickle_in_kendall_tau)

    tupVecOne = []
    for iter in range(len(vectorOne)):
        for iter2 in range(iter + 1, len(vectorOne)):
            tup = (vectorOne[iter],) + (vectorOne[iter2],)
            tupVecOne.append(tup)
    tupVecTwo = []
    for iter in range(len(vectorTwo)):
        for iter2 in range(iter + 1, len(vectorTwo)):
            tup = (vectorTwo[iter],) + (vectorTwo[iter2],)
            tupVecTwo.append(tup)
    x = 0.0
    y = 0.0

    for elt in tupVecOne:

        if elt in tupVecTwo:
            x = x + 1.0
        else:
            y = y + 1.0

    result = (x - y) / (x + y)
    kendall_tau_data.append(result)
    pickle_out_kendall_tau = open("kendall_tau_data.pickle", "wb")
    pickle.dump(kendall_tau_data, pickle_out_kendall_tau)
    print("Kendall Tau Value: " + str(result))


# MAP
def mean_average_precision(documents):
    mean_average_precisions = []

    if os.path.exists("mean_average_precisions.pickle"):
        mean_average_precisions = pickle.load(open("mean_average_precisions.pickle", "rb"))

    num_relevant_docs = 0
    running_total = 0
    precision_scores = []

    for doc in documents:
        if doc == 1:
            num_relevant_docs += 1

        running_total += 1
        precision = num_relevant_docs / running_total
        precision_scores.append(precision)

    mean_average_precision = 0
    if num_relevant_docs != 0:
        mean_average_precision = sum(precision_scores) / num_relevant_docs

    mean_average_precisions.append(mean_average_precision)
    pickle.dump(mean_average_precisions, open("mean_average_precisions.pickle", "wb"))

    return mean_average_precision


# R-precision
def r_precision(documents):
    r_precisions = []

    if os.path.exists("r_precisions.pickle"):
        mean_average_precisions = pickle.load(open("r_precisions.pickle", "rb"))

    most_recent_relevant_doc = 0
    running_total_documents = 0
    total_documents_R = 0

    for doc in documents:
        running_total_documents += 1
        if doc == 1:
            most_recent_relevant_doc += 1
            total_documents_R = running_total_documents

    r_precision = 0
    if most_recent_relevant_doc != 0:
        r_precision = most_recent_relevant_doc / total_documents_R

    r_precisions.append(r_precision)
    pickle.dump(r_precisions, open("r_precisions.pickle", "wb"))

    return r_precision


# MRR
def mean_reciprocal_rank(documents):
    mean_reciprocal_ranks = []

    if os.path.exists("mean_reciprocal_ranks.pickle"):
        mean_average_precisions = pickle.load(open("mean_reciprocal_ranks.pickle", "rb"))

    first_relevant_doc = 0
    running_total_documents = 0

    for doc in documents:
        running_total_documents += 1

        if doc == 1:
            first_relevant_doc = running_total_documents
            break

    mean_reciprocal_rank = 0
    if first_relevant_doc != 0:
        mean_reciprocal_rank = 1.0 / first_relevant_doc

    mean_reciprocal_ranks.append(mean_reciprocal_rank)
    pickle.dump(mean_reciprocal_ranks, open("mean_reciprocal_ranks.pickle", "wb"))

    return mean_reciprocal_rank


# movieIDs
def submit_feedback(user_relevance_info, profile, method_to_use="Rocchio"):
    pickle_in = open("recs.pickle", "rb")
    recs = pickle.load(pickle_in)

    alpha = 1.0
    beta = 1.0
    gamma = 1.0

    original_generated_ranking = [0] * 10
    user_ranking = [0] * 10
    relevantIDs = list()
    nonrelevantIDs = list()

    for elt in user_relevance_info:
        movieID, original_ranking, new_ranking, relevance = elt[0], elt[1], elt[2], elt[3]
        original_generated_ranking[original_ranking-1] = movieID
        user_ranking[new_ranking-1] = movieID
        if relevance:
            relevantIDs.append(movieID)
        else:
            nonrelevantIDs.append(movieID)

    kendallTau(original_generated_ranking, user_ranking, profile)






    mean_average_precision()
    mean_reciprocal_rank()
    r_precision()

    pickle_in = open("doc_term_weightings.pickle", "rb")
    doc_term_weightings = pickle.load(pickle_in)
    pickle_in = open("data/"+profile+"/query_weights.pickle", "rb")
    query_weights = pickle.load(pickle_in)
    new_query_weights = list()

    if method_to_use == "Rocchio":
        new_query_weights = rocchioModel(alpha, beta, gamma, query_weights, doc_term_weightings,
                                         relevantIDs, nonrelevantIDs)
    elif method_to_use == "IDE Dec Hi":
        new_query_weights = ide_dec_hi(alpha, beta, gamma, query_weights, doc_term_weightings,
                                       relevantIDs, nonrelevantIDs)
    elif method_to_use == "IDE Regular":
        new_query_weights = ide_regular(alpha, beta, gamma, query_weights, doc_term_weightings,
                                        relevantIDs, nonrelevantIDs)

    pickle_out = open("data/"+profile+"/query_weights.pickle", "wb")
    pickle.dump(new_query_weights, pickle_out)
    pickle_out.close()

