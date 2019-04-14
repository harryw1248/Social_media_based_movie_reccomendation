import pickle
import math
import collections
import time
import os
import sys
from inverted_index import *
'''
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

'''
def sum_vector(v1, v2):
    finalResult = [0.0] * len(v1)
    for elt in range(0, len(v1)):
        finalResult[elt] = v1[elt] + v2[elt]

    return finalResult


def ide_regular(alpha, beta, gamma, query_vec, doc_term_weightings, Dr, not_Dr):
    sum_relevant = [0.0] * len(doc_term_weightings[0].weights)
    sum_not_relevant = [0.0] * len(doc_term_weightings[0].weights)
    final_vec = [0.0] * len(doc_term_weightings[0].weights)

    for doc_j in Dr:
        sum_relevant = sum_vector(doc_term_weightings[doc_j].weights, sum_relevant)

    for doc_j in not_Dr:
        sum_not_relevant = sum_vector(doc_term_weightings[doc_j].weights, sum_not_relevant)

    for index in range(0, len(query_vec)):
        final_vec[index] = alpha * query_vec[index] + beta * sum_relevant[index] - gamma * sum_not_relevant[index]

    return final_vec

def ide_dec_hi(alpha, beta, gamma, query_vec, doc_term_weightings, Dr, not_Dr):
    sum_relevant = [0.0] * len(doc_term_weightings[0].weights)
    sum_not_relevant = [0.0] * len(doc_term_weightings[0].weights)
    final_vec = [0.0] * len(doc_term_weightings[0].weights)

    for doc_j in Dr:
        sum_relevant = sum_vector(doc_term_weightings[doc_j].weights, sum_relevant)

    if len(not_Dr):
        index_of_most_non_relevant = not_Dr[len(not_Dr) - 1]
        sum_not_relevant = doc_term_weightings[index_of_most_non_relevant].weights

    for index in range(0, len(query_vec)):
        final_vec[index] = alpha * query_vec[index] + beta * sum_relevant[index] - gamma * sum_not_relevant[index]

    return final_vec


def rocchio(alpha, beta, gamma, query_vec, doc_term_weightings, Dr, not_Dr):
    sum_relevant = [0.0] * len(doc_term_weightings[0].weights)
    sum_not_relevant = [0.0] * len(doc_term_weightings[0].weights)
    final_vec = [0.0] * len(doc_term_weightings[0].weights)

    if len(Dr):
        for doc_j in Dr:
            sum_relevant = sum_vector(doc_term_weightings[doc_j].weights, sum_relevant)
        sum_relevant = [x/len(Dr) for x in sum_relevant]

    if len(not_Dr):
        for doc_j in not_Dr:
            sum_not_relevant = sum_vector(doc_term_weightings[doc_j].weights, sum_not_relevant)
        sum_not_relevant = [x/len(not_Dr) for x in sum_not_relevant]

    for index in range(0, len(query_vec)):
        final_vec[index] = alpha * query_vec[index] + beta * sum_relevant[index] - gamma * sum_not_relevant[index]

    return final_vec


def kendallTau(vectorOne, vectorTwo):
    kendall_tau_data = list()

    if os.path.exists("kendall_tau_data.pickle"):
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

    if os.path.exists("mean_average_precision_data.pickle"):
        pickle_in_MAR = open("mean_average_precision_data.pickle", "rb")
        mean_average_precisions = pickle.load(pickle_in_MAR)

    num_relevant_docs = 0
    running_total = 0
    precision_scores = []

    for doc in documents:
        if doc == 1:
            num_relevant_docs += 1

        running_total += 1
        precision = num_relevant_docs / running_total
        precision_scores.append(precision)

    average_precision = 0
    if num_relevant_docs != 0:
        average_precision = sum(precision_scores) / num_relevant_docs

    mean_average_precisions.append(average_precision)
    pickle.dump(mean_average_precisions, open("mean_average_precision_data.pickle", "wb"))
    print("Mean Average Precision: " + str(average_precision))

    return mean_average_precision


# R-precision
def r_precision(documents):
    r_precisions = []

    if os.path.exists("r_precision_data.pickle"):
        pickle_in_r_precision = open("r_precision_data.pickle", "rb")
        r_precisions = pickle.load(pickle_in_r_precision)

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
    pickle.dump(r_precisions, open("r_precision_data.pickle", "wb"))
    print("R-Precision: " + str(r_precision))

    return r_precision


# MRR
def mean_reciprocal_rank(documents):
    mean_reciprocal_ranks = []

    if os.path.exists("mrr_data.pickle"):
        pickle_in_mrr_data = open("mrr_data.pickle", "rb")
        mean_reciprocal_ranks = pickle.load(pickle_in_mrr_data)

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
    pickle.dump(mean_reciprocal_ranks, open("mrr_data.pickle", "wb"))
    print("Mean Reciprocal Rank: " + str(mean_reciprocal_rank))

    return mean_reciprocal_rank


# movieIDs
def submit_feedback(user_relevance_info, profile, method_to_use="Rocchio"):
    alpha = 1.0
    beta = 1.0
    gamma = 1.0

    original_generated_ranking = [0] * 10
    user_ranking = [0] * 10
    relevantIDs = list()
    nonrelevantIDs = list()
    relevant_or_not_relevant = [0] * 10
    index = 0

    for elt in user_relevance_info:
        movieID, original_ranking, new_ranking, relevance = elt[0], elt[1], elt[2], elt[3]
        original_generated_ranking[original_ranking-1] = movieID
        user_ranking[new_ranking - 1] = movieID
        if relevance:
            relevantIDs.append(movieID)
            relevant_or_not_relevant[index] = 1
        else:
            nonrelevantIDs.append(movieID)
            relevant_or_not_relevant[index] = 0

        index += 1

    print(relevant_or_not_relevant)

    kendallTau(original_generated_ranking, user_ranking)

    mean_average_precision(relevant_or_not_relevant)
    mean_reciprocal_rank(relevant_or_not_relevant)
    r_precision(relevant_or_not_relevant)

    pickle_in = open("doc_term_weightings.pickle", "rb")
    doc_term_weightings = pickle.load(pickle_in)
    pickle_in = open("data/"+profile+"/query_weights.pickle", "rb")
    query_weights = pickle.load(pickle_in)
    new_query_weights = list()

    past_feedback = list()

    if not os.path.exists("past_feedback.pickle"):
        past_feedback_in = open("past_feedback.pickle", "wb")
        past_feedback_in.close()
    else:
        try:
            past_feedback_in = open("past_feedback.pickle", "rb")
            past_feedback = pickle.load(past_feedback_in)
        except:
            past_feedback = []

    past_feedback_out = open("past_feedback.pickle", "wb")
    new_elt = (query_weights, relevantIDs, nonrelevantIDs)
    past_feedback.append(new_elt)
    pickle.dump(past_feedback, past_feedback_out)

    if method_to_use == "Rocchio":
        new_query_weights = rocchio(alpha, beta, gamma, query_weights, doc_term_weightings,
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

