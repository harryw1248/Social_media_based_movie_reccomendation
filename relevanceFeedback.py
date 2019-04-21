'''
    relevanceFeedback.py

    Contains various relevance feedback methods (Rocchio, IDE Regular, IDE Dec Hi) used for query reformulation as
    well as functions that compute IR evaluation metrics (Kendall-Tau, MAP, MRR) and print the metrics to the console
    so they can be recorded. Everything is ran from the function submit_feedback(), which is called by the Flask app.
    Metrics will also be written to pickle files.
'''

import pickle
import math
import collections
import time
import os
import sys
from inverted_index import *


'''
    Sums two different vectors and returns the resulting vector
    
    Args:
        v1(list)
        v2(list)
    Return:
        finalResult(list)
'''
def sum_vector(v1, v2):
    finalResult = [0.0] * len(v1)
    for elt in range(0, len(v1)):
        finalResult[elt] = v1[elt] + v2[elt]

    return finalResult


'''
    Uses the IDE Regular query reformulation metric to alter the original query TF-IDF vector 
    
    Args:
        alpha(float)
        beta(float)
        gamma(float)
        query_vec(list of floats)
        doc_term_weightings(dict)
        Dr(list of ints)
        not_Dr(list of ints)
    
    Return:
        final_vec(list of floats)
'''
def ide_regular(alpha, beta, gamma, query_vec, doc_term_weightings, Dr, not_Dr):
    sum_relevant = [0.0] * len(doc_term_weightings[0].weights)
    sum_not_relevant = [0.0] * len(doc_term_weightings[0].weights)
    final_vec = [0.0] * len(doc_term_weightings[0].weights)

    # Sum all relevant document TF-IDF weights
    for doc_j in Dr:
        sum_relevant = sum_vector(doc_term_weightings[doc_j].weights, sum_relevant)

    # Sum all non-relevant document TF-IDF weights
    for doc_j in not_Dr:
        sum_not_relevant = sum_vector(doc_term_weightings[doc_j].weights, sum_not_relevant)

    # Iteratively compute each element using alpha, beta, and gamma as coefficients
    for index in range(0, len(query_vec)):
        final_vec[index] = alpha * query_vec[index] + beta * sum_relevant[index] - gamma * sum_not_relevant[index]

    return final_vec


'''
    Uses the IDE Dec Hi query reformulation metric to alter the original query TF-IDF vector 

    Args:
        alpha(float)
        beta(float)
        gamma(float)
        query_vec(list of floats)
        doc_term_weightings(dict)
        Dr(list of ints)
        not_Dr(list of ints)

    Return:
        final_vec(list of floats)
'''
def ide_dec_hi(alpha, beta, gamma, query_vec, doc_term_weightings, Dr, not_Dr):
    sum_relevant = [0.0] * len(doc_term_weightings[0].weights)
    sum_not_relevant = [0.0] * len(doc_term_weightings[0].weights)
    final_vec = [0.0] * len(doc_term_weightings[0].weights)

    # Sum all relevant document TF-IDF weights
    for doc_j in Dr:
        sum_relevant = sum_vector(doc_term_weightings[doc_j].weights, sum_relevant)

    # Check first to make sure that there is a non-relevant document to use
    # IDE Dec Hi subtracts the weights of the non-relevant doc with highest cosine similarity score
    if len(not_Dr):
        index_of_most_non_relevant = not_Dr[0]
        sum_not_relevant = doc_term_weightings[index_of_most_non_relevant].weights

    # Iteratively compute each element using alpha, beta, and gamma as coefficients
    for index in range(0, len(query_vec)):
        final_vec[index] = alpha * query_vec[index] + beta * sum_relevant[index] - gamma * sum_not_relevant[index]

    return final_vec


'''
    Uses the Rocchio query reformulation metric to alter the original query TF-IDF vector 

    Args:
        alpha(float)
        beta(float)
        gamma(float)
        query_vec(list of floats)
        doc_term_weightings(dict)
        Dr(list of ints)
        not_Dr(list of ints)

    Return:
        final_vec(list of floats)
'''
def rocchio(alpha, beta, gamma, query_vec, doc_term_weightings, Dr, not_Dr):
    sum_relevant = [0.0] * len(doc_term_weightings[0].weights)
    sum_not_relevant = [0.0] * len(doc_term_weightings[0].weights)
    final_vec = [0.0] * len(doc_term_weightings[0].weights)

    # Check to make sure there is at least one relevant doc
    if len(Dr):

        # Sums weights of all relevant docs
        for doc_j in Dr:
            sum_relevant = sum_vector(doc_term_weightings[doc_j].weights, sum_relevant)

        # Normalizes vector sum by dividing by number of non-relevant docs
        sum_relevant = [x/len(Dr) for x in sum_relevant]

    # Check to make sure there is at least one non-relevant doc
    if len(not_Dr):

        # Sums weights of all non-relevant docs
        for doc_j in not_Dr:
            sum_not_relevant = sum_vector(doc_term_weightings[doc_j].weights, sum_not_relevant)

        # Normalizes vector sum by dividing by number of non-relevant docs
        sum_not_relevant = [x/len(not_Dr) for x in sum_not_relevant]

    # Iteratively compute each element using alpha, beta, and gamma as coefficients
    for index in range(0, len(query_vec)):
        final_vec[index] = alpha * query_vec[index] + beta * sum_relevant[index] - gamma * sum_not_relevant[index]

    return final_vec


'''
    Computes the Kendall-Tau score between our system-generated ranking that used cosine similarity and the user-
    generated ranking that was given to the GUI during relevance feedback. 
    
    Args:
        vector_one (list of ints)
        vector_two (list of ints)
    
    Return:
        None
        
'''
def kendallTau(vector_one, vector_two):
    kendall_tau_data = list()

    # Check if pickle file already exists
    if os.path.exists("kendall_tau_data.pickle"):
        pickle_in_kendall_tau = open("kendall_tau_data.pickle", "rb")
        kendall_tau_data = pickle.load(pickle_in_kendall_tau)

    # Generate all possible pairs of two ranks for vector_one and append them to a list
    pairs_vec_one = []
    for iter in range(len(vector_one)):
        for iter2 in range(iter + 1, len(vector_one)):
            tup = (vector_one[iter],) + (vector_one[iter2],)
            pairs_vec_one.append(tup)

    # Generate all possible pairs of two ranks for vector_two and append them to a list
    pairs_vec_two = []
    for iter in range(len(vector_two)):
        for iter2 in range(iter + 1, len(vector_two)):
            tup = (vector_two[iter],) + (vector_two[iter2],)
            pairs_vec_two.append(tup)

    x = 0.0
    y = 0.0

    # Records number of agreements and disagreements between pairs in the two vectors
    for elt in pairs_vec_one:

        if elt in pairs_vec_two:
            x = x + 1.0
        else:
            y = y + 1.0

    # Calculates the actual result
    result = (x - y) / (x + y)

    # Write to pickle file and console
    kendall_tau_data.append(result)
    pickle_out_kendall_tau = open("kendall_tau_data.pickle", "wb")
    pickle.dump(kendall_tau_data, pickle_out_kendall_tau)
    print("Kendall Tau Value: " + str(result))


'''
    Computes the average precision for a certain query using user-generated relevance feedback. Takes in a list of 
    ints where 1 means the document at that rank/position was deemed relevant, whereas 0 means non-relevant.
    
    Args:
        documents(dict)
        
    Return:
        average_precision(float)
    
'''
def average_precision(documents):
    average_precisions = []

    # Check if pickle file already created
    if os.path.exists("mean_average_precision_data.pickle"):
        pickle_in_MAR = open("mean_average_precision_data.pickle", "rb")
        average_precisions = pickle.load(pickle_in_MAR)

    num_relevant_docs = 0.0
    running_total = 0.0
    precision_scores = []

    # Iterate through the documents vector and count number of relevant documents
    for doc in documents:
        if doc == 1:
            num_relevant_docs += 1.0

        running_total += 1.0
        precision = num_relevant_docs / running_total
        precision_scores.append(precision)

    average_precision = 0.0

    # If there is at least one relevant doc, AP = sum of each precision @ K divided by # relevant docs
    # If there are no relevant docs, AP = 0
    if num_relevant_docs != 0:
        average_precision = sum(precision_scores) / float(len(documents))

    # Record AP data
    average_precisions.append(average_precision)
    pickle.dump(average_precisions, open("mean_average_precision_data.pickle", "wb"))
    print("Average Precision: " + str(average_precision))

    return average_precision


'''
    Computes the reciprocal rank for a certain query using user-generated relevance feedback. Takes in a list of 
    ints where 1 means the document at that rank/position was deemed relevant, whereas 0 means non-relevant.

    Args:
        documents(dict)

    Return:
        reciprocal_rank(float)

'''
def reciprocal_rank(documents):
    reciprocal_ranks = []

    # Checks to make sure pickle file exists
    if os.path.exists("mrr_data.pickle"):
        pickle_in_mrr_data = open("mrr_data.pickle", "rb")
        reciprocal_ranks = pickle.load(pickle_in_mrr_data)

    first_relevant_doc = 0
    running_total_documents = 0

    # Searches for the first relevant document, and if it's found, breaks and records position
    for doc in documents:
        running_total_documents += 1

        if doc == 1:
            first_relevant_doc = running_total_documents
            break

    reciprocal_rank = 0.0
    # Checks to make sure there was at least one relevant document
    # RR is calculted as 1/(position of first relevant doc)
    if first_relevant_doc != 0:
        reciprocal_rank = 1.0 / first_relevant_doc


    # Record RR data
    reciprocal_ranks.append(reciprocal_rank)
    pickle.dump(reciprocal_ranks, open("mrr_data.pickle", "wb"))
    print("Reciprocal Rank: " + str(reciprocal_rank))

    return reciprocal_rank


'''
    Called by the Flask application after user enters in ranking and relevance judgments
    Calls function that compute IR evaluation metrics and call query reformulation method
'''
def submit_feedback(user_relevance_info, profile, method_to_use="Rocchio"):
    # Alpha, beta, and gamma always initialized to 1 for query reformulation
    alpha = 1.0
    beta = 1.0
    gamma = 1.0

    original_generated_ranking = [0] * 10
    user_ranking = [0] * 10
    relevantIDs = list()
    nonrelevantIDs = list()
    relevant_or_not_relevant = [0] * 10
    index = 0

    # Extracts relevancy information and compiles into two sorted vectors of relevant and non-relevant docs, where
    # lists are sorted by cosine similarity score
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

    # Computes metrics
    kendallTau(original_generated_ranking, user_ranking)
    average_precision(relevant_or_not_relevant)
    reciprocal_rank(relevant_or_not_relevant)

    # Opens up doc_term_weightings and original query vector from before relevance feedback
    pickle_in = open("doc_term_weightings.pickle", "rb")
    doc_term_weightings = pickle.load(pickle_in)
    pickle_in = open("data/"+profile+"/query_weights.pickle", "rb")
    query_weights = pickle.load(pickle_in)
    new_query_weights = list()

    past_feedback = list()

    # Writes to past feedback to hold collaborative filtering data
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

    # Query reformulation
    if method_to_use == "Rocchio":
        new_query_weights = rocchio(alpha, beta, gamma, query_weights, doc_term_weightings,
                                         relevantIDs, nonrelevantIDs)
    elif method_to_use == "IDE Dec Hi":
        new_query_weights = ide_dec_hi(alpha, beta, gamma, query_weights, doc_term_weightings,
                                       relevantIDs, nonrelevantIDs)
    elif method_to_use == "IDE Regular":
        new_query_weights = ide_regular(alpha, beta, gamma, query_weights, doc_term_weightings,
                                        relevantIDs, nonrelevantIDs)

    # Write new query weights and return
    pickle_out = open("data/"+profile+"/query_weights.pickle", "wb")
    pickle.dump(new_query_weights, pickle_out)
    pickle_out.close()
