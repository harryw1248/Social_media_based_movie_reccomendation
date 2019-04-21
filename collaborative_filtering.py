'''
This file implements the collaborative filtering mechanism.
The feature upweights the relevant movies of the previous user that is most similar to the current user
and downweights the irrelevant movies of the previous user that is most similar to the current user
'''
import numpy as np
import math

'''
Find the euclidean distance between the current query vector and a previous query vector
'''
def euclidean_distance(current_query, previous_query):

    distance = 0.0
    for index in range(0, len(current_query)):
        distance += (current_query[index] - previous_query[index]) ** 2

    distance = math.sqrt(distance)

    return distance

'''
Find the previous user that is the most similar to the current user
previous_queries is a list of previous queries where each item follows the following tuple format:
(query_vector_weights (list), list of relevant movie_ids, list of irrelevant movie_ids)
Returns a list of previously relevant movies and a list of previously irrelevant movies
'''
def find_nearest_neighbor(current_query, previous_queries):
    relevant_movie_ids = []
    irrelevant_movie_ids = []
    min_distance = np.inf

    for previous_query in previous_queries:
        query = previous_query[0]
        distance = euclidean_distance(current_query, query)
        if distance < min_distance:
            min_distance = distance
            relevant_movie_ids = previous_query[1]
            irrelevant_movie_ids = previous_query[2]

    return relevant_movie_ids, irrelevant_movie_ids


