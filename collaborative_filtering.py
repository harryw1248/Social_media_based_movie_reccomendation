'''
input data:
list of previous queries that where each item was
(query_vector_weights, list of relevant movie_ids, list of irrelevant movie_ids);
current query
'''
import numpy as np
import math

def euclidean_distance(current_query, previous_query):

    distance = 0.0
    for index in range(0, len(current_query)):
        distance += (current_query[index] - previous_query[index]) ** 2

    distance = math.sqrt(distance)

    return distance

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


