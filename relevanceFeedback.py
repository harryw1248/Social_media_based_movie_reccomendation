import pickle
import math
import collections
import time
import os


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


def optimalQuery(doc_term_weightings, Cr, notCr):
    # relevant set of documents (list of Ints containing movieID
    # Total Number of docs in collection:
    N = 10
    sumAllDj = [0.0] * len(doc_term_weightings[0].weights)
    for Dj in Cr:
        # Dj is already a weight, if not I will modify Dj to be a weight
        sumAllDj = sumVector(doc_term_weightings[Dj].weights, sumAllDj)
    leftSide = [x * 1.0 / len(Cr) for x in sumAllDj]
    sumNotRelevant = [0.0] * len(doc_term_weightings[0].weights)
    for Dj in notCr:
        # Dj is already a weight, if not I will modify Dj to be a weight
        sumNotRelevant = sumVector(doc_term_weightings[Dj].weights, sumNotRelevant)

    rightSide = [x * 1.0 / (N - len(Cr)) for x in sumNotRelevant]

    final_result = [0.0] * len(leftSide)
    for elt in range(len(leftSide)):
        final_result[elt] = leftSide[elt] - rightSide[elt]
    return final_result


def createNewRecommendations(list_of_relevant, list_of_nonrelevant):
    inverted_index = dict()
    synopsis_image_info = dict()
    index_to_movies = dict()
    query_appearances = collections.Counter()  # Initialize counter to hold appearances of each query term
    doc_term_weightings = dict()

    pickle_in = open("doc_term_weightings.pickle", "rb")
    # Dict of Movie ID to Document Weight Vectors[Vector of Doubles]
    doc_term_weightings = pickle.load(pickle_in)
    pickle_in = open("inverted_index.pickle", "rb")
    inverted_index = pickle.load(pickle_in)
    pickle_in = open("index_to_movies.pickle", "rb")
    index_to_movies = pickle.load(pickle_in)
    pickle_in = open("query_appearances.pickle", "rb")
    query_appearances = pickle.load(pickle_in)
    pickle_in = open("synopsis_image_info.pickle", "rb")
    synopsis_image_info = pickle.load(pickle_in)
    pickle_in = open("query_weights.pickle", "rb")
    original_query_weights = pickle.load(pickle_in)

    query = optimalQuery(doc_term_weightings, list_of_relevant, list_of_nonrelevant)

    query_length = 0.0
    for elt in query:
        query_length += elt * elt
    query_length = math.sqrt(query_length)

    # After calculating query weights and length, returns ranked list of documents by calculating similarity
    docs_with_scores = calculateDocumentSimilarity(doc_term_weightings, query_appearances, inverted_index, query, query_length)
    ordered_list = sorted(docs_with_scores.items(), key=lambda x: x[1])  # Order the list
    t0 = time.time()
    print("\nTotal time to make recommendation:" + str(time.time() - t0) + " seconds")  # Print computation time
    print("Your Top 10 Movie Recommendations:\n")

    out_file = open(os.getcwd() + "/" + "recommendations.txt", 'w')
    out_file.write("Your Top 10 Movie Recommendations:\n")

    rank = 1
    for (movieID, score) in reversed(ordered_list):  # Print each ranking member to the output file
        movie_title = index_to_movies[movieID]

        out_file.write(str(rank) + ". " + movie_title + " " + str(score) + '\n')
        print(str(rank) + ". " + movie_title + " " + str(score) + '\n')
        print(synopsis_image_info[movieID][0] + '\n')
        out_file.write(synopsis_image_info[movieID][0] + '\n')

        rank += 1
        if rank == 11:
            break

    out_file.close()


if __name__ == '__main__':
    Cr = [3, 6, 87, 9]
    notCr = [234, 644, 99, 433, 233, 67]
    createNewRecommendations(Cr, notCr)