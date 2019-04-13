from preprocess import removeStopWords, stemWords
import math
import collections
import pickle
import string
import time
import nltk
from nltk.corpus import wordnet
import os
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from relevanceFeedback import *
import collaborative_filtering as cf


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


# Computes the document vector weights using weighting scheme #1: TF-IDF
def computeDocWeightsTFIDF(inverted_index, num_files, doc_term_weightings):
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
def calculateQueryDataTFIDF(query_string, inverted_index, num_files, profile):
    tokens = nltk.word_tokenize(query_string)
    tokens = [x for x in tokens if x not in string.punctuation]
    query_tokens = removeStopWords(tokens)  # Remove the stopwords
    query_tokens = [x for x in query_tokens if (wordnet.synsets(x) and x != "birthday" and x != "bday")]
    query_tokens = stemWords(query_tokens)

    for i in range(0, len(query_tokens)):
        query_tokens[i] = query_tokens[i].lower()

    query_appearances = collections.Counter()
    query_weights = [0] * len(inverted_index)  # Initialize vector to hold query weights
    for query_token in query_tokens:
        query_appearances[query_token] += 1
    query_length = 0.0

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

    pickle_out = open("data/"+profile+"/query_appearances.pickle", "wb")
    pickle.dump(query_appearances, pickle_out)
    pickle_out.close()

    pickle_out2 = open("data/" + profile + "/query_weights.pickle", "wb")
    pickle.dump(query_weights, pickle_out2)
    pickle_out2.close()

    return (query_weights, query_length, query_appearances)


# Returns ordered ranking of retrieved documents
def calculateDocumentSimilarity(query_appearances, inverted_index, query_weights, query_length, doc_term_weightings,
                                upvoting_factor=1.05, downvoting_factor=0.95):
   docs_with_at_least_one_matching_query_term = set()
   docs_with_scores = dict()

   # Use a set to hold every movieID in which at least one query term appears
   for query_term in query_appearances:
       if query_term in inverted_index:
           for posting_data in inverted_index[query_term].posting_list:
               docs_with_at_least_one_matching_query_term.add(posting_data.movieID)

   past_feedback = list()
   if not os.path.exists("past_feedback.pickle"):
       past_feedback_in = open("past_feedback.pickle", "wb")
       past_feedback_in.close()

   else:
       try:
           past_feedback_in = open("past_feedback.pickle", "rb")
           past_feedback = pickle.load(past_feedback_in)
       except:
           past_feedback = list()

   #collaborative filtering: find nearest neighbor (previous user), extract list of relevant and irrelevant movies
   if len(past_feedback):
        relevant_movie_ids, irrelevant_movie_ids = cf.find_nearest_neighbor(query_weights, past_feedback)
   else:
       relevant_movie_ids = irrelevant_movie_ids = []

   # For each movieID in the set, calculate the cosine similarity and store in a map of movieID to similarity value
   for movieID in docs_with_at_least_one_matching_query_term:
       docs_with_scores[movieID] = calculateCosineSimilarity(doc_term_weightings[movieID].weights, query_weights,
                                                           doc_term_weightings[movieID].doc_length, query_length)
       if movieID in relevant_movie_ids:
           docs_with_scores[movieID] *= upvoting_factor

       if movieID in irrelevant_movie_ids:
           docs_with_scores[movieID] *= downvoting_factor

   return docs_with_scores


def indexDocument(document, inverted_index, movieID):
    tokens = nltk.word_tokenize(document)
    tokens = [x for x in tokens if x not in string.punctuation]
    tokens = removeStopWords(tokens)  # Remove the stopwords
    tokens = stemWords(tokens)      # PorterStemmer

    for i in range(0, len(tokens)):
        tokens[i] = tokens[i].lower()

    createInvertedIndex(inverted_index, movieID, tokens)   # Create the inverted index


def retrieveDocuments(profile, query, inverted_index, doc_term_weightings):
    query_weights = list()
    query_appearances = collections.Counter()
    query_length = 0.0

    if not os.path.exists("data/"+profile+"/query_appearances.pickle") and not\
        os.path.exists("data/"+profile+"/query_weights.pickle"):
        query_appearances = collections.Counter()
        num_files = len(doc_term_weightings) + 0.0
        (query_weights, query_length, query_appearances) = calculateQueryDataTFIDF(query, inverted_index, num_files, profile)

    else:
        pickle_in = open("data/"+profile+"/query_weights.pickle", "rb")
        query_weights = pickle.load(pickle_in)
        pickle_in = open("data/"+profile+"/query_appearances.pickle", "rb")
        query_appearances = pickle.load(pickle_in)

        for elt in query_weights:
            query_length += elt * elt

        query_length= math.sqrt(query_length)

    # After calculating query weights and length, returns ranked list of documents by calculating similarity
    return calculateDocumentSimilarity(query_appearances, inverted_index,
                                       query_weights, query_length, doc_term_weightings)


def correct_title_exceptions(title):
    title = title.replace(" ", "_")

    if ":" in title:
        title = title.replace(":", "")

    if "'" in title:
        title = title.replace("'", "")

    if "-" in title:
        title = title.replace("-", "")

    if "&" in title:
        title = title.replace("&", "and")

    if "__" in title:
        title = title.replace("__", "_")

    if title == "frozen_(disney)":
        title = "frozen_2013"

    elif title == "the_avengers_(2012)":
        title = "marvels_the_avengers"

    elif title == "star_wars_the_phantom_menace":
        title = "star_wars_episode_i_the_phantom_menace"

    elif title == "star_wars_attack_of_the_clones":
        title = "star_wars_episode_ii_attack_of_the_clones"

    elif title == "star_wars_revenge_of_the_sith":
        title = "star_wars_episode_iii_revenge_of_the_sith"

    elif title == "star_wars_a_new_hope":
        title = "star_wars"

    elif title == "star_wars_the_empire_strikes_back":
        title = "empire_strikes_back"

    elif title == "star_wars_return_of_the_jedi":
        title = "star_wars_episode_vi_return_of_the_jedi"

    elif title == "star_wars_the_force_awakens":
        title = "star_wars_episode_vii_the_force_awakens"

    elif title == "semipro":
        title = "semi_pro"

    return title

def create_data(inverted_index, num_files, synopsis_image_info, index_to_movies, doc_term_weightings):
    current_index = 0
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
    movie_num = 0
    pickle_tags = open("movie_attributes.pickle", "rb")
    movie_attributes = pickle.load(pickle_tags)

    doc_folder = ""

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

        if ", A" in movie_title:
            index = movie_title.find(", A")
            movie_title = "A " + movie_title[0: index]

        print(movie_title + ", Index: " + str(current_index))

        index_to_movies[current_index] = movie_title
        script = file.read()
        movieID = current_index
        movie_title = movie_title.lower()

        if movie_title in movie_attributes:
            for tag in movie_attributes[movie_title]:
                script = script + tag

        movie_title = correct_title_exceptions(movie_title)

        try:
            driver.get("https://www.rottentomatoes.com/m/" + movie_title)
            synopsis = driver.find_element_by_id('movieSynopsis').text
            url = driver.find_element_by_class_name('posterImage').get_attribute('src')

            if url is None:
                url = "No image available."

            synopsis_image_info[movieID] = (synopsis, url)
            script = script + synopsis

            num_found += 1
        except:
            if movie_title[0:4] == "the_":
                try:
                    movie_title = movie_title[4: len(movie_title)]
                    driver.get("https://www.rottentomatoes.com/m/" + movie_title)
                    synopsis = driver.find_element_by_id('movieSynopsis').text
                    url = driver.find_element_by_class_name('posterImage').get_attribute('src')

                    if url is None:
                        url = "Image not found."

                    script = script + synopsis
                    synopsis_image_info[movieID] = (synopsis, url)
                    num_found += 1
                except:
                    synopsis_image_info[movieID] = ("No synopsis found.", "No image available.")
                    print(str(movie_num) + " " + movie_title + " NOT FOUND")
            else:
                synopsis_image_info[movieID] = ("No synopsis found.", "No image available.")
                print(str(movie_num) + " " + movie_title + " NOT FOUND")

        movie_num += 1

        indexDocument(script, inverted_index, movieID)  # Update the inverted index
        file.close()
        num_files += 1
        current_index = current_index + 1

    # Once inverted index is complete, compute document weights using the appropriate weighting scheme
    computeDocWeightsTFIDF(inverted_index, num_files, doc_term_weightings)

    pickle_out1 = open("inverted_index.pickle", "wb")
    pickle.dump(inverted_index, pickle_out1)
    pickle_out1.close()
    pickle_out2 = open("doc_term_weightings.pickle", "wb")
    pickle.dump(doc_term_weightings, pickle_out2)
    pickle_out2.close()
    pickle_out3 = open("index_to_movies.pickle", "wb")
    pickle.dump(index_to_movies, pickle_out3)
    pickle_out3.close()

def generate_recommendations(profile):
    pickle_in = open("inverted_index.pickle", "rb")
    inverted_index = pickle.load(pickle_in)
    pickle_in = open("doc_term_weightings.pickle", "rb")
    doc_term_weightings = pickle.load(pickle_in)
    pickle_in = open("index_to_movies.pickle", "rb")
    index_to_movies = pickle.load(pickle_in)
    pickle_in = open("synopsis_image_info.pickle", "rb")
    synopsis = pickle.load(pickle_in)

    query = open("data/"+profile+"/fb_posts.txt").read()

    t0 = time.time()

    print("Searching for your recommended movies...\n")

    docs_with_scores = retrieveDocuments(profile, query, inverted_index, doc_term_weightings)

    recs = sorted(docs_with_scores.items(), key=lambda x: x[1], reverse=True)[:10]  # Order the list

    ranked_list = [(index_to_movies[movieID], synopsis[movieID][0],
                    synopsis[movieID][1], movieID, score) for movieID, score in recs]

    print("\nTotal time to make recommendation: " + str(time.time() - t0) + " seconds")    # Print computation time
    return ranked_list
