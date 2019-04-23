'''
    This file will compute the cosine similarity scores between the database of TF-IDF movie vectors by creating
    the TF-IDF query vector and will return a ranked list of 10 movie recommendations
'''

from preprocess import remove_stopwords, stem_words
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
from names_dataset import NameDataset
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

'''
    Computes the document vector weights using non-normalized TF-IDF weighting scheme and stores the necessary weights
    TF-IDF is calculated as (term frequency) * (N/document frequency)
    Args:
        inverted_index(dict)
        num_files(int)
        doc_term_weightings(dict)
    Return:
        None
'''
def compute_doc_weights_TFIDF(inverted_index, num_files, doc_term_weightings):
    # Intializes dict of document term weightings so each movieID maps to a SimilarityData() object
    for movieID in range(0, num_files + 1):
        doc_term_weightings[movieID] = SimilarityData(len(inverted_index))

    index = 0

    for term in inverted_index:
        for posting_data in inverted_index[term].posting_list:  # Iterates through each document in each posting list
            tf = posting_data.tf                                # tf is frequency of term in a document
            num_postings = inverted_index[term].length          # num_postings is document frequency
            movieID = posting_data.movieID

            if tf > 0.0:                                        # Only spend computational effort if tf > 0
                idf = math.log10(num_files / num_postings)
                doc_term_weightings[movieID].weights[index] = tf * idf
                doc_term_weightings[movieID].doc_length += ((tf * idf) * (tf * idf))  # Update running total for length

            else:
                continue

        index += 1

    for movieID in range(1, num_files + 1):   # Finish calculating doc lengths using sqrt() of the running total
        doc_term_weightings[movieID].doc_length = math.sqrt(doc_term_weightings[movieID].doc_length)


'''
    Calculates the cosine similarity of a vector of document weights and a vector of query weights
    Args:
        doc_weights(list)
        query_weights(list)
        doc_length(int)
        query_length(float)
    Return:
        cosine similarity value(float)
'''

# Function that calculates the cosine similarity of document and query vectors
def calculate_cosine_similarity(doc_weights, query_weights, doc_length, query_length):
    dot_product = 0.0

    # Iteratively computes dot product by multiplying respective elements in query and document vectors and summing
    for index in range(0, len(doc_weights)):
        dot_product += doc_weights[index] * query_weights[index]

    # Final value is the dot product divided by the product of document and query lengths
    return dot_product/(doc_length * query_length)


'''
    Updates the inverted index to include the inverted index entry for a given movie and its corresponding tokens 
    Args:
        inverted_index(dict)
        movieID(int)
        tokens(list of str)
    Return:
        None
'''

def create_inverted_index(inverted_index, movieID, tokens):
    for token in tokens:

        if token in inverted_index:
            current_posting_list_length = inverted_index[token].length
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


'''
    Computes the query vector weights using non-normalized TF-IDF weighting scheme
    TF-IDF is calculated as (term frequency) * (N/document frequency)
    Args:
        query_string(str)
        inverted_index(dict)
        num_files(int)
        profile(str)
    Return:
        tuple (list of float; float; dict)
'''
def calculate_query_TFIDF(query_string, inverted_index, num_files, profile):
    # List of words to remove words from profile text that appear often but have no bearing on user's likes/dislikes
    words_to_remove = ["birthday", "bday", "facebook", "lol", "thank", "christmas", "hanukkah", "happy"]

    # First we must preprocess the query (social media profile)
    m = NameDataset()
    tokens = nltk.word_tokenize(query_string)                           # Tokenizes the string using NLTK
    tokens = [x for x in tokens if x not in string.punctuation]         # Don't include punctuation
    query_tokens = remove_stopwords(tokens)                             # Remove the stopwords

    # Only includes words that are: 1.) In English 2.) Not in  words_to_remove 3.) Not a first name or last name
    query_tokens = [x for x in query_tokens if (wordnet.synsets(x) and x not in words_to_remove and
                                                not m.search_first_name(x)) and not m.search_last_name(x)]

    query_tokens = stem_words(query_tokens)                             # Stem words for preprocessing

    for i in range(0, len(query_tokens)):                               # Converts all tokens to lowercase
        query_tokens[i] = query_tokens[i].lower()

    query_tokens = [x for x in query_tokens if x != 'birthdai']         # Makes sure this common word doesn't appear
    query_appearances = collections.Counter()
    query_weights = [0] * len(inverted_index)                           # Initialize vector to hold query weights
    query_length = 0.0
    l = list(inverted_index.keys())                                     # Gets list of tuples (query_term, index)

    for query_token in query_tokens:                                    # Counter that keeps track of word appearances
        query_appearances[query_token] += 1

    # Iterate through each term in the query vector and assign nonzero weight if the term appears in inverted index
    for query_term in query_appearances:
        if query_term in inverted_index:
            index_of_word = l.index(query_term)                         # Since ordered dict, calculate index of term
            num_postings = inverted_index[query_term].length + 0.0      # Document frequency
            idf = math.log10(num_files / num_postings)                  # Inverse document frequency
            tf = query_appearances[query_term]                          # Term frequency
            query_weights[index_of_word] = tf * idf                     # Query weight
            query_length += (tf * idf) * (tf * idf)                     # Update running total for query length

    query_length = math.sqrt(query_length)                              # Calculate final query length

    # Writes the query data to pickles
    pickle_out = open("data/"+profile+"/query_appearances.pickle", "wb")
    pickle.dump(query_appearances, pickle_out)
    pickle_out.close()

    pickle_out2 = open("data/" + profile + "/query_weights.pickle", "wb")
    pickle.dump(query_weights, pickle_out2)
    pickle_out2.close()

    return (query_weights, query_length, query_appearances)             # Returns the tuple of necessary data


'''
    Uses the query data and collaborative filtering to retrieve movies with at least one matching query term
    and calls calculate_cosine_similarity()
    
    Args:
        query_appearances(dict)
        inverted_index(dict)
        query_weights(list of floats)
        query_length(float)
        doc_term_weightings(dict)
        upvoting_factor(float): 1.05 (default)
        downvoting_factor(float): 0.95 (default)
'''
def calculate_document_similarity(query_appearances, inverted_index, query_weights, query_length,
                                doc_term_weightings, upvoting_factor=1.05, downvoting_factor=0.95):

    docs_with_at_least_one_matching_query_term = set()
    docs_with_scores = dict()

    # Use a set to hold every movieID in which at least one query term appears
    for query_term in query_appearances:
        if query_term in inverted_index:
            for posting_data in inverted_index[query_term].posting_list:
                docs_with_at_least_one_matching_query_term.add(posting_data.movieID)

    past_feedback = list()

    # Does error checking for if the past_feedback exists already and whether or not it's empty
    if not os.path.exists("past_feedback.pickle"):
        past_feedback_in = open("past_feedback.pickle", "wb")
        past_feedback_in.close()

    else:
        try:
            past_feedback_in = open("past_feedback.pickle", "rb")
            past_feedback = pickle.load(past_feedback_in)
        except:
            past_feedback = list()

    # Collaborative filtering: find nearest neighbor (previous user), extract list of relevant and irrelevant movies
    if len(past_feedback):
        relevant_movie_ids, irrelevant_movie_ids = cf.find_nearest_neighbor(query_weights, past_feedback)
    else:
        relevant_movie_ids = irrelevant_movie_ids = []

    # For each movieID in the set, calculate the cosine similarity and store in a map of movieID to similarity value
    # If collaborative filtering is applicable, upvote or downvote the resulting cosine similarity score accordingly
    for movieID in docs_with_at_least_one_matching_query_term:
        docs_with_scores[movieID] = calculate_cosine_similarity(doc_term_weightings[movieID].weights, query_weights,
                                                                  doc_term_weightings[movieID].doc_length, query_length)
        if movieID in relevant_movie_ids:
            docs_with_scores[movieID] *= upvoting_factor

        if movieID in irrelevant_movie_ids:
            docs_with_scores[movieID] *= downvoting_factor

    return docs_with_scores


'''
    Preprocesses the relevant movie data (script, synopsis) using tokenization, stopword removal, and stemming
    and then sends the list of tokens to create_inverted_index()
    
    Args:
        document(list of str)
        inverted_index(dict)
        movieID(int)
    
    Return:
        None
'''
def index_document(document, inverted_index, movieID):
    tokens = nltk.word_tokenize(document)                               # Tokenize the script/synopsis
    tokens = [x for x in tokens if x not in string.punctuation]
    tokens = remove_stopwords(tokens)                                   # Remove the stopwords
    tokens = stem_words(tokens)                                         # Stem words

    for i in range(0, len(tokens)):
        tokens[i] = tokens[i].lower()                                   # Makes all words lowercase

    create_inverted_index(inverted_index, movieID, tokens)              # Create the inverted index


'''
    Retrieves previously created query TF-IDF vector or creates a new one and then calls calculate_document_similarity()
    
    Args:
        profile(str)
        query(str)
        inverted_index(dict)
        doc_term_weightings(dict)
    
    Return:
        dict
'''

def retrieve_documents(profile, query, inverted_index, doc_term_weightings):
    query_weights = list()
    query_appearances = collections.Counter()
    query_length = 0.0

    # If the query vector for this profile has never been computed
    if not os.path.exists("data/"+profile+"/query_appearances.pickle") and not\
        os.path.exists("data/"+profile+"/query_weights.pickle"):
        query_appearances = collections.Counter()
        num_files = len(doc_term_weightings) + 0.0
        (query_weights, query_length, query_appearances) = calculate_query_TFIDF(query, inverted_index,
                                                                                   num_files, profile)

    # The query vector for this profile was previously computed
    else:
        pickle_in = open("data/"+profile+"/query_weights.pickle", "rb")
        query_weights = pickle.load(pickle_in)
        pickle_in = open("data/"+profile+"/query_appearances.pickle", "rb")
        query_appearances = pickle.load(pickle_in)

        for elt in query_weights:
            query_length += elt * elt

        query_length = math.sqrt(query_length)

    # After calculating query weights and length, returns ranked list of documents by calculating similarity
    return calculate_document_similarity(query_appearances, inverted_index,
                                       query_weights, query_length, doc_term_weightings)


'''
    This function is only called in creation of the inverted index when scraping rottentomates.com for synopses.
    Since the web crawling relies on retrieving the correct movie name in the site, certain movies have to have
    their names modified to create the correct URL. The modification of certain punctuation or special case movie
    titles is included to get the appropriate data.
    
    Args:
        title (str)
    
    Return:
        title (str)
'''

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

'''
    -Creates all the necessary pickle files:
        1. index_to_movies maps each movieID to a string of the movie's name
        2. doc_term_weightings map each movieID to the TF-IDF weights and the length of the weights vector
        3. inverted_index maps each movieID to the list of words in the corpus that appear in the document
        4. synopsis_image_info maps each movieID to a synopsis (if found) and a movie poster image link (but 
            this was never accomplished because we could not figure out a way to do it)
            
    -Performs scraping of rotten_tomatoes using chromedriver
    -Uses the pickle file "movie_attributes.pickle" to retrieve available tags 
    -Manages the creation of the inverted index and the TF-IDF vectors for document weights
    
    Args:
        inverted_index(dict)
        num_files(int)
        synopsis_image_info (dict)
        index_to_movies(dict)
        doc_term_weightings(dict)
    
    Return:
        None
'''
def create_data(inverted_index, num_files, synopsis_image_info, index_to_movies, doc_term_weightings):
    # Sets up the chromedriver for web browsing
    global driver
    options = Options()

    chromedriver = "./chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver
    options.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

    #  Code to disable notifications pop up of Chrome Browser
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-infobars")
    options.add_argument("--mute-audio")
    options.add_argument("headless")

    driver = webdriver.Chrome(executable_path=chromedriver, options=options)
    num_found = 0
    movie_num = 0
    pickle_tags = open("movie_attributes.pickle", "rb")
    movie_attributes = pickle.load(pickle_tags)

    doc_folder = ""
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

        if ", A" in movie_title:
            index = movie_title.find(", A")
            movie_title = "A " + movie_title[0: index]

        print(movie_title + ", Index: " + str(current_index))               # Lets user know which movies have been seen

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
                    print(str(movie_num) + " " + movie_title + " NOT FOUND")    # If movie synopsis not found
            else:
                synopsis_image_info[movieID] = ("No synopsis found.", "No image available.")
                print(str(movie_num) + " " + movie_title + " NOT FOUND")        # If movie synopsis not found

        movie_num += 1
        index_document(script, inverted_index, movieID)                          # Update the inverted index
        file.close()
        num_files += 1
        current_index = current_index + 1

    # Once inverted index is complete, compute document weights using the appropriate weighting scheme
    compute_doc_weights_TFIDF(inverted_index, num_files, doc_term_weightings)

    # Write all info to pickle files to avoid having to recompute every time
    pickle_out1 = open("inverted_index.pickle", "wb")
    pickle.dump(inverted_index, pickle_out1)
    pickle_out1.close()
    pickle_out2 = open("doc_term_weightings.pickle", "wb")
    pickle.dump(doc_term_weightings, pickle_out2)
    pickle_out2.close()
    pickle_out3 = open("index_to_movies.pickle", "wb")
    pickle.dump(index_to_movies, pickle_out3)
    pickle_out3.close()


''' 
   Called by the Flask application to calculate cosine similarity measures for a given user's profile
   If create_new_pickle_files is set to True, then the entire movie database, tags, and synopses will be retrieved
   and scraped to create new inverted_index, doc_term_weightings, etc.
   
   Args:
        profile (str)
        create_new_pickle_files(bool): False(default)
    
    Return:
        ranked_list(dict) 
'''
def generate_recommendations(profile, create_new_pickle_files=False):
    inverted_index = dict()
    doc_term_weightings = dict()
    index_to_movies = dict()
    synopsis = dict()

    if create_new_pickle_files:
        create_data(inverted_index, 0, synopsis, index_to_movies, doc_term_weightings)

    else:
        # Opens all necessary pickle files
        pickle_in = open("inverted_index.pickle", "rb")
        inverted_index = pickle.load(pickle_in)
        pickle_in = open("doc_term_weightings.pickle", "rb")
        doc_term_weightings = pickle.load(pickle_in)
        pickle_in = open("index_to_movies.pickle", "rb")
        index_to_movies = pickle.load(pickle_in)
        pickle_in = open("synopsis_image_info.pickle", "rb")
        synopsis = pickle.load(pickle_in)
        print(len(doc_term_weightings))

    query = open("data/"+profile+"/posts.txt").read()                               # Open the social media text file
    t0 = time.time()                                                                # Keeps track of computation time
    print("Searching for your recommended movies...\n")

    # Retrieves a ranking of all movies in the database with cosine similarity scores
    docs_with_scores = retrieve_documents(profile, query, inverted_index, doc_term_weightings)

    # Sorts the retrieved list of movies by cosine similarity
    recs = sorted(docs_with_scores.items(), key=lambda x: x[1], reverse=True)[:10]  # Order the list

    # Contains each movie's name, synopsis, image link (not used), movieID, and cosine similarity score
    ranked_list = [(index_to_movies[movieID], synopsis[movieID][0],
                    synopsis[movieID][1], movieID, score) for movieID, score in recs]

    print("\nTotal time to make recommendation: " + str(time.time() - t0) + " seconds")    # Print computation time
    return ranked_list
