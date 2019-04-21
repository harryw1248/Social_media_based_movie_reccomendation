'''
This file was used to parse the data from Kaggle.
It parses the data from the movies_metadata.csv and keywords.csv files
'''

import pandas as pd
import numpy as np
import json
import pickle

'''
    Reads in a csv file and return a dataframe. A dataframe df is similar to dictionary.
    You can access the label by calling df['label'], the content by df['content']
    the rating by df['rating']
'''
def load_data(fname):
    return pd.read_csv(fname)

'''
Extracts movies ids from dataframe that was parsed from movies_metadata.csv
Returns a dictionary that has the movie id as the key and the movie title as the value
'''
def get_movie_ids(dataframe):
    movie_ids = {}
    ids = dataframe['id']
    names = dataframe['original_title']

    for id, name in zip(ids, names):
        movie_ids[id] = name.lower()

    return movie_ids
    
'''
Extracts movies genres from dataframe that was parsed from movies_metadata.csv
Returns a dictionary that has the movie id as the key and the list of corresponding movie genres as the value
'''
def get_movie_genres(dataframe):
    movie_genres = {}

    genres = dataframe.loc[:, 'genres']
    ids = dataframe.loc[:,'id']

    bad_symbols = "'[{}],"
    for id, genre in zip(ids, genres):
        for sym in bad_symbols:
            genre = genre.replace(sym, '')
        genre = genre.split(' ')
        movie_genres[id] = genre[3::4]
    
    for key in movie_genres:
        list = movie_genres[key]
        for item in list:
            item = item.lower()

    return movie_genres


'''
Extracts movies tags from dataframe that was parsed from keywords.csv
Returns a dictionary that has the movie id as the key and the list of corresponding movie tags as the value
'''
def get_movie_tags(dataframe):
    movie_tags = {}

    tags = dataframe.loc[:, 'keywords']
    ids = dataframe.loc[:,'id']

    bad_symbols = "'[{}],"
    for id, tag in zip(ids, tags):
        for sym in bad_symbols:
            tag = tag.replace(sym, '')
        tag = tag.split(' ')
        if len(tag) != 0:
            movie_tags[id] = tag[3::4]
        else:
            movie_tags[id] = []
    
    for key in movie_tags:
        list = movie_tags[key]
        for item in list:
            item = item.lower()

    return movie_tags

'''
Takes the dictionaris that has the movie genres and movie tags and combines their corresponding lists for each movies
Returns a dictionary that has the movie name as the key and the combined list of genres and tags
'''
def combine_attributes(movie_ids, movie_genres, movie_tags):
    movie_attributes = {}

    for key in movie_ids:
        name = movie_ids[key]
        genres = movie_ids[key]
        tags = movie_ids[key]
        attributes = genres + tags
        movie_attributes[name] = attributes
    
    return movie_attributes

if __name__ == '__main__':
    file_name = "movies_metadata.csv"
    dataframe = load_data(file_name)
    movie_ids = get_movie_ids(dataframe)
    movie_genres = get_movie_genres(dataframe)

    file_name = "keywords.csv"
    dataframe = load_data(file_name)
    movie_tags = get_movie_tags(dataframe)

    movie_attributes = combine_attributes(movie_ids, movie_genres, movie_tags)
    pickle.dump(movie_attributes, open("movie_attributes.pickle", "wb"))




