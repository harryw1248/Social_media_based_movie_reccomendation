from flask import Flask, render_template, request, session, url_for, redirect
import os, pickle, sys
import keyring
import inverted_index as recommender
from relevanceFeedback import submit_feedback
import scraper

# Run with: python3 app.py <path to /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome >
# (also make sure correct chromedriver executable is present in repo)

app = Flask(__name__)
service_id = 'eecs486project'

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


def parse_feedback(form):
    res = list()
    for i in range(1,11):
        movieID = int(form['movieID_'+str(i)])
        score = int(form['score_movie_'+str(i)])
        relevancy = 1 if form['relevancy_movie_'+str(i)] == 'relevant' else 0
        res.append((movieID, i, score, relevancy))
    return res

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/scrape', methods=['GET', 'POST'])
def scrape_profiles():
    login_email = request.form['username']
    keyring.set_password(service_id, login_email, request.form['password'])

    profile = request.form['fb_profile'].split("/")[-1]
    if request.form['twitter_profile']: # optional twitter link
        twitter_profile = request.form['twitter_profile'].split("?")[0].split("/")[-1]
    else:
        twitter_profile = None

    session['profile'] = profile
    if not os.path.isdir("data/"+profile):
        scraper.run_scraper(login_email, profile, twitter_profile, sys.argv[1])
    return redirect(url_for('recommendations'))

@app.route('/feedback', methods=['GET', 'POST'])
def post_feedback():
    feedback = parse_feedback(request.form)
    profile = session['profile']
    submit_feedback(feedback, profile, request.form['model'])
    return redirect(url_for('recommendations'))

@app.route('/recommendations')
def recommendations():
    global weightings, query, inverted_index, index_to_movies, synopsis
    profile = session['profile']
    if profile is None: return redirect(url_for('homepage'))
    movies = recommender.generate_recommendations(profile, False) # Note set to True in order to recreate inverted index
    return render_template('recommendations.html', user=profile, stories=movies)

if __name__ == '__main__':
    #app.debug=True
    app.secret_key = 'V\xfc\xa5\x04\x8ac\xa9#SU\x02*\x990\x9d\xb9\x08\xe6\xb5\x8d\xb9\xd2\xbe\x93\x94\xf1\xf2W7\xd6"\x0b\xe5\xc3{\xc7{U\xf8\xf4\xbc\xdd\xe6\x01\xea\t\\|<\xce'
    app.run()
