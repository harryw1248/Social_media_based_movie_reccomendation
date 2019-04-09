from flask import Flask, render_template, request, session, url_for, redirect
import os
import inverted_index
import scraper

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/scrape', methods=['GET', 'POST'])
def scrape_profiles():
    profile = request.form['fb_profile'].split("/")[-1]
    #twitter_profile = request.form['twitter_profile']
    session['profile'] = profile
    if not os.path.isdir("Data/"+profile):
        scraper.run_scraper(profile) #("https://www.facebook.com/anthony.liang8P")
    return redirect(url_for('recommendations'))

@app.route('/feedback', methods=['GET', 'POST'])
def post_feedback():
    print(request.form)
    return redirect(url_for('recommendations'))

@app.route('/recommendations')
def recommendations():
    profile = session['profile']
    if profile is None: return redirect(url_for('homepage'))
    movies = inverted_index.generate_recommendations(profile)
    #movies = [['Avengers', 'Adrift in space with no food or water, Tony Stark sends a message to Pepper Potts as his oxygen supply starts to dwindle. Meanwhile, the remaining Avengers -- Thor, Black Widow, Captain America and Bruce Banner -- must figure out a way to bring back their vanquished allies for an epic showdown with Thanos -- the evil demigod who decimated the planet and the universe.', 'avengers.jpg'] for i in range(5)]
    return render_template('recommendations.html', user=profile, stories=movies)

if __name__ == '__main__':
    app.debug=True
    app.secret_key = 'V\xfc\xa5\x04\x8ac\xa9#SU\x02*\x990\x9d\xb9\x08\xe6\xb5\x8d\xb9\xd2\xbe\x93\x94\xf1\xf2W7\xd6"\x0b\xe5\xc3{\xc7{U\xf8\xf4\xbc\xdd\xe6\x01\xea\t\\|<\xce'
    app.run()
