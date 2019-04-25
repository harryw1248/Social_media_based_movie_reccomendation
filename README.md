# A Social Media Based Recommendation System
### Vinny Ahluwalia, Shaeq Ahmed, Joshua Israel, Anthony Liang, Harry Wang  (Team 4)

## Dependencies

Run this command to install the dependencies needed for this project

`pip3 install -r requirements.txt`

Download pickles files from Google Drive or use the pickle files in the Canvas submission

Link:
https://drive.google.com/open?id=1oY1ObZNDpravOaSeeP-OIHsAS96xkBf7

## How to run:

(Steps 1 - 5 are necessary to run scraper)
1. Install `chromedriver` from https://sites.google.com/a/chromium.org/chromedriver/
2. Move chromedriver to root directory of project
3. Unzip the Kaggle data (kaggle_dataset.zip), the pickle files (pickle_files.zip), and the movie scripts (movies.zip) folder and put all those files in the root directory (doc_term_weightings.pickle, inverted_index.pickle, etc. must be in the root directory if you are not recreating the pickle data)
4. Make sure there is a folder entitled "data" in the root directory; it does not matter whether it's empty
5. Load the webapp: `python3 app.py [chromedriver path]`. This will load the flask app on localhost. Navigate to the path specified in the terminal.
      Ex: "python3 app.py Users/Vinchenzo4335/Documents/EECS_486_final_project-master/chromedriver"
6. You have reached the home page. Type in username and password to your Facebook.
7. Fill in relative or absolute path to Facebook and Twitter (optional) profile. `e.g. /shaeqahmed`.
8. Hit the Generate Recommendation button. You should see a Chrome window pop up and automate the Facebook scraping.
9. Generating recommendations usually takes ~45seconds - a minute. It should redirect to another route once the recommendations are generated.
10. Follow the instructions on the side panel of the page to submit feedback and update your recommendations.


![alt text](static/images/gui.png "GUI Picture")


## Repo Tree
```
├── README.md
├── app.py -> Driver, calls scraping and cosine similarity functions
├── chromedriver
├── collaborative_filtering.py --> runs collaborative filtering
├── dataset_parse.py --> parses Kaggle dataset
├── inverted_index.py --> calls preprocessing of text, creates pickle files, and computes cosine similarity
├── PorterStemmer.py --> stems words
├── preprocess.py --> helps preprocess text
├── relevanceFeedback.py --> query reformulation and metrics
├── scraper.py --> scrapes social media
├── static
│   ├── animate.min.css
│   ├── formstyle.css
│   ├── home_style.css
│   ├── images
│   │   ├── avengers.jpg
│   │   └── gui.png
│   └── rec_style.css
├── templates
│   ├── homepage.html
│   └── recommendations.html
└── utils
    ├── __init__.py
    ├── auth.py
    ├── story.py
```
