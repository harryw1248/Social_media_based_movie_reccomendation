import re
from requests_html import HTMLSession, HTML
from datetime import datetime

session = HTMLSession()

def get_tweets(user):
    url = f'https://twitter.com/i/profiles/show/{user}/timeline/tweets'
    r = session.get(url)
    tweets = ""
    for i in range(25):
        try:
            html = HTML(html=r.json()['items_html'], url='bunk', default_encoding='utf-8')
        except:
            break

        for tweet in html.find('html > .stream-item'):
            try:
                text = tweet.find('.tweet-text')[0].full_text
                tweets += text.strip() + " "
            except IndexError:  # issue #50
                continue

        last_tweet = html.find('.stream-item')[-1].attrs['data-item-id']
        r = session.get(url, params={'max_position': last_tweet})
    tweets = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", tweets)
    return tweets
