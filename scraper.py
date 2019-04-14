import os
import time
import re
import keyring
from getpass import getpass
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from requests_html import HTMLSession, HTML
from datetime import datetime

service_id = 'eecs486project'
session = HTMLSession()
driver = None

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
            except IndexError:
                continue

        last_tweet = html.find('.stream-item')[-1].attrs['data-item-id']
        r = session.get(url, params={'max_position': last_tweet})
    tweets = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", tweets)
    return tweets

def scroll():
    current_scrolls = 0
    old_height, new_height = 0, 0

    while True:
        try:
            if current_scrolls == 5000:
                return
            old_height = new_height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if old_height == new_height:
                break
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            current_scrolls += 1

        except TimeoutException:
            break
    return

def scrape(fb_profile, twitter_profile):
    url = "https://www.facebook.com/" + fb_profile
    driver.get(url)
    url = driver.current_url

    if not os.path.exists('data/'+fb_profile):
        os.mkdir(os.path.join('data', url.split('/')[-1]))

    scroll()
    data = driver.find_elements_by_xpath('//div[@class="_5pcb _4b0l _2q8l"]')
    status = None
    with open("data/"+fb_profile+'/posts.txt', "w") as f:
      for x in data:
          try:
              status = x.find_element_by_xpath(".//div[@class='_5wj-']").text
          except:
              try:
                  status = x.find_element_by_xpath(".//div[contains(@class, 'userContent') and contains(@class, '_5pbx')]").text
              except:
                  pass

          if status:
            status = status.replace("\n", " ")
            f.write(status)
          else:
            continue
      if twitter_profile: # append the tweets to fb posts if optional link to twitter profile provided
          tweets = get_tweets(twitter_profile)
          f.write(" "+tweets)
    return

def login(email):
    global driver
    driver.get("https://www.facebook.com")
    driver.find_element_by_name('email').send_keys(email)
    driver.find_element_by_name('pass').send_keys(keyring.get_password(service_id, email))
    driver.find_element_by_id('loginbutton').click()

def run_scraper(login_email, fb_profile, twitter_profile, bin_path):
    global driver
    options = Options()
    options.binary_location = bin_path # '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary'

    options.add_argument("--disable-notifications")
    options.add_argument("--disable-infobars")
    options.add_argument("--mute-audio")
    # soptions.add_argument("headless")

    chromedriver = "./chromedriver" # place driver in main repo directory
    os.environ["webdriver.chrome.driver"] = chromedriver

    driver = webdriver.Chrome(executable_path=chromedriver, options=options)


    login(login_email)
    scrape(fb_profile, twitter_profile)
    driver.close()
