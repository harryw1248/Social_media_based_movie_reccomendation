import os
import time
from getpass import getpass
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options

driver = None

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

def scrape(profile):
    url = "https://www.facebook.com/" + profile
    driver.get(url)
    url = driver.current_url

    if not os.path.exists('data/'+profile):
        os.mkdir(os.path.join('data', url.split('/')[-1]))

    scroll()
    data = driver.find_elements_by_xpath('//div[@class="_5pcb _4b0l _2q8l"]')
    status = None
    with open("data/"+profile+'/fb_posts.txt', "w") as f:
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

    return

def login(email, password):
    global driver
    driver.get("https://www.facebook.com")
    driver.find_element_by_name('email').send_keys(email)
    driver.find_element_by_name('pass').send_keys(password)
    driver.find_element_by_id('loginbutton').click()

def run_scraper(profile):
    global driver
    options = Options()
    options.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

    options.add_argument("--disable-notifications")
    options.add_argument("--disable-infobars")
    options.add_argument("--mute-audio")
    # options.add_argument("headless")

    chromedriver = "/Users/Vinchenzo4335/PycharmProjects/EECS486/Final_Project/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver

    driver = webdriver.Chrome(executable_path=chromedriver, options=options)

    email = 'vinchenzo.potato@gmail.com'
    password = getpass()
    login(email, password)
    scrape(profile)
    driver.close()

#run_scraper('shadman.habib.12')