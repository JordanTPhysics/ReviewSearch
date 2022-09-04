# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:49:37 2022
facebook selenium
@author: starg
"""
import selenium as sl
from selenium import webdriver
import time as t
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options

import pandas as pd
import dateutil.parser
from datetime import datetime, timedelta
import os


user = os.environ['fuser']
passwd = os.environ['fpass']

def parse_date(date):
    
    td = 0
    
    num = [int(s) for s in date.split() if s.isdigit()]
    if len(num) == 0:
        num = 1
    elif len(num) == 1:
        num = num[0]
        
    if 'hour' in date:
        td = 3600
    
    if 'week' in date:
        td = 86400 * 7
    
    if 'day' in date:
        td = 86400
    
    if 'month' in date:
        td = 86400 * 30
        
    if 'year' in date:
        td = 86400 * 365
        
    return datetime.now() - timedelta(seconds=td * num)


BASE_URL = "https://www.facebook.com"

option = Options()

option.add_argument("--disable-infobars")
option.add_argument("start-maximized")
option.add_argument("--disable-extensions")

# Pass the argument 1 to allow and 2 to block
option.add_experimental_option(
    "prefs", {"profile.default_content_setting_values.notifications": 1}
)

driver = webdriver.Chrome(
    options=option, executable_path="chromedriver.exe")


driver.implicitly_wait(10)


driver.get(BASE_URL)


cookies = driver.find_element(By.XPATH,"//button[contains(text(),'cookies')]")
print('cookies found')
cookies.click()

#target username
username = driver.find_element(By.ID,'email')
password = driver.find_element(By.ID,'pass')

#enter username and password
username.clear()
username.send_keys(user)
password.clear()
password.send_keys(passwd)

#target the login button and click it
button = driver.find_element(By.XPATH,"//button[contains(text(),'Log In')]")
button.click()


t.sleep(5)

driver.get("https://m.facebook.com/799845216729914/")

reviews = driver.find_element(By.XPATH,"//span[contains(text(),'Review')]")
reviews.click()
t.sleep(4)

while True:
    
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    ##setup
    driver.execute_script('window.scrollBy(0,400)')
    t.sleep(1)
    driver.execute_script('window.scrollBy(0,400)')
    new_height = driver.execute_script("return document.body.scrollHeight")
    
    
    if new_height == last_height: break
# Calculate new scroll height and compare with last scroll height



dates = []
reviews = []
#now = t.time()

    

for i in driver.find_elements(By.XPATH,"//a[@aria-label='Open story']"):
        
    parent = driver.execute_script("return arguments[0].parentNode;", i)
    rev = parent.find_element(By.XPATH,"./child::*").text
    reviews.append(rev)
    
    
    
date = driver.find_elements(By.XPATH,"//abbr[contains(text(),'about')]")
dates.extend(list(map(lambda x: x.text, date)))
    
    

dates = list(map(parse_date, dates))
    
    
data = list(zip(['CHRT' for i in range(len(reviews))], dates, reviews))

df = pd.DataFrame(data=data, index=None, columns=['company_id','date','review'])
company = 'CHRT'
df = df.dropna(how='any')
df.to_csv(f'../Data/{company}data.csv')






