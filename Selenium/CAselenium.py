# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:59:31 2022
consumer affairs website selenium
@author: starg
"""

from selenium import webdriver
import time as t
from selenium.webdriver.common.by import By
import os
import mysql.connector
from dateutil import parser
import re
import pandas as pd
from pathlib import Path  




######## INIT ########
sql_user = os.environ['MYSQL_USER']
sql_pass = os.environ['MYSQL_PASS']
BASE_URL = 'https://www.consumeraffairs.com/'

"""
before running the script, create a mysql table:

CREATE TABLE `consumeraffairs` (
  `review_id` int NOT NULL AUTO_INCREMENT,
  `company_id` varchar(45) NOT NULL,
  `preview` varchar(255) NOT NULL,
  `review` text NOT NULL,
  `date` datetime NOT NULL,
  `rating` varchar(45) NOT NULL,
  `likes` varchar(45) NOT NULL,
  UNIQUE KEY `review_id_UNIQUE` (`review_id`),
  UNIQUE KEY `preview_UNIQUE` (`preview`)
) ENGINE=InnoDB AUTO_INCREMENT=4283 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

"""
#department = input('type the department name: ')
#company = input('type company name: ')
#### here's one for reference

company = 'home_electronics/nintendo'

driver = webdriver.Chrome('chromedriver.exe')

driver.get(f'{BASE_URL}{company}.html')
pages = 1


########### WEBCRAWLER ###########
reviews = []
dates = []
ratings = []
likes = []


#use while True to crawl the entire page
while True:
#for i in range(pages):
    content = driver.find_elements(By.XPATH,"//div[@class='rvw-bd']")
    rating = driver.find_elements(By.XPATH,"//meta[@itemprop='ratingValue']")
    like = driver.find_elements(By.XPATH,"//span[@class='rvw-foot__helpful-count js-helpful-count']")
    review = list(map(lambda x: x.text,content))
    ratingLevel = list(map(lambda x: x.get_attribute('outerHTML'),rating))
    likesAmount = list(map(lambda x: x.text,like))
    date = [i.split('\n',1)[0] for i in review]
    comments = [i.split('\n',1)[1] for i in review]  
    reviews.extend(comments)
    dates.extend(date)
    ratings.extend(ratingLevel)
    likes.extend(likesAmount)
    
    try:
        nextbutton = driver.find_element(By.XPATH,"//a [@rel='next']")
        nextbutton.click()
    except:
        break
    
    
driver.quit()    

############# FORMAT DATA ###########


dates = list(map(lambda x: x[17:],dates))
dates = [re.sub('se: ','',i) for i in dates]
dates = [parser.parse(date) for date in dates]
ratings = list(map(lambda x: x[38:39],ratings))
likes = list(map(lambda x: x[:2],likes))
for i in range(len(likes)):
    if likes[i] == 'Be':
        likes[i] = 0
previews = list(map(lambda x: x[:254],reviews))

company_id = [company for i in range(len(reviews))]

datagrid = list(zip(company_id,previews,reviews,dates,ratings,likes))
df = pd.DataFrame(datagrid, columns=['company_id','preview','review','date','rating','likes'])

filepath = Path(f'../Data/{company}data.csv')
df.to_csv(f'../Data/{company}data.csv')

# ####### SQL COMMIT #######

# db = mysql.connector.connect(host="localhost", user=sql_user, passwd=sql_pass)
# pointer = db.cursor()
# pointer.execute("use reviewdata")  
    
# add_data = (
# "INSERT IGNORE INTO consumeraffairs " 
# "(company_id, preview, review, `date`, rating, likes) "
# "VALUES (%s, %s, %s, %s, %s, %s) "
# )



# pointer.executemany(add_data,datagrid)



# pointer.fetchall()
# db.commit()
# pointer.close()

# db.close()   
