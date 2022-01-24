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



######## INIT ########
sql_user = os.environ['MYSQL_USER']
sql_pass = os.environ['MYSQL_PASS']
BASE_URL = 'https://www.consumeraffairs.com/'

company = input('type the department, followed by a / then the company name: ')
#### here's one for reference

#company = 'entertainment/netflix'

driver = webdriver.Chrome('chromedriver.exe')

driver.get(f'{BASE_URL}{company}.html')
pages = 3


########### WEBCRAWLER ###########
reviews = []
dates = []
ratings = []
likes = []

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
for i in dates:
    if 'se:' in i:
        i = re.sub('se:','',i)

dates = [parser.parse(date) for date in dates]
ratings = list(map(lambda x: x[38:39],ratings))
likes = list(map(lambda x: x[:2],likes))

company_id = [company for i in range(len(reviews))]

datagrid = list(zip(company_id,reviews,dates,ratings,likes))



####### SQL COMMIT #######

db = mysql.connector.connect(host="localhost", user=sql_user, passwd=sql_pass)
pointer = db.cursor()
pointer.execute("use reviewdata")  
    
add_data = ("INSERT INTO consumeraffairs "
               "(company_id, review, date, rating, likes) "
               "VALUES (%s, %s, %s, %s, %s)")

pointer.executemany(add_data,datagrid)
db.commit()
    

    
pointer.close()
db.close()   
