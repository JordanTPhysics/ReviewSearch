# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:04:07 2021

@author: starg
"""

import numpy as np
import matplotlib.pyplot as plt
import selenium as sl
from selenium import webdriver
import time as t
from selenium.webdriver.common.by import By
import csv
import os
import pandas as pd
from dateutil import parser
import mysql.connector

"""
before running the script, create a mysql table:

CREATE TABLE `applestore` (
  `review_id` int NOT NULL AUTO_INCREMENT,
  `company_id` varchar(45) NOT NULL,
  `preview` varchar(255) NOT NULL,
  `review` text NOT NULL,
  `date` datetime NOT NULL,
  `rating` varchar(45) NOT NULL,
  UNIQUE KEY `review_id_UNIQUE` (`review_id`),
  UNIQUE KEY `preview_UNIQUE` (`preview`)
) ENGINE=InnoDB AUTO_INCREMENT=4283 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

"""


driver = webdriver.Chrome('chromedriver.exe')

#to make this work set sql username and password as environment variables
sql_user = os.environ['MYSQL_USER']
sql_pass = os.environ['MYSQL_PASS']

driver.get("https://apps.apple.com/gb/app/apple-store/id375380948#see-all/reviews")
t.sleep(1)
contents = []
dates = []
ratings = []


reviewDates = driver.find_elements(By.XPATH,"//time[@class='we-customer-review__date']")
reviewContents = driver.find_elements(By.XPATH,"//div[@class='we-clamp']")
reviewRatings = driver.find_elements(By.XPATH,"//figure[@class='we-star-rating we-customer-review__rating we-star-rating--large']")

for i in reviewDates:
        date = i.text
        dates.append(date)
for i in reviewContents:
        content = i.text
        contents.append(content)
for i in reviewRatings:
        source_code = i.get_attribute("outerHTML")
        ratings.append(source_code[92])    
   
    
driver.quit()

allReviews = zip(contents,dates,ratings)



Alldata = pd.DataFrame(allReviews)
Alldata.columns=['REVIEW','DATE','RATING']
Alldata = Alldata.apply(lambda x: x.str.strip()).replace('', np.nan)
Alldata = Alldata.dropna(axis=0)
dates = list(Alldata['DATE'].values)
dates = [parser.parse(date) for date in dates]
Alldata['DATE'] = dates
Alldata['PREVIEW'] = list(map(lambda x: x[:254],Alldata['REVIEW']))
Alldata['Company'] = 'Apple'
Alldata = Alldata[['Company','PREVIEW','REVIEW','DATE','RATING']]

x = Alldata.values.tolist()
data = []
for i in x:
    i = tuple(i)
    data.append(i)

db = mysql.connector.connect(host="localhost", user=sql_user, passwd=sql_pass)
pointer = db.cursor()
pointer.execute("use reviewdata")  
    
add_data = ("INSERT INTO applestore "
               "(company_id, preview, review, date, rating) "
               "VALUES (%s, %s, %s, %s, %s, %s)")

pointer.executemany(add_data,data)
db.commit()
    

    
pointer.close()
db.close()   





