# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 21:51:02 2021
Selenium in python time!!!!
##### EYE TEST DECEMBER 11 13:10 #####

@author: starg
"""

from selenium import webdriver
import time as t
import os
from selenium.webdriver.common.by import By
from dateutil import parser
import pandas as pd
import mysql.connector
import numpy as np




######## INIT #########


"""
before running the script, create a mysql table:

CREATE TABLE `customerservicescoreboard` (
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

sql_user = os.environ['MYSQL_USER']
sql_pass = os.environ['MYSQL_PASS']
company = input("enter Titled company name, make sure it's spelled the same on customerservicescoreboard.com':")
URL = "https://www.customerservicescoreboard.com/"+company
length = 9


contentsN = []
datesN = []
neg = [] 

####### SELENIUM CRAWLER ##############

driver = webdriver.Chrome('chromedriver.exe')
driver.get(URL)

##collecting negative reviews
for page in range(length):
    reviewDatesN = driver.find_elements(By.XPATH,"//span[@class='dtreviewed']")
    reviewContentsN = driver.find_elements(By.XPATH,"//span[@class='description item']")
    for i in reviewDatesN:
        dateN = i.text
        datesN.append(dateN)
    for i in reviewContentsN:
        contentN = i.text
        contentsN.append(contentN)
        neg.append("neg")
    pagenum = str(page+2)
    driver.get(URL+"/negative/?page="+pagenum)
    

reviewsNegative = zip(contentsN,datesN,neg)
reviewsNegativeList = list(reviewsNegative)

##collecting positive reviews
driver.get(URL)
positiveSwitch = driver.find_element(By.XPATH,"//a[contains(text(),'Positive')]")
positiveSwitch.click()
contentsP = []
datesP = []
pos = [] 
for page in range(length):
    reviewDatesP = driver.find_elements(By.XPATH,"//span[@class='dtreviewed']")
    reviewContentsP = driver.find_elements(By.XPATH,"//span[@class='description item']")
    for i in reviewDatesP:
        dateP = i.text
        datesP.append(dateP)
   
    for i in reviewContentsP:
        pos.append("pos")
        contentP = i.text
        contentsP.append(contentP)
    pagenum = str(page+2)
    driver.get(URL+"/positive/?page="+pagenum)
    

driver.quit()


############# REFORMAT DATA ############
    
reviewsPositive = zip(contentsP,datesP,pos)
reviewsPositiveList = list(reviewsPositive)

allReviews = reviewsNegativeList + reviewsPositiveList
Alldata = pd.DataFrame(allReviews)
Alldata.columns=['REVIEW','DATE','RATING']
Alldata = Alldata.apply(lambda x: x.str.strip()).replace('', np.nan)
Alldata = Alldata.dropna(axis=0)
dates = list(Alldata['DATE'].values)
dates = [parser.parse(date) for date in dates]
Alldata['DATE'] = dates
Alldata['Company'] = company
Alldata['PREVIEW'] = list(map(lambda x: x[:254],Alldata['REVIEW']))
Alldata = Alldata[['Company','PREVIEW','REVIEW','DATE','RATING']]

Dflist = Alldata.values.tolist()
data = list(map(lambda x: tuple(x),Dflist))


############ DATABASE PUSH ###############

db = mysql.connector.connect(host="localhost", user=sql_user, passwd=sql_pass)
pointer = db.cursor()
pointer.execute("use reviewdata")  
    
add_data = ("INSERT IGNORE INTO customerservicescoreboard "
               "(company_id, preview, review, `date`, rating) "
               "VALUES (%s, %s, %s, %s, %s)")

pointer.executemany(add_data,data)
pointer.fetchall()
db.commit()
    

    
pointer.close()
db.close()   

Alldata.to_csv(company+'Reviewdata.csv',header=False)


t.sleep(1)




