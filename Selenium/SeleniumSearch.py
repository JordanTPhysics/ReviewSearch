# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 21:51:02 2021
Selenium in python time!!!!
##### EYE TEST DECEMBER 11 13:10 #####

@author: starg
"""

from selenium import webdriver
import time as t
from selenium.webdriver.common.by import By
from dateutil import parser
import pandas as pd
import mysql.connector
import numpy as np





company = input("enter Titled company name, make sure it's spelled the same on customerservicescoreboard.com':")
URL = "https://www.customerservicescoreboard.com/"+company
length = 1
driver = webdriver.Chrome('chromedriver.exe')
driver.get(URL)
t.sleep(1)
contentsN = []
datesN = []
neg = [] 
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
Alldata = Alldata[['Company','REVIEW','DATE','RATING']]

x = Alldata.values.tolist()
data = []
for i in x:
    i = tuple(i)
    data.append(i)

db = mysql.connector.connect(host="localhost", user="admin", passwd="appletreez123")
pointer = db.cursor()
pointer.execute("use reviewdata")  
    
add_data = ("INSERT INTO customerservicescoreboard "
               "(company_id, review, date, rating) "
               "VALUES (%s, %s, %s, %s)")

pointer.executemany(add_data,data)
db.commit()
    

    
pointer.close()
db.close()   

Alldata.to_csv(company+'Reviewdata.csv',header=False)


#with open('reviewdata.csv','w') as out:
 #   csv_out=csv.writer(out)
  #  csv_out.writerows(Alldata)
t.sleep(1)




