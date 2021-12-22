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
import csv
import pandas as pd






company = input("enter Titled company name, make sure it's spelled the same on customerservicescoreboard.com':")
URL = "https://www.customerservicescoreboard.com/"+company

driver = webdriver.Chrome('chromedriver.exe')
driver.get(URL)
t.sleep(1)
contentsN = []
datesN = []
neg = [] 
##collecting negative reviews
for page in range (10):
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
    

reviewsNegative = zip(datesN,contentsN,neg)
reviewsNegativeList = list(reviewsNegative)

##collecting positive reviews
driver.get(URL)
positiveSwitch = driver.find_element(By.XPATH,"//a[contains(text(),'Positive')]")
positiveSwitch.click()
contentsP = []
datesP = []
pos = [] 
for page in range (10):
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
    
    
reviewsPositive = zip(datesP,contentsP,pos)
reviewsPositiveList = list(reviewsPositive)

allReviews = reviewsNegativeList + reviewsPositiveList
Alldata = pd.DataFrame(allReviews)
Alldata.columns=['DATE','REVIEW','RATING']
Alldata.dropna(subset =["DATE","REVIEW"],inplace=True)



Alldata.to_csv(company+'Reviewdata.csv',header=False)


#with open('reviewdata.csv','w') as out:
 #   csv_out=csv.writer(out)
  #  csv_out.writerows(Alldata)
t.sleep(1)

driver.quit()


