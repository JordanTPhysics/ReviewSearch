# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 21:51:02 2021
Selenium in python time!!!!
##### EYE TEST DECEMBER 11 13:10 #####

@author: starg
"""
import numpy as np
import matplotlib.pyplot as plt
import selenium as sl
from selenium import webdriver
import time as t
from selenium.webdriver.common.by import By
import csv
#from selenium.common.keys import Keys




driver = webdriver.Chrome('chromedriver.exe')



driver.get("https://www.customerservicescoreboard.com/NetFlix")
t.sleep(1)
contentsN = []
datesN = []
neg = [] 
##collecting negative reviews
for page in range (4):
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
    driver.get("https://www.customerservicescoreboard.com/NetFlix/negative/?page="+pagenum)
    

reviewsNegative = zip(datesN,contentsN,neg)
reviewsNegativeList = list(reviewsNegative)

##collecting positive reviews
driver.get("https://www.customerservicescoreboard.com/NetFlix")
positiveSwitch = driver.find_element(By.XPATH,"//a[contains(text(),'Positive')]")
positiveSwitch.click()
contentsP = []
datesP = []
pos = [] 
for page in range (4):
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
    driver.get("https://www.customerservicescoreboard.com/NetFlix/positive/?page="+pagenum)
    
    
reviewsPositive = zip(datesP,contentsP,pos)
reviewsPositiveList = list(reviewsPositive)

allReviews = reviewsNegativeList + reviewsPositiveList
with open('reviewdata.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerows(allReviews)
t.sleep(1)

driver.quit()


