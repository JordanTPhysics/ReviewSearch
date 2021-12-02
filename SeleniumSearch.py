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
#from selenium.common.keys import Keys
#PATH = "C:/Program Files (x86)/chromedriver.exe"

#chromedriver = open('chromedriver.exe', 'r').read()

driver = webdriver.Chrome('chromedriver.exe')



driver.get("https://www.customerservicescoreboard.com/NetFlix")
t.sleep(1)

##collecting negative reviews
for page in range (10):
    reviewDatesN = driver.find_elements(By.XPATH,"//span[@class='dtreviewed']")
    reviewContentsN = driver.find_elements(By.XPATH,"//span[@class='description item']")
    link_text = "/NetFlix/negative/?page="+str(page+2)
    nextPage = driver.find_element(By.PARTIAL_LINK_TEXT,link_text)
    nextPage.click()
    Alert = driver.switch_to().alert().dismiss()

contentsN = []
datesN = []

for i in reviewDatesN:
    dateN = i.text
    datesN.append(dateN)
    
for i in reviewContentsN:
    contentN = i.text
    contentsN.append(contentN)
    
reviewsNegative = zip(datesN,contentsN)
reviewsNegativeList = list(reviewsNegative)

##collecting positive reviews
positiveSwitch = driver.find_element(By.XPATH,"/html/body/div[1]/div/div/div/div[1]/div/ul/li[2]")
#for page in pages:
reviewDatesP = driver.find_elements(By.XPATH,"//span[@class='dtreviewed']")
reviewContentsP = driver.find_elements(By.XPATH,"//span[@class='description item']")

    #nextPage = driver.find_element(By.CLASS_NAME,"nav-next")
    #nextPage.click()

contentsP = []
datesP = []

for i in reviewDatesP:
    dateP = i.text
    datesP.append(dateP)
    
for i in reviewContentsP:
    contentP = i.text
    contentsP.append(contentP)
    
reviewsPositive = zip(datesP,contentsP)
reviewsPositiveList = list(reviewsPositive)


t.sleep(1)

driver.quit()


