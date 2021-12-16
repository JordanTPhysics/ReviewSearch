# -*- coding: utf-8 -*-
"""
Created on Wed Dec 2 2021 14:49
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


driver = webdriver.Chrome('chromedriver.exe')

reviewContents = []
reviewDates = []
starRating = []
posneg = []
pos = "pos"
neg = "neg"
driver.get("https://www.getapp.co.uk/reviews/2035411/whatsapp")
t.sleep(1)
cookies = driver.find_element(By.ID,"onetrust-accept-btn-handler")
cookies.click()






for page in range (10):
    reviewDates = driver.find_elements(By.XPATH,"//p[@class='small text-muted mb-3']")
    reviewContents = driver.find_elements(By.XPATH,"//div[@class='jss511']")
    
    
    for i in reviewDates: 
        reviewDates.append(i)
        
    
    for i in reviewContents:
        content = i.text
        reviewContents.append(content)
        if "Pros" in reviewContents:
            posneg.append(pos)
        else:
            posneg.append(neg)     
    
    pagenum = str(page+2)
    driver.get("https://www.getapp.com/customer-management-software/a/whatsapp/reviews/page-"+pagenum+"/")
    t.sleep(1)


driver.quit()


