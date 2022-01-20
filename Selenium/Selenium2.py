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


driver.get("https://www.getapp.com/customer-management-software/a/whatsapp/reviews/")
t.sleep(1)
try:
    cookies = driver.find_element(By.ID,"onetrust-accept-btn-handler")
    cookies.click()
except:
    pass



for page in range(3):
    
    reviewContent = driver.find_elements(By.XPATH,"//div[@class='MuiCollapse-wrapperInner']")
    reviewDate = driver.find_elements(By.XPATH,"//div[@class='MuiAccordionDetails-root jss463 Details jss484']")           
    reviewRating = driver.find_elements(By.XPATH,"//meta[@itemprop='ratingValue']")
    
    for i in reviewContent:
        reviewContents.append(i.text)
    for i in reviewDate:
        reviewDates.append(i.get_attribute('innerhtml'))
    for i in reviewRating:
        reviewRating.append(i.get_attribute('innerhtml'))
    
    
   
    
    pagenum = str(page+2)
    driver.get("https://www.getapp.com/customer-management-software/a/whatsapp/reviews/page-"+pagenum+"/")
    t.sleep(1)


driver.quit()


