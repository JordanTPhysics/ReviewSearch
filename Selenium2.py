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



driver.get("https://www.getapp.co.uk/reviews/2035411/whatsapp")



for page in range (10):
    reviewDates = driver.find_elements(By.XPATH,"//p[@class='small text-muted mb-3']")
    reviewContentsP = driver.find_elements(By.CLASS_NAME,"small")
    #reviewContentsN = driver.find_elements(By.XPATH,"//*[@id='apps']/div[3]/div[3]/div[1]/div/div[1]/div/div/div[1]/div[2]/div/div[2]/div/p/ ")
    #link_text = "/reviews/2035411/whatsapp?page="+str(page+2)
    #nextPage = driver.find_element(By.PARTIAL_LINK_TEXT,link_text)
    #nextPage.click()
    #Alert = driver.switch_to().alert().dismiss()

contents = []
dates = []

for i in reviewDates:
    date = i.text
    dates.append(date)
    
for i in reviewContentsP:
    content = i.text
    contents.append(content)
    
reviewsNegative = zip(dates,contents)
reviewsNegativeList = list(reviewsNegative)
print(contents)



driver.quit()


