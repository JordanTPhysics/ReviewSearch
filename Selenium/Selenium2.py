# -*- coding: utf-8 -*-
"""
Created on Wed Dec 2 2021 14:49
Selenium in python time!!!!


@author: starg
"""
import numpy as np
import matplotlib.pyplot as plt
import selenium as sl
from selenium import webdriver
import time as t
from selenium.webdriver.common.by import By
import os

sql_user = os.environ['MYSQL_USER']
sql_pass = os.environ['MYSQL_PASS']

driver = webdriver.Chrome('chromedriver.exe')

reviewContents = []
reviewDates = []
starRating = []


driver.get("https://www.getapp.com/customer-management-software/a/whatsapp/reviews/")
t.sleep(3)
try:
    cookies = driver.find_element(By.XPATH,"//button[@class='optanon-allow-all accept-cookies-button']")
    cookies.click()
except:
    pass




    
reviews = driver.find_elements(By.XPATH,"//p[contains(@class,'CollapsibleText')]")
reviewtexts = list(map(lambda x: x.text,reviews))
        
reviewDate = driver.find_elements(By.XPATH,"//span[contains(@class,'jss541')]")           
reviewRating = driver.find_elements(By.XPATH,"//meta[contains(@itemprop,'ratingValue')]")
                                                
    
    
    
   
    
#pagenum = str(page+2)
#driver.get("https://www.getapp.com/customer-management-software/a/whatsapp/reviews/page-"+pagenum+"/")
t.sleep(1)


driver.quit()


