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





driver = webdriver.Chrome('chromedriver.exe')



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
   
    

allReviews = zip(dates,contents)





with open('Deliveroodata.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerows(allReviews)
t.sleep(1)

driver.quit()