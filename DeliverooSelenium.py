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



driver.get("https://apps.apple.com/us/app/deliveroo-food-delivery/id1001501844#see-all/reviews")
t.sleep(1)
contents = []
dates = []



reviewDates = driver.find_elements(By.XPATH,"//time[@class='we-customer-review__date']")
reviewContents = driver.find_elements(By.XPATH,"//p[@dir='ltr']")
for i in reviewDates:
        date = i.text
        dates.append(date)
for i in reviewContents:
        content = i.text
        contents.append(content)
    
    #pagenum = str(page+2)
   
    

allReviews = zip(dates,contents)





#with open('reviewdata.csv','w') as out:
#    csv_out=csv.writer(out)
 #   csv_out.writerows(allReviews)
#t.sleep(1)

driver.quit()