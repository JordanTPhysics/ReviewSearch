# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 23:06:13 2021
selenium crawler on canterbury historic tours just for fun
@author: starg
"""
import numpy as np
import matplotlib.pyplot as plt
import selenium as sl
from selenium import webdriver
import time as t
from selenium.webdriver.common.by import By
import csv
import dateutil.parser
import os
import mysql.connector

######## INIT ########
sql_user = os.environ['MYSQL_USER']
sql_pass = os.environ['MYSQL_PASS']
BASE_URL = "https://www.tripadvisor.co.uk/Attraction_Review-g186311-d1235657-Reviews-Canterbury_Historic_River_Tours-Canterbury_Kent_England.html"

driver = webdriver.Chrome('chromedriver.exe')
driver.implicitly_wait(10)


driver.get(BASE_URL)




dates = []
contents = []
ratings = []
likes = []

while True:
    try:
        cookies = driver.find_element(By.ID,"onetrust-accept-btn-handler")
        
        cookies.click()
    except:
        pass
    
    
    readMore = driver.find_elements(By.XPATH,"//div[@class='dlJyA']")
   # for i in readMore:
  #      i.click()
    reviewRatings = driver.find_elements(By.XPATH,"//div[@class='emWez F1']")
    reviewDates = driver.find_elements(By.XPATH,"//span[@class='euPKI _R Me S4 H3']")
    reviewContents = driver.find_elements(By.XPATH,"//q[@class='XllAv H4 _a']")
    reviewLikes = driver.find_elements(By.XPATH,"//span[@class='ckXjS']")
    
    
    
    review = list(map(lambda x: x.text,reviewContents))
    date = list(map(lambda x: x.text,reviewDates))
    ratingLevel = list(map(lambda x: x.get_attribute('outerHTML')[92],reviewRatings))
    likesAmount = list(map(lambda x: x.text,reviewLikes))
    
     
    contents.extend(review)
    dates.extend(date)
    ratings.extend(ratingLevel)
    likes.extend(likesAmount)
         
    try:
        nextbutton = driver.find_element(By.XPATH,"//a[@class='ui_button nav next primary ']")
        nextbutton.click()
        t.sleep(1)
        
    except:
        break
    
dates = list(map(lambda x: x[20:],dates))
dates = [dateutil.parser.parse(date) for date in dates]
 


company_id = ["CHRT" for i in range(len(contents))]

datagrid = list(zip(company_id,contents,dates,ratings,likes))


db = mysql.connector.connect(host="localhost", user=sql_user, passwd=sql_pass)
pointer = db.cursor()
pointer.execute("use reviewdata")  
        
add_data = (
"INSERT IGNORE INTO chrt " 
"(company_id, review, `date`, rating, likes) "
"VALUES (%s, %s, %s, %s, %s) "
)



pointer.executemany(add_data,datagrid)



pointer.fetchall()
db.commit()
pointer.close()

db.close()   
