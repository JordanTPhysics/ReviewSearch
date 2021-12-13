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


driver = webdriver.Chrome('chromedriver.exe')
driver.get("https://www.tripadvisor.co.uk/Attraction_Review-g186311-d1235657-Reviews-Canterbury_Historic_River_Tours-Canterbury_Kent_England.html")


dates = []
contents = []
ratings = []

while True:
    
    
    reviewRatings = driver.find_elements(By.XPATH,"//div[@class='emWez F1']")
    reviewDates = driver.find_elements(By.XPATH,"//span[@class='euPKI _R Me S4 H3']")
    reviewContents = driver.find_elements(By.XPATH,"//q[@class='XllAv H4 _a']")

    for i in reviewDates:
        date = i.text
        dates.append(date)
    for i in reviewContents:
        content = i.text
        contents.append(content)
    for i in reviewRatings:
        source_code = i.get_attribute("outerHTML")
        ratings.append(source_code[92])  
        
    try:
        nextbutton = driver.find_element(By.XPATH,"//a[@class='ui_button nav next primary ']")
        nextbutton.click()
        t.sleep(1)
        cookies = driver.find_element(By.ID,"onetrust-accept-btn-handler")
        #'//button[text()="Some text"]')
        cookies.click()
    except:
        break