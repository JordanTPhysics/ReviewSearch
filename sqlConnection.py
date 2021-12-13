# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:12:57 2021

@author: starg
"""

import mysql.connector
import pandas as pd

df = pd.read_csv(r'reviewdata.csv',header=[0])
df.columns=['DATE','REVIEW','RATING']
#remove null values from dataframe
df.dropna(subset =["DATE","REVIEW"],inplace=True)
datalist = df.values.tolist()


    
db = mysql.connector.connect(
    host="localhost",
    user="admin",
    passwd="appletreez123",
    database="reviewdata")

mycursor = db.cursor()

add_query = "INSERT INTO netflixdata (review_date, content, rating) VALUES (%s,%s,%s)"
mycursor.executemany(add_query, datalist)
db.commit()

