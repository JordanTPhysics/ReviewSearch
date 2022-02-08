# -*- coding: utf-8 -*-
"""
Created on Thu Dec 9 11:12:57 2021

@author: starg
"""


import pandas as pd
import os
import sys
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

import mysql.connector
from dateutil import parser
sql_user = os.environ['MYSQL_USER']
sql_pass = os.environ['MYSQL_PASS']
#table_name = input('Enter the name of the table: ')
df = pd.read_csv(r'Selenium/CanonReviewdata.csv',header=[0])


df.columns=['Index','Review','Rating','Date']
df.drop(columns=['Index'])
#df = df[['Review','Rating','Date']]
df['Company']='canon'


#remove null values from dataframe
df.dropna(subset =["Date","Review"],inplace=True)

Base = declarative_base()

class Review(Base):
    __tablename__ = 'CanonReviews'
    # Here we define columns for the table
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    #site_id = Column(Integer, nullable=False)
    review = Column(String(250), nullable=False)
    rating = Column(String(250))
    date = Column(DATETIME(), nullable=False)
    polarity = Column(FLOAT)
    subjectivity = Column(FLOAT)
    dominant_topic = Column(Integer)
    topic_contribution = Column(FLOAT)
 

 
engine = create_engine(f'mysql://{sql_user}:{sql_pass}localhost:3306/reviewdata', echo=True)    

#df.to_sql('canon2',engine)
TABLE_NAME = input('Choose a table to query: ')
db = mysql.connector.connect(host="localhost", user=sql_user, passwd=sql_pass)
pointer = db.cursor()
pointer.execute("use reviewdata")

#pointer.execute("SELECT * FROM all_reviews WHERE company = 'canon'")
#df = pointer.fetchall()
#df = pd.DataFrame(df,columns =['Company_id','Company','Review','idx'])
del df['Index']
x = df.values.tolist()
data = []
for i in x:
    i = tuple(i)
    data.append(i)
add_data = (f"INSERT INTO {TABLE_NAME} "
               "(review, rating, date, company_id) "
               "VALUES (%s, %s, %s, %s)")
try:
    enter = input('ARE YOU SURE??? (y)')
    if enter == 'y':
        pointer.executemany(add_data,data)
        db.commit()
    
except:
    db.rollback()
    
pointer.close()
db.close()   
    
    
    
    
    
    