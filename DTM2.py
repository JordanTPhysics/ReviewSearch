# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 18:48:43 2021
text cleaning techniques 2.0, copied from Alice Zhao's lecture and github: https://github.com/adashofdata/nlp-in-python-tutorial
@author: starg
"""

from bs4 import BeautifulSoup
import re
import pickle
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv(r'reviewdata.csv',header=[0])

df.columns=['DATE','REVIEW','RATING']
df.dropna(subset =["DATE","REVIEW"],inplace=True)
df = df.sort_index()

def clean_text_round1(text):
    #lowercase
    text = text.lower()
    #replace square brackets and content inside with ''
    text = re.sub('\[.*?\]', '', text)
    #remove instances of punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    #remove numbers and words attached to numbers
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)

data_clean = pd.DataFrame(df.REVIEW.apply(round1))

def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)

data_clean = pd.DataFrame(data_clean.REVIEW.apply(round2))



vect = CountVectorizer()
DTMvectors = vect.fit_transform(data_clean.REVIEW)


td = pd.DataFrame(DTMvectors.todense()).iloc[:259]
td.columns = vect.get_feature_names()



