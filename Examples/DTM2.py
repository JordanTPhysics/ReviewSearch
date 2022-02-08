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
import scipy
from gensim import matutils, models


df = pd.read_csv(r'Selenium/reviewdata.csv',header=[0])

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


td = pd.DataFrame(DTMvectors.todense()).iloc[:len(data_clean)]
td.columns = vect.get_feature_names()

sparse_counts = scipy.sparse.csr_matrix(td)
corpus = matutils.Sparse2Corpus(sparse_counts)

cv = pickle.load(open("cv_stop.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=6, passes=10)
lda.print_topics()





