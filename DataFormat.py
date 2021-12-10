# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:47:02 2021
This is simply where the retrieved data will be reformatted for language processing
@author: starg
"""



import random
import numpy as np
import csv
from string import punctuation
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize as wt
stopList = list(nltk.corpus.stopwords.words('english'))
stopList.append('HubSpot')

#Ka Po's hubspot data
df = pd.read_csv(r'hubspotCombined.csv',header=[0])

#random.shuffle(df)

TheReviews = list(df['User Review'].values)
TheRatings = list(df['Star Rating'].values)

posneg = []
for rating in TheRatings:
   if rating.startswith('1' or '2' or '3'):
       posneg.append("neg")
   elif rating.startswith('4' or '5'):
       posneg.append("pos")
           
reviews_in_words = []  
for review in TheReviews:
    
    tokens = wt(review)
    reviews_in_words.append(tuple(tokens))
    
#retrieve each review as list of words all cat into single list   
words = []
for i in reviews_in_words:
    for word in i:
        if word not in punctuation and word not in stopList:
            words.append(word)
            
        
#remove stop words and punctuation from word list        
filtered = []
for w in words:
    if w.casefold() not in punctuation and w not in stopList:
        filtered.append(w)
        
all_words = nltk.FreqDist(filtered)
all_words.plot(20, cumulative=False)  

#lists the 2000 most used words
word_features = list(all_words)[:2000]
def review_features(review):
    review_words = set(review)
    
    features = {}
    #scans each document to find whether or not a given word was featured
    for word in word_features:
        features['contains({})'.format(word)] = (word in review_words)
        
    return features
    
#for each document record which words featured and which class
#(pos,neg) the document falls into
featuresets = [(review_features(d), c) for (d,c) in tokenized_and_tagged]
    
train_set, test_set = featuresets[100:], featuresets[:100]  

#trains the classifier to class reviews good and bad by showing it good and bad reviews.
classifier =  nltk.NaiveBayesClassifier.train(train_set)


print(nltk.classify.accuracy(classifier, test_set))
#5 features most skewed to pos or neg  
classifier.show_most_informative_features(5)       
