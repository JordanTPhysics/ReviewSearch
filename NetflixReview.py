# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:34:39 2021
https://github.com/joeyajames/Python/blob/master/NLTK/NLTK.ipynb
@author: starg
"""

import nltk, re, pprint

import random
import numpy as np

from nltk.tokenize import word_tokenize as wt
from nltk.probability import FreqDist
from nltk.cluster import euclidean_distance, KMeansClusterer
import matplotlib.pyplot as plt
import csv
from string import punctuation
#from SeleniumSearch import allReviews
import pandas as pd
from nltk import sent_tokenize, word_tokenize, pos_tag
from gensim.models import Word2Vec


#import data as dataframe
df = pd.read_csv(r'reviewdata.csv',header=[0])
#remove null values from dataframe
df.dropna(subset =["DATE","REVIEW"],inplace=True)
#remove date column as it has no use in sentiment analysis
del df['DATE']
#dataframe to list
allReviews = df.values.tolist()
random.shuffle(allReviews)
stopList = list(nltk.corpus.stopwords.words('english'))




#extract the reviews from tuple
reviews = []
ratings = []
for review, rating in allReviews:
    reviews.append(review)
    ratings.append(rating)
    
  
    
#tokenize each review   
reviews_in_words = []  
for review in reviews:
    
    tokens = wt(review)
    reviews_in_words.append(tuple(tokens))
     
tokenized_and_tagged = zip(reviews_in_words,ratings)

#retrieve each review as list of words all cat into single list   
words = []
for i in reviews_in_words:
    for word in i:
        words.append(word)
        
#remove stop words and punctuation from word list        
filtered = []
for w in words:
    if w.casefold() not in punctuation and w not in stopList:
        filtered.append(w)
    
    
    
#plot frequency distribution from most common
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


vectorfilter = [] 
for r in reviews_in_words:
    if r not in stopList:
        vectorfilter.append(r)

#each word represented by a vector based on it's relation to other words
netFlix_vec = Word2Vec(vectorfilter)


#6 most similar words to bad and good respectively
print(netFlix_vec.wv.most_similar('bad',topn=6))
print(netFlix_vec.wv.most_similar('good',topn=6))


vectors = [np.array(f) for f in [[2, 1], [1, 3], [4, 7], [6, 7]]]
means = [[4, 3], [5, 5]]

clusterer = KMeansClusterer(2, euclidean_distance, initial_means=means)
clusters = clusterer.cluster(vectors, True, trace=True)






 
