# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:34:39 2021

@author: starg
"""

import nltk
from nltk.corpus import movie_reviews
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

ReviewList = open('output10.txt','r').read()



stop_words = set(stopwords.words('english'))

word_tokens = wt(ReviewList)
 
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

features = {}
    #scans each document to find whether or not a given word was featured
for word in word_tokens:
   features['contains({})'.format(word)] = (word in filtered_sentence)
       
classifier = nltk.NaiveBayesClassifier.train()   
print(nltk.classify.accuracy(classifier, filtered_sentence))  
classifier.show_most_informative_features(5)

freqDist = nltk.FreqDist(w.lower() for w in filtered_sentence)

freqDist.plot(20, cumulative=False);


 
