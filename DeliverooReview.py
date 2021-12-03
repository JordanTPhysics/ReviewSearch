# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:34:39 2021

@author: starg
"""

import nltk

import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import csv

#from SeleniumSearch import allReviews
import pandas as pd
from nltk import sent_tokenize, word_tokenize, pos_tag




#reader = csv.reader(open('reviewdata.csv', 'rU'), delimiter= ",",quotechar='|')
df = pd.read_csv(r'reviewdata.csv',header=[0])


dict = {'Time': 'Date',
		'Review': 'Content',
		'Rating': 'Sentiment'}


df.rename(columns=dict,
		inplace=True)




#documents=[(list(df.words()), category) 
            #for category in df.neg()
            #for fileid in df.fileids(category)]

stopSet = set(nltk.corpus.stopwords.words('english'))
xtraStops = []

#stopSet.append(xtraStops)
# Split the dataset by class values, returns a dictionary
# def separate_by_class(allReviews):
# 	separated = dict()
# 	for i in range(len(allReviews)):
# 		vector = allReviews[i]
# 		class_value = vector[-1]
# 		if (class_value not in separated):
# 			separated[class_value] = list()
# 		separated[class_value].append(vector)

# 	return separated



# word_tokens = wt(sets)
 
# filtered_sentence = [w for w in word_tokens if not w.lower() in stopSet]

# features = {}
#     #scans each document to find whether or not a given word was featured
# for word in word_tokens:
#    features['contains({})'.format(word)] = (word in filtered_sentence)
       
# classifier = nltk.NaiveBayesClassifier.train()   
# print(nltk.classify.accuracy(classifier, filtered_sentence))  
# classifier.show_most_informative_features(5)

# freqDist = nltk.FreqDist(w.lower() for w in filtered_sentence)

# freqDist.plot(20, cumulative=False);


 
