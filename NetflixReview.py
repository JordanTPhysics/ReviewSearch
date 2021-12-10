# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:34:39 2021
https://github.com/joeyajames/Python/blob/master/NLTK/NLTK.ipynb
@author: starg
"""

import nltk#, re, pprint

import random
import numpy as np

from nltk.tokenize import word_tokenize as wt
import matplotlib.pyplot as plt
#import csv
from string import punctuation
from sklearn.decomposition import PCA
import pandas as pd
#from nltk import sent_tokenize, word_tokenize, pos_tag
from gensim.models import Word2Vec#, KeyedVectors
#from textblob import TextBlob


#import data as dataframe
df = pd.read_csv(r'reviewdata.csv',header=[0])
df.columns=['DATE','REVIEW','RATING']
#remove null values from dataframe
df.dropna(subset =["DATE","REVIEW"],inplace=True)
#remove date column as it has no use in sentiment analysis
del df['DATE']
#dataframe to list
allReviews = df.values.tolist()
random.shuffle(allReviews)
stopList = list(nltk.corpus.stopwords.words('english'))
stopList.extend(['Netflix','netflix','account','...','I'])
stopList.extend(punctuation)




#extract the reviews from tuple
reviews = list(df['REVIEW'].values)
ratings = list(df['RATING'].values)

    
def applystopwords(aReview):
    aReview = list(aReview)
    for token in aReview:
       if token in stopList or token in punctuation:
           aReview.remove(token)
    
    return aReview
    
#tokenize each review   
reviews_in_words = []  
for review in reviews:
    
    tokens = wt(review)
    reviews_in_words.append(tuple(tokens))
    #The following prints a number from 1 to -1 based on the sentiment of the text
    #blob = TextBlob(review)
    #sentiment = blob.sentiment.polarity
    #print(sentiment)
tokenized_and_tagged = zip(reviews_in_words,ratings)

#retrieve each review as list of words all cat into single list   
filtered_reviews = []
for i in reviews_in_words:
    filtered_reviews.append(applystopwords(i))
        
   

    
  

TheReviews = list(df['REVIEW'].values)

tokenized = [nltk.word_tokenize(review) for review in TheReviews]
    
model = Word2Vec(filtered_reviews, min_count=2, vector_size=32)
model.wv.most_similar('happy')   

#plot frequency distribution from most common
all_words = nltk.FreqDist(filtered_reviews)
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





#each word represented by a vector based on it's relation to other words




#6 most similar words to bad and good respectively
print(model.wv.most_similar('good',topn=6))
print(model.wv.most_similar('bad',topn=6))

X = model.wv.index_to_key


pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.scatter(result[:, 0], result[:, 1])
wordvecs = model.wv.index_to_key
for i, wordvec in enumerate(wordvecs):
	plt.annotate(i, xy=(result[i, 0], result[i, 1]))
plt.show()







