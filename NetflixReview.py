# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:34:39 2021
https://github.com/joeyajames/Python/blob/master/NLTK/NLTK.ipynb
@author: starg
"""

import nltk, re, pprint

import random
import numpy as np
import pyttsx3
from nltk.tokenize import word_tokenize as wt, sent_tokenize as st
import matplotlib.pyplot as plt

from string import punctuation
import pandas as pd

from gensim.models import Word2Vec#, KeyedVectors
from textblob import TextBlob
#text2speech = pyttsx3.init()
#text2speech.say("welcome to the jungle, we got fun and games, we got everything you want honey we got the names")
# text2speech.runAndWait()


#import data as dataframe
df = pd.read_csv(r'Selenium/reviewdata.csv',header=[0])
df.columns=['DATE','REVIEW','RATING']
#remove null values from dataframe
df.dropna(subset =["DATE","REVIEW"],inplace=True)

stopList = list(nltk.corpus.stopwords.words('english'))
#add additional stop words relevant to this context
stopList.extend(['Netflix','netflix','account','...','I','And','service','movies','customer','\'ve','``','\'\''])
stopList.extend(punctuation)




#extract the reviews from tuple
reviews = list(df['REVIEW'].values)
ratings = list(df['RATING'].values)
split = len(reviews)//2
positive, negative = reviews[100:], reviews[:100]
  


#tokenize each review   
reviews_in_words = []  
for review in reviews:
    output = ''.join(c for c in review if not c.isdigit())
    tokens = wt(output)
    reviews_in_words.append(tuple(tokens))
    #The following prints a number from 1 to -1 based on the sentiment of the text
    #blob = TextBlob(review)
    #sentiment = blob.sentiment.polarity
    #print(sentiment)
tokenized_and_tagged = zip(reviews_in_words,ratings)
tnt = list(tokenized_and_tagged)
random.shuffle(tnt)
filtered_words = []
filtered_reviews = []
for i in reviews_in_words:
    filtered_review = [w.lower() for w in i if w not in stopList and w not in punctuation]
    filtered_reviews.append(filtered_review)
    for w in filtered_review:
        filtered_words.append(w)
        

    #each word represented by a vector based on it's relation to other words
happy, sad = reviews_in_words[:100], reviews_in_words[100:]
happytokens = []
sadtokens = []
for review in happy:
    for i in review:
        if i not in stopList:
            happytokens.append(i)
        
for review in sad:
    for i in review:
        if i not in stopList:
            sadtokens.append(i)        
model = Word2Vec(filtered_reviews, min_count=2)
model.wv.most_similar('happy')   

#plot frequency distribution from most common
all_words = nltk.FreqDist(filtered_words)
#all_words.plot(20, cumulative=False,title='all words') 
happyplot = nltk.FreqDist(happytokens)
sadplot = nltk.FreqDist(sadtokens)        
happyplot.plot(20, cumulative=False,title='positive')    
sadplot.plot(20, cumulative=False,title='negative')                       
#lists the 2000 most used words
word_features = list(filtered_words)[:2000]
def review_features(review):
    review_words = set(review)
    
    features = {}
    #scans each document to find whether or not a given word was featured
    for word in word_features:
        features['contains({})'.format(word)] = (word in review_words)
        
    return features
    
#for each document record which words featured and which class
#(pos,neg) the document falls into
featuresets = [(review_features(d), c) for (d,c) in tnt]
    
train_set, test_set = featuresets[100:], featuresets[:100]  

#trains the classifier to class reviews good and bad by showing it good and bad reviews.
classifier =  nltk.NaiveBayesClassifier.train(train_set)


print(nltk.classify.accuracy(classifier, test_set))
#5 features most skewed to pos or neg  
classifier.show_most_informative_features(5)




#6 most similar words to bad and good respectively
print(model.wv.most_similar('good',topn=6))
print(model.wv.most_similar('bad',topn=6))

plt.scatter(model.wv.vectors,model.wv.vectors)
#plt.show()










