# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:12:36 2021
a classifier I found online, for training purposes

@author: starg
We want to automate review collection from websites like amazon, collect the rating,
the date of the review, and the number of people who echo the review contents.
This word processor will then take the reviews and will be able to categorize good and bad reviews,
then display a word frequency chart detailing the common words and phrases found in positive and
 negative reviews respectively

"""

import nltk
from nltk.corpus import movie_reviews
import random


ReviewList = open('output10.txt')





#lists out all of the documents and categorizes them
documents=[(list(movie_reviews.words(fileid)), category) 
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
    
random.shuffle(documents)
#displayes a lowercase frequency distribution of all the words detected in the documents
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
#lists the 2000 most used words
word_features = list(all_words)[:2000]


def document_features(document):
    document_words = set(document)
    
    features = {}
    #scans each document to find whether or not a given word was featured
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
        
    return features
#defines the set of word features in the documents
featuresets = [(document_features(d), c) for (d,c) in documents]
#divides the set in 2 so one can be used to train the classifier and one can be used to test
train_set, test_set = featuresets[100:], featuresets[:100]  

#trains the classifier to class reviews good and bad by showing it good and bad reviews.
classifier =  nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))  
classifier.show_most_informative_features(5)

