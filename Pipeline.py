# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:51:09 2021
finding an optimal pipeline for useful language data
@author: starg
"""

import nltk, re, numpy as np, pandas as pd
from nltk.tokenize import word_tokenize as wt, sent_tokenize as st
from nltk import pos_tag
import gensim
import string
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

df = pd.read_csv(r'Selenium/CanonReviewdata.csv',header=[0])
df.columns=['INDEX','DATE','REVIEW','RATING']
#remove null values from dataframe
df.dropna(subset =["DATE","REVIEW"],inplace=True)
stopList = list(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

pospron = 'PRP$'
    
reviews = list(df['REVIEW'].values)
ratings = list(df['RATING'].values)


reviewWords = [wt(review) for review in reviews]

#apply sentence tokenizing before punctuation removal to preserve sentence structure
sents = [st(review) for review in reviews]




bigram_phrases = gensim.models.Phrases(reviewWords, min_count=3, threshold=50)    
trigram_phrases = gensim.models.Phrases(bigram_phrases[reviewWords],min_count=1, threshold=50)
bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram =  gensim.models.phrases.Phraser(trigram_phrases)

def make_bigrams(texts):
    return(bigram[doc] for doc in texts)

def make_trigrams(texts):
    return(trigram[doc] for doc in texts)
  
data_bigrams = make_bigrams(reviewWords)
data_bigrams_trigrams = make_trigrams(data_bigrams)
bigrams_list = []
for i in data_bigrams:
    bigrams_list.append(i)

filtered_bigrams = []
for review in bigrams_list:
    filtered = [w.lower() for w in review if w not in stopList]
    filtered_bigrams.append(filtered)
    
def clean_text_round1(text):
    #lowercase
    #text = text.lower()
    #replace square brackets and content inside with ''
    text = re.sub('\[.*?\]', '', text)
    #remove instances of punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    #remove numbers and words attached to numbers
    text = re.sub('\w*\d\w*', '', text)
    
    return text
clean_bigrams = []
for doc in filtered_bigrams:
    doc = [clean_text_round1(word) for word in doc]
    doc = list(filter(None, doc))
    #clean_bigram = pos_tag(doc)
    clean_bigrams.append(doc)
    
def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)    
    


lems = []
all_words = []

for doc in clean_bigrams:
    lem = [lemmatize_with_postag(word) for word in doc]
    for word in doc:
        all_words.append(word)
    lems.append(lem)

frequency_graph = nltk.FreqDist(all_words)
frequency_graph.plot(20,cumulative=False)
    