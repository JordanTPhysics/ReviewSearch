# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 16:03:22 2022
Writing about physics is hard when it's so much easier to see in the world around us.
 Some people think it's the answer for everything, some see it as nature's artform in its purest state...
 I see it as a way to put the laws of nature into a context relatable to humans, a philosophy not unlike many others, a framework from which to draw morals, fundamentals and inspiration.
 The natural world is filled with patterns and routines which we call 'laws'.
some are simple and seemingly 'intuitive', like gravity, and some are deeply complex and seem to make no sense at all, like wave-particle duality.
Tied with such laws are fundamental constants of nature that govern and more importantly, limit the scope of phenomena and effects around us.
One might ask the question of why these limits are set upon us... and why they hold these specific numerical values?
The easiest answer to give is that these concepts only hold the values we observe comes from the act of observation itself:
If the constants held different values, even by the tiniest margin, then the evolution of the universe would have unfolded in such a radical and unpredictably different way that we, the observer and questioner would not be present to experience, or let alone question why.
This asks the further question of whether other 'beings', 'existing' in their own universe who may be considering this very 'concept' at this very 'moment' despite the unexplainably vast difference between labels and experiences. 
The thing is, these constants seem to bring a precision, or dare I say perfection to the world around us which may make the observer believe this is the way things should be.
These kinds of ideals are not far from those of religious people.
Many of the faithful accommodate the idea of a greater purpose and are determined to be a part of destiny to give their lives a sense of direction.
My only qualm with religion is the absolutist mindset they adopt, leaving little room for imagination and mystery, they claim to have all the answers, or at least that god does.
Contrarily, scientists hold the idea that the knowledge of the universe is a bottomless abyss of discovery and anyone who thinks having all the answers is achievable is no less than insane.
Recently I have explored and taken an interest in religion simply because of the concept of god itself.
Many of the world's religions conceptualize their god(s) as humans or animals or as a hybrid inbetween, but I think this is a purely simplifying approach to make them appear more relatable, the irony of which is not lost on me.
Personally, I think the concept of god extends beyond worldly and universal concepts out to an unexplainable context: more like an entity that pervades every physical, metaphysical, and any other type of 'physical' thing, whether or not it could be described in words.
Furthermore, all things share an equal part of god, who in turn is made of equal parts of us. To put it simply, just exising, consciously or unconsciously, is divine.



@author: starg
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:51:09 2021
finding an optimal pipeline for useful language data
try to find the optimal coherence and perplexity for an LDA model
Add a method to filter reviews with sentiment scores at certain thresholds

@author: starg
"""

import nltk, re, numpy as np, pandas as pd
from nltk.tokenize import word_tokenize as wt, sent_tokenize as st
import pyLDAvis
import pyLDAvis.gensim_models
import gensim
import string
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import gensim.corpora
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.dates
from wordcloud import WordCloud
from dateutil import parser


#import venn

########## INIT DATA ############

df = pd.read_csv(r'Selenium/reviewdata.csv',header=[0])
df.columns=['Date','Review','Rating']
#df.drop(columns=['Index'])
df = df[['Review','Rating','Date']]
#df.columns=['AUTHOR','COMMENT','REVIEW','RATING']
#remove null values from dataframe
df.dropna(subset =["Date","Review"],inplace=True)
stopList = list(nltk.corpus.stopwords.words('english'))
#additional stopwords
stopList.extend(['get','netflix','u','ve'])
lemmatizer = WordNetLemmatizer()
punct = string.punctuation +"``“”£"

def untokenize(doc):
    review = " ".join(doc)
    
    return review 
   
reviews = list(df['Review'].values)
ratings = list(df['Rating'].values)
dates = list(df['Date'].values)
dates = [parser.parse(date) for date in dates]
df['Date'] = dates

########### SENTIMENT ANALYSIS ##################
sents = list(map(lambda x: TextBlob(x).sentiment.polarity, reviews))
subs = list(map(lambda x: TextBlob(x).sentiment.subjectivity, reviews))

df['Polarity'] = sents
df['Subjectivity'] = subs

## plot of review sentiment as a function of time

dates = matplotlib.dates.date2num(df['Date'])
plt.plot_date(dates, df['Polarity'])
plt.title("sentiment over time")
plt.ylabel('Polarity')
plt.show()

reviewWords = [wt(review) for review in reviews]
all_unfiltered = []
for doc in reviewWords:
    for word in doc:
        all_unfiltered.append(word)


        
############# NGRAMMER ###############


bigram_phrases = gensim.models.Phrases(reviewWords, min_count=6, threshold=50)    
trigram_phrases = gensim.models.Phrases(bigram_phrases[reviewWords],min_count=3, threshold=50)
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

# pos_revs = []
# neg_revs = []
# for i in range(len(lems)):
#         if ratings[i] == 'neg':
#             neg_revs.append(lems[i])
#         elif ratings[i] == 'pos':
#             pos_revs.append(lems[i])
# cs_pos = 0
# cs_neg = 0
# test = 'problem'
# for doc in pos_revs:
    
#     if test in doc:
#         cs_pos = cs_pos + 1
    
#     pos_percent = cs_pos/len(pos_revs)*100    
        
# for doc in neg_revs:
    
#     if test in doc:
#         cs_neg = cs_neg + 1
#     neg_percent = cs_neg/len(neg_revs)*100    
    
    
# plt.bar(['positive','negative'],[pos_percent,neg_percent])
# plt.title("occurrences of: "+test+" in doc")
# plt.ylabel("percentage occurence")       
# plt.show() 
#random.shuffle(lems)  

############ WORD2VEC MODEL ##############


# model = Word2Vec(lems, workers=4,  min_count=5, window=10, sample=1e-3)
# #print("Words that are similar to customerservice:" , model.wv.most_similar('service',topn=6))
# #print("Words that are similar to problem:" , model.wv.most_similar('problem',topn=6))

# vocab = list(model.wv.key_to_index)
# X = model.wv[vocab]

# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X)

# w2v_dataframe = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
# Xvec = w2v_dataframe['x']
# Yvec = w2v_dataframe['y']
# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.scatter(Xvec,Yvec)
# ax.scatter(Xvec[10],Yvec[10],s=10, c='r', marker="o", label='second')
# plt.show()