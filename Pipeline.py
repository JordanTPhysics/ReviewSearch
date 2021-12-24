# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:51:09 2021
finding an optimal pipeline for useful language data
@author: starg
"""

import nltk, re, numpy as np, pandas as pd
from nltk.tokenize import word_tokenize as wt, sent_tokenize as st
import gensim
import string
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import gensim.corpora
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv(r'Selenium/reviewdata.csv',header=[0])
df.columns=['DATE','REVIEW','RATING']
#df.columns=['AUTHOR','COMMENT','REVIEW','RATING']
#remove null values from dataframe
df.dropna(subset =["DATE","REVIEW"],inplace=True)
stopList = list(nltk.corpus.stopwords.words('english'))
#additional stopwords
stopList.extend(['get'])
lemmatizer = WordNetLemmatizer()
vect = CountVectorizer()

    
reviews = list(df['REVIEW'].values)
ratings = list(df['RATING'].values)


reviewWords = [wt(review) for review in reviews]

#apply sentence tokenizing before punctuation removal to preserve sentence structure

sents = [st(review) for review in reviews]

        
############## NGRAMMER ###############


bigram_phrases = gensim.models.Phrases(reviewWords, min_count=6, threshold=50)    
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
    doc = [clean_text_round1(word) for word in doc if word not in stopList]
    doc = list(filter(None, doc))
    #clean_bigram = pos_tag(doc)
    clean_bigrams.append(doc)



############### LEMMATIZER #########################

    
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
    for word in lem:
        if word not in stopList:
            all_words.append(word)
    lems.append(lem)
    
random.shuffle(lems)  


################## LDA MODEL #################

words = gensim.corpora.Dictionary(lems)

corpus = [words.doc2bow(doc) for doc in lems]

LDA = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                      id2word = words,
                                      num_topics = 2,
                                      random_state=2,
                                      update_every=1,
                                      passes=10,
                                      alpha='auto',
                                      per_word_topics=True)
LDA.print_topics()

def format_topics_sentences(ldamodel=LDA, corpus=corpus, texts=lems):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=LDA, corpus=corpus, texts=lems)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)


################## WORDCLOUD ##########################


cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stopList,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = LDA.show_topics(formatted=False)

fig, axes = plt.subplots(1, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


########### DOCUMENT-TERM MATRIX #############

df['Lemmas'] = lems

DTMvectors = vect.fit_transform(df.Lemmas)



############ FREQUENCY PLOT #################


frequency_graph = nltk.FreqDist(all_words)
frequency_graph.plot(20,cumulative=False)
    