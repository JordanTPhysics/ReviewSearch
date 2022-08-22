# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:21:58 2022

Application to process review dataframe and serve analytics

@author: starg
"""

import nltk, re, numpy as np, pandas as pd
from nltk.tokenize import word_tokenize as wt#, sent_tokenize as st
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
#import WordCloud
import os


from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    elif treebank_tag.startswith('P'):
        return wordnet.PRP
    elif treebank_tag.startswith('C'):
        return wordnet.CC
    else:
        return ''

punct = string.punctuation + "``“”£"

def make_bigrams(texts, stopList):
    
    bigram_phrases = gensim.models.Phrases(texts, min_count=6, threshold=50)    
    
    bigram = gensim.models.phrases.Phraser(bigram_phrases)  
    data_bigrams = make_bigrams(texts, bigram)
    
    bigrams_list = []
    for i in data_bigrams:
        bigrams_list.append(i)
    
    filtered_bigrams = []
    for review in bigrams_list:
        filtered = [w.lower() for w in review if w not in stopList]
        filtered_bigrams.append(filtered)
    
    clean_bigrams = []
    print('cleaning texts...')
    for doc in filtered_bigrams:
            doc = [clean_text_round1(word) for word in doc if word not in stopList]
            doc = list(filter(None, doc))
            #clean_bigram = pos_tag(doc)
            clean_bigrams.append(doc)
    
    
    return clean_bigrams
    
def make_trigrams(texts, trigram):
    
        return(trigram[doc] for doc in texts)

def untokenize(doc):
    review = " ".join(doc)
    
    return review





def prepare_data(df):
    
    reviews = list(df['reviews'])
    data = df
    
    clean_bigrams = []

    
    return data

def format_topics_sentences(ldamodel, corpus, texts):
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

def clean_text_round1(text):
    
    #replace square brackets and content inside with ''
    text = re.sub('\[.*?\]', '', text)
    #remove instances of punctuation
    text = re.sub('[%s]' % re.escape(punct), '', text)
    #remove numbers and words attached to numbers
    text = re.sub('\w*\d\w*', '', text)
    #remove /r and /n
    text = re.sub('\w*/\D*\w*', '', text)
    
    return text


def lemmatize_with_postag(texts):
    
    lems = []
    
    for text in texts:
        lem = []
        for word in text:
            pos = get_wordnet_pos()
    
    return  


def calc_average(topic):
    total_sent = 0
    for i in topic:
        total_sent += i[1]
    avg_sent = total_sent/len(topic)
    
    return avg_sent

def plt_polarity(df):
    
    reviews = list(df['review'].values)
    ratings = list(df['rating'].values)
    
    sents = list(map(lambda x: TextBlob(x).sentiment.polarity, reviews))
    subs = list(map(lambda x: TextBlob(x).sentiment.subjectivity, reviews))
    
    df['polarity'] = sents
    df['subjectivity'] = subs
    
    dates = matplotlib.dates.date2num(df['date'])
    plt.plot_date(dates, sents)
    plt.title("sentiment over time")
    plt.ylabel('Polarity')
    plt.xlabel('review time')
    plt.savefig('./figures/polarity.png', )
    plt.clf()
    
    return


def lda_graph(df, ldamodel):
    
    return

def topiclize(df):
    
    
    
    lems = df['lems']
    #create a dictionary of all the words found 
    words = gensim.corpora.Dictionary([d for d in lems])
    #change the number of topics to look for here
    LDAtopics = 16
    #converts to bag of words
    corpus = [words.doc2bow(doc) for doc in lems]
    
    LDA = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                          id2word = words,
                                          num_topics = LDAtopics,
                                          random_state=2,
                                          update_every=1,
                                          passes=10,
                                          alpha='auto',
                                          per_word_topics=True)
    
    doms = list(df['Topic_number'])
    sents = list(df['polarity'])
    
    doc_groups = [[] for i in range(LDAtopics)]
    for i in range(len(doms)):
        dom = int(doms[i])
        doc_groups[dom].append((lems[i],sents[i]))
        
        
    averages = []
    for i in range(len(doc_groups)):
        topic = doc_groups[i]
        average = calc_average(topic)
        
        
        averages.append(average)
    topnum = list(df['Topic_number'])
    average_values = []
    for i in range(len(topnum)):
        topic = topnum[i]
        average_values.append(averages[int(topic)])
        
    df['Topic_Sentiment_Average'] = average_values
    
    LDA.print_topics()
    #minimize this for maximum efficiency of LDA model
    print('LDA model perplexity: ', LDA.log_perplexity(corpus))
    
    
    
    
    
    lda_vis = pyLDAvis.gensim_models.prepare(LDA, corpus, words)
    pyLDAvis.display(lda_vis)
    #pyLDAvis.save_html(lda_vis, './FileModel'+ str(LDAtopics) +'.html')
    df_topic_sents_keywords = format_topics_sentences(ldamodel=LDA, corpus=corpus, texts=lems)
    
    df['Topic_words'] = df_topic_sents_keywords.Topic_Keywords
    df['Topic_number'] = df_topic_sents_keywords.Dominant_Topic
    df.dropna(subset =['Topic_words','Topic_number'],inplace=True)
    
    return df

def main(df):
    
    if df == None:
        df = pd.read_csv('Data/data.csv')

    stopList = list(nltk.corpus.stopwords.words('english'))
    #additional stopwords

    stopList.extend(['get','netflix','u','ve','\'t','s','nt','ye'])
    lemma = WordNetLemmatizer()
    
    
    reviewWords = [wt(review) for review in df['review']]
    
    
    
    
    
    
    ############### LEMMATIZER #########################

    lems = []
    all_words = []
    pos_words = []
    neg_words = []
    for i in range(len(clean_bigrams)):
        lem = [lemmatize_with_postag(word) for word in clean_bigrams[i]]
        for word in lem:
            if word not in stopList:
                all_words.append(word)
                try:
                    if  df['rating'][i] == 'pos':
                        pos_words.append(word)
                    else:
                        neg_words.append(word)
                except:
                    if  df['rating'][i] > 3:
                        pos_words.append(word)
                    else:
                        neg_words.append(word)
        lems.append(lem)
        
        
        
    pos_freq = nltk.FreqDist(pos_words)
    pos_freq.plot(20,cumulative=False, title = 'positive feedback words')
    plt.clf()
    
    neg_freq = nltk.FreqDist(neg_words)
    neg_freq.plot(20,cumulative=False, title = 'negative feedback words')
    plt.clf()
    
        
        
    df['lemmas'] = lems    
    
    
    

    df.to_csv(f"./Data/analyzed.csv")
    
    ################## WORDCLOUD ##########################
    
    
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    
    # cloud = WordCloud(stopwords=stopList,
    #                   background_color='white',
    #                   width=2500,
    #                   height=1800,
    #                   max_words=10,
    #                   colormap='tab10',
    #                   color_func=lambda *args, **kwargs: cols[i],
    #                   prefer_horizontal=1.0)
    
    # topics = LDA.show_topics(formatted=False)
    
    # fig, axes = plt.subplots(1, 2, figsize=(10,10), sharex=True, sharey=True)
    
    # for i, ax in enumerate(axes.flatten()):
    #     fig.add_subplot(ax)
    #     topic_words = dict(topics[i][1])
    #     cloud.generate_from_frequencies(topic_words, max_font_size=300)
    #     plt.gca().imshow(cloud)
    #     plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    #     plt.gca().axis('off')
    
    
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.axis('off')
    # plt.margins(x=0, y=0)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()














