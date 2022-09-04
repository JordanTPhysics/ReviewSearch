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

from textblob import TextBlob
import gensim.corpora
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates
from matplotlib.figure import Figure
import os


from nltk.corpus import wordnet

LDA_TOPICS = 10

FIGURES = []

FIG_SIZE = (8, 4)

class NLP:

    def get_wordnet_pos(treebank_tag):
    
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return 'n'
    
    


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
    
    def clean_text(text):
        assert type(text) == str
        punct = string.punctuation + "``“”£"
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
        from nltk.stem import WordNetLemmatizer
        lems = []
        lemma = WordNetLemmatizer()
        
        for text in texts:
            pos = nltk.pos_tag(text)
            
            lemtext = []
            
            for word, tag in pos:
                lemmed = lemma.lemmatize(word, NLP.get_wordnet_pos(tag))
                lemtext.append(lemmed)
              
            lems.append(lemtext)
        return lems


    def plt_polarity(df):
        
        sents = df['polarity']
        sent_moving_average = []
        total = 0
        for i, s in enumerate(sents):
            total += s/(i+1)
            sent_moving_average.append(total)
        dates = matplotlib.dates.date2num(df['date'])
        fig = plt.figure(figsize = FIG_SIZE)
        
        plt.plot_date(dates, sent_moving_average)
        plt.title("average sentiment over time")
        plt.ylabel('Sentiment')
        plt.xlabel('review date')
        fig.savefig('static/figures/polarity.png')
        plt.show()
        #FIGURES.append(fig)
        plt.clf()
        return fig

    def calc_average(topic):
        total_sent = 0
        for i in topic:
            total_sent += i
        avg_sent = total_sent/len(topic)
        
        return avg_sent        



    def topic_freq(df, ldamodel):
        #currently returns a table with topic numbers
        
        
        data = df['processed']
        colors = plt.cm.BuPu(np.linspace(0.5, 1, LDA_TOPICS))
        words = gensim.corpora.Dictionary([d for d in data])
        corpus = [words.doc2bow(doc) for doc in data]
        labels = [str(i) for i in range(LDA_TOPICS)]
        
        
        #displays how many documents are associated with each topic
        doc_model = [ldamodel.get_document_topics(doc) for doc in corpus]
        topics = [0 for i in range(LDA_TOPICS)]
        
        for i in doc_model:
        
            theta = i[0][0]
            topics[theta] += 1
        
        fig = plt.figure(1, FIG_SIZE)
        plt.bar(labels,topics, color=colors)
        plt.title("Document-topic allocation")
        plt.ylabel(f"doc count. Total: {len(df)}")
        plt.xlabel('topic number')
        fig.savefig('static/figures/lda_dist.png')
        #FIGURES.append(fig)
        plt.show()
        plt.clf()
        return fig
        
    def topic_sent(df, ldamodel):
        
        data = df['processed']
        colors = plt.cm.BuPu(np.linspace(0.5, 1, LDA_TOPICS))
        words = gensim.corpora.Dictionary([d for d in data])
        corpus = [words.doc2bow(doc) for doc in data]
        labels = [str(i) for i in range(LDA_TOPICS)]
        
        doms = list(df['Topic_number'])
        sents = list(df['polarity'])
        topic_groups = [[] for i in range(LDA_TOPICS)]
        #grouping docs by main topic in LDA model
        for i, dom in enumerate(doms):
            topic_groups[int(dom)].append(sents[i])
        
        
        #calculates the average sentiment of docs for all given topic
        averages = list(map(NLP.calc_average,topic_groups))
        
        fig = plt.figure(2, FIG_SIZE)
        plt.bar(labels, averages, color=colors)
        plt.title('average sentiment per topic')
        plt.xlabel('topic number')
        plt.ylabel('average sentiment')
        fig.savefig('static/figures/topicsentavg.png')
        #FIGURES.append(fig)
        plt.show()
        plt.clf()
        
        return fig
        
       
        
    def format_topics(ldamodel):
        topics = []    
        for i, topic in ldamodel.show_topics(formatted=False):
            words = [word[0] for word in topic]
            topics.append(words)
        return [(i, topic) for i, topic in enumerate(topics)]

    def topiclize(df):
        
        
        
        data = df['processed']
        #create a dictionary of all the cleaned words found 
        words = gensim.corpora.Dictionary([d for d in data])
        corpus = [words.doc2bow(doc) for doc in data]
        print('Number of unique tokens: %d' % len(words))
        print('Number of documents: %d' % len(corpus))
        #converts to bag of words
        
        
        LDA = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                              id2word = words,
                                              num_topics = LDA_TOPICS,
                                              random_state=2,
                                              update_every=1,
                                              passes=10,
                                              alpha='auto',
                                              per_word_topics=True)
       
        
            
        
        
        
        #minimize this for maximum efficiency of LDA model
        print('LDA model perplexity: ', LDA.log_perplexity(corpus))
        
        df_topic_sents_keywords = NLP.format_topics_sentences(ldamodel=LDA,
                                                          corpus=corpus,
                                                          texts=data)
        
        df['Topic_words'] = df_topic_sents_keywords.Topic_Keywords
        df['Topic_number'] = df_topic_sents_keywords.Dominant_Topic
        
        
        sents = df['polarity']
        topic_groups = [[] for i in range(LDA_TOPICS)]
        #grouping docs by main topic in LDA model
        for i, dom in enumerate(df['Topic_number']):
            topic_groups[int(dom)].append(sents[i])
        
        averages = list(map(NLP.calc_average,topic_groups))
        
        average_values = []
        for i, dom in enumerate(df['Topic_number']):
            average_values.append(averages[int(dom)])
        
        df['Topic_Sentiment_Average'] = average_values
        #print(average_values, labels)
        #null values were appearing
        df.dropna(subset =['Topic_words','Topic_number'],inplace=True)
        return LDA
    
    def freqdist(df):
        all_words = []
        neg_words = []
        pos_words = []
        for i, words in enumerate(df['processed']):
            all_words.extend(words)
            
            if df['polarity'][i] > 0.2:
               pos_words.extend(words) 
            elif df['polarity'][i] < 0.2:
                neg_words.extend(words)
        
        fig = plt.figure(3, FIG_SIZE)
        plt.gcf().subplots_adjust(bottom=0.15) # to avoid x-ticks cut-off
        fdist = nltk.FreqDist(all_words)
        pdist = nltk.FreqDist(pos_words)
        ndist = nltk.FreqDist(neg_words)
        fdist.plot(20, cumulative=False, title='cleaned word dist for {company}', show=False)
        #pdist.plot(20, cumulative=False, title='positive words', show=False)
        #ndist.plot(20, cumulative=False, title='negative words', show=False)
        
        fig.savefig('static/figures/freqDist.png', bbox_inches = "tight")
        plt.clf()
        
        return fig
    
    def lda_vis(LDA, df):
        

        
        data = df['processed']
        
        words = gensim.corpora.Dictionary([d for d in data])
        corpus = [words.doc2bow(doc) for doc in data]
        
        lda_vis = pyLDAvis.gensim_models.prepare(LDA, corpus, words)
        pyLDAvis.display(lda_vis)
        pyLDAvis.save_html(lda_vis, f'./FileModel{LDA_TOPICS}.html')
        
        
        return df

    def wordcloud(LDA, stopList):
        
        from wordcloud import WordCloud
        
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
        plt.savefig('static/figures/wordcloud.png')
        
        #plt.show()
        plt.clf()

#this function performs data preprocessing and returns a df with cleaned texts
# and polarity rating.
    def preprocess(df):
        
        reviews = df['review']
        
        stopList = list(nltk.corpus.stopwords.words('english'))
        #additional stopwords
    
        stopList.extend(['get','u','ve','\'t','s','nt','ye'])
        
        cleaned = list(map(lambda x: NLP.clean_text(x), reviews))
        
        reviewWords = [wt(review) for review in cleaned]
        
        def remove_stops(text):
            return [w for w in text if w.lower() not in stopList]
        
        cleaned_stops = list(map(remove_stops,reviewWords))
        ############### LEMMATIZER #########################
        print('lemmatizing...')
        lemms = NLP.lemmatize_with_postag(cleaned_stops)
        
        
        print('making bigrams....')
        bigram_phrases = gensim.models.Phrases(lemms, min_count=6, threshold=50)    
        
        bigram = gensim.models.phrases.Phraser(bigram_phrases)  
        
        def make_bigrams(texts):
            
            return (bigram[doc] for doc in texts)
        
        data_bigrams = make_bigrams(lemms)
        print('data processing complete')
        df['processed'] = list(data_bigrams)
        
        sents = list(map(lambda x: TextBlob(x).sentiment.polarity, reviews))
            
        df['polarity'] = sents
        
        
        return df

def main(df):
    print(df['review'].head())
    
    
    #plots polarity as time function,
    #returns df with polarity and subjectivity values
    processed_data = NLP.preprocess(df)
    
    
    print('creating LDA model...')
    model = NLP.topiclize(processed_data)
    
    print('plotting polarity')
    NLP.plt_polarity(processed_data)
    NLP.topic_freq(processed_data, model)
    NLP.topic_sent(processed_data)
    NLP.format_topics(model)
    
    #plotting lda
    #returns a 2d list of topics and words
    NLP.lda_graph(processed_data, model)
    
    
    return
    
    
    
    #df.to_csv(f"./Data/analyzed.csv")
    
    ################## WORDCLOUD ##########################
    



if __name__ == "__main__":
    main(pd.read_csv('../Data/CHRTdata.csv'))
    














