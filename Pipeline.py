
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:51:09 2021
finding an optimal pipeline for useful language data
try to find the optimal coherence and perplexity for an LDA model
Add a method to filter reviews with sentiment scores at certain thresholds

for initializing class variables fast
class A(object):
    def __init__(self, a, b, c, d, e, f):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})


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
from wordcloud import WordCloud
import os
import mysql.connector
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
import warnings
warnings.filterwarnings("ignore")


############ INIT DATA ############

USERNAME = os.environ['MYSQL_USER']
PASSWORD = os.environ['MYSQL_PASS']


#USERNAME = input('enter mysql username: ')
#PASSWORD = input('enter mysql password: ')
db = mysql.connector.connect(host="localhost", user=f"{USERNAME}", passwd=f"{PASSWORD}")
pointer = db.cursor()
pointer.execute("use reviewdata")
TABLE_NAME = input('Choose an sql table: ')
COMPANY = input('choose a company: ')
pointer.execute(f"SELECT * FROM {TABLE_NAME} WHERE company_id='{COMPANY}'")
df = pointer.fetchall()
try:
    df = pd.DataFrame(df,columns =['Index','Company','Preview','Review','Date','Rating','Likes'])
    print('likes included')
except:
    df = pd.DataFrame(df,columns =['Index','Company','Preview','Review','Date','Rating'])
    

stopList = list(nltk.corpus.stopwords.words('english'))
#additional stopwords
stopList.extend(['get','netflix','u','ve','\'t'])
lemmatizer = WordNetLemmatizer()
punct = string.punctuation +"``“”£"

def untokenize(doc):
    review = " ".join(doc)
    
    return review 
   
reviews = list(df['Review'].values)
ratings = list(df['Rating'].values)

#df.to_csv('CanonReviewdata.csv',header=False)
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
plt.xlabel('review time')
plt.show()

df['Month'] = df['Date'].dt.month
data_score = df.groupby("Month", as_index=False) \
.agg(('count', 'mean')) \
.reset_index()



# plt.plot(data_score['Month'], data_score['Rating']['mean'])
# plt.title('Average Star Rating per month')
# plt.xlabel('last 12 months')
# plt.ylabel('Rating out 5')
# plt.axhline(data_score['Rating']['mean'].mean(), color='green', linestyle='--')
# plt.show()

##plot of review length against sentiment
lengths = list(map(lambda x: len(x),reviews))

plt.scatter(sents, lengths)
plt.ylabel('review length')
plt.xlabel('review sentiment')
plt.title('sentiment for review length')
plt.show()
reviewWords = [wt(review) for review in reviews]
all_unfiltered = []
for doc in reviewWords:
    for word in doc:
        all_unfiltered.append(word)


        
############## NGRAMMER ###############


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
    
    
    


    
def clean_text_round1(text):
    #lowercase
    
    #replace square brackets and content inside with ''
    text = re.sub('\[.*?\]', '', text)
    #remove instances of punctuation
    text = re.sub('[%s]' % re.escape(punct), '', text)
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
pos_words = []
neg_words = []
for i in range(len(clean_bigrams)):
    lem = [lemmatize_with_postag(word) for word in clean_bigrams[i]]
    for word in lem:
        if word not in stopList:
            all_words.append(word)
            try:
                if  df['Rating'][i] == 'pos':
                    pos_words.append(word)
                else:
                    neg_words.append(word)
            except:
                if  df['Rating'][i] > 3:
                    pos_words.append(word)
                else:
                    neg_words.append(word)
    lems.append(lem)
    
pos_freq = nltk.FreqDist(pos_words)
pos_freq.plot(20,cumulative=False, title = 'positive feedback words')

neg_freq = nltk.FreqDist(neg_words)
neg_freq.plot(20,cumulative=False, title = 'negative feedback words') 
    
    
df['Lemmas'] = lems    



################## LDA MODEL #################

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
LDA.print_topics()
#minimize this for maximum efficiency of LDA model
print('LDA model perplexity: ', LDA.log_perplexity(corpus))





lda_vis = pyLDAvis.gensim_models.prepare(LDA, corpus, words)
pyLDAvis.display(lda_vis)
pyLDAvis.save_html(lda_vis, './FileModel'+ str(LDAtopics) +'.html')

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

df['Topic_words'] = df_topic_sents_keywords.Topic_Keywords
df['Topic_number'] = df_topic_sents_keywords.Dominant_Topic


df.dropna(subset =['Topic_words','Topic_number'],inplace=True)


doms = list(df['Topic_number'])

doc_groups = [[] for i in range(LDAtopics)]
for i in range(len(doms)):
    dom = int(doms[i])
    doc_groups[dom].append((lems[i],sents[i]))
    
    
def calc_average(topic):
    total_sent = 0
    for i in topic:
        total_sent += i[1]
    avg_sent = total_sent/len(topic)
    
    return avg_sent
    

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

Base = declarative_base()

class Review(Base):
    __tablename__ = f'{COMPANY}analysis'
    # Here we define columns for the table
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    company_id = Column(String(250), nullable=False)
    review = Column(String(250), nullable=False)
    date = Column(DATETIME(), nullable=False)
    polarity = Column(FLOAT)
    topic_words = Column(String(500))
    dominant_topic = Column(Integer)
    topic_average = Column(FLOAT)

SQLdf = pd.DataFrame(data=[df['Company'],df['Review'],df['Date'],df['Polarity'],df['Topic_words'],df['Topic_number'],df['Topic_Sentiment_Average']])
SQLdf = SQLdf.T






doc_model = [LDA.get_document_topics(doc) for doc in corpus]
xs = []
ys = []
multis = []
topics = [0 for i in range(LDAtopics)]
labels = [str(i) for i in range(LDAtopics)]
for i in doc_model:
    
        theta = i[0][0]
        topics[theta] += 1
        r = i[0][1]
        xs.append(theta)
        ys.append(r)
area = 200
colors = 2 * np.pi * np.random.rand(len(xs))

"Uncomment to see a pretty plot"
#plots = plt.figure()
#ax = plots.add_subplot(projection='polar',label="Document-topic allocation")

#c = ax.scatter(xs, ys, c=colors, s=area, cmap='hsv', alpha=0.75)



plt.bar(labels,topics)
plt.title("Document-topic allocation")
plt.ylabel("doc count. Total: "+str(len(xs)))
plt.show()

for topic in LDA.print_topics():
    print("Topic: ")
    print(topic)

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


############ FREQUENCY PLOT #################
print(len(all_unfiltered)-len(all_words))

frequency_graph = nltk.FreqDist(all_words)
frequency_graph.plot(20,cumulative=False)

engine = create_engine(f'mysql://{USERNAME}:{PASSWORD}@localhost:3306/reviewdata', echo=True)
SQLdf.to_sql(f'{COMPANY}analysis',engine)




    