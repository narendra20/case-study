#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:22:14 2019

@author: anishumesh
"""

import pandas as pd 
import numpy as np
import json

import requests, re
import pandas as pd
import seaborn as sns
import nltk
import string, itertools
from collections import Counter, defaultdict
from nltk.text import Text
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from sklearn.cluster import KMeans
from wordcloud import WordCloud

review = [json.loads(line) for line in open('/Users/anishumesh/Downloads/yelp_dataset/review.json', 'r')]
bd = [json.loads(line) for line in open('/Users/anishumesh/Downloads/yelp_dataset/business.json', 'r')]
rdf = pd.DataFrame(review)
bdf = pd.DataFrame(bd)

txt = ['Restaurants']
bdf['categories_new']= bdf['categories'].str.split(', ')
bdf1 = bdf.dropna(subset=['categories_new'])
all_restaurants=bdf1[bdf1['categories'].str.contains('Restaurants')]
#all_restaurants.head()
restaurants_reviews = pd.merge(all_restaurants, rdf, on = 'business_id')
#restaurants_reviews.to_csv('restaurant_reviews', encoding='utf-8', index=False)
#check for null reviews
#restaurants_reviews['text'].isna().sum()
#restaurants_reviews['category'].head
#restaurants_reviews_head

#Filter Cuisines
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('American'),'category'] = 'American'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Mexican'), 'category'] = 'Mexican'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Italian'), 'category'] = 'Italian'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Japanese'), 'category'] = 'Japanese'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Chinese'), 'category'] = 'Chinese'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Thai'), 'category'] = 'Thai'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Mediterranean'), 'category'] = 'Mediterranean'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('French'), 'category'] = 'French'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Vietnamese'), 'category'] = 'Vietnamese'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Greek'),'category'] = 'Greek'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Indian'),'category'] = 'Indian'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Korean'),'category'] = 'Korean'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Hawaiian'),'category'] = 'Hawaiian'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('African'),'category'] = 'African'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Spanish'),'category'] = 'Spanish'
restaurants_reviews.loc[restaurants_reviews.categories.str.contains('Middle_eastern'),'category'] = 'Middle_eastern'
# label reviews as positive or negative
restaurants_reviews['labels'] = ''
restaurants_reviews.loc[restaurants_reviews.stars_y >=4, 'labels'] = 'positive'
restaurants_reviews.loc[restaurants_reviews.stars_y ==3, 'labels'] = 'neutral'
restaurants_reviews.loc[restaurants_reviews.stars_y <3, 'labels'] = 'negative'
restaurants_reviews_head = restaurants_reviews.head(50000)

#Top 10 cities with most restaurants
import seaborn as sns
import matplotlib.pyplot as plt
import tslib
plt.style.use('ggplot')
plt.figure(figsize=(11,6))
grouped = restaurants_reviews.city.value_counts()[:10]
sns.barplot(grouped.index, grouped.values, palette=sns.color_palette("GnBu_r", len(grouped)))
plt.ylabel('Number of restaurants', fontsize=14, labelpad=10)
plt.xlabel('City', fontsize=14, labelpad=10)
plt.title('Count of Restaurants by City (Top 10)', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=15)
for  i, v in enumerate(grouped):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)

#Positive and Negative words
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
pd.set_option('display.float_format', lambda x: '%.4f' % x)
## convert text to lower case
restaurants_reviews.text = restaurants_reviews.text.str.lower()
## remove unnecessary punctuation
restaurants_reviews['removed_punct_text']= restaurants_reviews.text.str.replace('\n','').str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','')
## import positive file which contains common meaningless positive words such as good
file_positive = open('/Users/anishumesh/Downloads/positive-words.txt')
reader =csv.reader(file_positive)
positive_words=[]
for word in reader:
    if word:
        positive_words.append(word[0])

#import negative words
file_negative = open('/Users/anishumesh/Downloads/negative-words.txt')
reader2 =csv.reader(file_negative)
negative_words=[]
for word in reader2:
    if word:
        negative_words.append(word[0])
#Get dataset by category
def get_dataset(category):
    df = restaurants_reviews[['removed_punct_text','labels']][restaurants_reviews.category==category]
    df.reset_index(drop=True, inplace =True)
    df.rename(columns={'removed_punct_text':'text'}, inplace=True)
    return df
## only keep positive and negative words
def filter_words(review):
    words = [word for word in review.split() if word in positive_words + negative_words]
    words = ' '.join(words)
    return words

#Korean
Korean_reviews = get_dataset('Korean')
Korean_train, Korean_test = train_test_split(Korean_reviews[['text','labels']],test_size=0.5)
print('Total %d number of reviews' % Korean_train.shape[0])
def split_data(dataset, test_size):
    df_train, df_test = train_test_split(dataset[['text','labels']],test_size=test_size)
    return df_train
## construct features and labels
terms_train=list(Korean_train['text'])
class_train=list(Korean_train['labels'])

terms_test=list(Korean_test['text'])
class_test=list(Korean_test['labels'])
## get bag of words : the frequencies of various words appeared in each review
vectorizer = CountVectorizer()
feature_train_counts=vectorizer.fit_transform(terms_train)
feature_train_counts.shape
## run model
svm = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
svm.fit(feature_train_counts, class_train)
## create dataframe for score of each word in a review calculated by svm model
coeff = svm.coef_[0]
Korean_words_score = pd.DataFrame({'score': coeff, 'word': vectorizer.get_feature_names()})
## get frequency of each word in all reviews in specific category
Korean_reviews = pd.DataFrame(feature_train_counts.toarray(), columns=vectorizer.get_feature_names())
Korean_reviews['labels'] = class_train
Korean_frequency = Korean_reviews[Korean_reviews['labels'] =='positive'].sum()[:-1]
Korean_words_score.set_index('word', inplace=True)
Korean_polarity_score = Korean_words_score
Korean_polarity_score['frequency'] = Korean_frequency
## calculate polarity score 
Korean_polarity_score['polarity'] = Korean_polarity_score.score * Korean_polarity_score.frequency / Korean_reviews.shape[0]
## drop unnecessary words
unuseful_positive_words = Korean_polarity_score.loc[['great','amazing','love','best','awesome','excellent','good',
                                                    'favorite','loved','perfect','gem','perfectly','wonderful',
                                                    'happy','enjoyed','nice','well','super','like','better','decent','fine',
                                                    'pretty','enough','excited','impressed','ready','fantastic','glad','right',
                                                    'fabulous']]
unuseful_negative_words =  Korean_polarity_score.loc[['bad','disappointed','unfortunately','disappointing','horrible',
                                                     'lacking','terrible','sorry', 'disappoint']]

Korean_polarity_score.drop(unuseful_positive_words.index, axis=0, inplace=True)
Korean_polarity_score.drop(unuseful_negative_words.index, axis=0, inplace=True)
Korean_polarity_score.polarity = Korean_polarity_score.polarity.astype(float)
Korean_polarity_score.frequency = Korean_polarity_score.frequency.astype(float)
Korean_polarity_score[Korean_polarity_score.polarity>0].sort_values('polarity', ascending=False)[:20]
## filter words
Korean_train.text = Korean_train.text.apply(filter_words)


#for only head


bdf2 = bdf1[pd.DataFrame(bdf1.categories_new.tolist()).contains(txt).any(1)]

# for all restaurants

bdf_res[pd.DataFrame(bdf_res.categories.tolist()).isin(txt).any(1)]
bdf2_res['categories'].head(5)
bdf_res['categories'].replace('', np.nan, inplace=True)
bdf_res['categories'] = bdf_res['categories'].replace('', np.nan, inplace=True)
bdf_res['categories'] = bdf_res['categories'].dropna()
bdf2_res['categories'].isnull()








#Positive and Negative words
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
pd.set_option('display.float_format', lambda x: '%.4f' % x)
## convert text to lower case
restaurants_reviews_head.text = restaurants_reviews_head.text.str.lower()
## remove unnecessary punctuation
restaurants_reviews_head['removed_punct_text']= restaurants_reviews_head.text.str.replace('\n','').str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','')
## import positive file which contains common meaningless positive words such as good
file_positive = open('/Users/anishumesh/Downloads/positive-words.txt')
reader =csv.reader(file_positive)
positive_words=[]
for word in reader:
    if word:
        positive_words.append(word[0])

#import negative words
file_negative = open('/Users/anishumesh/Downloads/negative-words.txt')
reader2 =csv.reader(file_negative)
negative_words=[]
for word in reader2:
    if word:
        negative_words.append(word[0])
#Get dataset by category
def get_dataset(category):
    df = restaurants_reviews_head[['removed_punct_text','labels']][restaurants_reviews_head.category==category]
    df.reset_index(drop=True, inplace =True)
    df.rename(columns={'removed_punct_text':'text'}, inplace=True)
    return df
## only keep positive and negative words
def filter_words(review):
    words = [word for word in review.split() if word in positive_words + negative_words]
    words = ' '.join(words)
    return words

#Korean
Korean_reviews = get_dataset('Korean')
Korean_train, Korean_test = train_test_split(Korean_reviews[['text','labels']],test_size=0.5)
print('Total %d number of reviews' % Korean_train.shape[0])
def split_data(dataset, test_size):
    df_train, df_test = train_test_split(dataset[['text','labels']],test_size=test_size)
    return df_train
## construct features and labels
terms_train=list(Korean_train['text'])
class_train=list(Korean_train['labels'])

terms_test=list(Korean_test['text'])
class_test=list(Korean_test['labels'])
## get bag of words : the frequencies of various words appeared in each review
vectorizer = CountVectorizer()
feature_train_counts=vectorizer.fit_transform(terms_train)
feature_train_counts.shape
## run model
svm = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
svm.fit(feature_train_counts, class_train)
## create dataframe for score of each word in a review calculated by svm model
coeff = svm.coef_[0]
Korean_words_score = pd.DataFrame({'score': coeff, 'word': vectorizer.get_feature_names()})
## get frequency of each word in all reviews in specific category
Korean_reviews = pd.DataFrame(feature_train_counts.toarray(), columns=vectorizer.get_feature_names())
Korean_reviews['labels'] = class_train
Korean_frequency = Korean_reviews[Korean_reviews['labels'] =='positive'].sum()[:-1]
Korean_words_score.set_index('word', inplace=True)
Korean_polarity_score = Korean_words_score
Korean_polarity_score['frequency'] = Korean_frequency
## calculate polarity score 
Korean_polarity_score['polarity'] = Korean_polarity_score.score * Korean_polarity_score.frequency / Korean_reviews.shape[0]
Korean_polarity_score.frequency = pd.to_numeric(Korean_polarity_score.frequency, errors='coerce')
Korean_polarity_score.score
## drop unnecessary words
unuseful_positive_words = Korean_polarity_score.loc[['great','amazing','love','best','awesome','excellent','good',
                                                    'favorite','loved','perfect','gem','perfectly','wonderful',
                                                    'happy','enjoyed','nice','well','super','like','better','decent','fine',
                                                    'pretty','enough','excited','impressed','ready','fantastic','glad','right',
                                                    'fabulous']]
unuseful_negative_words =  Korean_polarity_score.loc[['bad','disappointed','unfortunately','disappointing','horrible',
                                                     'lacking','terrible','sorry', 'disappoint']]

Korean_polarity_score.drop(unuseful_positive_words.index, axis=0, inplace=True)
Korean_polarity_score.drop(unuseful_negative_words.index, axis=0, inplace=True)
Korean_polarity_score.polarity = Korean_polarity_score.polarity.astype(float)
Korean_polarity_score.frequency = Korean_polarity_score.frequency.astype(float)
Korean_polarity_score[Korean_polarity_score.polarity>0].sort_values('polarity', ascending=False)[:20]
## filter words
Korean_train.text = Korean_train.text.apply(filter_words)







from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(restaurants_reviews_head['removed_punct_text'])
X = cv.transform(restaurants_reviews_head['removed_punct_text'])
X_test = cv.transform(restaurants_reviews_head['removed_punct_text'])


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))
    
    
from flask import Flask
from flask.ext import restful
from flask.ext.restful import Resource, Api   
from flask import Response

import requests
from monkeylearn import MonkeyLearn

ml = MonkeyLearn('aa989be08a06899dfe7f7a85b4534573a4a784d1')
response = restaurants_reviews_head['removed_punct_text'].tolist()
resp2 = response[:100]
resp = Response(response=restaurants_reviews_head['removed_punct_text'].to_json(), status=200, mimetype="application/json")
return(resp)
data = restaurants_reviews_head['removed_punct_text']
model_id = 'cl_pi3C7JiL'
result = ml.classifiers.classify(model_id, resp2)
print(result.body)
resultdf = pd.DataFrame(result.body)



from textblob import TextBlob

def sentiment_calc(text):
    try:
        return TextBlob(text).sentiment
    except:
        return None

restaurants_reviews_head['sentiment'] = restaurants_reviews_head['removed_punct_text'].apply(sentiment_calc)

testimonial = TextBlob(restaurants_reviews_head['removed_punct_text'])
testimonial.sentiment
testimonial.sentiment.polarity



#VADER 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
sentiment = restaurants_reviews_head['removed_punct_text'].apply(lambda x: analyser.polarity_scores(x))
restaurants_reviews_head = pd.concat([restaurants_reviews_head,sentiment.apply(pd.Series)],1)