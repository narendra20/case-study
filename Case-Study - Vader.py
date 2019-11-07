#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 18:51:16 2019

@author: anishumesh
"""

import pandas as pd


df=pd.read_csv('/Users/anishumesh/Downloads/Picking-dishes-from-Yelp-master/triples.csv')
food_reviews = df.groupby('restaurant_id').agg({'menu_item':','.join,'text': 'first','review_rating':'first','review_date':'first','user_name':'first' }).reset_index()

#VADER 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
sentiment = food_reviews['text'].apply(lambda x: analyser.polarity_scores(x))
food_reviews = pd.concat([food_reviews,sentiment.apply(pd.Series)],1)