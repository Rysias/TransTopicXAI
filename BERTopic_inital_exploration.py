# -*- coding: utf-8 -*-
"""
Initial exploration of the BERTopic package for my b.sc. thesis project.

Main goals of exploration: 
    - Looking at computational cost (is it feasible to run on a CPU?)
    - Exploring flexibility (how much info can I get out?)
    - Familiarizing myself with the API 

Created on Fri Sep  3 16:11:04 2021

@author: jhr
"""
import random
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Getting the Data #
docs = fetch_20newsgroups(subset='test',  remove=('headers', 'footers', 'quotes'))['data']
docs = random.sample(docs, 1000)

# initializing the model #
topic_model = BERTopic(embedding_model="paraphrase-MiniLM-L6-v2")
topics, probs = topic_model.fit_transform(docs)
