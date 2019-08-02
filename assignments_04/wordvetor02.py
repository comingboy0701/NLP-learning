# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:34:19 2019

@author: dell
"""

from gensim.models import word2vec
import os
import logging

root = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.LineSentence('cut_std_zh_wiki_01')
model = word2vec.Word2Vec(sentences,size=10000,window=10,min_count=10,workers=4)
model.save('WikiCHModel03')