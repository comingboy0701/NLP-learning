# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:33:51 2019

@author: dell
"""

from gensim.models import word2vec
import os


root = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)

sentences = word2vec.LineSentence('cut_std_zh_wiki_00')
model = word2vec.Word2Vec(sentences,size=10000,window=10,min_count=10,workers=4)
model.save('WikiCHModel01')