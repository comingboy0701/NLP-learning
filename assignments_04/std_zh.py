# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:03:11 2019

@author: ASUSNB
"""
import re
import jieba

input_file = 'zh_wiki_01'

p1 = re.compile(r'(<doc id=.*?>)|(</doc>)')
p2 = re.compile(r'\W') #[，；。？！\[\]（）\(\)《 》]

outfile= open('cut_std_'+input_file,encoding='utf-8', mode = 'w')
with open(input_file, encoding='utf-8', mode = 'r') as f:
    for line in f:
        line = p1.sub(r' ', line)
        line = p2.sub(r' ', line)
        line = line.strip()
        line = ' '.join([i for i in (list(jieba.cut(line))) if i.strip()])
        if line:
            outfile.write(line+'\n')
        print(line)
outfile.close()

from gensim.models import word2vec

sentences = word2vec.LineSentence('cut_std_zh_wiki_01')
model = word2vec.Word2Vec(sentences,size=10000,window=10,min_count=10,workers=4)
model.save('WikiCHModel')

#
model = word2vec.Word2Vec.load('WikiCHModel')
print(model.wv.similarity('奥运会','金牌')) #两个词的相关性
print(model.wv.most_similar(['伦敦','中国'],['北京'])) # 北京is to中国 as 伦敦is to？


from fig_model import tsne_plot

tsne_plot(model)


