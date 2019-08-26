# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:18:54 2019

@author: comingboy
"""
import pandas as pd
import re
import jieba
# 清洗数据，把文本数据变成 tf-idf 权重稀疏权重矩阵
fname = 'sqlResult_1558435.csv'
content = pd.read_csv(fname, encoding='gb18030')

content = content[['source','content']]
content = content.dropna()

p2 = re.compile(r'\W') #[，；。？！\[\]（）\(\)《 》]
def clean_word(line):
    line = p2.sub(r' ', line)
    line = line.strip()
    line = ' '.join([i for i in (list(jieba.cut(line))) if i.strip()])
    return line

content['content2'] = content['content'].apply(lambda x: clean_word(x))
content = content.dropna()
content_test = content.iloc[:100,:]
content_train = content.iloc[100:,:]
content_test.to_csv('content_test.csv',index = False)
content_train.to_csv('content_train.csv',index = False)
