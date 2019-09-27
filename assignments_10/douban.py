# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


# 清洗数据，把文本数据变成 tf-idf 权重稀疏权重矩阵
fname = 'movie_comments.csv'
content = pd.read_csv(fname)

content = content[['comment','star']]
content = content.dropna()

p2 = re.compile(r'\W') #[，；。？！\[\]（）\(\)《 》]
def clean_word(line):
    line = p2.sub(r' ', line)
    line = line.strip()
    line = ' '.join([i for i in (list(jieba.cut(line))) if i.strip()])
    return line

content['comment'] = content['comment'].apply(lambda x: clean_word(x))
content = content.dropna()
content.to_csv('content_test.csv',index = False)

## TIF 的值
# content = pd.read_csv('content_train.csv')
X_test = content['comment']
tfidf = TfidfVectorizer(max_features=4000)
tfidf_trian= tfidf.fit_transform(X_test)   ## 拟合模型，返回tf-idf权重稀疏权重矩阵
weight_train= tfidf_trian.toarray() 


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pylab as plt

X, y = weight_train,content['star']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 666)

## 降维，首先确定方差涵盖了了总体的多少？
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
plt.plot([i for i in range(X_train.shape[1])],np.cumsum(pca.explained_variance_ratio_ ))


## (1)然后降维，按百分比降维，
pca = PCA(0.90)
pca.fit(X_train)
#print(pca.n_components_) # 2010 涵盖了90% 的信息量，在4000维度的时候
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

