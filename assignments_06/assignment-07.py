# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:09:26 2019

@author: comingboy
"""

import pandas as pd
import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

## 清洗数据，把文本数据变成 tf-idf 权重稀疏权重矩阵
#fname = 'sqlResult_1558435.csv'
#content = pd.read_csv(fname, encoding='gb18030')
#
#content = content[['source','content']]
#content = content.dropna()
#
#p2 = re.compile(r'\W') #[，；。？！\[\]（）\(\)《 》]
#def clean_word(line):
#    line = p2.sub(r' ', line)
#    line = line.strip()
#    line = ' '.join([i for i in (list(jieba.cut(line))) if i.strip()])
#    return line
#
#content['content2'] = content['content'].apply(lambda x: clean_word(x))
#content = content.dropna()
#content_test = content.iloc[:100,:]
#content_train = content.iloc[100:,:]
#content_test.to_csv('content_test.csv',index = False)
#content_train.to_csv('content_train.csv',index = False)
## 预处理之后的数据,直接加载
content = pd.read_csv('content_train.csv')
content = content.dropna()
content['source2'] = content['source'].apply(lambda x:np.where(x=='新华社',1,0))

X_test = content['content2']
tfidf = TfidfVectorizer(max_features=4000)
tfidf_trian= tfidf.fit_transform(X_test)   ## 拟合模型，返回tf-idf权重稀疏权重矩阵
weight_train= tfidf_trian.toarray()   ## 拟合模型，返回tf-idf权重稀疏权重矩阵

## 根据 tf-idf权重稀疏权重矩阵 进行分类
# 维度太大，首先进行降维
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pylab as plt

X, y = weight_train,content['source2']

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

## (2) 或者按维度降维
#pca = PCA(n_components=500)
#pca.fit(X_train)
#X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)

## 1. 利用　KNN　进行分类, 时间太长
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train_pca,y_train)
score = knn_clf.score(X_test_pca,y_test)
y_predict = knn_clf.predict(X_test_pca)
knn_clf.score(X_test_pca,y_test)
print("精确率，召回率，F1：", classification_report(y_test, y_predict))



## 2. 利用　logistic　进行分类
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train_pca,y_train)
log_reg.score(X_test_pca,y_test) # 0.9726
y_predict = log_reg.predict(X_test_pca)
log_reg.score(X_test_pca,y_test)
print("精确率，召回率，F1：", classification_report(y_test, y_predict))



## 3. 利用 决策树 进行分类
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(max_depth = 2,criterion='entropy')
dt_reg.fit(X_train_pca,y_train)
dt_reg.score(X_test_pca,y_test) # 太低了吧 0.0125
y_predict = dt_reg.predict(X_test_pca)
dt_reg.score(X_test_pca,y_test)
print("精确率，召回率，F1：", classification_report(y_test, y_predict))



## 4. 利用 随机森林 进行分类
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=4,random_state=666,
                               oob_score=True,n_jobs=-1)
rf_clf.fit(X,y)
rf_clf.oob_score_ # 0.81283
y_predict = rf_clf.predict(X_test_pca)
rf_clf.score(X_test_pca,y_test)
print("精确率，召回率，F1：", classification_report(y_test, y_predict))


## 5. 利用 支持向量机 进行分类
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train_pca,y_train)
y_predict = svc.predict(X_test_pca)
svc.score(X_test_pca,y_test) #0.9824
print("精确率，召回率，F1：", classification_report(y_test, y_predict))

## 根据分类的模型 找出抄袭的文章
content_test = pd.read_csv('content_test.csv')
model = svc # svc 比较好
txt_test = content_test['content2']
content_test_idf = tfidf.transform(txt_test).toarray()

content_test_idf_pca = pca.transform(content_test_idf)

predict_source = model.predict(content_test_idf_pca)
index = np.vstack([predict_source ==1,content_test['source']!='新华社']).all(axis=0)

copy_xinhua_news = content_test.loc[index,:] # 抄袭的文章


