# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import jieba
import re,os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from six.moves import cPickle as pickle

# 清洗数据，把文本数据变成 tf-idf 权重稀疏权重矩阵
fname = 'movie_comments.csv'
content = pd.read_csv(fname)
content = content[['comment','star']]
content = content.dropna()
# 只对中文进行情感的分类
#p2 = re.compile(r'\W') #[，；。？！\[\]（）\(\)《 》]# 只需要中文
def clean_word(line):
#    line = p2.sub(r' ', line)
    line = re.findall(r"[\u4e00-\u9fa5]+",line)
    if bool(line):
        line = " ".join(line)
        line = ' '.join([i for i in (list(jieba.cut(line))) if i.strip()])
    else:
        line = np.nan
    return line

content['comment'] = content['comment'].apply(lambda x: clean_word(x))
content = content.dropna(axis=0)
content["star"] = content["star"].apply(lambda x: int(x))
content.to_csv('content_test.csv',index = False)

## 2.利用 IDf 计算每条评论的向量值 
content = pd.read_csv('content_test.csv')
X_test = content['comment']
tfidf = TfidfVectorizer(max_features=1000)
tfidf_trian= tfidf.fit_transform(X_test)   ## 拟合模型，返回tf-idf权重稀疏权重矩阵
weight_train= tfidf_trian.toarray() 

# 3 PCA 降维，分层取样得到训练数据和验证数据
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from collections import Counter

X, y = weight_train,content['star']
ycounters = Counter(y)
plt.bar(ycounters.keys(),ycounters.values())

## 降维，首先确定方差涵盖了了总体的多少？
pca = PCA(n_components=X.shape[1])
pca.fit(X)
plt.plot([i for i in range(X.shape[1])],np.cumsum(pca.explained_variance_ratio_ ))


## (1)然后降维，按百分比降维，
pca = PCA(0.90)
pca.fit(X)
print(pca.n_components_) # 736 涵盖了90% 的信息量，在2000维度的时候
datasets = pca.transform(X)

## (2) 或者按维度降维

#pca = PCA(n_components=736)
#pca.fit(X_train)
#datasets = pca.transform(X)

# 4. 分成取样，由于数据的不平衡问题
train_dataset =  np.ndarray((20000*5, 735), dtype=np.float32)
train_labels = np.ndarray(20000*5, dtype=np.int32)
valid_dataset = np.ndarray((1000*5, 735), dtype=np.float32)
valid_labels = np.ndarray(1000*5, dtype=np.int32)
test_dataset =  np.ndarray((1000*5, 735), dtype=np.float32)
test_labels = np.ndarray(1000*5, dtype=np.int32)
 
num_eve = 20000
test_eve = 1000

for i,j in enumerate(range(1,6)):
    data_class = y[y==j]
    index = data_class.index
    train_dataset[i*num_eve:i*num_eve+num_eve,:]= datasets[index[:num_eve],:]
    train_labels[i*num_eve:i*num_eve+num_eve]= data_class[:num_eve]
    
    valid_dataset[i*test_eve:i*test_eve+test_eve,:] = datasets[index[num_eve:num_eve+test_eve],:]
    valid_labels[i*test_eve:i*test_eve+test_eve] = data_class[num_eve:num_eve+test_eve]
    test_dataset[i*test_eve:i*test_eve+test_eve,:] = datasets[index[num_eve+test_eve:num_eve+2*test_eve],:]
    test_labels[i*test_eve:i*test_eve+test_eve] = data_class[num_eve+test_eve:num_eve+2*test_eve]

## 数据预处理有没有错误
plt.bar(Counter(train_labels).keys(),Counter(train_labels).values())
plt.bar(Counter(test_labels).keys(),Counter(test_labels).values())
plt.bar(Counter(valid_labels).keys(),Counter(valid_labels).values())
plt.plot(train_dataset[:2000,1])


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


pickle_file = os.path.join( 'douban.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
  


## 3. 利用 支持向量机 进行分类,为啥 score 这么低？？？？
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(train_dataset,train_labels)
y_predict = svc.predict(valid_dataset)
svc.score(valid_dataset,valid_labels) #0.9824
print("精确率，召回率，F1：", classification_report(valid_labels, y_predict))

## 4. 利用tensorflow 进行分类



