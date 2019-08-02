# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:07:06 2019

@author: ASUSNB
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    size = 0
    for word in model.wv.vocab: # fig first 100
        if size >=300:
            break
        tokens.append(model[word])
        labels.append(word)
        size +=1
    # PCA 2 main
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()