# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:30:15 2019

@author: comingboy
"""

# 1随机选取 
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import random
data = load_boston()
X,y = data['data'], data['target']
X_rm = X[:,5]

def price(rm,k,b):
    return k*rm+b
def loss(y,y_hat):
    return sum((y_i-y_hat_i)**2 for y_i,y_hat_i in zip(list(y),list(y_hat)))/len(list(y))

try_times = 20000
min_loss= float('inf')
best_k, best_b = None,None
for i in range(try_times):
    k = random.random()*200-100
    b = random.random()*200-100
    price_by_random_k_and_b = [price(r,k,b) for r in X_rm]
    current_loss = loss(y, price_by_random_k_and_b)
    if current_loss < min_loss:
        min_loss = current_loss
        best_k, best_b = k, b
        print('When time is : {}, get best_k: {} best_b: {}, and the loss is: {}'.format(i, best_k, best_b, min_loss))
plt.figure(1)
price_by_random_k_and_b = [price(r,best_k,best_b) for r in X_rm]
plt.scatter(X_rm,y)
plt.scatter(X_rm,price_by_random_k_and_b)

# 2监督方向

try_times = 2000
min_loss =float('inf')
best_k = random.random() * 200 - 100
best_b = random.random() * 200 - 100
direction = [(1,1),
             (1,-1),
             (-1,1),
             (-1,-1)]
next_direction = random.choice(direction)
scalar = 0.1
for i in range(try_times):
    k_direction,b_direction = next_direction
    current_k,current_b = best_k+scalar*k_direction,best_b+ scalar*b_direction
    price_by_k_and_b = [price(r,current_k,current_b) for r in X_rm]
    current_loss = loss(y,price_by_k_and_b)
    if current_loss<min_loss:
        min_loss = current_loss
        best_k,best_b = current_k,current_b
        next_direction = next_direction
        print('When time is : {}, get best_k: {} best_b: {}, and the loss is: {}'.format(i, best_k, best_b, min_loss))
    else:
        next_direction = random.choice(direction)
plt.figure(2)
price_by_k_and_b = [price(r,best_k,best_b) for r in X_rm]
plt.scatter(X_rm,y)
plt.scatter(X_rm,price_by_k_and_b)
    