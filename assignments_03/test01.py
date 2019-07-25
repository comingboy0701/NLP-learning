# encoding: utf-8
"""
@author: chen km
@contact: 760855003@qq.com
@file: test01.py
@time: 2019/7/20 16:47
"""
# 1. 梯度下降 拟合 函数
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import random

data = load_boston()
X, y = data['data'], data['target']
X_rm = X[:, 5]
plt.scatter(X_rm, y)
plt.show()

# 损失函数假设是(y-yi)^2


def price(rm, k, b):
    """f(x) = k * x + b"""
    return k * rm + b


def loss(y, y_hat):  # to evaluate the performance
    return sum((y_i - y_hat_i)**2 for y_i,
               y_hat_i in zip(list(y), list(y_hat))) / len(list(y))


def partial_k(x, y, y_hat):
    n = len(y)
    gradient = 0
    for x_i, y_i, y_hat_i in zip(list(x), list(y), list(y_hat)):
        gradient += (y_i - y_hat_i) * x_i
    return -2 / n * gradient


def partial_b(x, y, y_hat):
    n = len(y)
    gradient = 0
    for y_i, y_hat_i in zip(list(y), list(y_hat)):
        gradient += (y_i - y_hat_i)
    return -2 / n * gradient



trying_times = 2000
X, y = data['data'], data['target']
min_loss = float('inf')
current_k = random.random() * 200 - 100
current_b = random.random() * 200 - 100
learning_rate = 1e-04
update_time = 0
for i in range(trying_times):
    price_by_k_and_b = [price(r, current_k, current_b) for r in X_rm]
    current_loss = loss(y, price_by_k_and_b)
    if current_loss < min_loss:  # performance became better
        min_loss = current_loss
        if i % 50 == 0:
            print(
                'When time is : {}, get best_k: {} best_b: {}, and the loss is: {}'.format(
                    i,
                    current_k,
                    current_b,
                    min_loss))
    k_gradient = partial_k(X_rm, y, price_by_k_and_b)
    b_gradient = partial_b(X_rm, y, price_by_k_and_b)
    current_k = current_k + (-1 * k_gradient) * learning_rate
    current_b = current_b + (-1 * b_gradient) * learning_rate

# 2. 动态规划

