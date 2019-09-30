# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:58:29 2019

@author: comingboy
"""

from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.utils import resample
import numpy as np


class Node:
    def __init__(self, inputs=[]):
        self.inputs = inputs  # 输入节点
        self.value = []  # 当前的节点的值
        self.outputs = []  # 输出节点
        self.gradients = {}  # 对于每个 参数的偏导数
        for node in self.inputs:
            node.outputs.append(self)  # 加上一个连接关系，把所有的点连接起来

    def forward(self):  # 根据输入的点，计算输出点的值，存贮在self.value
        raise NotImplemented  # 虚类，必须在子类中实现

    def backward(self):  # 计算偏导数，存贮在self.gradients
        raise NotImplemented


class Input(Node):
    def __init__(self, name=''):
        Node.__init__(self, inputs=[])
        self.name = name

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self] = grad_cost

    def __repr__(self):
        return "Input Node:{}".format(self.name)


class Linear(Node):
    def __init__(self, nodes, weights, bias):
        self.x_node = nodes
        self.w_node = weights
        self.b_node = bias
        Node.__init__(self, inputs=[nodes, weights, bias])

    def forward(self):
        self.value = np.dot(
            self.x_node.value,
            self.w_node.value) + self.b_node.value

    def backward(self):
        for node in self.outputs:
            grad_cost = node.gradients[self]
            self.gradients[self.w_node] = np.dot(
                self.x_node.value.T, grad_cost)
            self.gradients[self.b_node] = np.sum(
                grad_cost * 1, axis=0, keepdims=False)
            self.gradients[self.x_node] = np.dot(
                grad_cost, self.w_node.value.T)


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
        self.x_node = node

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-1 * x))

    def forward(self):
        self.value = self._sigmoid(self.x_node.value)

    def backward(self):
        y = self.value
        self.partial = y * (1 - y)
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.x_node] = grad_cost * self.partial


class MSE(Node):
    def __init__(self, y_true, y_hat):
        self.y_true_node = y_true
        self.y_hat_node = y_hat
        Node.__init__(self, inputs=[y_true, y_hat])

    def forward(self):
        y_true_flatten = self.y_true_node.value.reshape(-1, 1)
        y_hat_flatten = self.y_hat_node.value.reshape(-1, 1)

        self.diff = y_true_flatten - y_hat_flatten

        self.value = np.mean(self.diff ** 2)

    def backward(self):
        n = self.y_hat_node.value.shape[0]

        self.gradients[self.y_true_node] = (2 / n) * self.diff
        self.gradients[self.y_hat_node] = (-2 / n) * self.diff


def training_one_batch(topological_sorted_graph):
    # graph 是经过拓扑排序之后的 一个list
    for node in topological_sorted_graph:
        node.forward()

    for node in topological_sorted_graph[::-1]:
        node.backward()


def topological_sort(data_with_value):
    feed_dict = data_with_value
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outputs:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]
            # if n is Input Node, set n'value as
            # feed_dict[n]
            # else, n's value is caculate as its
            # inbounds

        L.append(n)
        for m in n.outputs:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def sgd_update(trainable_nodes, learning_rate=1e-2):
    for t in trainable_nodes:
        t.value += -1 * learning_rate * t.gradients[t]


data = load_boston()
X_ = data['data']
y_ = data['target']
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)
n_features = X_.shape[1]
n_hidden = 10
n_hidden_2 = 10
W1_, b1_ = np.random.randn(n_features, n_hidden), np.zeros(n_hidden)
W2_, b2_ = np.random.randn(n_hidden, 1), np.zeros(1)

# 1st Build Nodes in this graph

X, y = Input(name='X'), Input(name='y')  # tensorflow -> placeholder
W1, b1 = Input(name='W1'), Input(name='b1')
W2, b2 = Input(name='W2'), Input(name='b2')

# build connection relationship

linear_output = Linear(X, W1, b1)
sigmoid_output = Sigmoid(linear_output)
yhat = Linear(sigmoid_output, W2, b2)
loss = MSE(y, yhat)


input_node_with_value = {  # -> feed_dict
    X: X_,
    y: y_,
    W1: W1_,
    W2: W2_,
    b1: b1_,
    b2: b2_
}

graph = topological_sort(input_node_with_value)


def run(dictionary):
    return topological_sort(dictionary)


losses = []
epochs = 5000

batch_size = 64

steps_per_epoch = X_.shape[0] // batch_size

for i in range(epochs):
    loss = 0

    for batch in range(steps_per_epoch):
        # indices = np.random.choice(range(X_.shape[0]), size=10, replace=True)
        # X_batch = X_[indices]
        # y_batch = y_[indices]
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        X.value = X_batch
        y.value = y_batch

        #         input_node_with_value = {  # -> feed_dict
        #             X: X_batch,
        #             y: y_batch,
        #             W1: W1.value,
        #             W2: W2.value,
        #             b1: b1.value,
        #             b2: b2.value,
        #         }

        #         graph = topological_sort(input_node_with_value)

        training_one_batch(graph)

        learning_rate = 1e-3

        sgd_update(
            trainable_nodes=[
                W1,
                W2,
                b1,
                b2],
            learning_rate=learning_rate)

        loss += graph[-1].value

    if i % 100 == 0:
        print('Epoch: {}, loss = {:.3f}'.format(i + 1, loss / steps_per_epoch))
        losses.append(loss)


plt.plot(losses)
plt.show()

# 测试实例

def _sigmoid(x):
    return 1. / (1 + np.exp(-1 * x))

y_predict = np.dot(_sigmoid(np.dot(X_,W1.value)+b1.value),W2.value)+b2.value

y_true = np.concatenate([y_predict, np.array(y_).reshape(-1,1)],axis=1)