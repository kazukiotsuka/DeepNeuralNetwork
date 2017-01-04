#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# layers.py
#

import numpy as np
from nnoperations import NNOperations


class MultiplicationLayer():
    def __init__(self):
        self.in_1 = None
        self.in_2 = None

    def forward(self, in_1, in_2):
        self.in_1 = in_1
        self.in_2 = in_2
        return in_1 * in_2

    def backward(self, out):
        d1 = out * self.in_2
        d2 = out * self.in_1
        return d1, d2


class AdditionLayer():
    def __init__(self):
        pass

    def forward(self, in_1, in_2):
        return in_1 + in_2

    def backward(self, out):
        d1 = out * 1
        d2 = out * 1
        return d1, d2


class ReLULayer():
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X <= 0)
        out = X.copy()
        out[self.mask] = 0
        return out

    def backward(self, out):
        out[self.mask] = 0
        dx = out
        return dx


class SigmoidLayer():
    def __init__(self):
        self.out = None

    def forward(self, X):
        out = 1 / (1 + np.exp(-X))
        self.out = out
        return out

    def backward(self, out):
        dx = out * (1.0 - self.out) * self.out
        return dx


class HiddenLayer():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.original_x_shape = X.shape
        X = X.reshape(X.shape[0], -1)
        self.X = X
        out = np.dot(X, self.W) + self.b
        return out

    def backward(self, out):
        self.dW = np.dot(self.X.T, out)
        self.db = np.sum(out, axis=0)
        dx = np.dot(out, self.W.T)
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLossLayer(NNOperations):
    def __init__(self):
        self.cost = None
        self.out = None
        self.labels = None

    def forward(self, X, labels):
        self.labels = labels
        self.out = self.softmax(X)
        self.cost = self.crossEntropyError(self.out, self.labels)
        return self.cost

    def backward(self, out=1):
        batch_size = self.labels.shape[0]
        if self.labels.size == self.out.size:  # one-hot
            dx = (self.out - self.labels) / batch_size
        else:
            dx = self.labels.copy()
            dx[np.arange(batch_size), self.labels] -= 1
            dx = dx / batch_size
        return dx
