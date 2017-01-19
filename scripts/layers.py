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

    def backward(self, dout):
        d1 = dout * self.in_2
        d2 = dout * self.in_1
        return d1, d2


class AdditionLayer():
    def __init__(self):
        pass

    def forward(self, in_1, in_2):
        return in_1 + in_2

    def backward(self, dout):
        d1 = dout * 1
        d2 = dout * 1
        return d1, d2


class ReLULayer():
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X <= 0)
        out = X.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class SigmoidLayer():
    def __init__(self):
        self.out = None

    def forward(self, X):
        out = 1 / (1 + np.exp(-X))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
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

    def backward(self, dout):
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
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

    def backward(self, dout=1):
        batch_size = self.labels.shape[0]
        if self.labels.size == self.out.size:  # one-hot
            dx = (self.out - self.labels) / batch_size
        else:
            dx = self.labels.copy()
            dx[np.arange(batch_size), self.labels] -= 1
            dx = dx / batch_size
        return dx


class ConvolutionLayer(NNOperations):
    def __init__(
            self,
            init_std,
            filter_num=16,
            channel_num=1,
            filter_size=3,
            stride=1,
            padding=0):
        self.W = self._W(
            filter_num, channel_num, filter_size, stride, padding, init_std)
        self.b = self._b(filter_num)
        self.stride = stride
        self.padding = padding

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def _W(
            self,
            filter_num,
            channel_num,
            filter_size,
            stride,
            padding,
            init_std):
        return init_std * np.random.randn(
            filter_num, channel_num, filter_size, filter_size)

    def _b(self, filter_num):
        return np.zeros(filter_num)

    def forward(self, x):
        """Applies product-sum operation filter.

        Computes inner products between input image matrix and filter matrix.
        im2col() function is used for transposing input as 2 dim matrix
        to culculate them efficiently.


        x : 4 dimentional matrix as (num, channels, height, width)

        Returns
          4 dimentional matrix as (num, channels, height, width)
        """
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.padding - FH) / self.stride)
        out_w = int(1 + (W + 2*self.padding - FW) / self.stride)

        col = self.im2col(x, FH, FW, self.stride, self.padding)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = self.col2im(dcol, self.x.shape, FH, FW, self.stride, self.padding)
        return dx


class PoolingLayer(NNOperations):
    def __init__(self, pool_h, pool_w, stride=1, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

        self.x = None
        self.arg_max = None

    def forward(self, x):
        """Applies maximum extraction operation filter.

        Computes inner products between input image matrix and filter matrix.
        im2col() function is used for transposing input as 2 dim matrix
        to culculate them efficiently.


        x : 4 dimentional matrix as (num, channels, height, width)

        Returns
          4 dimentional matrix as (num, channels, height, width)
        """
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = self.im2col(
            x, self.pool_h, self.pool_w, self.stride, self.padding)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] =\
            dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = self.col2im(
            dcol,
            self.x.shape,
            self.pool_h,
            self.pool_w,
            self.stride,
            self.padding)
        return dx


class DropoutLayer(NNOperations):
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, is_train=True):
        if is_train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
