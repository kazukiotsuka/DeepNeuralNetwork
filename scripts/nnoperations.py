#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# nnoperation.py
#

import numpy as np
from enum import Enum


class ActivationType(Enum):
    ReLU = 1
    Sigmoid = 2


class NNOperations():
    __activation_type__ = None

    def softmax(self, x):
        """Softmax function.
        """
        if x.ndim == 1:
            e_x = np.exp(x - np.max(x, axis=0))
            return e_x / np.sum(e_x, axis=0)
        elif x.ndim == 2:
            x = x.T
            e_x = np.exp(x - np.max(x, axis=0))
            return (e_x / np.sum(e_x, axis=0)).T

    def ReLU(self, x):
        """ReLU function.
        """
        return np.max(0, x)

    def sigmoid(self, x):
        """Sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def meanSquearedError(self, y, t):
        """Mean Squared Error.

        y: labeled data
        t: training data
        """
        return 0.5 * np.sum((y-t)**2)

    def crossEntropyError(self, y, t):
        """Cross Entropy Error.

        y: labeled data (one-hot)
        t: training data
        """
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        if t.size == y.size:  # one-hot-vector
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

    def numericalGradient(self, f, x):
        """Numerical Gradient.

        f: function of x
        x: variable vector like [x0, x1, x2,...xi]

        Returns gradients of each variables.
        """
        h = 1e-4  # small distance 0.0001
        x_shape = x.shape
        x = x.reshape(x.size)
        grad = np.zeros_like(x)  # gradients of each x
        for i in range(x.size):
            xi = x[i]
            x[i] = xi + h  # replace x[i] with xi+h
            fx_plus_h = f(x)
            x[i] = xi - h  # replace x[i] with xi-h
            fx_minus_h = f(x)
            grad[i] = (fx_plus_h - fx_minus_h) / (2*h)
            x[i] = xi  # reset x
        grad = grad.reshape(x_shape)
        return grad

    def gradientDescent(self, f, init_x, alpha=0.01, step=100):
        """Gradient Descent.

        f: optimized function
        init_x: initial variables x
        alpha: learning rate
        step: compute times
        """
        x = init_x
        for i in range(step):
            grad = self.numericalGradient(f, x)
            x -= alpha * grad
            print(x)
        return x

    def im2col(
            self, input_data, filter_h, filter_w, stride=1, padding=0):
        """Convert image matrix to 2 dimention column.

        input_data : 4 dim input data as (data length, channels, h, w)
        filter_h : filter height
        filter_w : filter width
        stride : stride
        padding : padding

        Returns
            col : 2 dim array
        """
        N, C, H, W = input_data.shape

        out_h = (H + 2*padding - filter_h)//stride + 1
        out_w = (W + 2*padding - filter_w)//stride + 1

        img = np.pad(
            input_data,
            [(0, 0), (0, 0), (padding, padding), (padding, padding)],
            'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):

            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] =\
                    img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    def col2im(
            self, col, input_shape, filter_h, filter_w, stride=1, padding=0):
        """Revert 2 dimention column to image matrix data.

        col : 2 dim column
        input_shape : input shape e.g. (10, 1, 28, 28)
        filter_h : filter height
        filter_w : filter width
        stride : stride
        padding : padding

        Returns
            4 dimentions image data as (data length, channels, h, w)
        """
        N, C, H, W = input_shape
        out_h = (H + 2*padding - filter_h)//stride + 1
        out_w = (W + 2*padding - filter_w)//stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)\
            .transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros(
            (N, C, H + 2*padding + stride - 1, W + 2*padding + stride - 1))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] +=\
                    col[:, :, y, x, :, :]

        return img[:, :, padding:H + padding, padding:W + padding]

    def initialWeightStd(
            self,
            activation_type: ActivationType,
            node_num: int):
        """Returns initial weight std.

        node_num: The number of nodes connected with the weights.

        When the activation type is ..
        ReLU -> He
        Sigmoid -> Xavier
        """
        if activation_type in (ActivationType.ReLU, 'ReLU'):
            return np.sqrt(2.0 / node_num)
        elif activation_type in (ActivationType.Sigmoid, 'Sigmoid'):
            return np.sqrt(1.0 / node_num)
        else:
            print('[WARNING] {} is invalid activation type'.format(
                activation_type))
