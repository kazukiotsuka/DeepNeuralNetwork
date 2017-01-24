#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# layers.py
#

import numpy as np
from nnoperations import NNOperations
from config import Config
from config import DebugLevel


class NNLayer():
    """Layer Base Class.
    """
    __layer_name__ = None  # layer name
    index = None  # index in layers
    name = None

    def __init__(self, index=None):
        self.index = index
        if not index:
            self.name = self.__layer_name__
        else:
            self.name = '{}{}'.format(self.__layer_name__, index)


def debug(is_debug=False, debug_level=DebugLevel.INFO):
    """Stdout layer info.

    Output by DebugLevel
    - INFO : only label and nonzero info
    - DETAILS : label, data, nonzero info
    """
    def _debug(func):
        def wrapper(*args, **kwargs):
            if is_debug:
                layer_name = args[0].name
                func_name = func.__name__
                data = args[1]
                if debug_level.value > DebugLevel.INFO.value:
                    print('{} {} input {}'.format(layer_name, func_name, data))
                else:
                    print('{} {}'.format(layer_name, func_name))

                if isinstance(data, (list, np.ndarray)):
                    non_zero_num = len(np.nonzero(data.flatten())[0])
                    print('nonzero {}/{}'.format(
                        non_zero_num, len(data.flatten())))
                    if non_zero_num == 0:
                        print('!!!!!!!!!!!! all zero !!!!!!!!!!!!!!!')
                        print('[WARN] a suspected case of vanishing gradient.')
            return func(*args, **kwargs)
        return wrapper
    return _debug


class MultiplicationLayer(NNLayer):
    __layer_name__ = 'MaltiplicationLayer'

    def __init__(self, index=None):
        super().__init__(index)
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


class AdditionLayer(NNLayer):
    __layer_name__ = 'AdditionLayer'

    def __init__(self, index=None):
        super().__init__(index)

    def forward(self, in_1, in_2):
        return in_1 + in_2

    def backward(self, dout):
        d1 = dout * 1
        d2 = dout * 1
        return d1, d2


class ReLULayer(NNOperations, NNLayer):
    __layer_name__ = 'ReLULayer'

    def __init__(self, index=None):
        super().__init__(index)
        self.mask = None

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def forward(self, X):
        self.mask = (X <= 0)
        out = X.copy()
        out[self.mask] = 0
        return out

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class SigmoidLayer(NNLayer):
    __layer_name__ = 'SigmoidLayer'

    def __init__(self, index=None):
        super().__init__(index)
        self.out = None

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def forward(self, X):
        out = 1 / (1 + np.exp(-X))
        self.out = out
        return out

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class HiddenLayer(NNOperations, NNLayer):
    __layer_name__ = 'HiddenLayer'

    def __init__(
            self,
            index,
            activation_type,
            pre_node_num,
            next_node_num):
        super().__init__(index)
        init_std = self.initialWeightStd(
            activation_type=activation_type,
            pre_node_num=pre_node_num)
        self.W = self._W(init_std, pre_node_num, next_node_num)
        self.b = self._b(next_node_num)
        self.original_x_shape = None
        self.X = None
        self.dW = None
        self.db = None

    def _W(self, init_std, pre_node_num, next_node_num):
        return init_std * np.random.randn(pre_node_num, next_node_num)

    def _b(self, next_node_num):
        return np.zeros(next_node_num)

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def forward(self, X):
        self.original_x_shape = X.shape
        X = X.reshape(X.shape[0], -1)
        self.X = X
        out = np.dot(X, self.W) + self.b
        return out

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def backward(self, dout):
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLossLayer(NNOperations, NNLayer):
    __layer_name__ = 'SoftmaxWithLossLayer'

    def __init__(self, index=None):
        super().__init__(index)
        self.cost = None
        self.dout = None
        self.labels = None

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def forward(self, X, labels):
        self.labels = labels
        self.dout = self.softmax(X)
        self.cost = self.crossEntropyError(self.dout, self.labels)
        return self.cost

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def backward(self, dout=1):
        batch_size = self.labels.shape[0]
        if self.labels.size == self.dout.size:  # one-hot
            dx = (self.dout - self.labels) / batch_size
        else:
            dx = self.dout.copy()
            dx[np.arange(batch_size), self.labels] -= 1
            dx = dx / batch_size
        return dx


class ConvolutionLayer(NNOperations, NNLayer):
    __layer_name__ = 'ConvolutionLayer'

    def __init__(
            self,
            index,
            activation_type,
            filter_num=16,
            channel_num=1,
            filter_size=3,
            stride=1,
            padding=0):
        super().__init__(index)
        init_std = self.initialWeightStd(
            activation_type=activation_type,
            pre_node_num=channel_num*filter_size*filter_size)
        self.W = self._W(
            filter_num, channel_num, filter_size, stride, padding, init_std)
        self.b = self._b(filter_num)
        self.filter_num = filter_num
        self.filter_size = filter_size
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

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
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

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = self.col2im(dcol, self.x.shape, FH, FW, self.stride, self.padding)
        return dx


class PoolingLayer(NNOperations, NNLayer):
    __layer_name__ = 'PoolingLayer'

    def __init__(self, index=None, pool_h=0, pool_w=0, stride=1, padding=0):
        super().__init__(index)
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

        self.x = None
        self.arg_max = None

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
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

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
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


class DropoutLayer(NNOperations, NNLayer):
    __layer_name__ = 'DropoutLayer'

    def __init__(self, index=None, dropout_ratio=0.5):
        super().__init__(index)
        self.dropout_ratio = dropout_ratio
        self.mask = None

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def forward(self, x, is_train=True):
        if is_train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    @debug(is_debug=Config.IS_DEBUG, debug_level=Config.DEBUG_LEVEL)
    def backward(self, dout):
        return dout * self.mask
