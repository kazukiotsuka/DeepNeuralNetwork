#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# neuralnetwork.py
#

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from load_image_data import LoadImageData
from nnoperations import NNOperations
from layers import ReLULayer
from layers import HiddenLayer
from layers import SoftmaxWithLossLayer


class NeuralNetwork(NNOperations):

    def __init__(
            self, input_size, hidden_size, output_size, init_weight=0.01):
        self.input_size = input_size
        self.params = {}
        self.params['W1'] = init_weight * np.random.randn(
            input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = init_weight * np.random.randn(
            hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # setup layers
        self.layers = OrderedDict()
        self.layers['Hidden1'] = HiddenLayer(
            self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLULayer()
        self.layers['Hidden2'] = HiddenLayer(
            self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLossLayer()

    def forward(self, x):
        """Forward propagation.

        x: input data
        """
        for key, layer in self.layers.items():
            x = layer.forward(x)
        return x

    def computeCost(self, y, t):
        """Compute Cost.

        y: output result
        t: labeled data
        """
        return self.lastLayer.forward(y, t)

    def computeAccuracy(self, x, labels):
        """Compute Accuracy.
        """
        ans = np.argmax(self.forward(x), axis=1)
        if labels.ndim != 1:
            label = np.argmax(labels, axis=1)
        accuracy = np.sum(ans == label) / float(x.shape[0])
        return accuracy

    def computeGradientWithBackPropagation(self, x, t):
        """Compute Gradient with Back Propagation.

        x: input data
        t: labeled data
        """

        # forward
        self.computeCost(self.forward(x), t)

        # backward
        layers = list(self.layers.values())
        layers.append(self.lastLayer)
        layers.reverse()
        out = 1
        for layer in layers:
            out = layer.backward(out)

        grads = {}
        grads['W1'] = self.layers['Hidden1'].dW
        grads['b1'] = self.layers['Hidden1'].db
        grads['W2'] = self.layers['Hidden2'].dW
        grads['b2'] = self.layers['Hidden2'].db

        return grads


if __name__ == '__main__':

    # settings
    IMAGE_SIZE = 784
    ITERS_NUM = 10000
    BATCH_SIZE = 100
    LEARNING_RATE = 0.1

    # load training and test data
    loader = LoadImageData('mnist')
    train_images, train_labels, test_images, test_labels = loader.load(
        image_size=IMAGE_SIZE,
        should_normalize=True,
        should_flatten=True,
        should_label_be_one_hot=True)

    # create neural network with 784 input dimentions
    nn = NeuralNetwork(
        input_size=IMAGE_SIZE, hidden_size=50, output_size=10)

    # learn & update weights
    costs = []
    train_accuracies = []
    test_accuracies = []
    ITER_PER_EPOC = max(train_images.shape[0]/BATCH_SIZE, 1)

    for i in range(ITERS_NUM):
        batch_mask = np.random.choice(train_images.shape[0], BATCH_SIZE)
        x_batch = train_images[batch_mask]
        t_batch = train_labels[batch_mask]

        # grads = nn.computeNumericalGradients(x_batch, t_batch)
        grads = nn.computeGradientWithBackPropagation(x_batch, t_batch)

        for key in ('W1', 'b1', 'W2', 'b2'):
            nn.params[key] -= LEARNING_RATE * grads[key]

        costs.append(nn.computeCost(nn.forward(x_batch), t_batch))
        print('cost {}'.format(costs[-1]))

        # check accuracy
        if i % ITER_PER_EPOC == 0:
            train_accuracies.append(
                nn.computeAccuracy(train_images, train_labels))
            test_accuracies.append(
                nn.computeAccuracy(test_images, test_labels))
            print("train accuracy {}, test accuracy {}".format(
                train_accuracies[-1], test_accuracies[-1]))

    plt.plot(np.arange(ITERS_NUM), costs)
    plt.axis([0, ITERS_NUM, 0, np.max(costs)])
    plt.show()
