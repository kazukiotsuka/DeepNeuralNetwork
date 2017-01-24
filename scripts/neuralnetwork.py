#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# neuralnetwork.py
#

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from config import Config
from config import DebugLevel
from collections import OrderedDict
from load_image_data import LoadImageData
from nnoperations import NNOperations
from nnoperations import ActivationType
from layers import ReLULayer
from layers import SigmoidLayer
from layers import HiddenLayer
from layers import SoftmaxWithLossLayer
from layers import ConvolutionLayer
from layers import PoolingLayer
from layers import DropoutLayer
from optimizers import Optimizer
from optimizers import OptimizationMethod


class NeuralNetwork(NNOperations):
    __activation_type__ = None
    PARAMS_KEYS = ('W1', 'b1', 'W2', 'b2')

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

    def forward(self, X, is_activation_check=False):
        """Forward propagation.

        x: input data
        """
        activations = []
        for key, layer in self.layers.items():
            X = layer.forward(X)

            # save activations
            if is_activation_check and isinstance(
                    layer, (ReLULayer, SigmoidLayer)):
                activations.append((layer.name, X))

        if is_activation_check:
            self.showActivationsDistribution(activations)

        return X

    def computeCost(self, y, t):
        """Compute Cost.

        y: output result
        t: labeled data
        """
        return self.lastLayer.forward(y, t)

    def computeAccuracy(self, x, labels):
        """Compute Accuracy.

        x: training data
        labels: correct labeles
        """
        out = self.forward(x)
        predictions = np.argmax(out, axis=1)
        if labels.ndim != 1:  # one-hot
            labels = np.argmax(labels, axis=1)
        accuracy = np.sum(predictions == labels) / float(x.shape[0])
        return accuracy

    def computeGradientWithBackPropagation(
            self, x, t, is_activation_check=False):
        """Compute Gradient with Back Propagation.

        x: input data
        t: labeled data
        """

        # forward
        self.computeCost(
            self.forward(x, is_activation_check=is_activation_check), t)

        # backward
        layers = list(self.layers.values())
        layers.append(self.lastLayer)
        layers.reverse()
        dout = 1
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Hidden1'].dW
        grads['b1'] = self.layers['Hidden1'].db
        grads['W2'] = self.layers['Hidden2'].dW
        grads['b2'] = self.layers['Hidden2'].db

        return grads

    def fit(
            self,
            train_images,
            train_labels,
            test_images,
            test_labels,
            optimization_method=OptimizationMethod.Adam,
            optimization_params={
                'learning_rate': 0.01,
                'momentum': 0.9,
                'beta1': 0.9,
                'beta2': 0.999},
            iters_num=10000,
            mini_batch_size=100,
            samples_num_evaluated_per_epoc=100,
            is_activation_check=False):
        """Fits weight paramters by using optimization algorithm.
        """

        costs = []
        train_accuracies = []
        test_accuracies = []

        ITERS_NUM = iters_num
        MINI_BATCH_SIZE = mini_batch_size
        TRAIN_IMAGES_NUM = train_images.shape[0]
        ITER_PER_EPOC = max(TRAIN_IMAGES_NUM/MINI_BATCH_SIZE, 1)

        optimizer = Optimizer().optimizer(
            optimization_method=optimization_method,
            learning_rate=optimization_params.get('learning_rate'),
            momentum=optimization_params.get('momentum'),
            beta1=optimization_params.get('beta1'),
            beta2=optimization_params.get('beta2'),
            )

        for i in range(ITERS_NUM):
            batch_mask = np.random.choice(TRAIN_IMAGES_NUM, MINI_BATCH_SIZE)
            x_batch = train_images[batch_mask]
            t_batch = train_labels[batch_mask]

            grads = nn.computeGradientWithBackPropagation(
                x_batch, t_batch, is_activation_check=is_activation_check)
            optimizer.update(nn.params, grads)

            costs.append(nn.computeCost(nn.forward(x_batch), t_batch))
            print('cost {}'.format(costs[-1]))

            # check accuracy
            if i % ITER_PER_EPOC == 0:
                print('=========ITERATION {}=========='.format(i))
                if samples_num_evaluated_per_epoc is None:
                    samples_num_evaluated_per_epoc = -1
                train_accuracies.append(
                    nn.computeAccuracy(
                        train_images[:samples_num_evaluated_per_epoc],
                        train_labels[:samples_num_evaluated_per_epoc]))
                test_accuracies.append(
                    nn.computeAccuracy(
                        test_images[:samples_num_evaluated_per_epoc],
                        test_labels[:samples_num_evaluated_per_epoc]))
                print("train accuracy {}, test accuracy {}".format(
                    train_accuracies[-1], test_accuracies[-1]))

        return costs, train_accuracies, test_accuracies

    def activationLayerFromType(self, activation_type, index=None):
        """Return activation layer from ActivationType.

        Returns an instance of
        - ReLULayer
        - SigmoidLayer
        """
        if activation_type == ActivationType.ReLU:
            return ReLULayer(index=index)
        elif activation_type == ActivationType.Sigmoid:
            return SigmoidLayer(index=index)
        else:
            print('[WARNING] activation_type {} is not recognized.'.format(
                activation_type))

    def activationLayer(self, index=None):
        """Returns activation layer of the NN instance.
        """
        return self.activationLayerFromType(self.__activation_type__, index)

    def activationNameFromType(self, activation_type) -> str:
        """Returns activation name from ActivationType.

        Returns
        - 'ReLU'
        - 'Sigmoid;
        """
        if activation_type == ActivationType.ReLU:
            return 'ReLU'
        elif activation_type == ActivationType.Sigmoid:
            return 'Sigmoid'

    def activationName(self):
        """Returns activation layer of the NN instance.
        """
        return self.activationNameFromType(self.__activation_type__)

    def showActivationsDistribution(self, activations: list):
        """See activations distribution as hisgram.

        activations: touple of (layer_name, layer_output)
        """
        for i, activation in enumerate(activations):
            plt.subplot(1, len(activations), i+1)
            plt.title(activation[0])
            if i != 0:
                plt.yticks([], [])
            plt.hist(activation[1].flatten(), 30, range=(0, 1))
        plt.show()
        return


class ConvolutionNetwork(NeuralNetwork):
    PARAMS_KEYS = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')
    __activation_type = None

    def __init__(
            self,
            input_size=(1, 28, 28),
            activation_type=ActivationType.ReLU,
            filter_num=30,
            filter_size=5,
            filter_padding=0,
            filter_stride=1,
            hidden_size=100,
            output_size=10):

        self.__activation_type__ = activation_type

        convolution_output_size =\
            (input_size[1] - filter_size + 2*filter_padding) /\
            filter_stride + 1
        pooling_output_size =\
            int(filter_num *
                (convolution_output_size / 2) * (convolution_output_size / 2))

        convolution_layer_1 = ConvolutionLayer(
            index=1,
            activation_type=self.__activation_type__,
            filter_num=filter_num,
            channel_num=input_size[0],
            filter_size=filter_size)

        # set params
        self.params = {}
        hidden_layer_1 = HiddenLayer(
            index=1,
            activation_type=self.__activation_type__,
            pre_node_num=pooling_output_size,
            next_node_num=hidden_size)
        hidden_layer_2 = HiddenLayer(
            index=2,
            activation_type=self.__activation_type__,
            pre_node_num=hidden_size,
            next_node_num=output_size)

        self.params['W1'] = convolution_layer_1.W
        self.params['b1'] = convolution_layer_1.b
        self.params['W2'] = hidden_layer_1.W
        self.params['b2'] = hidden_layer_1.b
        self.params['W3'] = hidden_layer_2.W
        self.params['b3'] = hidden_layer_2.b

        # save init weights
        self.init_weights = []
        self.init_weights.append(self.params['W1'])
        self.init_weights.append(self.params['W2'])
        self.init_weights.append(self.params['W3'])

        # set layers
        self.layers = OrderedDict()
        self.layers['Convolution1'] = convolution_layer_1
        self.layers['ReLU1'] = self.activationLayer(index=1)
        self.layers['Pooling1'] = PoolingLayer(
            index=1, pool_h=2, pool_w=2, stride=2)
        self.layers['Hidden1'] = hidden_layer_1
        self.layers['ReLU2'] = self.activationLayer(index=2)
        self.layers['Hidden2'] = hidden_layer_2
        self.lastLayer = SoftmaxWithLossLayer()

    def computeGradientWithBackPropagation(
            self, x, t, is_activation_check=False):
        """Compute Gradient with Back Propagation.

        x: input data
        t: labeled data
        """

        # forward
        self.computeCost(
            self.forward(x, is_activation_check=is_activation_check), t)

        # backward
        layers = list(self.layers.values())
        layers.append(self.lastLayer)
        layers.reverse()
        dout = 1
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Convolution1'].dW
        grads['b1'] = self.layers['Convolution1'].db
        grads['W2'] = self.layers['Hidden1'].dW
        grads['b2'] = self.layers['Hidden1'].db
        grads['W3'] = self.layers['Hidden2'].dW
        grads['b3'] = self.layers['Hidden2'].db

        return grads


class DeepNeuralNetwork(ConvolutionNetwork):
    """Deep Convolution Neural Network

    *Network Structure

    (Convolution1|filter num x 16, size 3 x 3, padding 1 stride 1)
    (ReLU1)
    (Convolution2|filter num x 16, size 3 x 3, padding 1 stride 1)
    (ReLU2)
    (Pooling1)
    (Convolution3|filter num x 32, size 3 x 3, padding 1 stride 1)
    (ReLU3)
    (Convolution4|filter num x 32, size 3 x 3, padding 2 stride 1)
    (ReLU4)
    (Pooling|H:2, W:2, stride:2)

    (Convolution5|filter num x 64, size 3 x 3, padding 1 stride 1)
    (ReLU5)
    (Convolution6|filter num x 64, size 3 x 3, padding 1 stride 1)
    (ReLU6)
    (Pooling|H:2, W:2, stride:2)

    (Hidden1|hidden size:50)
    (ReLU7)
    (Dropout1)
    (Hidden2|hidden size:50)
    (Dropout2)

    (Softmax)
    """

    __activation_type__ = None

    def __init__(
            self,
            input_size=(1, 28, 28),
            activation_type=ActivationType.ReLU,
            hidden_size=50,
            output_size=10):

        # basic paramters
        self.__activation_type__ = activation_type
        self.params = {}
        self.layers = []

        # set layers
        channel_num = input_size[0]
        for i, param in enumerate([
                {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
                {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'pre_node_num': 64*4*4, 'next_node_num': hidden_size},
                {'pre_node_num': hidden_size, 'next_node_num': output_size},
                ]):

            # layer 1 ~ 6 Convolution Layer & ReLU Layer
            if i+1 in range(1, 7):
                # create convolution layer
                convolution_layer = ConvolutionLayer(
                    index=i+1,
                    activation_type=self.__activation_type__,
                    filter_num=param['filter_num'],
                    channel_num=channel_num,
                    filter_size=param['filter_size'],
                    stride=param['stride'],
                    padding=param['pad'])
                self.layers.append(convolution_layer)
                self.layers.append(
                    self.activationLayerFromType(activation_type, index=i+1))
                # layer 2, 4, 6 Pooling Layer
                if i+1 in (2, 4, 6):
                    self.layers.append(
                        PoolingLayer(index=i+1, pool_h=2, pool_w=2, stride=2))
                # update next channel num
                channel_num = convolution_layer.filter_num
                layer = convolution_layer

            # layer 7, 8 Hidden Layer & ReLU Layer & Dropout Layer
            if i+1 in (7, 8):
                hidden_layer = HiddenLayer(
                    index=i+1,
                    activation_type=self.__activation_type__,
                    pre_node_num=param['pre_node_num'],
                    next_node_num=param['next_node_num'])
                self.layers.append(hidden_layer)
                if i+1 == 7:
                    self.layers.append(
                        self.activationLayerFromType(
                            activation_type, index=i+1))
                self.layers.append(DropoutLayer(index=i+1, dropout_ratio=0.5))
                layer = hidden_layer

            # set W,b
            self.params['W{}'.format(i+1)] = layer.W
            self.params['b{}'.format(i+1)] = layer.b

            print('layer {} created'.format(i+1))

            if Config.IS_DEBUG:
                print('W{} shape : {}'.format(
                    i+1, self.params['W{}'.format(i+1)].shape))
                print('b{} shape : {}'.format(
                    i+1, self.params['W{}'.format(i+1)].shape))

        # output created layer structures
        for layer in self.layers:
            print(layer.name)

        # keep weight required layer indexes
        self.weight_layer_indexes = []
        for j, layer in enumerate(self.layers):
            if isinstance(layer, (ConvolutionLayer, HiddenLayer)):
                self.weight_layer_indexes.append(j)
        self.debug('weight_layer_indexes {}'.format(self.weight_layer_indexes))

        print('{} layers created'.format(len(self.layers)))
        # last layer SoftmaxWithLoss Layer
        self.lastLayer = SoftmaxWithLossLayer()

    def forward(self, X, is_train=False, is_activation_check=False):
        """Forward propagation.

        x: input data
        """
        activations = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DropoutLayer):
                X = layer.forward(X, is_train)
            else:
                X = layer.forward(X)
            # save activations
            if is_activation_check and isinstance(
                    layer, (ReLULayer, SigmoidLayer)):
                activations.append((layer.name, X))

        if is_activation_check:
            self.showActivationsDistribution(activations)

        return X

    def backward(self):
        """Back propagation.
        """
        # backward
        layers = self.layers.copy()
        layers.append(self.lastLayer)
        layers.reverse()
        dout = 1
        for i, layer in enumerate(layers):
            dout = layer.backward(dout)
        return

    def computeGradientWithBackPropagation(
            self, x, t, is_activation_check=False):
        """Compute Gradient with Back Propagation.

        x: input data
        t: labeled data
        """

        # forward
        self.computeCost(
            self.forward(
                x,
                is_train=True,
                is_activation_check=is_activation_check),
            t)

        # backward
        self.backward()

        grads = {}
        for i, layer_index in enumerate(self.weight_layer_indexes):
            grads['W{}'.format(i+1)] = self.layers[layer_index].dW
            grads['b{}'.format(i+1)] = self.layers[layer_index].db

        return grads


if __name__ == '__main__':

    # settings
    IMAGE_SIZE = 784
    ITERS_NUM = 1000
    MINI_BATCH_SIZE = 100

    # load training and test data
    loader = LoadImageData('mnist')
    train_images, train_labels, test_images, test_labels = loader.load(
        image_size=IMAGE_SIZE,
        should_normalize=True,
        should_flatten=False,
        should_label_be_one_hot=False)

    # create neural network with 784 input dimentions

    # nn = NeuralNetwork(
    #     input_size=IMAGE_SIZE, hidden_size=50, output_size=10)

    # nn = ConvolutionNetwork(
    #     input_size=(1, 28, 28),
    #     filter_num=30,
    #     filter_size=5,
    #     filter_padding=0,
    #     filter_stride=1,
    #     hidden_size=100,
    #     output_size=10)

    nn = DeepNeuralNetwork(
        input_size=(1, 28, 28),
        activation_type=ActivationType.ReLU,
        hidden_size=50,
        output_size=10)

    # learn & update weights
    costs, train_accuracies, test_accuracies = nn.fit(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        optimization_method=OptimizationMethod.Adam,
        optimization_params={
            'learning_rate': 0.01,
            'beta1': 0.9,
            'beta2': 0.999},
        iters_num=ITERS_NUM,
        mini_batch_size=MINI_BATCH_SIZE,
        samples_num_evaluated_per_epoc=100,
        is_activation_check=(
            Config.IS_DEBUG and
            Config.DEBUG_LEVEL.value > DebugLevel.INFO.value))

    plt.plot(np.arange(ITERS_NUM), costs)
    plt.axis([0, ITERS_NUM, 0, np.max(costs)])
    plt.show()
