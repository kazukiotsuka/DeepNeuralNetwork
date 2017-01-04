#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# data/{image_key}/test_images
#                 /test_labels
#                 /train_images
#                 /train_labels
#

import os
import numpy as np


class LoadImageData(object):
    __data_key__ = None  # e.g. mnist
    __data_path_dir__ = (os.path.dirname(__file__) + '/../data/')
    __data_path_root__ = None

    def __init__(self, data_key):
        self.__data_key__ = data_key
        self.__data_path_root__ = self.__data_path_dir__ + data_key

    def loadImages(self, file_name, image_size=784, offset=16):
        """Load Images.

        Load image data put as below.

        data/{data_key}/{file_name}
        """
        file_path = '{}/{}'.format(self.__data_path_root__, file_name)
        with open(file_path, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=offset)
            images = images.reshape(-1, image_size)
        return images

    def loadLabels(self, file_name, offset=8):
        """Load Labels.

        Load image data put as below.

        data/{data_key}/{label_name}
        """
        file_path = '{}/{}'.format(self.__data_path_root__, file_name)
        with open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=offset)
        return labels

    def convertLabelsToOntHot(self, labels):
        """Convert Label to One Hot expression.

        [1, 2, 5, 10, 3, 4]
        ->
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        """
        one_hot_matrix = np.zeros((labels.size, 10))
        for i, row in enumerate(one_hot_matrix):
            row[labels[i]] = 1
        return one_hot_matrix

    def load(
            self,
            image_size=784,
            should_normalize=True,
            should_flatten=True,
            should_label_be_one_hot=False):
        """Load data.

        should_normalize : If True, normalized to 0.0~1.0.
        should_flatten : If True, images becomes 1 dimentional array.
        should_label_be_one_hot : If True, label becomes like [0, 0, 0, 1, 0,]

        Returns
            train_images, train_labels, test_images, test_labels
        """

        test_images = self.loadImages('test_images', image_size)
        test_labels = self.loadLabels('test_labels')
        train_images = self.loadImages('train_images', image_size)
        train_labels = self.loadLabels('train_labels')

        if should_normalize:
            test_images = test_images.astype(np.float32)
            train_images = train_images.astype(np.float32)
            test_images /= 255.0
            train_images /= 255.0

        if not should_flatten:
            test_images.reshape(-1, 1, 28, 28)
            train_images.reshape(-1, 1, 28, 28)

        if should_label_be_one_hot:
            test_labels = self.convertLabelsToOntHot(test_labels)
            train_labels = self.convertLabelsToOntHot(train_labels)

        return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    loader = LoadImageData('mnist')
    train_images, train_labels, test_images, test_labels = loader.load(
        should_normalize=True,
        should_flatten=True,
        should_label_be_one_hot=True)
    print(train_images.shape)
    print(np.where(train_images != 0))
    print(train_labels.shape)
    print(train_labels[0])
    print(test_images.shape)
    print(np.where(test_images != 0))
    print(test_labels.shape)
    print(test_labels[0])
