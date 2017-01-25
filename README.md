# DeepNeuralNetwork
Deep Neural Network (python)

This is a software enable deep learning with Deep Convolution Neural Network.  
The structure of network is as below.

```
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
```

### implemented features

- debug mode  
- initialize weight std by the theory of He or Xavier
- showing activation output distribution histogram    
- vanishing gradient alert system
- minibatch
