from builtins import object
import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim

        # Initialize weights and biases for first convolutional layer
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)

        # Initialize weights and biases for hidden affine layer
        # Assuming suitable padding and stride so that width and height are preserved after first convolutional layer
        # 2x2 max pool decreases height and width by half, which is then received by hidden affine layer
        self.params['W2'] = np.random.randn(num_filters * int(H/2) * int(W/2), hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)

        # Initialize weights and biases for output affine layer
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # First conv - ReLU - 2x2 Max Pool
        conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(x=X, w=W1, b=b1, conv_param=conv_param, pool_param=pool_param)

        # Hidden affine - ReLU
        affine_relu_out, affine_relu_cache = affine_relu_forward(x=conv_relu_pool_out, w=W2, b=b2)

        # Output affine
        scores, scores_cache = affine_forward(x=affine_relu_out, w=W3, b=b3)

        if y is None:
            return scores

        # Compute softmax loss, using scores and known labels contained in y
        loss, grads = 0, {}

        data_loss, dscores = softmax_loss(x=scores, y=y)
        reg_loss = 0.5 * self.reg * np.sum(W1*W1) + 0.5 * self.reg * np.sum(W2*W2) + 0.5 * self.reg * np.sum(W3*W3)
        loss = data_loss + reg_loss

        # Backprop through output affine, including L2 regularization
        daffine_relu_out, dW3, db3 = affine_backward(dout=dscores, cache=scores_cache)
        dW3 += self.reg * W3

        # Backprop through hidden affine - ReLU, including L2 regularization
        dconv_relu_pool_out, dW2, db2 = affine_relu_backward(dout=daffine_relu_out, cache=affine_relu_cache)
        dW2 += self.reg * W2

        # Backprop through first conv - ReLU - 2x2 max pool, including L2 regularization
        dx, dW1, db1 = conv_relu_pool_backward(dout=dconv_relu_pool_out, cache=conv_relu_pool_cache)
        dW1 += self.reg * W1

        # Update grads dictionary
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3

        return loss, grads
