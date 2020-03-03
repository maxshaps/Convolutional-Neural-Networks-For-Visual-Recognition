from builtins import range
from builtins import object
import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # Forward pass: affine - relu - affine
        hidden_layer, cache_hidden_layer = affine_relu_forward(x=X, w=W1, b=b1)
        scores, cache_scores = affine_forward(x=hidden_layer, w=W2, b=b2)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        # Compute softmax loss, using scores and known labels contained in y
        loss, grads = 0, {}

        data_loss, dscores = softmax_loss(x=scores, y=y)
        reg_loss = 0.5 * self.reg * np.sum(W1*W1) + 0.5 * self.reg * np.sum(W2*W2)
        loss = data_loss + reg_loss

        # Backprop into second layer, including L2 regularization
        dx2, dW2, db2 = affine_backward(dout=dscores, cache=cache_scores)
        dW2 += self.reg * W2

        # Backprop into first layer, including L2 regularization
        dx1, dW1, db1 = affine_relu_backward(dout=dx2, cache=cache_hidden_layer)
        dW1 += self.reg * W1

        # Update grads dictionary
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Initialize network parameters and store in self.params dictionary
        dimensions = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W%d' % (i+1)] = np.random.randn(dimensions[i], dimensions[i + 1]) * weight_scale
            self.params['b%d' % (i+1)] = np.zeros(dimensions[i + 1])
            if self.normalization in ['batchnorm', 'layernorm'] and i != (self.num_layers-1):
                self.params['gamma%d' % (i+1)] = np.ones(dimensions[i+1])
                self.params['beta%d' % (i+1)] = np.zeros(dimensions[i+1])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        # Compute forward pass through entire network
        scores = None
        cache_layers = {}
        cache_layers[0] = out = X.copy()

        # Create a separate dropout_caches dict if using dropout
        if self.use_dropout:
            dropout_caches = {}

        # Forward pass through all layers except the last
        for i in range(1, self.num_layers):
            # Forward pass with batchnorm
            if self.normalization=='batchnorm':
                out, cache_layers[i] = affine_batchnorm_relu_forward(x=out, w=self.params['W%d' % i], b=self.params['b%d' % i], gamma=self.params['gamma%d' % i], beta=self.params['beta%d' % i], bnparams=self.bn_params[i-1])
            # Forward pass with layernorm
            elif self.normalization=='layernorm':
                out, cache_layers[i] = affine_layernorm_relu_forward(x=out, w=self.params['W%d' % i], b=self.params['b%d' % i], gamma=self.params['gamma%d' % i], beta=self.params['beta%d' % i], bnparams=self.bn_params[i-1])
            # Forward pass without batchnorm nor layernorm
            else:
                out, cache_layers[i] = affine_relu_forward(x=out, w=self.params['W%d' % i], b=self.params['b%d' % i])
            # Forward pass with dropout
            if self.use_dropout:
                out, dropout_caches[i] = dropout_forward(x=out, dropout_param=self.dropout_param)

        # Forward pass through final layer, returning scores
        scores, cache_layers[self.num_layers] = affine_forward(x=out, w=self.params['W%d' % self.num_layers], b=self.params['b%d' % self.num_layers])

        # If test mode return early
        if mode == 'test':
            return scores

        # Compute backward pass through entire network
        loss, grads = 0.0, {}

        # Compute softmax loss
        loss, dx = softmax_loss(scores, y)

        # Add L2 regularization loss:
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params['W%d' % i]**2)

        # Backprop from last layer, including gradient from L2 regularization
        dx, grads['W%d' % self.num_layers], grads['b%d' % self.num_layers] = affine_backward(dout=dx, cache=cache_layers[self.num_layers])
        grads['W%d' % self.num_layers] += self.reg * self.params['W%d' % self.num_layers]

        # Backprop through remaining layers, including gradient from L2 regularization
        for i in range(self.num_layers-1,0,-1):
            # Backward pass with dropout
            if self.use_dropout:
                dx = dropout_backward(dout=dx, cache=dropout_caches[i])
            # Backward pass with batchnorm
            if self.normalization=='batchnorm':
                dx, grads['W%d' % i], grads['b%d' % i], grads['gamma%d' % i], grads['beta%d' % i] = affine_batchnorm_relu_backward(dout=dx, cache=cache_layers[i])
                grads['W%d' % i] += self.reg * self.params['W%d' % i]
            # Backward pass with layernorm
            if self.normalization=='layernorm':
                dx, grads['W%d' % i], grads['b%d' % i], grads['gamma%d' % i], grads['beta%d' % i] = affine_layernorm_relu_backward(dout=dx, cache=cache_layers[i])
                grads['W%d' % i] += self.reg * self.params['W%d' % i]
            # Backward pass without batchnorm nor layernorm
            else:
                dx, grads['W%d' % i], grads['b%d' % i] = affine_relu_backward(dout=dx, cache=cache_layers[i])
                grads['W%d' % i] += self.reg * self.params['W%d' % i]

        return loss, grads


def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bnparams):
    """
    Convenience layer that performs an affine transform, followed by batchnorm, followed by a ReLU
    """
    affine_out, affine_cache = affine_forward(x, w, b)
    batchnorm_out, batchnorm_cache = batchnorm_forward(affine_out, gamma, beta, bnparams)
    out, relu_cache = relu_forward(batchnorm_out)
    cache = (affine_cache, batchnorm_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    """
    affine_cache, batchnorm_cache, relu_cache = cache
    drelu = relu_backward(dout, relu_cache)
    dbatchnorm, dgamma, dbeta = batchnorm_backward_alt(drelu, batchnorm_cache)
    dx, dw, db = affine_backward(dbatchnorm, affine_cache)
    return dx, dw, db, dgamma, dbeta


def affine_layernorm_relu_forward(x, w, b, gamma, beta, lnparams):
    """
    Convenience layer that performs an affine transform, followed by layernorm, followed by a ReLU
    """
    affine_out, affine_cache = affine_forward(x, w, b)
    layernorm_out, layernorm_cache = layernorm_forward(affine_out, gamma, beta, lnparams)
    out, relu_cache = relu_forward(layernorm_out)
    cache = (affine_cache, layernorm_cache, relu_cache)
    return out, cache


def affine_layernorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-layernorm-relu convenience layer
    """
    affine_cache, layernorm_cache, relu_cache = cache
    drelu = relu_backward(dout, relu_cache)
    dlayernorm, dgamma, dbeta = layernorm_backward(drelu, layernorm_cache)
    dx, dw, db = affine_backward(dlayernorm, affine_cache)
    return dx, dw, db, dgamma, dbeta
