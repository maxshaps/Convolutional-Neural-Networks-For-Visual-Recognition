from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    out = np.dot(x.reshape(x.shape[0], w.shape[0]), w) + b
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(x.shape[0], w.shape[0]).T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    out = np.maximum(0, x)
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    dx = dout
    dx[x < 0] = 0

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    # For more on this implementation of forward pass:
    # http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    if mode == 'train':
        # Compute batch mean
        batch_mean = (1.0 / N) * np.sum(x, axis=0)

        # Center batch data
        batch_centered = x - batch_mean

        # Square batch_centered for variance computation
        squared_batch_centered = batch_centered ** 2

        # Compute batch variance
        batch_var = (1.0 / N) * np.sum(squared_batch_centered, axis=0)

        # Batchnorm denominator for standardization, with eps for numeric stability
        batch_denom = np.sqrt(batch_var + eps)

        # Inverse of batch_denom
        inv_batch_denom = 1.0 / batch_denom

        # Standardize batch data
        batch_standardized = batch_centered * inv_batch_denom

        # Scale batch_standardized
        scaled_batch_standardized = gamma * batch_standardized

        # Shift scaled_batch_standardized to output
        out = scaled_batch_standardized + beta

        # Update running_mean and running_var
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var

        # Store intermediate variables needed for backprop
        cache = (gamma, eps, batch_centered, batch_var, batch_denom, inv_batch_denom, batch_standardized)

    elif mode == 'test':
        # Standardize incoming data
        batch_standardized = (x - running_mean) / np.sqrt(running_var + eps)

        # Scale and shift standardized data to output
        out = gamma * batch_standardized + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape

    # Unpack cache variable
    gamma, eps, batch_centered, batch_var, batch_denom, inv_batch_denom, batch_standardized = cache

    # For more on this implementation of backward pass:
    # http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

    # Backprop through + beta sum gate
    dbeta = np.sum(dout, axis=0)

    # Backprop through gamma * \hat{x} multiplication gate
    dgamma = np.sum(dout * batch_standardized, axis=0)
    dbatch_standardized = dout * gamma

    # Backprop through batch_centered * inv_batch_denom multiplication gate
    dinv_batch_denom = np.sum(dbatch_standardized * batch_centered, axis = 0)
    dbatch_centered_1 = dbatch_standardized * inv_batch_denom

    # Backprop through inverse of denominator
    dbatch_denom = - dinv_batch_denom / (batch_denom ** 2)

    # Backprop through denominator computation
    dbatch_var = 0.5 * dbatch_denom / np.sqrt(batch_var + eps)

    # Backprop through variance computation
    dsquared_batch_centered = (1.0 / N) * np.ones((N, D)) * dbatch_var

    # Backprop through squaring of batch_centered
    dbatch_centered_2 = 2 * batch_centered * dsquared_batch_centered

    # Backprop through centering of batch data
    dbatch_mean = -1.0 * np.sum(dbatch_centered_1 + dbatch_centered_2, axis=0)
    dx_1 = dbatch_centered_1 + dbatch_centered_2

    # Backprop through batch mean computation
    dx_2 = (1.0 / N) * np.ones((N, D)) * dbatch_mean

    dx = dx_1 + dx_2

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape

    # Unpack cache variable
    gamma, eps, batch_centered, batch_var, batch_denom, inv_batch_denom, batch_standardized = cache

    # For more on this implementation of backward pass:
    # https://kevinzakka.github.io/2016/09/14/batch_normalization/

    # I found this explanation more intuitive than above, although there was some issue with matrix transposition that made this version not work when adapting for layernorm_backward
    # http://cthorey.github.io./backpropagation/

    dbeta = np.sum(dout, axis=0)

    dgamma = np.sum(dout * batch_standardized, axis=0)

    dbatch_standardized = dout * gamma
    dx = (1.0 / N) * inv_batch_denom * (N * dbatch_standardized - np.sum(dbatch_standardized, axis=0) - batch_standardized * np.sum(dbatch_standardized * batch_standardized, axis=0))

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)

    # Compute layer norm mean
    ln_mean = np.mean(x, axis=1, keepdims=True)

    # Center layer norm data
    ln_centered = x - ln_mean

    # Compute layer norm variance
    ln_var = np.var(x, axis=1, keepdims=True)

    # Compute inverse layer norm denominator, with eps for numerical stability
    inv_ln_denom = 1.0 / np.sqrt(ln_var + eps)

    # Standardize layer norm input
    ln_standardized = ln_centered * inv_ln_denom

    # Scale and shift standardized data to output
    out = gamma * ln_standardized + beta

    # Store intermediate variables needed for backprop
    cache = (gamma, inv_ln_denom, ln_standardized)

    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape

    # Unpack cache variable
    gamma, inv_ln_denom, ln_standardized = cache

    dbeta = np.sum(dout, axis=0)

    dgamma = np.sum(dout * ln_standardized, axis=0)

    dln_standardized = dout * gamma

    # Transpose matrix quantities before applying formula from batchnorm_backward_alt
    dln_standardized = dln_standardized.T
    ln_standardized = ln_standardized.T
    inv_ln_denom = inv_ln_denom.T

    # Plug in transposed quantities to formula from batchnorm_backward_alt to compute transpose of dx
    dx = (1.0 / D) * inv_ln_denom * (D * dln_standardized - np.sum(dln_standardized, axis=0) - ln_standardized * np.sum(dln_standardized * ln_standardized, axis=0))

    # Transpose back to get final gradient
    dx = dx.T

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keeping** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        # Generate dropout mask, including dividing by p for inverted dropout
        mask = (np.random.rand(*x.shape) < p) / p

        # Multiply input by dropout mask to get output
        out = x * mask

    elif mode == 'test':
        # Inverted droupout doesn't require scaling at test time
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        # Backprop multiplication by droupout mask
        dx = dout * mask

    elif mode == 'test':
        # Inverted dropout doesn't involve any computation at testing
        dx = dout

    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    Hprime = int(1 + (H + 2 * pad - HH) / stride)
    Wprime = int(1 + (W + 2 * pad - WW) / stride)

    # Initialize output array
    out = np.zeros((N, F, Hprime, Wprime))

    # Zero pad input along height and width borders
    x_pad = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant', constant_values=0)

    # For all data points
    for n in range(N):
        # For all filters
        for f in range(F):
            # Height stride
            for i in range(Hprime):
                # Width stride
                for j in range(Wprime):
                    # Multiply padded input volume by weights and then add biases
                    padded_input_volume = x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] # input spans all color channels
                    out[n, f, i, j] = np.sum(padded_input_volume * w[f, :]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    # Unpack cache variable
    x, w, b, conv_param = cache

    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, Hprime, Wprime = dout.shape

    # Zero pad input along height and width borders
    x_pad = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant', constant_values=0)

    # Initialize output gradients
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # For conceptual understanding of backward pass for conv layer
    # https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
    # For all data points
    for n in range(N):
        # For all filters
        for f in range(F):
            db[f] += np.sum(dout[n, f])
            # Height stride
            for i in range(Hprime):
                # Width stride
                for j in range(Wprime):
                    dw[f] += x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * dout[n, f, i, j]
                    dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[f] * dout[n, f, i, j]

    # Index slice dx_pad to yield dx
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None

    stride, pool_height, pool_width = pool_param['stride'], pool_param['pool_height'], pool_param['pool_width']
    N, C, H, W = x.shape
    Hprime = int(1 + (H - pool_height) / stride)
    Wprime = int(1 + (W - pool_width) / stride)

    # Initialize output array
    out = np.zeros((N, C, Hprime, Wprime))

    # Height stride
    for i in range(Hprime):
        # Width stride
        for j in range(Wprime):
            out[:, :, i, j] = np.max(x[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width], axis=(2,3))

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    # Unpack cache variable
    x, pool_param = cache

    stride, pool_height, pool_width = pool_param['stride'], pool_param['pool_height'], pool_param['pool_width']
    N, C, H, W = x.shape
    Hprime = int(1 + (H - pool_height) / stride)
    Wprime = int(1 + (W - pool_width) / stride)

    # Initialize output gradient
    dx = np.zeros_like(x)

    # For all data points
    for n in range(N):
        # For all color channels
        for c in range(C):
            # Height stride
            for i in range(Hprime):
                # Width stride
                for j in range(Wprime):
                    # Window over which max_pool_forward is taken
                    window = x[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]

                    # Array where True for element of x taken during max_pool_forward, False else
                    binary = window==np.max(window)

                    # Pass gradient only to those elements taken during max_pool_forward
                    dx[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width] = binary * dout[n, c, i, j]

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N, C, H, W = x.shape

    # Reorder input x (N, C, H, W) to x_reordered (N, H, W, C)
    x_reordered = np.transpose(x, (0, 2, 3, 1))

    # Reshape x_reordered (N, H, W, C) to x_reshaped (N * H * W, C) in order to call batchnorm_forward
    x_reshaped = np.reshape(x_reordered, (-1, C))

    # Call batchnorm_forward on x_reshaped to perform batch normalization
    out_batchnorm, cache = batchnorm_forward(x=x_reshaped, gamma=gamma, beta=beta, bn_param=bn_param)

    # Reshape out_batchnorm (N * H * W, C) to out_batchnorm_reshaped (N, H, W, C)
    out_batchnorm_reshaped = np.reshape(out_batchnorm, (N, H, W, C))

    # Reorder out_batchnorm_reshaped (N, H, W, C) to out (N, C, H, W)
    out = np.transpose(out_batchnorm_reshaped, (0, 3, 1, 2))

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape

    # Reorder input dout (N, C, H, W) to dout_reordered (N, H, W, C)
    dout_reordered = np.transpose(dout, (0, 2, 3, 1))

    # Reshape dout_reordered (N, H, W, C) to dout_reshaped (N * H * W, C) in order to call batchnorm_backward_alt
    dout_reshaped = np.reshape(dout_reordered, (-1, C))

    # Call batchnorm_backward_alt on dout_reshaped to perform backward pass for batch normalization
    dx_batchnorm, dgamma, dbeta = batchnorm_backward_alt(dout=dout_reshaped, cache=cache)

    # Reshape dx_batchnorm (N * H * W, C) to dx_reshaped (N, H, W, C)
    dx_reshaped = np.reshape(dx_batchnorm, (N, H, W, C))

    # Reorder dx_reshaped (N, H, W, C) to dx (N, C, H, W)
    dx = np.transpose(dx_reshaped, (0, 3, 1, 2))

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner
    identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)

    N, C, H, W = x.shape

    # Reshape x (N, C, H, W) to x_reshaped (N * G, C * H * W / G) in order to do standardization along axis=1
    x_reshaped = np.reshape(x, (N * G, int(C * H * W / G)))

    # Adapt code from layernorm_forward

    ## Compute mean along axis=1
    gn_mean = np.mean(x_reshaped, axis=1, keepdims=True)

    ## Center group norm data
    gn_centered = x_reshaped - gn_mean

    ## Compute group norm variance
    gn_var = np.var(x_reshaped, axis=1, keepdims=True)

    ## Compute inverse group norm denominator, with eps for numerical stability
    inv_gn_denom = 1.0 / np.sqrt(gn_var + eps)

    ## Standardize group norm input
    gn_standardized = gn_centered * inv_gn_denom

    # Reshape gn_standardized back to original size of x (N, C, H, W)
    gn_standardized = np.reshape(gn_standardized, (N, C, H, W))

    # Make sure scaling and shifting to output is broadcast along the C channel
    out = gamma[None, :, None, None] * gn_standardized + beta[None, :, None, None]

    # Store intermediate variables needed for backprop
    cache = (G, gamma, inv_gn_denom, gn_standardized)

    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape

    # Unpack cache variable
    G, gamma, inv_gn_denom, gn_standardized = cache

    # Adapt code from layernorm_backward

    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    dgamma = np.sum(dout * gn_standardized, axis=(0, 2, 3), keepdims=True)

    dgn_standardized = dout * gamma[None, :, None, None]

    # Reshape gn_standardized, dgn_standardized (N, C, H, W) to (N * G, C * H * W / G)
    gn_standardized = np.reshape(gn_standardized, (N * G, int(C * H * W / G)))
    dgn_standardized = np.reshape(dgn_standardized, (N * G, int(C * H * W / G)))

    ## Transpose matrix quantities before applying formula from batchnorm_backward_alt
    gn_standardized = gn_standardized.T
    dgn_standardized = dgn_standardized.T
    inv_gn_denom = inv_gn_denom.T
    D = int(C * H * W / G)

    ## Plug in transposed quantities to formula from batchnorm_backward_alt to compute transpose of dx
    dx = (1.0 / D) * inv_gn_denom * (D * dgn_standardized - np.sum(dgn_standardized, axis=0) - gn_standardized * np.sum(dgn_standardized * gn_standardized, axis=0))

    # Transpose and reshape to get final gradient
    dx = np.reshape(dx.T, (N, C, H, W))

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
