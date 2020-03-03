import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros(W.shape)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  # For each data point X[i] over all examples, compute contribution to loss and gradient
  for i in range(num_train):
      # Compute vector of scores
      scores = np.dot(X[i], W)

      # Normalization trick to avoid numerical instability (http://cs231n.github.io/linear-classify/#softmax)
      scores -= np.max(scores)

      # Compute loss (average over num_train examples later)
      probs = np.exp(scores) / np.sum(np.exp(scores))
      loss += -np.log(probs[y[i]])

      # Compute gradient
      for j in range(num_classes):
          dW[:, j] += (probs[j] - (j == y[i])) * X[i] # subtract 1 when j corresponds to correct class stored in y[i]

  # Average loss and gradient over num_train examples
  loss /= num_train
  dW /= num_train

  # Add regularization to loss and gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros(W.shape)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  # Compute matrix of scores, each row corresponding to a data point
  scores = np.dot(X, W)

  # Normalization trick to avoid numerical instability (http://cs231n.github.io/linear-classify/#softmax)
  scores -= np.max(scores, axis=1, keepdims=True)

  # Compute loss
  probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  loss = np.mean(-np.log(probs[range(num_train), y]))

  # Compute gradient
  dscores = probs
  dscores[range(num_train), y] -= 1
  dscores /= num_train
  dW = np.dot(X.T, dscores)

  # Add regularization to loss and gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW
