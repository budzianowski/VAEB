import theano.tensor as T
from theano import function
import theano.tensor.nnet as nnet
import math

# Transform the final layer of outputs onto the Dout-dimensional simplex.
#
# Inputs:
# H - outputs from final layer. (N,D)
# W - matrix mapping from final layer to reals. (D,Dout)
# b - bias vector (Dout)
#
# Outputs:
# stochastic matrix of outputs. (N,Dout)
#
def OutToSimplex(H, W, b):
  return nnet.softmax(T.dot(H, W) + b)


# Tensor-equivalent to OutToSimplex.
#
# Inputs:
# H - outputs from final layer. (K,N,D)
# W - matrix mapping from final layer to reals. (D,Dout)
# K - no. dimensions in first dimension of H.
#
# Outputs:
# stochastic tensor of outputs.
#
def TensorOutToSimplex(H, W, K):
  ret = T.tensor3('Simplex')
  for k in range(K):
    ret[k] = OutToSimplex(H[k], W)
  return ret


# Maps each dimension of the MLP output to [0, 1).
#
# Inputs:
# H - outputs from final layer (N,D)
# W - matrix mapping from final layer to reals. (D,Dout)
#
# Outputs:
# matrix of outputs on [0,1)^{N x Dout}
#
def OutToProbs(H, W, b):
  return nnet.sigmoid(T.dot(H, W) + b)


# T.tensor3 equivalent of OutToProbs.
#
# Inputs:
# H - output from MLP. (K,N,D)
# W - matrix mapping from final layer to pre-nonlinearity output. (D,Dout)
# K - size of first dim of H.
#
# Outputs:
# tensor of outputs in [0, 1)^{K x N x Dout}
#
def TensorOutToProbs(H, W, K):
  return nnet.sigmoid(T.batched_dot(H, T.tile(W.dimshuffle(('x',0,1)),[K,1,1])))


# Transforms final layer of outputs onto the real line.
#
# Inputs:
# H - output of final MLP layer. (N,D)
# W - matrix mapping from final output layer to reals. (D,Dout)
#
# Outputs:
# matrix of outputs. (N,Dout)
def OutToReal(H, W, b):
  return T.dot(H, W) + b


# Log pdf of the Bernoulli distribution. Applies softmax to H.
#
# Inputs:
# Y - matrix of observations. (N,D)
# P - probability distributions over outputs. (N,D)
#
# Outputs:
# logp - log probability of the observations Y. (1,1)
#
def bernoulli(Y, P):
  return T.sum(Y * T.log(P + 1e-7) + (1 - Y) * T.log(1.0 - P + 1e-7))


# Compute the elementwise Bernoulli (ie: bernoulli without the sum.)
#
# Inputs:
# Y - observations.
# P - probability distributions over outputs.
#
# Outputs:
# logp - log probability of each observation under the probability.
#
def bernoulli_elemwise(Y, P):
  return Y * T.log(P + 1e-10) + (1 - Y) * T.log(1 - P + 1e-10)


# Log pdf of the independent multivariate Normal distribution.
#
# Inputs:
# Y - matrix of observations. (N,D)
# mu - means. (N,D)
# logs2 - log of the variances. (N,D)
#
# Outputs:
# logp - log probability of the observations Y. (1,1)
#
def indep_normal(Y, mu, logs2):
  log2pi = math.log(2 * math.pi)
  return - 0.5 * T.sum(log2pi + logs2 + T.sqr(Y - mu) / T.exp(logs2))


# Unit testing for the pdfs. Print out some computation graphs for visual
# debugging and stuff.
def main():
  x = T.dmatrix('x')
  p = T.dmatrix('p')
  f = function([x, p], bernoulli(x, p))
  print(f([[0, 0, 1], [0, 0, 1]], [[0.01, 0.01, 0.99], [0.01, 0.01, 0.99]]))

if __name__ == '__main__':
  main()
