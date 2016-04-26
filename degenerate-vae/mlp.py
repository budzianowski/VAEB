import theano as th
import theano.tensor as T
from theano import function
from theano import shared
import logpdf as lpdf
import numpy.random as rnd
import time
import numpy as np
import math
import data
import matplotlib.pyplot as plt
from infalg import AdaDelta
from infalg import AdaGrad
from theano.printing import pydotprint
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

stream = RandomStreams(seed=15485863)
floatX = th.config.floatX

# Helper function to merge a set of dictionaries.
def merge_dicts(dict_list):
  z = {}
  for dct in dict_list:
    z.update(dct)
  return z


# Construct a list of bias vectors using BiasVector. D should be a list
# containing the lengths of the bias vectors to use.
def BiasVectors(D):
  return [BiasVector(D[i], 'b' + str(i)) for i in range(len(D))]


# Generate a single bias vector object. D is its length, name its string id.
def BiasVector(D, name):
  return shared(rnd.normal(0.0, 0.01, size=(D)).astype(floatX), name)


# Construct a list of weight matrices using WeightMatrix. D should be a list
# of dimensions whose length is one greater than the length of the returned
# list of matrices.
def WeightMatrices(D):
  return [WeightMatrix(D[i], D[i+1], 'W' + str(i)) for i in range(len(D) - 1)]


# Construct a matrix of size Din x Dout with name=name.
def WeightMatrix(Din, Dout, name):
  return shared(rnd.normal(0.0, 0.01, size=(Din, Dout)).astype(floatX), name)


# Iteratively construct the output of the final layer of an MLP.
# Number of hidden layers is defined implicitly by length of W.
# (M := # hidden layers = len(W)).
#
# Inputs:
# X - input matrix. (N, D_in)
# W - weight matrices.
# b  - bias vectors.
# f - activation function.
#
# Outputs:
# H - network output at final layer.
# theta - network parameters.
#
def ConstructMLP(X, W, b, f):

  # Use forward recursion to define the MLP.
  def Construct(H, W, b):
    if len(W) > 1:
      return Construct(f(T.dot(H, W[0]) + b[0]), W[1:], b[1:])
    else:
      return f(T.dot(H, W[0]) + b[0])
  return Construct(X, W, b)


# Construct the Gaussian log prior for an MLP with weight matrices given
# by the list W and variances s2.
#
# Inputs:
# theta - list of parameter tensors.
# s2 - prior variance.
#
# Outputs:
# log prior function.
#
def ConstructNormalPrior(theta, s2):
  logp = 0
  for par in theta:
    logp += T.sum(T.sqr(par) / s2 + T.log(2 * math.pi * s2))
  return -0.5 * logp


# Construct a probabilistic MLP for classification. 1-hot repr. assumed for Y.
# MAP inference is performed.
#
# Inputs:
# Din - input dimensionality.
# Dh  - hidden dimensionalities (list of length # hidden layers)
# Dout - dimensionality of output layer.
# f   - activation function for hidden layers.
# s2  - variance for each hidden layer prior.
#
def ConstructPMLPClassifier(Din, Dh, Dout, f, s2, inf):

  # Construct probabilistic MLP.
  X, Y = T.matrix('X'), T.matrix('Y')
  W, b = WeightMatrices([Din] + Dh), BiasVectors(Dh)
  H = ConstructMLP(X, W, b, f)
  Wout, bout = WeightMatrix(Dh[-1], Dout, 'Wout'), BiasVector(Dout, 'bout')
  Ypr = lpdf.OutToSimplex(H, Wout, bout)
  loglik = lpdf.bernoulli(Y, Ypr)
  logprior = ConstructNormalPrior(W + b + [Wout, bout], s2)
  logjoint = loglik #+ logprior

  # Build function to construct training procedure (requires data).
  def buildtraining(Xtr, Ytr):

    # Define inference procedure and mini-batching variables.
    updates = inf.construct(logjoint, W + b + [Wout, bout])
    lb, ub = T.iscalar('lb'), T.iscalar('ub')

    # Construct training Theano function.
    trfunc = function(
      inputs=[lb, ub],
      outputs=logjoint,
      name='train',
      updates=updates,
      givens={ X : Xtr[lb:ub], Y : Ytr[lb:ub]}
    )

    # Return function to train on a specified subset of the data.    
    return lambda lb, ub : trfunc(lb, ub)

  # Build function to make predictions (distributions over outputs).
  predict = function(
    inputs=[X],
    outputs=Ypr,
    name='predict',
    updates=[],
    givens = {}
  )

  # Build a function to compute the log joint on the provided data.
  computelogjoint = function(
    inputs=[X, Y],
    outputs=logjoint,
    name='logjoint',
    updates=[],
    givens=[]
  )

  return buildtraining, predict, computelogjoint, W + b + [Wout, bout]


# Compute the DKL between two sets of independent Gaussians.
def GaussDKL(mu0, s20, mu1, s21):
  return 0.5 * T.sum(s20 / s21 + T.sqr((mu1 - mu0)) / s21 \
    - 1.0 + T.log(s21) - T.log(s20))

"""# Construct a probabilistic MLP for classification. 1-hot repr. assumed for Y.
# Variational Inference is performed.
#
# Inputs:
# Din - input dimensionality.
# Dh  - hidden dimensionalities (list of length # hidden layers)
# Dout - dimensionality of output layer.
# f   - activation function for hidden layers.
# s2  - variance for each hidden layer prior.
#
def ConstructBayesianMLPClassifier(Din, Dh, Dout, f, s2, inf, mu0):

  # Define process to draw parameters.
  W, b, mus, logsgs = [], [], [], []
  D = [Din] + Dh
  h = len(Dh)
  logsginit = -1.0
  for i in range(len(Dh)):
    mu = shared(rnd.normal(mu0[i].get_value(), 0.01, size=(D[i], D[i+1])).astype(floatX), 'mu')
    logsg = shared(rnd.normal(logsginit, 0.01, size=(D[i], D[i+1])).astype(floatX), 'logsg')
    W.append(mu + T.exp(logsg) * stream.normal(size=mu.shape))
    mus.append(mu)
    logsgs.append(logsg)
    mu = shared(rnd.normal(mu0[i+h].get_value(), 0.01, size=(D[i+1])).astype(floatX), 'mu')
    logsg = shared(rnd.normal(logsginit, 0.01, size=(D[i+1])).astype(floatX), 'logsg')
    b.append(mu + T.exp(logsg) * stream.normal(size=mu.shape))
    mus.append(mu)
    logsgs.append(logsg)
  mu = shared(rnd.normal(mu0[-2].get_value(), 0.01, size=(D[-1], Dout)).astype(floatX), 'mu')
  logsg = shared(rnd.normal(logsginit, 0.01, size=(D[-1], Dout)).astype(floatX), 'logsg')
  Wout = mu + T.exp(logsg) * stream.normal(size=mu.shape)
  mus.append(mu)
  logsgs.append(logsg)
  mu = shared(rnd.normal(mu0[-1].get_value(), 0.01, size=(Dout)).astype(floatX), 'mu')
  logsg = shared(rnd.normal(logsginit, 0.01, size=(Dout)).astype(floatX), 'logsg')
  bout = mu + T.exp(logsg) * stream.normal(size=mu.shape)
  mus.append(mu)
  logsgs.append(logsg)

  # Construct probabilistic MLP.
  X, Y = T.matrix('X'), T.matrix('Y')
  H = ConstructMLP(X, W, b, f)
  Ypr = lpdf.OutToSimplex(H, Wout, bout)
  loglik = lpdf.bernoulli(Y, Ypr)

  # Construct elbo.
  priordkl = 0.0
  for i in range(len(s2)):
    priordkl += GaussDKL(mus[i], T.exp(logsgs[i]), 0.0, s2[i])
  elbo = loglik - priordkl

  # Build function to construct training procedure (requires data).
  def buildtraining(Xtr, Ytr):

    # Define inference procedure and mini-batching variables.
    updates = inf.construct(elbo, mus + logsgs)
    lb, ub = T.iscalar('lb'), T.iscalar('ub')

    # Construct training Theano function.
    trfunc = function(
      inputs=[lb, ub],
      outputs=elbo,
      name='train',
      updates=updates,
      givens={ X : Xtr[lb:ub], Y : Ytr[lb:ub]}
    )

    # Return function to train on a specified subset of the data.    
    return lambda lb, ub : trfunc(lb, ub)

  # Build function to make predictions (distributions over outputs).
  predict = function(
    inputs=[X],
    outputs=Ypr,
    name='predict',
    updates=[],
    givens = {}
  )

  # Build a function to compute the log joint on the provided data.
  computeelbo = function(
    inputs=[X, Y],
    outputs=elbo,
    name='elbo',
    updates=[],
    givens=[]
  )
      

  return buildtraining, predict, computeelbo, mus, logsgs"""


# Construct a probabilistic MLP for classification. 1-hot repr. assumed for Y.
# Variational Inference is performed (dropout).
#
# Inputs:
# Din - input dimensionality.
# Dh  - hidden dimensionalities (list of length # hidden layers)
# Dout - dimensionality of output layer.
# f   - activation function for hidden layers.
# s2  - variance for each hidden layer prior.
#
def ConstructBayesianMLPClassifier(Din, Dh, Dout, f, s2, inf):

  def bernoulli(size):
    return stream.binomial(size, p=0.5, dtype=floatX)

  # Define process to draw parameters.
  W, b, = [], []
  Ws, bs = WeightMatrices([Din] + Dh), BiasVectors(Dh)
  for weight, bias in zip(Ws, bs):
    W.append(weight * bernoulli(weight.get_value().shape))
    b.append(bias * bernoulli(bias.get_value().shape))
  Wouts, bouts = WeightMatrix(Dh[-1], Dout, 'Wout'), BiasVector(Dout, 'bout')
  Wout = Wouts * bernoulli(size=Wouts.get_value().shape)
  bout = bouts * bernoulli(size=bouts.get_value().shape)
  theta = Ws + bs + [Wouts, bouts]

  # Construct probabilistic MLP.
  X, Y = T.matrix('X'), T.matrix('Y')
  H = ConstructMLP(X, W, b, f)
  Ypr = lpdf.OutToSimplex(H, Wout, bout)
  loglik = lpdf.bernoulli(Y, Ypr)

  # Construct elbo.
  priordkl = 0.0
  #for par in theta:
  #  priordkl += GaussDKL(mus[i], T.exp(logsgs[i]), 0.0, s2[i])
  elbo = loglik - priordkl

  # Build function to construct training procedure (requires data).
  def buildtraining(Xtr, Ytr):

    # Define inference procedure and mini-batching variables.
    updates = inf.construct(elbo, theta)
    lb, ub = T.iscalar('lb'), T.iscalar('ub')

    # Construct training Theano function.
    trfunc = function(
      inputs=[lb, ub],
      outputs=elbo,
      name='train',
      updates=updates,
      givens={ X : Xtr[lb:ub], Y : Ytr[lb:ub]}
    )

    # Return function to train on a specified subset of the data.    
    return lambda lb, ub : trfunc(lb, ub)

  # Build function to make predictions (distributions over outputs).
  predict = function(
    inputs=[X],
    outputs=Ypr,
    name='predict',
    updates=[],
    givens = {}
  )

  # Build a function to compute the log joint on the provided data.
  computeelbo = function(
    inputs=[X, Y],
    outputs=elbo,
    name='elbo',
    updates=[],
    givens=[]
  )
      

  return buildtraining, predict, computeelbo, theta


def LearnMLPracticalClassification(Xtr, Ytr, Xte, Yte, bayesian=0,\
  epochs=100, Dh=[500,500,500], infer=AdaGrad(0.01), L=1):

  # Build classifier learning procedure.
  print('Building classifier graph.')
  Ntr, Din = Xtr.get_value().shape
  Dout = Ytr.get_value().shape[1]
  f = T.tanh
  s2 = [1.0] * (2 * (len(Dh) + 1))
  if bayesian:
    buildtraining, predict, logjoint, theta = \
      ConstructBayesianMLPClassifier(Din, Dh, Dout, f, s2, infer)
  else:
    buildtraining, predict, logjoint, theta = \
      ConstructPMLPClassifier(Din, Dh, Dout, f, s2, infer)

  print('Compiling MLP.')
  train = buildtraining(Xtr, Ytr)

  # Perform inference.
  print('Performing inference.')
  bs, logp = 100, []
  for i in range(epochs):
    lb = 0
    while lb < Ntr:
      ub = lb + bs
      if ub > Ntr:
        ub = Ntr
      logp.append(train(lb, ub))
      lb = ub
    musg = 0
    print('Epoch ' + str(i) + '. logjoint = ' + str(logp[-1]) + '.')

  # Compute accuracies.
  print('Computing predictions under posterior.')
  def accuracy(Y, Ypr):
    correct = 0.0
    idx = np.argmax(Ypr, 1)
    for i in range(idx.shape[0]):
      if Y[i,idx[i]] == 1.0:
        correct += 1.0
    return correct / Ypr.shape[0]
  Yprtr, Yprte = 0.0, 0.0
  for t in range(L):
    Yprtr += predict(Xtr.get_value())
    Yprte += predict(Xte.get_value())
  Yprtr /= L
  Yprte /= L
  print('training accuracy = ' + str(accuracy(Ytr.get_value(), Yprtr)))
  print('testing accuracy = ' + str(accuracy(Yte.get_value(), Yprte)))

  # Plot log probability over time.
  plt.plot(logp)
  plt.savefig('logp.pdf')
  plt.close()

  return theta

def main():

  def TestBiasVectors():
    biases = BiasVectors([5, 4])
    for bias in biases:
      print(bias.get_value().shape)
    return biases
  #biases = TestBiasVectors()

  def TestWeightMatrices():
    weights = WeightMatrices([10, 5, 4])
    for weight in weights:
      print(weight.get_value().shape)
    return weights
  #weights = TestWeightMatrices()

  def TestMLP():
    X = T.matrix('X')
    f = T.nnet.sigmoid
    mlp, W, b = ConstructMLP(X, 10, [5, 4], f)
    out = function(
      inputs=[X],
      outputs=mlp)
    pydotprint(mlp, 'mlptest.png')
  #TestMLP()

  def TestNormalPrior():
    theta = biases + weights
    s2 = [1.0] * len(theta)
    logprior = ConstructNormalPrior(theta, s2)
    pydotprint(logprior, 'priortest.png')
  #TestNormalPrior()


  rnd.seed(15485863)

  """Xtr, Ytr, Xte, Yte = data.GenerateGaussians(1000, 1000)
  Dh = [500]
  theta = LearnMLPracticalClassification(Xtr, Ytr, Xte, Yte, \
    bayesian=1, epochs=100, Dh=Dh, L=100, infer=AdaGrad(0.1))
  return"""

  # Load data.
  print('Loading data.')
  Xtr, Ytr, Xte, Yte = data.LoadMLPracticalClassification(0)
  Dh = [1000, 1000]
  theta = LearnMLPracticalClassification(Xtr, Ytr, Xte, Yte, bayesian=0,\
    epochs=250, L=100, infer=AdaGrad(0.1), Dh=Dh)

  """Xtr, Ytr, Xte, Yte = data.LoadMNIST()
  theta = LearnMLPracticalClassification(Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte, \
    epochs=75, bayesian=0, Dh = [500, 300], infer=AdaGrad(0.01))"""



if __name__ == '__main__':
  main()
