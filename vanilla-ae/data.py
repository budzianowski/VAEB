from theano import shared
import theano as th
import numpy as np
import cPickle, gzip
import numpy as np
import numpy.random as rnd

floatX = th.config.floatX

# Load the mnist data set (scaled between 0 and 1).
def LoadMNIST(Ntrout=60000):

  # Load as per Theano instructions.
  f = gzip.open('../mnist.pkl.gz', 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  f.close()

  # Extract input data sets.
  Xtrnp = np.vstack((train_set[0], valid_set[0]))
  Xtr = shared(Xtrnp[:Ntrout], 'Xtr')
  Xte = shared(test_set[0], 'Xte')
  Ntr, Nte = Xtr.get_value().shape[0], Xte.get_value().shape[0]

  # Extract train label data and map to 1-hot representation.
  Ytr_tmp = np.hstack((train_set[1], valid_set[1]))
  Ytr = np.zeros((Ytr_tmp.shape[0], 10))
  Ytr[np.arange(Ytr.shape[0]), Ytr_tmp] = 1
  Ytr = Ytr.astype(floatX)

  # Extract test label data and map to 1-hot representation.
  Yte_tmp = test_set[1]
  Yte = np.zeros((Nte, 10))
  Yte[np.arange(Nte), Yte_tmp] = 1
  Yte = Yte.astype(floatX)

  # Return data, building label DataMatrix objects.
  Ytr = shared(Ytr[:Ntrout], 'Ytr')
  Yte = shared(Yte, 'Yte')
  return Xtr, Ytr, Xte, Yte



# Load the Frey Face data with the first 1500 images as training by default.
def LoadFreyFace(Ntrout=1500):
  Xtmp = np.genfromtxt('freyface.csv', delimiter=',') / 255.0
  Xtr, Xte = Xtmp[:Ntrout], Xtmp[Ntrout:]
  return shared(Xtr.astype(floatX), 'Xtr'), shared(Xte.astype(floatX), 'Xte')



def GenerateGaussians(N, Ntr):

  # Generate the data.  
  mu0, mu1 = np.array([1.0, 1.0]), np.array([-1.0, -1.0])
  sg = np.array([[1.0, -0.75], [-0.75, 1.0]])
  X0 = rnd.multivariate_normal(mu0, sg, size=(N)).astype(floatX)
  X1 = rnd.multivariate_normal(mu1, sg, size=(N)).astype(floatX)
  Y0 = np.hstack((np.ones((N, 1)), np.zeros((N, 1)))).astype(floatX)
  Y1 = np.hstack((np.zeros((N, 1)), np.ones((N, 1)))).astype(floatX)

  # Permute the data and split to return.
  idx = rnd.permutation(2*N)
  X, Y = np.vstack((X0, X1))[idx], np.vstack((Y0, Y1))[idx]
  return shared(X[:Ntr], 'Xtr'), shared(Y[:Ntr], 'Ytr'), \
    shared(X[Ntr:], 'Xte'), shared(Y[Ntr:], 'Yte')


def main():

  rnd.seed(15485863)

  """Xtr, Ytr, Xte, Yte = LoadMLPracticalClassification(0)
  Xtr, Ytr, Xte = LoadMLPracticalClassification()
  Xtr, Ytr, Xte, Yte = LoadMNIST()
  print(Xtr.get_value().shape)
  print(Yte.get_value().shape)
  print(Xte.get_value().shape)
  print(Yte.get_value().shape)"""
  Xtr, Ytr, Xte, Yte = GenerateGaussians(100, 100)

if __name__ == '__main__':
  main()  
