import data, mlp
import theano as th
import theano.tensor as T
from theano import function
import logpdf as lpdf
from infalg import AdaGrad
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import VAEBImage

floatX = th.config.floatX

# Construct a plain old autoencoder.
#
# Inputs:
# Xtr - training data (Ntr,D)
# Denc - list containing dims of hidden layers. eg. [500,400]=2 hidden layers
#        in the encoder mlp, the first having 500 units and the second 400.
# Dz   - the number of "latent" dimensions in the middle layer.
# Ddec - same as Denc, but for the decoder network.
# f - activation function. (tanh, logistic, relu etc)
# s2 - variance for prior on parameters (ie. controls "weight decay")
# inf - inference object, used to compile training function.
# otype - type of output from decoder network. Supports binary | cont. (The
#       names are slightly misleading. "binary" actually means "valued on [0,1]"
#       (ie. for mnist) and "cont" means "Gaussian output distribution with mean
#       constrained to lie on [0,1] and unconstrained variance" (ie. for frey
#       faces).
#
# Outputs:
# train - training function. Returns the mean log likelihood of the observations
#         under the current model parameters. (ish, doesn't technically make
#         sense but is a good convergence metric).
# reconstruct - function to pass data through encoder and decoder.
# encode - function to run an input through the encoder network and produce the
#          latent representation.
# decode - function to run a point in the latent space through the decoder
#          network and produce the expected location of the output.
# theta - Theano shared variables.
# 
def ConstructAE(Xtr, Denc=[500], Dz=20, Ddec=[500], f=T.tanh, s2=1.0, \
  inf=AdaGrad(0.01)):

  theta = []

  # Construct encoder network and 'latent' variables.
  X, Dobs = T.matrix('X'), Xtr.get_value().shape[1]
  Wenc, benc = mlp.WeightMatrices([Dobs] + Denc), mlp.BiasVectors(Denc)
  Hz = mlp.ConstructMLP(X, Wenc, benc, f)
  Wz, bz = mlp.WeightMatrix(Denc[-1], Dz, 'Wz'), mlp.BiasVector(Dz, 'bz')
  Z = f(T.dot(Hz, Wz) + bz)
  theta += Wenc + benc + [Wz, bz]

  # Construct decoder network. Type of output depends upon otype.
  Wdec, bdec = mlp.WeightMatrices([Dz] + Ddec), mlp.BiasVectors(Ddec)
  Hx = mlp.ConstructMLP(Z, Wdec, bdec, f)
  theta += Wdec + bdec

  # Map to output. Looking to minimise the squared-error (se).
  Wout = mlp.WeightMatrix(Ddec[-1], Dobs, 'Wout')
  bout = mlp.BiasVector(Dobs, 'bout')
  Xpr = lpdf.OutToProbs(Hx, Wout, bout)
  se = T.sum(T.sqr(X - Xpr))
  theta += [Wout, bout]

  # Add weight decay (equivalent to normal prior).
  weightdecay = mlp.ConstructNormalPrior(theta, s2)
  logjoint = -se + weightdecay

  # Build function to construct training procedure.
  updates = inf.construct(logjoint, theta)
  idx = T.ivector('idx')
  train = function(
    inputs=[idx],
    outputs=se / X.shape[0],
    name='train',
    updates=updates,
    givens= { X : Xtr[idx] }
  )

  # Build function to make predictions (distributions over outputs).
  reconstruct = function(
    inputs=[X],
    outputs=Xpr,
    name='reconstruct',
    updates=[],
    givens = {}
  )

  # Build a function to encode data.
  encode = function(
    inputs=[X],
    outputs=Z,
    name='encode',
    updates=[],
    givens={}
  )

  # Build a function to decode the data.
  decode = function(
    inputs=[Z],
    outputs=Xpr,
    name='decode',
    updates=[],
    givens={}
  )

  return train, reconstruct, encode, decode, theta


# Compute the root-mean-square-error between X and Xpr. Rows are observations.
def rmse(X, Xpr):
  return np.sqrt(np.mean(np.sum((X - Xpr)**2, 1)))

# Compute the root-mean-square-error between X and Xpr. Rows are observations.
def mse(X, Xpr):
  return np.mean(np.sum((X - Xpr)**2, 1))


# Learn an autoencoder for the Frey Face data. Ntr = no. train data, Dz = no.
# latent dims (no. units in the squished layer between the encoder and decoder).
# Saves the learning curve as frey-loglik.pdf and computes + prints the training 
# / test rmse. Returns function to pass data through the entire autoencoder as
# well as to encode and decode.
def LearnFreyFace(epochs=100, Dz=20, Ntr=1500):

  # Load Frey Face data with 1500 of the faces as training data. nb. returns
  # theano shared variable objects. To access the actual data use
  # Xtr.get_value() or Xte.get_value().
  Xtr, Xte = data.LoadFreyFace(Ntr)

  # Construct autoencoder.
  train, reconstruct, encode, decode, theta = ConstructAE( \
    Xtr, Denc=[200], Dz=Dz, Ddec=[200], inf=AdaGrad(0.01))

  # Train the autoencoder. Permute the order of the data after each epoch - 
  # if you don't do this then you get weird periodicities in the learning curve.
  print('Training the autoencoder.')
  batch_size, mse = 100, []
  for i in range(epochs):
    idx = rnd.permutation(np.arange(Ntr)).astype(np.int32())
    lb = 0
    while lb < Ntr:
      ub = lb + batch_size
      if ub > Ntr:
        ub = Ntr
      mse.append(train(idx[lb:ub]))
      lb = ub
    print('Epoch ' + str(i) + '. mse = ' + str(mse[-1]))

  # Save the learning curve.
  #plt.plot(mse)
  #plt.savefig('frey-loglik.pdf')
  #plt.close()

  # Compute reconstruction error. I'm using rmse.
  Xtrpr, Xtepr = reconstruct(Xtr.get_value()), reconstruct(Xte.get_value())
  rmsetr, rmsete = rmse(Xtr.get_value(), Xtrpr), rmse(Xte.get_value(), Xtepr)
  print('training rmse = ' + str(rmsetr))
  print('testing rmse = ' + str(rmsete))

  return reconstruct, encode, decode, Xtr, Xte


# Learn an autoencoder for the mnist data. Ntr = no. training data, Dz = no.
# latent dims (no. units in the squished layer between the encoder and decoder).
# Saves the learning curve as mnist-loglik.pdf and computes + prints the train
# / test rmse. Returns function to pass data through the entire autoencoder as
# well as to encode and decode.
def LearnMNIST(epochs=25, Dz=20, Ntr=50000):

  # Load mnist data with 50000 training images. Note: returns
  # theano shared variable objects. To access the actual data use
  # Xtr.get_value() or Xte.get_value().
  Xtr, Ytr, Xte, Yte = data.LoadMNIST(Ntr)

  # Construct autoencoder.
  train, reconstruct, encode, decode, theta = ConstructAE( \
    Xtr, Denc=[500], Dz=Dz, Ddec=[500], inf=AdaGrad(0.01))

  # Train the autoencoder. Permute the order of the data after each epoch - 
  # if you don't do this then you get weird periodicities in the learning curve.
  print('Training the autoencoder.')
  batch_size, mse = 100, []
  for i in range(epochs):
    idx = rnd.permutation(np.arange(Ntr)).astype(np.int32())
    lb = 0
    while lb < Ntr:
      ub = lb + batch_size
      if ub > Ntr:
        ub = Ntr
      mse.append(train(idx[lb:ub]))
      lb = ub
    print('Epoch ' + str(i) + '. mse = ' + str(mse[-1]))

  # Save the learning curve.
  #plt.plot(mse)
  #plt.savefig('mnist-mse.pdf')
  #plt.close()

  # Compute reconstruction error. I'm using rmse.
  Xtrpr, Xtepr = reconstruct(Xtr.get_value()), reconstruct(Xte.get_value())
  rmsetr, rmsete = rmse(Xtr.get_value(), Xtrpr), rmse(Xte.get_value(), Xtepr)
  print('training rmse = ' + str(rmsetr))
  print('testing rmse = ' + str(rmsete))

  return reconstruct, encode, decode, Xtr, Xte


# Learns an autoencoder with Dz=20 dimensions for both Frey Faces and mnist.
# Uses 1500 images as training data for Frey Face and 50000 for mnist.
def main():

  # Seed with a large-ish prime.
  rnd.seed(15485863)

  with open('../reconstruction_res/AE_MSE.res', 'w') as f :
    f.write('data_type,latent_size,MSE\n')


  # Appears to still be improving after 200 iterations.
  for Dz in [2, 10, 20] :
    epochs = 550 if Dz > 2 else 100
    reconstruct, encode, decode, Xtrain, XTest = LearnFreyFace(epochs=epochs, Dz=Dz, Ntr=1500)

    # Spit out some faces. (Close the window to iterate.)
    for t in range(8):
      reconstructed = reconstruct(XTest.get_value()[t:t+1])
      VAEBImage.save_image(XTest.get_value()[t:t+1], '../reconstruction_res/AE_continuous_{0}_orig_{1}.jpg'.format(Dz, t))
      VAEBImage.save_image(reconstructed, '../reconstruction_res/AE_continuous_{0}_recon_{1}.jpg'.format(Dz, t))

    mseVal = mse(XTest.get_value(), reconstruct(XTest.get_value()))
    with open('../reconstruction_res/AE_MSE.res', 'a') as f :
      f.write('{0},{1},{2}\n'.format('continuous',Dz,mseVal))

    # Generally requires fewer epochs owing to being a smaller data set.
    reconstruct, encode, decode, XTrain, XTest = LearnMNIST(epochs=1000, Dz=Dz, Ntr=50000)

    # Spit out some faces. (Close the window to iterate.)
    for t in range(8):
      reconstructed = reconstruct(XTest.get_value()[t:t+1])
      VAEBImage.save_image(XTest.get_value()[t:t+1], '../reconstruction_res/AE_discrete_{0}_orig_{1}.jpg'.format(Dz, t))
      VAEBImage.save_image(reconstructed, '../reconstruction_res/AE_discrete_{0}_recon_{1}.jpg'.format(Dz, t))

    mseVal = mse(XTest.get_value(), reconstruct(XTest.get_value()))
    with open('../reconstruction_res/AE_MSE.res', 'a') as f :
      f.write('{0},{1},{2}\n'.format('discrete',Dz,mseVal))

if __name__ == '__main__':
  main()
