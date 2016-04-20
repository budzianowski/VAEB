#!/usr/bin/env python
#

import theano as th
import theano.tensor as T
import numpy as np
import gzip
import time
import cPickle
import sys
import copy

floatX = th.config.floatX

#to add another command line argument, simply add:
#   its name as a key
#   value as a tuple of its default value and the argument type (e.g. int, string, float)
command_line_args = {"seed" : (10, int),
                     "n_latent" : (10, int),
                     "n_epochs" : (2000, int)}
#to add a new flag, simply add its name
command_line_flags = {"continuous" : True}

class VAE(object):
    def __init__(self, x_train, continuous=False, hidden_units=500, latent_size=10,
                 batch_size=100, L=1, learning_rate=0.01):

        [self.N, self.input_size] = x_train.shape  # number of observations and features
        self.n_hidden_units = hidden_units
        self.n_latent = latent_size  # size of z
        self.continuous = continuous  # if we want to use MNIST or Frey data set
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.prng = np.random.RandomState(10)
        self.sigmaInit = 0.01    # variance to initialize parameters, from pg. 7
        self.L = L  # number of samples from p(z|x)

        # Initialization of weights (notation from pg.11):
        initW = lambda dimIn, dimOut: self.prng.normal(0,  self.sigmaInit, (dimIn, dimOut)).astype(floatX)
        initB = lambda dimOut: np.zeros((dimOut, )).astype(floatX)

        # Notation as in the pg. 11
        # Encoder
        # h
        W_values = initW(self.input_size, self.n_hidden_units)
        b_values = initB(self.n_hidden_units)
        self.W3 = th.shared(value=W_values, name='W3')
        self.b3 = th.shared(value=b_values, name='b3')

        # mu
        W_values = initW(self.n_hidden_units, self.n_latent)
        b_values = initB(self.n_latent)
        self.W4 = th.shared(value=W_values, name='W4')
        self.b4 = th.shared(value=b_values, name='b4')

        # sigma
        W_values = initW(self.n_hidden_units, self.n_latent)
        b_values = initB(self.n_latent)
        self.W5 = th.shared(value=W_values, name='W5')
        self.b5 = th.shared(value=b_values, name='b5')

        # Decoder
        # tanh layer
        W_values = initW(self.n_latent,  self.n_hidden_units)
        b_values = initB(self.n_hidden_units)
        self.W1 = th.shared(value=W_values, name='W1')
        self.b1 = th.shared(value=b_values, name='b1')

        W_values = initW(self.n_hidden_units, self.input_size)  # or mu for continuous output
        b_values = initB(self.input_size)
        self.W2 = th.shared(value=W_values, name='W2')
        self.b2 = th.shared(value=b_values, name='b2')

        if self.continuous: # for Freyfaces
            W_values = initW(self.n_hidden_units, self.input_size)  # sigma for gaussian output
            b_values = initB(self.input_size)
            self.W6 = th.shared(value=W_values, name='W6')
            self.b6 = th.shared(value=b_values, name='b6')

            self.params = [self.W3, self.W4, self.W5, self.W1, self.W2, self.W6, self.b3, self.b4, self.b5, self.b1, self.b2, self.b6]
        else:
            self.params = [self.W3, self.W4, self.W5, self.W1, self.W2, self.b3, self.b4, self.b5, self.b1, self.b2]

        # ADA-GRAD parameters
        self.ADA = []
        for param in self.params:
            eps_p = np.zeros_like(param.get_value(borrow=True), dtype=floatX)
            self.ADA.append(th.shared(eps_p, borrow=True))

        # # TODO: ADA-DELTA parameters
        # self.ADA1 = []
        # self.ADA2 = []
        # for param in self.params:
        #     eps_p = np.zeros_like(param.get_value(borrow=True), dtype=floatX)
        #     self.ADA1.append(th.shared(eps_p, borrow=True))
        #     self.ADA2.append(th.shared(eps_p, borrow=True))

        x_train = th.shared(np.asarray(x_train, dtype=floatX), name="x_train")

        # UPDATE and VALIDATE FUNCTION
        self.update, self.validate = self.getGradient(x_train)

    def encoder(self, x):
        h = T.tanh(T.dot(x, self.W3) + self.b3)

        mu = T.dot(h, self.W4) + self.b4
        log_sigma = T.dot(h, self.W5) + self.b5

        return mu, log_sigma

    def decoder(self, x, z):
        h = T.tanh(T.dot(z, self.W1) + self.b1)

        if self.continuous:
            mu = T.nnet.sigmoid(T.dot(h, self.W2) + self.b2)
            log_sigma = T.dot(h, self.W6) + self.b6

            # Log-likelihood for Gaussian
            logpXgivenZ = (- 0.5 * np.log(2 * np.pi) - 0.5 * log_sigma \
                              - 0.5 * (x - mu) ** 2 / T.exp(log_sigma)).sum(axis=1, keepdims=True)

        else:
            y = T.nnet.sigmoid(T.dot(h, self.W2) + self.b2)
            # Cross entropy
            logpXgivenZ = -T.nnet.binary_crossentropy(y, x).sum(axis=1, keepdims=True)  # pg.11 for MNIST

        return logpXgivenZ

    # MAIN function for feed-forwarding and getting update
    def getGradient(self, x_train):
        x = T.matrix('x')   # creating Theano variable for input
        index = T.iscalar('index')  # creating Theano variable for batching

        # encoding
        mu, log_sigma = self.encoder(x)

        # creating random variable for reparametrization trick
        srng = T.shared_randomstreams.RandomStreams(seed=10)
        eps = srng.normal(mu.shape)  # shared random variable, Theano magic

        # reparametrization trick
        z = mu + T.exp(0.5 * log_sigma) * eps

        # decoding
        logpXgivenZ = self.decoder(x, z)

        # KL
        KL = 0.5 * T.sum(1 + log_sigma - mu ** 2 - T.exp(log_sigma), axis=1, keepdims=True)

        # SGVB = KL + p(x|z) , eq. 10
        logpx = T.mean(KL + logpXgivenZ)

        # gradients
        gradients = T.grad(logpx, self.params)

        # update of parameters
        updates = self.getUpdates(gradients)

        # update function
        update = th.function(
            inputs=[index],
            outputs=logpx,
            updates=updates,
            givens={
                x: x_train[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        # getting likelihood for validation set
        validate = th.function(
            [x],
            logpx,
            allow_input_downcast=True
        )

        return update, validate

    def getUpdates(self, gradients):
        eps = 0.000001  # fudge factor for for ADA-GRAD and MAP

        # # SGD with prior (MAP) or L2 regularisation
        # updates = [
        #     (param, param + self.learning_rate * (gradient - eps * param ** 2))
        #     for param, gradient in zip(self.params, gradients)
        # ]

        # ADA-GRAD
        updates = []
        for param, gradient, ada in zip(self.params, gradients, self.ADA):
            acc = ada + T.sqr(gradient)   # squared!

            updates.append((param, param + self.learning_rate * gradient / (T.sqrt(acc) + eps) -
                             self.learning_rate * eps * (param ** 2)))  # MAP
            updates.append((ada, acc))

        # # TODO: ADA-DELTA
        # decay = 0.9
        # updates = []
        # for param, gradient, ada1, ada2 in zip(self.params, gradients, self.ADA1, self.ADA2):
        #     accGradient = decay * ada1 + (1 - decay) * T.sqr(gradient)   # squared!
        #     upd = (T.sqrt(ada2 + eps) / T.sqrt(accGradient + eps)) * gradient
        #     accUpdate = decay * ada2 + (1 - decay) * T.sqr(upd)  # squared!
        #
        #     updates.append((param, param - upd \
        #                     - self.learning_rate * eps * (param ** 2)))  # MAP
        #
        #     updates.append((ada1, accGradient))
        #     updates.append((ada2, accUpdate))

        return updates

def get_arg(arg, args, default, type_) :
  arg = '--'+arg
  if arg in args :
    index = args.index(arg)
    value = args[args.index(arg) + 1]
    del args[index]   #remove arg-name
    del args[index]   #remove value
    return type_(value)
  else :
    return default


def get_flag(flag, args, default_value) :
  flag = '-'+flag
  have_flag = flag in args
  if have_flag :
    args.remove(flag)
    return True

  return default_value

def parse_args() :
  args = copy.deepcopy(sys.argv[1:])
  arg_dict = {}
  for (arg_name, arg_args) in command_line_args.iteritems() :
    (arg_defalut_val, arg_type) = arg_args
    arg_dict[arg_name] = get_arg(arg_name, args, arg_defalut_val, arg_type)

  for (flag_name, default_value) in command_line_flags.iteritems() :
    arg_dict[flag_name] = get_flag(flag_name, args, default_value)

  return arg_dict


if __name__ == '__main__':
    # model specification
    args = parse_args()
    np.random.seed(args['seed'])
    n_latent = args['n_latent']
    n_epochs = args['n_epochs']
    continuous = args['continuous']

    print("loading data")
    if continuous:
        hu_N = 200
        f = open('freyfaces.pkl', 'rb')
        x = cPickle.load(f)  # only 1965 observations
        f.close()
        x_train = x[:1500]  # about 90% of the data
        x_valid = x[1500:]
    else:
        hu_N = 500
        f = gzip.open('mnist.pkl.gz', 'rb')
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = cPickle.load(f)  # 50000/10000/10000 observations
        f.close()

    print("creating the model")
    model = VAE(x_train, continuous, hu_N, n_latent)

    print("learning")
    batch_order = np.arange(int(model.N / model.batch_size))  # ordering of the batches
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        start = time.time()
        np.random.shuffle(batch_order)
        LB = 0.0

        for batch in batch_order:
            batch_LB = model.update(batch)
            LB += batch_LB

        LB /= len(batch_order)

        print("Epoch %s : [Lower bound: %s, time: %s]" % (epoch, LB, time.time() - start))
        LBvalidation = model.validate(x_valid)
        print("          [Lower bound on validation set: %s]" % LBvalidation)
