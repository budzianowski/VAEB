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
import pickle
import os


floatX = th.config.floatX

#   to add another command line argument, simply add:
#   its name as a key
#   value as a tuple of its default value and the argument type (e.g. int, string, float)
command_line_args = {'seed' : (10, int),
                     'n_latent' : (2, int),
                     'n_epochs' : (2000, int),
                     'batch_size' : (100, int),
                     'L' : (1, int),
                     'hidden_unit' : (-1, int),
                     'learning_rate' : (0.01, float),
                     'trace_file' : ('', str)}          #if set, trace information will be written about number of training
                                                        #samples and lower bound
#to add a new flag, simply add its name
command_line_flags = ['continuous']

class VAE(object):
    def __init__(self, x_train, continuous, hidden_units, latent_size,
                 batch_size, L, learning_rate):

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

        if os.path.isfile('modelFrey.pkl'):
            if continuous:
                pkl_file = open('modelFrey.pkl', 'rb')
                states = pickle.load(pkl_file)
                self.W3 = th.shared(value=states[0], name='W3')
                self.W4 = th.shared(value=states[1], name='W4')
                self.W5 = th.shared(value=states[2], name='W5')
                self.W1 = th.shared(value=states[3], name='W1')
                self.W2 = th.shared(value=states[4], name='W2')
                self.W6 = th.shared(value=states[5], name='W6')
                self.b3 = th.shared(value=states[6], name='b3')
                self.b4 = th.shared(value=states[7], name='b4')
                self.b5 = th.shared(value=states[8], name='b5')
                self.b1 = th.shared(value=states[9], name='b1')
                self.b2 = th.shared(value=states[10], name='b2')
                self.b6 = th.shared(value=states[11], name='b6')
                self.params = [self.W3, self.W4, self.W5, self.W1, self.W2, self.W6, self.b3, self.b4, self.b5, self.b1, self.b2, self.b6]
            else:
                pkl_file = open('modelMNIST.pkl', 'rb')
                states = pickle.load(pkl_file)
                self.W3 = th.shared(value=states[0], name='W3')
                self.W4 = th.shared(value=states[1], name='W4')
                self.W5 = th.shared(value=states[2], name='W5')
                self.W1 = th.shared(value=states[3], name='W1')
                self.W2 = th.shared(value=states[4], name='W2')
                self.b3 = th.shared(value=states[5], name='b3')
                self.b4 = th.shared(value=states[6], name='b4')
                self.b5 = th.shared(value=states[7], name='b5')
                self.b1 = th.shared(value=states[8], name='b1')
                self.b2 = th.shared(value=states[9], name='b2')
                self.params = [self.W3, self.W4, self.W5, self.W1, self.W2, self.b3, self.b4, self.b5, self.b1, self.b2]

        else:
            # normal implementation
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

            if self.continuous:  # for Freyfaces
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


        x_train = th.shared(np.asarray(x_train, dtype=floatX), name="x_train")

        # UPDATE and VALIDATE FUNCTION
        self.update, self.validate, self.freyFace = self.getGradient(x_train)

    def saveState(self, filename):
        state = []
        for param in self.params:
            state.append(param.get_value())
        output = open(filename, 'wb')
        pickle.dump(state, output)
        output.flush()

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

    def image(self, z):
        h = T.tanh(T.dot(z, self.W1) + self.b1)

        if self.continuous:
            mu = T.nnet.sigmoid(T.dot(h, self.W2) + self.b2)
            log_sigma = T.dot(h, self.W6) + self.b6

            x = [mu, log_sigma]

        else:
            y = T.nnet.sigmoid(T.dot(h, self.W2) + self.b2)

            # return random bivariate
            x = y

        return x

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

        # Frey Faces
        f = T.dmatrix('f')
        face = self.image(f)
        freyFace = th.function(
            [f],
            face,
            allow_input_downcast=True
        )
        return update, validate, freyFace

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


def get_flag(flag, args) :
    flag = '-'+flag
    have_flag = flag in args
    if have_flag :
        args.remove(flag)

    return have_flag

def parse_args() :
    args = copy.deepcopy(sys.argv[1:])
    arg_dict = {}
    for (arg_name, arg_args) in command_line_args.iteritems() :
        (arg_defalut_val, arg_type) = arg_args
        arg_dict[arg_name] = get_arg(arg_name, args, arg_defalut_val, arg_type)

    for flag_name in command_line_flags :
        arg_dict[flag_name] = get_flag(flag_name, args)

    return arg_dict


def print_args(args) :
    print('Parameters used:')
    print('--------------------------------------')
    for (k, v) in args.iteritems() :
        print('\t{0}: {1}'.format(k, v))
    print('--------------------------------------')


if __name__ == '__main__':
    # model specification
    args = parse_args()
    print_args(args)

    np.random.seed(args['seed'])
    n_latent = args['n_latent']
    n_epochs = args['n_epochs']
    continuous = args['continuous']
    batch_size = args['batch_size']
    L = args['L']
    hidden_unit = args['hidden_unit']
    learning_rate = args['learning_rate']
    trace_file = args['trace_file']

    continuous = False
    print("loading data")
    if continuous:
        if hidden_unit < 0:
            hidden_unit = 200
        f = open('freyfaces.pkl', 'rb')
        x = cPickle.load(f)  # only 1965 observations
        f.close()
        x_train = x[:1500]  # about 90% of the data
        x_valid = x[1500:]
    else:
        if hidden_unit < 0:
            hidden_unit = 500
        f = gzip.open('mnist.pkl.gz', 'rb')
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = cPickle.load(f)  # 50000/10000/10000 observations
        f.close()

    print("creating the model")
    model = VAE(x_train, continuous, hidden_unit, n_latent, batch_size, L, learning_rate)

    import os
    import VAEBImage
    import scipy.stats
    os.chdir("freyFaces")
    # for ii in [0.1, 0.15, 0.2, 0.25, 0.3, 0.75, 0.8, 0.85, 0.9, 0.95]:
    #     for jj in [0.1, 0.15, 0.2, 0.25, 0.3, 0.75, 0.8, 0.85, 0.9, 0.95]:
    for ii in range(10):
        for jj in range(10):
            z = np.asarray([scipy.stats.norm.ppf((ii+0.9)/10.), scipy.stats.norm.ppf((jj+0.9)/10.)]).reshape(1, 2)
            #z = np.asarray([scipy.stats.norm.ppf(ii), scipy.stats.norm.ppf(jj)]).reshape(1, 2)
            if continuous:
                mu, log_sigma = model.freyFace(z)
                I = np.eye(mu.shape[1])
                cov = (np.exp(log_sigma)**2) * I
                # return random normal
                face = np.random.multivariate_normal(mu.reshape(560), cov)  # 560 pixels
                cmd = 'FREY' + str(ii) + str(jj) + '.jpg'
            else:
                face = model.freyFace(z)
                cmd = 'MNIST' + str(ii) + str(jj) + '.jpg'

            VAEBImage.save_image(face, cmd)

    VAEBImage.multipleImages('MNIST')

