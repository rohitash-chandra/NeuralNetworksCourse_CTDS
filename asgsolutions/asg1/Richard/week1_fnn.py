#!/usr/bin/env python

"""
RS 2017/06/25:  Assignment #1 -- Feed-Forward Neural Network

I used Rohit's code as a reference for writing this, but thought it might be
worthwhile for me to structure the code in my own way, so here's my version.
"""

import copy
import time
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
#                               Helper functions
# ----------------------------------------------------------------------------

def logistic(x):
    """
    Activation function for a single neuron
    """
    return 1.0/(1 + np.exp(-x))

def normalize(D):
    """
    Normalizes datasets so that all features lie between 0 and 1.
    """
    Dmin, Dmax = np.amin(D, axis=0), np.amax(D, axis=0)
    return (D - Dmin)/(Dmax - Dmin)

def bit_encode(fset):
    """
    Transforms categorical variables into bit sequences
    """
    # Here I use a few incantations to just format the string in binary,
    # and then pull apart the binary digit string into an array of bits.
    sset = np.unique(sorted(fset))
    intlabels = range(len(sset))
    nbits = len("{:b}".format(len(sset)))
    fmt = "{{:0{}b}}".format(nbits)
    strlabels = [ fmt.format(i) for i in intlabels ]
    binlabels = [ [ int(c) for c in s ] for s in strlabels ]
    for i, s in enumerate(sset):
        print "{:20} {}".format(s, binlabels[i])
    return { s:l for s, l in zip(sset, binlabels) }

# ----------------------------------------------------------------------------
#                                The main class
# ----------------------------------------------------------------------------

class NeuralNet(object):
    """
    Basic feed-forward neural net.  It has layers, weights and biases,
    can be trained.
    """

    def __init__(self, learning_rate=0.5, topology=[]):
        """
        Constructor.
            learning_rate:  dimensionless learning rate between 0 and 1,
                the fraction of the parameter gradient vector to take as
                a single step during stochastic gradient descent
            topology:  iterable of integers specifying the number of neurons
                in each layer a fully connected neural net architecture
        """
        self._layer_size = [ ]
        self._weights = [ ]
        self._biases = [ ]
        self.learning_rate = learning_rate
        self.debug = False
        for n in topology:
            self.add_layer(n)

    def add_layer(self, n):
        """
        Adds a fully connected neural net layer.  Layers are evaluated via
            a[l] = f(dot(w[l-1], a[l-1]) + b[l-1])
        so if a[l-1].shape = (n[l-1],), then w[l-1].shape = (n[l], n[l-1]).
            n:  number of neurons in layer

        """
        self._layer_size.append(n)
        if len(self._layer_size) > 1:
            nlast = self._layer_size[-2]
            self._weights.append(np.random.uniform(size=(n, nlast)))
            self._biases.append(np.random.uniform(size=(n,)))

    def _forward_pass(self, x):
        """
        Runs a forward pass of the network.  This simply evaluates the network
        by successively applying weights and biases of successive layers.
        Returns the whole state of the network.  Assumes logistic activation,
        but will eventually generalize.  Fills self._nodes with the actual
        values of the nodes in each layer.
            x:  training example, shape = (n_features,)
        """
        self._nodes = [ np.array(x) ]
        for l in range(1, len(self._layer_size)):
            xt = np.dot(self._weights[l-1], self._nodes[-1]) - self._biases[l-1]
            self._nodes.append(logistic(xt))
        return self._nodes[-1]

    def _backward_pass(self, x, y):
        """
        Runs a backward pass of the network.  This involves working backwards
        through the list of compositions.  Assumes logistic activation and
        mean square error as objective function, but will generalize someday.
        Fills self.deltas, with the same structure as nodes except for the
        original input layer (which is given).
            x:  training example input, shape = (n_features,)
            y:  labeled output for x, shape = (n_outputs,)
        """
        # Gradient for output layer:  dC/dx = dC/da * da/dx
        # where a(x) = activation on inputs x
        yout = self._nodes[-1]
        delta = (y - yout) * yout * (1 - yout)
        self._deltas = [ np.array(delta) ]
        # Gradients for hidden layers, working backwards from output layer
        # w[l].shape = (n[l], n[l-1]), so w[l].T.shape = (n[l-1], n[l])
        for l in range(len(self._layer_size)-2, 0, -1):
            yhid = self._nodes[l]
            delta = np.dot(self._weights[l].T, delta) * yhid * (1 - yhid)
            self._deltas.insert(0, delta)

    def _update_weights(self):
        """
        Update weights based on forward and backward passes through network.
        Assumes methods _forward_pass() and _backward_pass() have already been
        called on this training iteration.
        """
        eta = self.learning_rate
        for l in range(len(self._layer_size)-1, 0, -1):
            Nl, Dl = self._nodes[l-1], self._deltas[l-1]
            self._weights[l-1] += eta * Nl[None, :] * Dl[:, None]
            self._biases[l-1] += -eta * Dl
        if self.debug:
            print "*** nodes =", self._nodes
            print "*** deltas =", self._deltas
            print "*** weights =", self._weights
            print "*** biases =", self._biases

    def train(self, Xtrain, Ytrain, batchsize=None, iterations=100,
              msetol=0.0, verbose=True):
        """
        Train for N iterations or until a specified tolerance is reached.
        Results in a trained classifier which can be applied or tested.
            Xtrain:  training inputs, shape = (n_samples, n_features)
            Ytrain:  training outputs, shape = (n_samples, n_outputs)
            batchsize:  number of training examples per batch; defaults to
                10% of all training examples
            iterations:  maximum number of iterations to train
            msetol:  closest mean square error allowed to count a training
                example as being "correct"
            verbose:  print status messages during training?
        """
        # Sensible default batch size
        N = len(Xtrain)
        if batchsize is None:
            batchsize = N/10 + 1
        if verbose:
            print "NeuralNet.train():  starting training --",
            print "network layer sizes =", self._layer_size
        # Iterate until we're either done or sick of doing.
        self._objmu_best = np.inf
        self._objmu_hist = [ ]
        for i in range(iterations):
            # The heart of the method:  forward pass, backward pass, train
            # for each training example in a random batch w/replacement
            resids = [ ]
            batchidx = np.random.randint(N, size=(batchsize,))
            for x, y in zip(Xtrain[batchidx], Ytrain[batchidx]):
                resids.append(y - self._forward_pass(x))
                self._backward_pass(x, y)
                self._update_weights()
            # Calculate the objective function inline so we don't have to
            # redo forward passes in a separate call to the test() method.
            # Keep objective function history in case we want to plot vs time.
            resids = np.array(resids)
            objfunc = np.mean(resids**2, axis=1)
            objmu = np.mean(objfunc)
            self._objmu_hist.append(objmu)
            # If this is our best objective function, save parameters
            if objmu < self._objmu_best:
                self._objmu_best = objmu
                self._weights_best = copy.deepcopy(self._weights)
                self._biases_best = copy.deepcopy(self._biases)
                if verbose:
                    print "NeuralNet.train(), iteration {:4d}: ".format(i),
                    print "new MSE minimum =", objmu
            # If we're good enough, quit
            if objmu < msetol:
                break
        if verbose:
            print "NeuralNet.train():  done after {:d} iterations".format(i+1)

    def test(self, Xtest, Ytest, msetol):
        """
        Test performance of network according to MSE objective function
        without backpropagating or optimizing.  As a side effect, sets all
        weights and biases to the values that achieved the lowest MSE results
        during the training process.
            Xtest:  testing inputs, shape = (n_samples, n_features)
            Ytest:  testing outputs, shape = (n_samples, n_outputs)
            msetol:  closest mean square error allowed to count a training
                example as being "correct"
        """
        # Load best available knowledge
        self._weights, self._weights_best = self._weights_best, self._weights
        self._biases, self._biases_best = self._biases_best, self._biases
        # Calculate objective function and fraction of correct answers
        resids = [ ]
        for x, y in zip(Xtest, Ytest):
            resids.append(y - self._forward_pass(x))
        resids = np.array(resids)
        objfunc = np.mean(resids**2, axis=1)
        correct = np.isclose(0, np.amin(resids, axis=1), atol=msetol)
        return np.mean(objfunc), np.mean(correct)

    def predict(self, x):
        """
        Apply the classifier.
            x:  training example, normalized np.array of shape (n_features,)
        """
        return self._forward_pass(x)


def test_shapes():
    """
    Test of constructor and basic training.  Just makes sure all arrays
    throughout the network have compatible dimensions.  If they don't, all
    will probably crash.  Nothing to do with accuracy yet.
    """
    net = NeuralNet(topology=[9,7,5,3,1])
    Xtrain = np.random.uniform(size=(1, net._layer_size[0]))
    net.train(Xtrain, Xtrain[:,-1], iterations=1, verbose=False)
    print "net._layer_size =", net._layer_size
    for i, w in enumerate(net._weights):
        print "between layers", i, "and", i+1, ":",
        print "net._weights.shape = ", net._weights[i].shape,
        print "net._biases.shape = ", net._biases[i].shape
        print "                        ",
        print "net._nodes.shape = ", net._nodes[i].shape,
        print "net._deltas.shape = ", net._deltas[i].shape


# ----------------------------------------------------------------------------
#                                  Exercises
# ----------------------------------------------------------------------------


def run_network(Xtrain, Xtest, Ytrain, Ytest, topology):
    """
    Test of performance.  How well are we doing on a well-known dataset?
    """
    mse, pct, itc = [ ], [ ], [ ]
    # Run over several training sessions to see how consistent performance is
    start_time = time.time()
    for i in range(30):
        net = NeuralNet(topology=topology, learning_rate=0.2)
        net.train(Xtrain, Ytrain, msetol=0.1,
                  batchsize=len(Xtrain), iterations=2000, verbose=True)
        msei, pcti = net.test(Xtest, Ytest, 0.49)
        mse.append(msei)
        pct.append(100.0*pcti)
        itc.append(len(net._objmu_hist))
    elapsed = time.time() - start_time
    print "total training + eval time:  {:.3f} sec".format(elapsed)
    print "final performance on {:d} test instances:  ".format(len(Xtest)),
    print "mse = {:.3f} (+/- {:.3f}),".format(np.mean(mse), np.std(mse)),
    print "acc = {:.1f}% (+/- {:.1f}),".format(np.mean(pct), np.std(pct)),
    print "itc = {:.1f} (+/- {:.1f})".format(np.mean(itc), np.std(itc))

def main_iris():
    """
    Problem 1:  run on the iris dataset with one and two hidden layers
    Problem 2:  try more hidden layers
    Problem 3:  futz around with the architecture more broadly
    """
    # For the iris dataset, Rohit is just using the first 4 columns as the
    # features and the last 2 as the labels, so we'll do that here.
    Dtrain = normalize(np.loadtxt('datasets/iris_train.csv', delimiter=','))
    Dtest = normalize(np.loadtxt('datasets/iris_test.csv', delimiter=','))
    Xtrain, Xtest = Dtrain[:, :4], Dtest[:, :4]
    Ytrain, Ytest = Dtrain[:, 4:], Dtest[:, 4:]
    # Run different topologies
    run_network(Xtrain, Xtest, Ytrain, Ytest, topology=[4,6,2])
    print "-" * 72
    run_network(Xtrain, Xtest, Ytrain, Ytest, topology=[4,6,6,2])
    print "-" * 72
    run_network(Xtrain, Xtest, Ytrain, Ytest, topology=[4,4,3,2])

def main_wine():
    """
    Problem 4:  try five other datasets (or at least one)
    """
    # For the wine dataset, the labels are in the first column and the rest
    # of the features follow.  I'll hold out 1/4 of the data for testing.
    data = normalize(np.loadtxt('datasets/wine.data', delimiter=','))
    class_labels = bit_encode(range(3))
    relabel = np.array([class_labels[int(i)] for i in data[:,0]])
    idx = (np.random.permutation(len(data)) < 130)
    Xtrain, Xtest = data[idx, 1:], data[~idx, 1:]
    Ytrain, Ytest = relabel[idx], relabel[~idx]
    # Run different topologies
    run_network(Xtrain, Xtest, Ytrain, Ytest, topology=[13,6,2])

def main_forestfires():
    """
    Problem 4:  try five other datasets (or at least one)
    """
    # The forest fire dataset is a regression task, and they recommend
    # log-transforming the response variable (burned area) before training.
    # There are several categorical variables I'll need to bit-encode.
    # As before, I'll hold out 1/4 of the data for testing.
    data = normalize(np.loadtxt('datasets/forestfires.csv', delimiter=','))

if __name__ == "__main__":
    main_wine()
