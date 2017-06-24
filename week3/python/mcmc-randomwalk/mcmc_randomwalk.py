# Random Walk MCMC for Weighted Mixture of Distributions for  Curve Fitting.
# Rohitash Chandra and Sally Cripps (2017).
# CTDS, UniSYD. c.rohitash@gmail.com
# Simulated data is used.


# Ref: https://en.wikipedia.org/wiki/Dirichlet_process
# https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.random.dirichlet.html

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import matplotlib.mlab as mlab
import time


def fx_func(nModels, x, mu, sig, w):
    fx = np.zeros(x.size)
    for i in range(nModels):
        fx = fx + w[i] * mlab.normpdf(x, mu[i], np.sqrt(sig[i]))
    return fx


def likelihood_func(nModels, x, y, mu, sig, w, tau):
    tausq = tau
    fx = fx_func(nModels, x, mu, sig, w)
    loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
    return [np.sum(loss), fx]


def sampler(samples, nModels, x, ydata):
    # memory for posterior mu, sig, w and tau - also fx
    pos_mu = np.ones((samples, nModels))
    pos_sig = np.ones((samples, (nModels)))
    pos_w = np.ones((samples, (nModels)))
    pos_tau = np.ones((samples,  1))

    fx_samples = np.ones((samples, ydata.size))

    mu_current = np.zeros(nModels)
    mu_proposal = np.zeros(nModels)

    sig_current = np.zeros(nModels)  # to get sigma
    nu_current = np.zeros(nModels)  # to get sigma
    sig_proposal = np.zeros(nModels)  # sigma
    nu_proposal = np.zeros(nModels)
    w_current = np.zeros(nModels)
    w_proposal = np.zeros(nModels)

    step_size_mu = 0.1  # need to choose these values according to the problem
    step_size_nu = 0.2
    step_size_eta = 0.1

    for i in range(nModels):
        sig_current[i] = np.var(x)
        mu_current[i] = np.mean(x)
        nu_current[i] = np.log(sig_current[i])
        w_current[i] = 1.0 / nModels

    fx = fx_func(nModels, x, mu_current, sig_current, w_current)

    t = np.var(fx - ydata)
    tau_current = t
    eta_current = np.log(t)
    eta_proposal = 0.1
    tau_proposal = 0   # vector notation is used although size is 1

    likelihood_current, fx = likelihood_func(nModels, x, ydata, mu_current,
                                             sig_current, w_current,
                                             tau_current)

    print(t, eta_current)
    print(likelihood_current)

    naccept = 0
    print('begin sampling')
    plt.plot(x, ydata)
    plt.plot(x, fx)
    plt.title("Plot of Data vs Initial Fx")
    plt.savefig('mcmcresults/begin.png')
    plt.clf()

    plt.plot(x, ydata)

    for i in range(samples - 1):

        # print(likelihood_current, mu_current, nu_current, eta_current,
        #       'current')

        if nModels == 1:
            weights = [1]

        if nModels == 2:
            # (genreate vector that  adds to 1)
            weights = np.random.dirichlet((1, 1), 1)
        if nModels == 3:
            weights = np.random.dirichlet((1, 1, 1), 1)  # (vector adds to 1)
        if nModels == 4:
            weights = np.random.dirichlet(
                (1, 1, 1, 1), 1)  # (vector adds to 1)
        if nModels == 5:
            weights = np.random.dirichlet(
                (1, 1, 1, 1, 1), 1)  # (vector adds to 1)

        nu_proposal = nu_current + np.random.normal(0, step_size_nu, nModels)
        sig_proposal = np.exp(nu_proposal)

        mu_proposal = mu_current + np.random.normal(0, step_size_mu, nModels)

        for j in range(nModels):
            # ensure they stay between a range
            if mu_proposal[j] < 0 or mu_proposal[j] > 1:
                mu_proposal[j] = random.uniform(np.min(x), np.max(x))

            w_proposal[j] = weights[0, j]  # just for vector consistency

        eta_proposal = eta_current + np.random.normal(0, step_size_eta, 1)
        tau_proposal = math.exp(eta_proposal)

        likelihood_proposal, fx = likelihood_func(nModels, x, ydata,
                                                  mu_proposal, sig_proposal,
                                                  w_proposal, tau_proposal)

        diff = likelihood_proposal - likelihood_current

        mh_prob = min(1, math.exp(diff))

        u = random.uniform(0, 1)

        if u < mh_prob:
            # Update position
            print(i, ' is accepted sample')
            naccept += 1
            likelihood_current = likelihood_proposal
            mu_current = mu_proposal
            nu_current = nu_proposal
            eta_current = eta_proposal

            print(likelihood_current, mu_current, nu_current,  eta_current,
                  'accepted')

            # print(pos_mu[i])
            pos_mu[i + 1, ] = mu_proposal
            pos_sig[i + 1, ] = sig_proposal
            pos_w[i + 1, ] = w_proposal
            pos_tau[i + 1, ] = tau_proposal
            fx_samples[i + 1, ] = fx
            plt.plot(x, fx)

        else:
            pos_mu[i + 1, ] = pos_mu[i, ]
            pos_sig[i + 1, ] = pos_sig[i, ]
            pos_w[i + 1, ] = pos_w[i, ]
            pos_tau[i + 1, ] = pos_tau[i, ]
            fx_samples[i + 1, ] = fx_samples[i, ]

            # print(i, 'rejected and retained')

    print(naccept, ' num accepted')
    print(naccept / samples, '% was accepted')

    plt.title("Plot of Accepted Proposals")
    plt.savefig('mcmcresults/proposals.png')
    plt.clf()

    return (pos_mu, pos_sig, pos_w, pos_tau, fx_samples)


def main():
    random.seed(time.time())
    nModels = 2

    # load univariate data in same format as given
    modeldata = np.loadtxt('simdata.txt')
    # print modeldata

    ydata = modeldata  #
    print(ydata.size)
    x = np.linspace(1 / ydata.size, 1, num=ydata.size)  # (input x for ydata)

    NumSamples = 50000   # need to pick yourself

    pos_mu, pos_sig, pos_w, pos_tau, fx_samples = sampler(NumSamples, nModels,
                                                          x, ydata)
    print('sucessfully sampled')

    burnin = 0.05 * NumSamples   # use post burn in samples
    pos_mu = pos_mu[int(burnin):]
    pos_sig = pos_sig[int(burnin):]
    pos_w = pos_w[int(burnin):]
    pos_tau = pos_tau[int(burnin):]

    fx_mu = fx_samples.mean(axis=0)
    fx_high = np.percentile(fx_samples, 95, axis=0)
    fx_low = np.percentile(fx_samples, 5, axis=0)

    plt.plot(x, ydata)
    plt.plot(x, fx_mu)

    plt.plot(x, fx_low)
    plt.plot(x, fx_high)
    plt.fill_between(x, fx_low, fx_high, facecolor='g', alpha=0.4)

    plt.title("Plot of Data vs MCMC Uncertainty ")
    plt.savefig('mcmcresults/mcmcres.png')
    plt.clf()


if __name__ == "__main__":
    main()
