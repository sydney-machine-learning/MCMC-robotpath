import sympy as sym
from sympy import *
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import os
from pylab import *
import random as rd

# The Lypunov model  
def distance(beta, delta):
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    v = sym.Symbol('v')
    w = sym.Symbol('w')
    ntime = 1000
    stepsize = 0.05
    X = [-15, -15, 0.1, 0.1]
    tau = [15, 15]
    LM = [-5, 5]
    o = [0, 1]
    rad = 2
    V = 0.5 * (((x - tau[0]) ** 2) + ((y - tau[1]) ** 2) + (v ** 2) + (w ** 2))
    F = 0.5 * (((x - tau[0]) ** 2) + ((y - tau[1]) ** 2))
    W = 0.5 * (((x - o[0]) ** 2) + ((y - o[1]) ** 2) - (rad ** 2))
    L = V + F * beta / W
    rho1 = -(delta * v + sym.diff(L, x))
    rho2 = -(delta * w + sym.diff(L, y))
    Ldot = -((delta) * (v ** 2)) - ((delta) * (w ** 2))
    FX = [rho1] * 4
    Xnew = [X[0]]
    Ynew = [X[1]]
    s = 0
    flag1 = 0
    for t in range(ntime):
        FX[0] = X[2]
        FX[1] = X[3]
        FX[2] = rho1.subs([(x, X[0]), (y, X[1]), (v, X[2]), (w, X[3])])
        FX[3] = rho2.subs([(x, X[0]), (y, X[1]), (v, X[2]), (w, X[3])])
        kX1 = [0] * len(X)
        kX2 = [0] * len(X)
        kX3 = [0] * len(X)

        for I in range(len(X)):
            kX1[I] = stepsize * FX[I]
            X[I] = X[I] + kX1[I] / 2
            kX2[I] = stepsize * FX[I]
            X[I] = X[I] + kX2[I] / 2
            kX3[I] = stepsize * FX[I]
            X[I] = X[I] + kX3[I] / 2
            X[I] = X[I] + (kX1[I] + 2 * (kX2[I] + kX3[I]) + stepsize * FX[I]) / 6

        lyap = L.subs([(x, X[0]), (y, X[1]), (v, X[2]), (w, X[3])])
        lyapdot = Ldot.subs([(x, X[0]), (y, X[1]), (v, X[2]), (w, X[3])])
        s = s + ((((Xnew[-1] - X[0]) ** 2) + ((Ynew[-1] - X[1]) ** 2)) ** 0.5)
        Xnew.append(X[0])
        Ynew.append(X[1])
        val1 = W.subs([(x, X[0]), (y, X[1])])  # checks whether it hit the obstacle or not
        val2 = F.subs([(x, X[0]), (y, X[1])])  # checks how close obj is to the target
        if (val1 <= 0):
            flag1 = 1

        if (val2 <= 0.1):
            break

    flag2 = 0;
    val2 = F.subs([(x, Xnew[-1]), (y, Ynew[-1])])  # checks whether the robot has reached the target or not
    if (val2 > 0.1):
        flag2 = 1
    return [s, flag1, flag2, Xnew, Ynew];


def plot_curve(Xnew, Ynew):
    X = [-15, -15, 0.1, 0.1];
    tau = [15, 15];
    rad = 2;
    o = [0, 1]
    figure, ax = plt.subplots(figsize=(8, 6))
    h, = ax.plot(X[0], X[1], '-');
    tt = np.linspace(0, 2 * mt.pi, 1000);
    ht, = ax.plot(tau[0], tau[1], '.');
    plt.setp(ht, markersize=10);
    cos_tt = []
    for i in range(len(tt)):
        cos_tt.append((rad * mt.cos(tt[i])) + o[0])
        i += 1
    sin_tt = []
    for i in range(len(tt)):
        sin_tt.append((rad * mt.sin(tt[i])) + o[1])
        i += 1

    hobs, = ax.fill(cos_tt, sin_tt, '.')
    plt.setp(hobs, facecolor='r')
    plt.grid()
    plt.axis('square')
    plt.axis([-20, 20, -20, 20])
    b, = ax.plot(Xnew, Ynew)
    plt.show()


def prior_func(sigma1_squared, sigma2_squared, m1, m2, nu_1, nu_2, w, tausq):
    # number of parameters in model
    part1 = -1 * (1 / 2) * (np.log(2 * math.pi * sigma2_squared) + np.log(2 * math.pi * sigma1_squared))
    part2 = (1 / (2 * sigma1_squared) * ((w[0] - m1) ** 2)) + (1 / (2 * sigma2_squared) * ((w[1] - m2) ** 2))
    log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
    return log_loss


def likelihood_func(displacement, w, tausq):
    # The distance covered by the robot in its path - the distance between start pt and end point is taken as the error measure 
    y = displacement
    [fx, flag1, flag2, X, Y] = distance(w[0], w[1])
    loss = 0
    if (flag1 + flag2 >= 1):
        loss = loss - 20
    # Added penalty to the loss fn for hitting the target
    loss = loss - 0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
    return [loss, fx, abs(fx - y), X, Y]

# MCMC part
def sampler(displacement):
    w_size = 2
    #sample size can be varied
    samples = 1000

    pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
    pos_tau = np.ones((samples, 1))
    error_train = np.zeros(samples)
    error_train = np.empty(samples)
    error_train.fill(1000)
    # w can be initialized as per our choice
    w = [0.7, 11]
    w_proposal = np.random.randn(w_size)

    step_w = 0.5;  # defines how much variation you need in changes to w
    step_eta = 0.1;
    X = []
    Y = []
    [pred_train, flag1, flag2, x, y] = distance(w[0], w[1])
    eta = mt.log((pred_train - displacement) ** 2)
    tau_pro = mt.exp(eta)
    print('Process Begins')
    # sigma1 and m1 represent the sd and mean of beta and sigma2 and m2 represent the sd and mean of delta
    # can be inititalised as per our convenience 
    sigma1_squared = 0.65
    sigma2_squared = 1.5
    m1 = 0.55
    m2 = 10
    # considered by looking at distribution of  similar trained  models - i.e distribution of weights and bias
    nu_1 = 0
    nu_2 = 0
    naccept = 0

    prior_likelihood = prior_func(sigma1_squared, sigma2_squared, m1, m2, nu_1, nu_2, w,
                                  tau_pro)  # takes care of the gradients
    [likelihood, pred_train, error, x, y] = likelihood_func(displacement, w, tau_pro)
    X.append(x)
    Y.append(y)

    for i in range(samples - 1):

        w_proposal = w + np.random.normal(0, step_w, w_size)
        eta_pro = eta + np.random.normal(0, step_eta, 1)
        tau_pro = math.exp(eta_pro)

        [likelihood_proposal, pred_train, error, xnew, ynew] = likelihood_func(displacement, w_proposal, tau_pro)
        prior_prop = prior_func(sigma1_squared, sigma2_squared, m1, m2, nu_1, nu_2, w_proposal, tau_pro)

        diff_likelihood = likelihood_proposal - likelihood
        diff_priorliklihood = prior_prop - prior_likelihood

        mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))
        u = rd.uniform(0, 1)

        if u < mh_prob:
            
            naccept += 1
            likelihood = likelihood_proposal
            prior_likelihood = prior_prop
            w = w_proposal
            eta = eta_pro
            error_train[i + 1,] = error
            print(i, likelihood, prior_likelihood, error, w, 'accepted')
            pos_w[i + 1,] = w_proposal
            pos_tau[i + 1,] = tau_pro
            X.append(xnew)
            Y.append(ynew)

        else:
            pos_w[i + 1,] = pos_w[i,]
            pos_tau[i + 1,] = pos_tau[i,]
            error_train[i + 1,] = error_train[i,]
            X.append(X[-1])
            Y.append(Y[-1])

    accept_ratio = naccept / (samples * 1.0) * 100

    print(accept_ratio, '% was accepted')

    burnin = 0.25 * samples  # use post burn in samples

    pos_w = pos_w[int(burnin):, ]
    pos_tau = pos_tau[int(burnin):, ]
    error_train = error_train[int(burnin):]
    return (pos_w, pos_tau, X, Y, error_train, accept_ratio)


def histogram_trace(pos_points, fname):
    size = 15
    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.grid(alpha=0.75)
    plt.hist(pos_points, bins=20, color='#0504aa', alpha=0.7)
    plt.title("Posterior distribution ", fontsize=size)
    plt.xlabel(' Parameter value  ', fontsize=size)
    plt.ylabel(' Frequency ', fontsize=size)
    plt.tight_layout()
    plt.savefig(fname + '_posterior.png')
    plt.clf()
    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.grid(alpha=0.75)
    plt.plot(pos_points)
    plt.title("Parameter trace plot", fontsize=size)
    plt.xlabel(' Number of Samples  ', fontsize=size)
    plt.ylabel(' Parameter value ', fontsize=size)
    plt.tight_layout()
    plt.savefig(fname + '_trace.png')
    plt.clf()


def main():
    X = [-15, -15, 0.1, 0.1]
    tau = [15, 15]
    rad = 2;
    o = [0, 1]
    displacement = ((((tau[0] - X[0]) ** 2) + ((tau[1] - X[1]) ** 2)) ** 0.5)
    [pos_w, pos_tau, Xcord, Ycord, error, accept_ratio] = sampler(displacement)

    error_mu = error.mean(axis=0)
    error_high = np.percentile(error, 95, axis=0)
    error_low = np.percentile(error, 5, axis=0)

    figure, ax = plt.subplots(figsize=(8, 6))
    h, = ax.plot(X[0], X[1], '-');
    tt = np.linspace(0, 2 * mt.pi, 1000);
    ht, = ax.plot(tau[0], tau[1], '.');
    plt.setp(ht, markersize=10);
    cos_tt = []
    for i in range(len(tt)):
        cos_tt.append((rad * mt.cos(tt[i])) + o[0])
        i += 1
    sin_tt = []
    for i in range(len(tt)):
        sin_tt.append((rad * mt.sin(tt[i])) + o[1])
        i += 1

    hobs, = ax.fill(cos_tt, sin_tt, '.')
    plt.setp(hobs, facecolor='r')

    p = 50
    pcen = np.percentile(error, p, interpolation='nearest')
    i_near = abs(error - pcen).argmin()
    X_med = Xcord[i_near]
    Y_med = Ycord[i_near]
    print(error[i_near])
    print("\n")
    p = 95
    pcen = np.percentile(error, p, interpolation='nearest')
    i_near = abs(error - pcen).argmin()
    X_high = Xcord[i_near]
    Y_high = Ycord[i_near]
    print(error[i_near])
    print("\n")
    p = 5
    pcen = np.percentile(error, p, interpolation='nearest')
    i_near = abs(error - pcen).argmin()
    X_low = Xcord[i_near]
    Y_low = Ycord[i_near]
    print(error[i_near])
    print("\n")
    plt.plot(X_med, Y_med, label='pred. (mean)')
    plt.plot(X_low, Y_low, label='pred.(5th percen.)')
    plt.plot(X_high, Y_high, label='pred.(95th percen.)')
    plt.legend(loc='upper right')

    plt.title("Path Uncertainty")
    plt.savefig('mcmctrain1000.png')
    plt.clf()
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)
    ax.boxplot(pos_w)
    ax.set_xlabel('beta delta')
    ax.set_ylabel('Posterior')
    plt.legend(loc='upper right')
    plt.title("Posterior")
    plt.savefig('w_pos1000.png')
    plt.clf()
    import os
    folder = 'posterior_1000'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(pos_w.shape[1]):
        histogram_trace(pos_w[:, i], folder + '/' + str(i))


if __name__ == "__main__": main()