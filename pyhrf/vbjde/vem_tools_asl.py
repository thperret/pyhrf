# -*- coding: utf-8 -*-
"""TOOLS and FUNCTIONS for VEM JDE
Used in different versions of VEM
"""

#import os.path as op
import numpy as np
from numpy.matlib import *
#import scipy as sp
#from scipy.linalg import toeplitz
#import time
#import UtilsC
#import pyhrf
#from pyhrf.boldsynth.hrf import getCanoHRF
#from pyhrf.paradigm import restarize_events
#from pyhrf.tools import format_duration
import vem_tools as vt
try:
    from collections import OrderedDict
except ImportError:
    from pyhrf.tools.backports import OrderedDict

# Tools
##############################################################

eps = 1e-6


def error(estimated, simulated):
    e = np.mean(np.mean((estimated - simulated) ** 2) / \
                    np.mean(simulated ** 2))
    return e


def mult(v1, v2):
    matrix = np.zeros((len(v1), len(v2)), dtype=float)
    for i in xrange(len(v1)):
        for j in xrange(len(v2)):
                matrix[i, j] += v1[i] * v2[j]
    return matrix


def normpdf(x, mu, sigma):
    u = (x - mu) / np.abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * np.abs(sigma))) * np.exp(-u * u / 2)
    return y


def covariance_matrix(order, D, dt):
    D2 = vt.buildFiniteDiffMatrix(order, D)
    R = np.dot(D2, D2) / pow(dt, 2 * order)
    return R


def create_conditions(Onsets, M, N, D, TR, dt):
    condition_names = []
    X = OrderedDict([])
    for condition, Ons in Onsets.iteritems():
        X[condition] = vt.compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = np.zeros((M, N, D), dtype=np.int32)
    nc = 0
    for condition, Ons in Onsets.iteritems():
        XX[nc, :, :] = X[condition]
        nc += 1
    return X, XX


def create_neighbours(graph, J):
    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = np.zeros((J, maxNeighbours), dtype=np.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i, :len(graph[i])] = graph[i]
    return maxNeighbours, neighboursIndexes

def _trap_area( p1, p2 ):
    """
    Calculate the area of the trapezoid defined by points
    p1 and p2

    `p1` - left side of the trapezoid
    `p2` - right side of the trapezoid
    """
    base = abs( p2[ 0 ] - p1[ 0 ] )
    avg_ht = ( p1[ 1 ] + p2[ 1 ] ) / 2.0

    return base * avg_ht

def roc_curve(dvals, labels, rocN=None, normalize=True):
    """
    Compute ROC curve coordinates and area

    - `dvals`  - a list with the decision values of the classifier
    - `labels` - list with class labels, \in {0, 1}

    returns (FP coordinates, TP coordinates, AUC )
    """
    if rocN is not None and rocN < 1:
        rocN = int(rocN * np.sum(np.not_equal(labels, 1)))

    TP = 0.0  # current number of true positives
    FP = 0.0  # current number of false positives

    fpc = [0.0]  # fp coordinates
    tpc = [0.0]  # tp coordinates
    dv_prev = - np.inf  # previous decision value
    TP_prev = 0.0
    FP_prev = 0.0
    area = 0.0

    #num_pos = labels.count(1)  # num pos labels np.sum(labels)
    #num_neg = labels.count(0)  # num neg labels np.prod(labels.shape)-num_pos

    #if num_pos == 0 or num_pos == len(labels):
    #    raise ValueError, "There must be at least one example from each class"

    # sort decision values from highest to lowest
    indices = np.argsort(dvals)[::-1]

    for idx in indices:
        # increment associated TP/FP count
        if labels[idx] == 1:
            TP += 1.
        else:
            FP += 1.
            if rocN is not None and FP == rocN:
                break
        # Average points with common decision values
        # by not adding a coordinate until all
        # have been processed
        if dvals[idx] != dv_prev:
            if len(fpc) > 0 and FP == fpc[-1]:
                tpc[-1] = TP
            else:
                fpc.append(FP)
                tpc.append(TP)
            dv_prev = dvals[idx]
            area += _trap_area((FP_prev, TP_prev), (FP, TP))
            FP_prev = FP
            TP_prev = TP

    #area += _trap_area( ( FP, TP ), ( FP_prev, TP_prev ) )
    #fpc.append( FP  )
    #tpc.append( TP )
    if normalize:
        fpc = [np.float64(x) / FP for x in fpc]
        if TP > 0:
            tpc = [np.float64(x) / TP for x in tpc]
        if area > 0:
            area /= (TP * FP)

    return fpc, tpc, area


# Expectation functions
##############################################################

def expectation_A(H, m_A, G, m_C, W, X, Gamma, q_Z, mu_Ma, sigma_Ma,
                  D, J, M, K, y_tilde, Sigma_A, sigma_epsilone):
    X_tilde = np.zeros((J, M, D), dtype=float)
    y_tildeH = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        for m, k1 in enumerate(X):
            # from X, we take k1=cond_name, m=index_cond
            y_tildeH[:, i] -= m_C[i, m] * np.dot(np.dot(W, X[k1]), G)
            for m2, k2 in enumerate(X):
                Sigma_A[m, m2, i] = np.dot(np.dot(np.dot(np.dot(H.T, \
                                                 X[k1].T), Gamma_i), X[k2]), H)
            X_tilde[i, m, :] = np.dot(np.dot(Gamma_i, y_tildeH[:, i]).T, X[k1])
        tmp = np.dot(X_tilde[i, :, :], H)
        for k in xrange(0, K):
            Delta = np.diag(q_Z[:, k, i] / (sigma_Ma[:, k] + eps))
            tmp += np.dot(Delta, mu_Ma[:, k])
            Sigma_A[:, :, i] += Delta
        tmp2 = np.linalg.inv(Sigma_A[:, :, i])
        Sigma_A[:, :, i] = tmp2
        m_A[i, :] = np.dot(Sigma_A[:, :, i], tmp)
    return m_A, Sigma_A


def expectation_C(G, m_C, H, m_A, W, X, Gamma, q_Z, mu_Mc, sigma_Mc,
                  D, J, M, K, y_tilde, Sigma_C, sigma_epsilone):
    X_tilde = np.zeros((J, M, D), dtype=float)
    y_tildeG = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        for m, k1 in enumerate(X):
            y_tildeG[:, i] -= m_A[i, m] * np.dot(X[k1], H)
            for m2, k2 in enumerate(X):
                Sigma_C[m, m2, i] = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(\
                                  G.T, X[k1].T), W.T), Gamma_i), W), X[k2]), G)
            X_tilde[i, m, :] = np.dot(np.dot(np.dot(Gamma_i, \
                                    y_tildeG[:, i]).T, W) ,X[k1])
        tmp = np.dot(X_tilde[i, :, :], G)
        for k in xrange(0, K):
            Delta = np.diag(q_Z[:, k, i] / (sigma_Mc[:, k] + eps))
            tmp += np.dot(Delta, mu_Mc[:, k])
            Sigma_C[:, :, i] += Delta
        tmp2 = np.linalg.inv(Sigma_C[:, :, i])
        Sigma_C[:, :, i] = tmp2
        m_C[i, :] = np.dot(Sigma_C[:, :, i], tmp)
    return m_C, Sigma_C


def expectation_H(Sigma_A, m_A, m_C, G, X, W, Gamma, D, J, N, y_tilde,
                  sigma_epsilone, scale, R, sigmaH):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_a = scale * R / sigmaH
    y_tildeH = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            tmp += m_A[i, m] * X[k]
            y_tildeH[:, i] -= m_C[i, m] * np.dot(np.dot(W, X[k]), G)
        Y_bar_tilde += np.dot(np.dot(tmp.T, Gamma_i), y_tildeH[:, i])
        S_a += np.dot(np.dot(tmp.T, Gamma_i), tmp)
        for m1, k1 in enumerate(X):                # Loop over the M conditions
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                S_a += Sigma_A[m1, m2, i] * np.dot(np.dot(X[k1].T,
                                                          Gamma_i), X[k2])
    Sigma_H = np.linalg.inv(S_a)
    m_H = np.dot(Sigma_H, Y_bar_tilde)
    return m_H, Sigma_H


def expectation_G(Sigma_C, m_C, m_A, H, X, W, Gamma, D, J, N, y_tilde,
                  sigma_epsilone, scale, R, sigmaG):
    Y_bar_tilde = np.zeros((D), dtype=float)
    S_c = scale * R / sigmaG
    y_tildeG = y_tilde.copy()
    for i in xrange(0, J):
        Gamma_i = Gamma / max(sigma_epsilone[i], eps)
        tmp = np.zeros((N, D), dtype=float)
        for m, k in enumerate(X):                  # Loop over the M conditions
            tmp += m_C[i, m] * np.dot(W, X[k])
            y_tildeG[:, i] -= m_A[i, m] * np.dot(X[k], H)
        Y_bar_tilde += np.dot(np.dot(tmp.T, Gamma_i), y_tildeG[:, i])
        S_c += np.dot(np.dot(tmp.T, Gamma_i), tmp)
        for m1, k1 in enumerate(X):                # Loop over the M conditions
            for m2, k2 in enumerate(X):            # Loop over the M conditions
                S_c += Sigma_C[m1, m2, i] * np.dot(np.dot(np.dot(np.dot( \
                                            X[k1].T, W.T), Gamma_i), W), X[k2])
    Sigma_G = np.linalg.inv(S_c)
    m_G = np.dot(Sigma_G, Y_bar_tilde)
    return m_G, Sigma_G


def constraint_norm1(Ftilde, Sigma_F, positivity=False):
    """ Constrain with optimization strategy """
    import cvxpy as cvx
    m, n = Sigma_F.shape
    Sigma_F_inv = np.linalg.inv(Sigma_F)
    zeros_F = np.zeros_like(Ftilde[:, np.newaxis])
    F = cvx.Variable(n)      # Construct the problem. PRIMAL
    expression = cvx.quad_form(F - Ftilde[:, np.newaxis], Sigma_F_inv)
    objective = cvx.Minimize(expression)
    if positivity:
        constraints = [F[0] == 0, F[-1] == 0, F >= zeros_F,
                       cvx.square(cvx.norm(F, 2)) <= 1]
    else:
        constraints = [F[0] == 0, F[-1] == 0, cvx.square(cvx.norm(F, 2)) <= 1]
    prob = cvx.Problem(objective, constraints)
    prob.solve(verbose=0, solver=cvx.CVXOPT)
    return np.squeeze(np.array((F.value)))


def constraint_norm1_b(Ftilde, Sigma_F, positivity=False, perfusion=None):
    """ Constrain with optimization strategy """
    from scipy.optimize import fmin_l_bfgs_b, fmin_slsqp
    Sigma_F_inv = np.linalg.inv(Sigma_F)
    zeros_F = np.zeros_like(Ftilde)
    #print 'm_H = ', Ftilde
    #print 'Sigma_H = ', Sigma_F_inv

    def fun(F):
        'function to minimize'
        return np.dot(np.dot((F - Ftilde).T, Sigma_F_inv), (F - Ftilde))

    def fung(F):
        'function to minimize'
        mean = np.dot(np.dot((F - Ftilde).T, Sigma_F_inv), (F - Ftilde)) * 0.5
        Sigma = np.dot(Sigma_F_inv, (F - Ftilde))
        return mean, Sigma

    def ec1(F):
        'Norm2(F)==1'
        return - 1 + np.linalg.norm(F, 2)

    if perfusion is not None:
        def ec2(F):
            'F>=perfusion'
            return F - [perfusion[0]] * (len(zeros_F))
        #print 'SLSQP method: '
        y = fmin_slsqp(fun, zeros_F, eqcons=[ec1],# ieqcons=[ec2],
                       bounds=[(None, None)] * (len(zeros_F)))
        #y = fmin_slsqp(fun, zeros_F, eqcons=[ec1], ieqcons=[ec2],
        #               bounds=[(None, None)] * (len(zeros_F)))
    else:
        #print 'SLSQP method: '
        y = fmin_slsqp(fun, zeros_F, eqcons=[ec1],
                       bounds=[(None, None)] * (len(zeros_F)))

    #print y
    if 0:
        print y
        print len(y)
        print fun(y)
        print "Feasibility residue: %g" % ec1(y)
        print "Random perturbations of optimal point"
        for _ in xrange(20):
            z = y + np.random.randn(*y.shape) * 1e-3
            print "fun(z) = %g" % fun(z)

        print 'L-BFGS-B method: '
        y2 = fmin_l_bfgs_b(fung, zeros_F, bounds=[(-1, 1)] * (len(zeros_F)))
        print y2[0]
        print len(y2)
        print fung(y2[0])
        print "Feasibility residue: %g" % ec1(y2[0])
        print "Random perturbations of optimal point"
        for _ in xrange(20):
            z = y2[0] + np.random.randn(*y2[0].shape) * 1e-3
            w = fung(z)
            print "fung(z) = %g" % w[0]
    return y
    #fun = lambda F: np.dot(np.dot((F - Ftilde).T, Sigma_F_inv),
    #                              (F - Ftilde))
    #cons = ({'type': 'ineq', 'fun': lambda F: - np.linalg.norm(F, 2) + 1})
    #res = fmin_slsqp(fun, zeros_F, constraints=cons)
    #res = minimize(fun, zeros_F, constraints=cons)
    #print res.message
    #return res.x

def expectation_Zb(Sigma_A, m_A, Sigma_C, m_C, sigma_Ma, mu_Ma, sigma_Mc, \
                  mu_Mc, Beta, Z_tilde, q_Z, graph, M, J, K):
    energy = np.zeros(K)
    Gauss = energy.copy()
    # Compute Z_tilde
    for i in xrange(0, J):
        for m in xrange(0, M):
            alpha = - 0.5 * Sigma_A[m, m, i] / (sigma_Ma[m, :] + eps) \
                    - 0.5 * Sigma_C[m, m, i] / (sigma_Mc[m, :] + eps)
            #alpha /= alpha.mean()
            local_energy = Beta[m] * sum(Z_tilde[m, graph[i]])
            for k in xrange(0, K):
                extern_field = alpha[k] \
                     + np.log(normpdf(m_A[i, m], mu_Ma[m, k], sigma_Ma[m, k]))\
                     + np.log(normpdf(m_C[i, m], mu_Mc[m, k], sigma_Mc[m, k]))
                # check if the sigma is sqrt or not!!
                energy[k] = extern_field + local_energy
            Probas = np.exp(energy - max(energy))
            Z_tilde[m, i] = Probas[0] / (sum(Probas))
    # Compute q_Z
    for i in xrange(0, J):
        for m in xrange(0, M):
            alpha = - 0.5 * Sigma_A[m, m, i] / (sigma_Ma[m, :] + eps) \
                    - 0.5 * Sigma_C[m, m, i] / (sigma_Mc[m, :] + eps)
            alpha /= alpha.mean()
            local_energy = Beta[m] * sum(Z_tilde[m, graph[i]], 0)
            for k in xrange(0, K):
                extern_field = alpha[k]
                energy[k] = extern_field + local_energy
                Gauss[k] = normpdf(m_A[i, m], mu_Ma[m, k], sigma_Ma[m, k]) \
                + normpdf(m_C[i, m], mu_Mc[m, k], sigma_Mc[m, k])
            Probas = np.exp(energy - max(energy))
            aux = (Gauss * Probas / sum(Probas))
            q_Z[m, i] = (aux / sum(aux))[0]
            q_Z[m, i] /= sum(q_Z[m, i])
    return q_Z, Z_tilde


def expectation_Z(Sigma_A, m_A, Sigma_C, m_C, sigma_Ma, mu_Ma, sigma_Mc, \
                  mu_Mc, Beta, Z_tilde, q_Z, graph, M, J, K):
    energy = np.zeros(K)
    Gauss = energy.copy()
    # Compute Z_tilde
    for i in xrange(0, J):
        for m in xrange(0, M):
            alpha = - 0.5 * Sigma_A[m, m, i] / (sigma_Ma[m, :] + eps) \
                    - 0.5 * Sigma_C[m, m, i] / (sigma_Mc[m, :] + eps)
            alpha /= alpha.mean()
            tmp = sum(Z_tilde[m, :, graph[i]], 0)
            for k in xrange(0, K):
                extern_field = alpha[k] \
                            + max(np.log(normpdf(m_A[i, m], mu_Ma[m, k],
                                         sigma_Ma[m, k]) + eps), -100)\
                            + max(np.log(normpdf(m_C[i, m], mu_Mc[m, k],
                                         sigma_Mc[m, k]) + eps), -100)
                # check if the sigma is sqrt or not!!
                local_energy = Beta[m] * tmp[k]
                energy[k] = extern_field + local_energy
            Probas = np.exp(energy - max(energy))
            Z_tilde[m, :, i] = Probas / (sum(Probas) + eps)
    # Compute q_Z
    for i in xrange(0, J):
        for m in xrange(0, M):
            alpha = - 0.5 * Sigma_A[m, m, i] / (sigma_Ma[m, :] + eps) \
                    - 0.5 * Sigma_C[m, m, i] / (sigma_Mc[m, :] + eps)
            alpha /= alpha.mean()
            tmp = sum(Z_tilde[m, :, graph[i]], 0)
            for k in xrange(0, K):
                extern_field = alpha[k]
                local_energy = Beta[m] * tmp[k]
                energy[k] = extern_field + local_energy
                Gauss[k] = normpdf(m_A[i, m], mu_Ma[m, k], \
                                    np.sqrt(sigma_Ma[m, k])) \
                            + normpdf(m_C[i, m], mu_Mc[m, k], \
                                    np.sqrt(sigma_Mc[m, k]))
            Probas = np.exp(energy - max(energy))
            q_Z[m, :, i] = Gauss * Probas / sum(Probas)
            q_Z[m, :, i] /= sum(q_Z[m, :, i])
    return q_Z, Z_tilde


# Maximization functions
##############################################################

def maximization_mu_sigma(Mu, Sigma, q_Z, m_X, K, M, Sigma_X):
    for m in xrange(0, M):
        for k in xrange(0, K):
            #S = sum( q_Z[m,k,:] ) + eps
            S = sum(q_Z[m, k, :])
            if S == 0.:
                S = eps
            #Sigma[m, k] = sum(q_Z[m, k, :] * (pow(m_X[:, m] - Mu[m, k], 2) +
            #                        Sigma_X[m, m, :])) / S
            Sigma[m, k] = sum(q_Z[m, k, :] * (pow(m_X[:, m] - Mu[m, k], 2))) / S
            if Sigma[m, k] < eps:
                Sigma[m, k] = eps
            if k != 0:          # mu_0 = 0 a priori
                # Mu[m,k] = eps + sum( q_Z[m,k,:] * m_A[:,m] ) / S
                Mu[m, k] = sum(q_Z[m, k, :] * m_X[:, m]) / S
            else:
                Mu[m, k] = 0.
    return Mu, Sigma


def maximization_L_alpha(Y, m_A, m_C, X, W, w, Ht, Gt, L, P, alpha):
    for i in xrange(0, Y.shape[1]):
        S = np.zeros((P.shape[0]), dtype=np.float64)  # zerosP.copy()
        S1 = S.copy()
        S2 = S.copy()
        for m, k in enumerate(X):
            S += m_A[i, m] * np.dot(X[k], Ht)
            S += m_C[i, m] * np.dot(np.dot(W, X[k]), Gt)
        S1 += S + np.dot(w, alpha[i])
        S2 += S + np.dot(P, L[:, i])
        L[:, i] = np.dot(P.T, Y[:, i] - S1)
        alpha[i] = np.dot(w.T, Y[:, i] - S2) / (np.dot(w.T, w))
    return L, alpha


def maximization_sigma_prior(D, R, m_X, gamma_x):
    R_inv = np.linalg.inv(R)
    alpha = (np.dot(mult(m_X, m_X), R_inv)).trace()
    #sigmaH = (D + sqrt(D * D + 8 * gamma_h * alpha)) / (4* gamma_h)
    sigmaX = (-D + sqrt(D * D + 8 * gamma_x * alpha)) / (4 * gamma_x)
    return sigmaX


def maximization_sigma_noise(Y, X, m_A, Sigma_A, Ht, m_C, Sigma_C, Gt, W, \
                             M, N, J, y_tilde, sigma_eps):
    hXXh = np.zeros((M, M), dtype=float)
    gXXg = hXXh.copy()
    gXXh = hXXh.copy()
    for i in xrange(0, J):
        S = np.zeros((N), dtype=float)
        for m, k in enumerate(X):
            for m2, k2 in enumerate(X):
                hXXh[m, m2] = np.dot(np.dot(np.dot(Ht.T, X[k].T), X[k2]), Ht)
                gXXg[m, m2] = np.dot(np.dot(np.dot(np.dot(np.dot( \
                                           Gt.T, X[k].T), W.T), W), X[k2]), Gt)
                gXXh[m, m2] = np.dot(np.dot(np.dot(np.dot( \
                                               Gt.T, X[k].T), W.T), X[k2]), Ht)
            S += m_A[i, m] * np.dot(X[k], Ht) + \
                 m_C[i, m] * np.dot(np.dot(W, X[k]), Gt)
        sigma_eps[i] = np.dot(np.dot(m_A[i, :].T, hXXh), m_A[i, :])
        sigma_eps[i] += (np.dot(Sigma_A[:, :, i], hXXh)).trace()
        sigma_eps[i] += np.dot(np.dot(m_C[i, :].T, gXXg), m_C[i, :])
        sigma_eps[i] += (np.dot(Sigma_C[:, :, i], gXXg)).trace()
        sigma_eps[i] += np.dot(y_tilde[:, i].T, y_tilde[:, i])
        sigma_eps[i] -= 2 * np.dot(np.dot(m_C[i, :].T, gXXh), m_A[i, :])
        sigma_eps[i] -= 2 * np.dot(S.T, y_tilde[:, i])
        sigma_eps[i] /= N
    return sigma_eps


def gradient(q_Z, Z_tilde, J, m, K, graph, beta, gamma):
    Gr = gamma
    print 'Gr initial = ', Gr
    for i in xrange(0, J):
        print '*** voxel ', i
        print 'neighbours = ', graph[i]
        print 'Ztilde = ', Z_tilde[m, :, graph[i]]
        tmp2 = beta * sum(Z_tilde[m, :, graph[i]], 0)
        print 'beta * sum_j\inN(i) = ', tmp2
        Emax = max(tmp2)
        print 'Emax = ', Emax
        #Sum = sum(np.exp(tmp2 - Emax))
        Sum = sum(np.exp(tmp2))
        print 'exp(beta * sum_k\inN(i) - Emax)', Sum
        for k in xrange(0, K):
            print 'class ', k
            tmp = sum(Z_tilde[m, k, graph[i]], 0)
            print 'sum_j\inN(i) class = ', tmp
            energy = beta * tmp
            print 'beta * sum_j\inN(i) class = ', energy
            #Pmf_ik = np.exp(energy - Emax)
            Pmf_ik = np.exp(energy) / (Sum + eps)
            print 'Pmf_i = ', Pmf_ik
            #Gr += tmp * (-q_Z[m, k, i] + Pmf_ik)
            Gr += (-q_Z[m, k, i] + Pmf_ik)
    print 'Gr = ', Gr
    return Gr


def maximization_beta(beta, q_Z, Z_tilde, J, K, m, graph, gamma, neighbour,
                      maxNeighbours):
    Gr = 100
    step = 0.005
    ni = 1
    while ((abs(Gr) > 0.0001) and (ni < 200)):
        Gr = gradient(q_Z, Z_tilde, J, m, K, graph, beta, gamma)
        beta -= step * Gr
        print 'beta[%d] = %f' % (ni, beta)
        ni += 1
    return max(beta, eps)


# Other functions
##############################################################

def computeFit(m_H, m_A, m_G, m_C, W, X, J, N):
    # print 'Computing Fit ...'
    stimIndSignal = np.zeros((N, J), dtype=np.float64)
    for i in xrange(0, J):
        m = 0
        for k in X:
            stimIndSignal[:, i] += m_A[i, m] * np.dot(X[k], m_H) \
                                 + m_C[i, m] * np.dot(np.dot(W, X[k]), m_G)
            m += 1
    return stimIndSignal
