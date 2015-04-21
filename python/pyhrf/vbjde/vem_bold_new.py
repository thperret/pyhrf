# -*- coding: utf-8 -*-

"""This module implements the VEM for BOLD data.

The function uses the C extension for expectation and maximization steps (see
src/pyhrf/vbjde/utilsmodule.c file).


See Also
--------
pyhrf.ui.analyser_ui, pyhrf.ui.treatment, pyhrf.ui.jde, pyhrf.ui.vb_jde_analyser

Notes
-----
TODO: add some refs?

Attributes
----------
eps : float
    mimics the mechine epsilon to avoid zero values
logger : logger
    logger instance identifying this module to log informations

"""

import os
import time
import logging

from collections import OrderedDict

import numpy as np

try:
    os.environ["DISPLAY"]
except KeyError:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib
import matplotlib.pyplot as plt

import pyhrf.vbjde.UtilsC as UtilsC
import pyhrf.vbjde.vem_tools as vt

from pyhrf.tools.aexpression import ArithmeticExpression as AExpr
from pyhrf.boldsynth.hrf import getCanoHRF


logger = logging.getLogger(__name__)
eps = 1e-4


def jde_vem_bold(graph, Y, Onsets, Thrf, K, TR, beta, dt,
                 scale=1, estimateSigmaH=True, sigmaH=0.05,
                 NitMax=-1, NitMin=1, estimateBeta=True,
                 PLOT=False, contrasts=None, computeContrast=False,
                 gamma_h=0, seed=6537546):
    """This is the main function that compute the VEM analysis on BOLD data.

    Parameters
    ----------
    graph : TODO
        TODO
    Y : TODO
        TODO
    Onsets : dict
        dictionnary of onsets
    Thrf : float
        hrf total time duration
    K : TODO
        TODO
    TR : float
        repetition time
    beta : TODO
        TODO
    dt : float
        hrf temporal precision
    scale : float, optional
        scale factor for datas ? TODO: check
    estimateSigmaH : bool, optional
        toggle estimation of sigma H
    sigmaH : float, optional
        initial or fixed value of sigma H
    NitMax : int, optional
        maximal computed iteration number
    NitMin : int, optional
        minimal computed iteration number
    estimateBeta : bool, optional
        toggle the estimation of Beta
    PLOT : bool, optional
        if True, plot some images of some variables (TODO: describe, or better,
        remove)
    contrasts : list, optional
        list of contrasts to compute
    computeContrast : bool, optional
        if True, compute the contrasts defined in contrasts
    gamma_h : float (TODO: check)
        TODO
    seed : int, optional
        seed used by numpy to initialize random generator number

    Returns
    -------
    tuple
        tuple of several variables (TODO: describe)
            - ni : TODO
            - m_A : TODO
            - m_H : TODO
            - q_Z : TODO
            - sigma_epsilone : TODO
            - mu_M : TODO
            - sigma_M : TODO
            - Beta : TODO
            - L : TODO
            - PL : TODO
            - CONTRAST : TODO
            - CONTRASTVAR : TODO
            - cA : TODO
            - cH : TODO
            - cZ : TODO
            - cAH : TODO
            - cTime : TODO
            - cTimeMean : TODO
            - Sigma_A : TODO
            - StimulusInducedSignal : TODO
    """

    logger.info("Fast EM with C extension started.")

    if not contrasts:
        contrasts = []

    np.random.seed(seed)

    if NitMax < 0:
        NitMax = 100
    gamma = 7.5
    gradientStep = 0.003
    MaxItGrad = 200
    Thresh = 1e-5

    # Initialize sizes vectors
    D = np.int(np.ceil(Thrf / dt)) + 1
    M = len(Onsets)
    N = Y.shape[0]
    J = Y.shape[1]
    condition_names = []

    maxNeighbours = max([len(nl) for nl in graph])
    neighboursIndexes = np.zeros((J, maxNeighbours), dtype=np.int32)
    neighboursIndexes -= 1
    for i in xrange(J):
        neighboursIndexes[i, :len(graph[i])] = graph[i]

    X = OrderedDict([])
    for condition, Ons in Onsets.iteritems():
        X[condition] = vt.compute_mat_X_2(N, TR, D, dt, Ons)
        condition_names += [condition]
    XX = np.zeros((M, N, D), dtype=np.int32)
    nc = 0
    for condition, Ons in Onsets.iteritems():
        XX[nc, :, :] = X[condition]
        nc += 1

    order = 2
    D2 = vt.buildFiniteDiffMatrix(order, D)
    R = np.dot(D2, D2) / pow(dt, 2 * order)
    invR = np.linalg.inv(R)
    Det_invR = np.linalg.det(invR)

    Gamma = np.identity(N)
    Det_Gamma = np.linalg.det(Gamma)

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_AH = 1
    AH = np.zeros((J, M, D), dtype=np.float64)
    AH1 = np.zeros((J, M, D), dtype=np.float64)
    Crit_FreeEnergy = 1
    cTime = []
    cA = []
    cH = []
    cZ = []
    cAH = []

    CONTRAST = np.zeros((J, len(contrasts)), dtype=np.float64)
    CONTRASTVAR = np.zeros((J, len(contrasts)), dtype=np.float64)
    Q_barnCond = np.zeros((M, M, D, D), dtype=np.float64)
    XGamma = np.zeros((M, D, N), dtype=np.float64)
    m1 = 0
    for k1 in X:  # Loop over the M conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1, m2, :, :] = np.dot(
                np.dot(X[k1].transpose(), Gamma), X[k2])
            m2 += 1
        XGamma[m1, :, :] = np.dot(X[k1].transpose(), Gamma)
        m1 += 1

    sigma_epsilone = np.ones(J)

    logger.info(
        "Labels are initialized by setting active probabilities to ones ...")
    q_Z = np.zeros((M, K, J), dtype=np.float64)
    q_Z[:, 1, :] = 1

    q_Z1 = np.zeros((M, K, J), dtype=np.float64)
    Z_tilde = q_Z.copy()

    TT, m_h = getCanoHRF(Thrf, dt)  # TODO: check
    m_h = m_h[:D]
    m_H = np.array(m_h).astype(np.float64)
    m_H1 = np.array(m_h)
    sigmaH1 = sigmaH
    Sigma_H = np.ones((D, D), dtype=np.float64)

    Beta = beta * np.ones((M), dtype=np.float64)
    P = vt.PolyMat(N, 4, TR)
    L = vt.polyFit(Y, TR, 4, P)
    PL = np.dot(P, L)
    y_tilde = Y - PL
    Ndrift = L.shape[0]

    sigma_M = np.ones((M, K), dtype=np.float64)
    sigma_M[:, 0] = 0.5
    sigma_M[:, 1] = 0.6
    mu_M = np.zeros((M, K), dtype=np.float64)
    for k in xrange(1, K):
        mu_M[:, k] = 1  # InitMean
    Sigma_A = np.zeros((M, M, J), np.float64)
    for j in xrange(0, J):
        Sigma_A[:, :, j] = 0.01 * np.identity(M)
    m_A = np.zeros((J, M), dtype=np.float64)
    m_A1 = np.zeros((J, M), dtype=np.float64)
    for j in xrange(0, J):
        for m in xrange(0, M):
            for k in xrange(0, K):
                m_A[j, m] += np.random.normal(
                    mu_M[m, k], np.sqrt(sigma_M[m, k])) * q_Z[m, k, j]
    m_A1 = m_A

    t1 = time.time()

    ni = 0
    while (ni < NitMin + 1) or (Crit_AH > Thresh and ni < NitMax):

        logger.info("{:-^80}".format(" Iteration n°"+str(ni+1)+" "))

        logger.info("Expectation A step...")
        UtilsC.expectation_A(q_Z, mu_M, sigma_M, PL, sigma_epsilone, Gamma,
                             Sigma_H, Y, y_tilde, m_A, m_H, Sigma_A,
                             XX.astype(np.int32), J, D, M, N, K)
        logger.info("Expectation H step...")
        UtilsC.expectation_H(XGamma, Q_barnCond, sigma_epsilone, Gamma, R,
                             Sigma_H, Y, y_tilde, m_A, m_H, Sigma_A,
                             XX.astype(np.int32), J, D, M, N, scale, sigmaH)
        m_H[0] = 0
        m_H[-1] = 0

        DIFF = np.reshape(m_A - m_A1, (M * J))
        Crit_A = (np.linalg.norm(DIFF) /
                  np.linalg.norm(np.reshape(m_A1, (M * J)))) ** 2
        cA += [Crit_A]
        m_A1[:, :] = m_A[:, :]

        Crit_H = (np.linalg.norm(m_H - m_H1) / np.linalg.norm(m_H1)) ** 2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0, D):
            AH[:, :, d] = m_A[:, :] * m_H[d]
        DIFF = np.reshape(AH - AH1, (M * J * D))
        Crit_AH = (np.linalg.norm(
            DIFF) / (np.linalg.norm(np.reshape(AH1, (M * J * D))) + eps)) ** 2
        logger.info("Convergence criteria: %f (Threshold = %f)",
                    Crit_AH, Thresh)
        cAH += [Crit_AH]
        AH1[:, :, :] = AH[:, :, :]

        logger.info("Expectation Z step...")
        UtilsC.expectation_Z(Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M, q_Z,
                             neighboursIndexes.astype(np.int32), M, J, K,
                             maxNeighbours)

        DIFF = np.reshape(q_Z - q_Z1, (M * K * J))
        Crit_Z = (np.linalg.norm(DIFF) /
                  (np.linalg.norm(np.reshape(q_Z1, (M * K * J))) + eps)) ** 2
        cZ += [Crit_Z]
        q_Z1[:, :, :] = q_Z[:, :, :]

        if estimateSigmaH:
            logger.info("Maximization sigma_H step...")
            if gamma_h > 0:
                sigmaH = vt.maximization_sigmaH_prior(
                    D, Sigma_H, R, m_H, gamma_h)
            else:
                sigmaH = vt.maximization_sigmaH(D, Sigma_H, R, m_H)
            logger.info('sigmaH = %s', str(sigmaH))

        logger.info("Maximization (mu,sigma) step...")
        mu_M, sigma_M = vt.maximization_mu_sigma(mu_M, sigma_M, q_Z, m_A, K,
                                                 M, Sigma_A)

        logger.info("Maximization L step...")
        UtilsC.maximization_L(Y, m_A, m_H, L, P, XX.astype(np.int32), J, D, M,
                              Ndrift, N)

        PL = np.dot(P, L)
        y_tilde = Y - PL
        if estimateBeta:
            logger.info("estimating beta")
            for m in xrange(0, M):
                Beta[m] = UtilsC.maximization_beta(
                    Beta[m], q_Z[m, :, :].astype(np.float64),
                    Z_tilde[m, :, :].astype(np.float64), J, K,
                    neighboursIndexes.astype(np.int32), gamma, maxNeighbours,
                    MaxItGrad, gradientStep)
            logger.info("End estimating beta")
            logger.info("Beta = %s", str(Beta))

        logger.info("Maximization sigma noise step...")
        UtilsC.maximization_sigma_noise(Gamma, PL, sigma_epsilone, Sigma_H, Y,
                                        m_A, m_H, Sigma_A, XX.astype(np.int32),
                                        J, D, M, N)

        ni += 1
        t02 = time.time()
        cTime += [t02 - t1]

    #logger.info("------------------------------ Iteration n° " +
                #str(ni + 2) + " ------------------------------")
    #UtilsC.expectation_A(q_Z, mu_M, sigma_M, PL, sigma_epsilone, Gamma, Sigma_H,
                         #Y, y_tilde, m_A, m_H, Sigma_A, XX.astype(np.int32), J,
                         #D, M, N, K)
    #UtilsC.expectation_H(XGamma, Q_barnCond, sigma_epsilone, Gamma, R, Sigma_H,
                         #Y, y_tilde, m_A, m_H, Sigma_A, XX.astype(np.int32), J,
                         #D, M, N, scale, sigmaH)
    #m_H[0] = 0
    #m_H[-1] = 0

    #DIFF = np.reshape(m_A - m_A1, (M * J))
    #Crit_A = (np.linalg.norm(DIFF) /
              #(np.linalg.norm(np.reshape(m_A1, (M * J))) + eps)) ** 2
    #cA += [Crit_A]
    #m_A1[:, :] = m_A[:, :]

    #Crit_H = (np.linalg.norm(m_H - m_H1) / (np.linalg.norm(m_H1) + eps)) ** 2
    #cH += [Crit_H]
    #m_H1[:] = m_H[:]
    #for d in xrange(0, D):
        #AH[:, :, d] = m_A[:, :] * m_H[d]
    #DIFF = np.reshape(AH - AH1, (M * J * D))
    #Crit_AH = (np.linalg.norm(DIFF) /
               #(np.linalg.norm(np.reshape(AH1, (M * J * D))) + eps)) ** 2
    #cAH += [Crit_AH]
    #AH1[:, :, :] = AH[:, :, :]

    #UtilsC.expectation_Z(Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M, q_Z,
                         #neighboursIndexes.astype(np.int32), M, J, K,
                         #maxNeighbours)

    #DIFF = np.reshape(q_Z - q_Z1, (M * K * J))
    #Crit_Z = (np.linalg.norm(DIFF) /
              #(np.linalg.norm(np.reshape(q_Z1, (M * K * J))) + eps)) ** 2
    #cZ += [Crit_Z]
    #q_Z1[:, :, :] = q_Z[:, :, :]

    #if estimateSigmaH:
        #logger.info("M sigma_H step ...")
        #if gamma_h > 0:
            #sigmaH = vt.maximization_sigmaH_prior(D, Sigma_H, R, m_H, gamma_h)
        #else:
            #sigmaH = vt.maximization_sigmaH(D, Sigma_H, R, m_H)
        #logger.info('sigmaH = %s', str(sigmaH))

    #mu_M, sigma_M = vt.maximization_mu_sigma(
        #mu_M, sigma_M, q_Z, m_A, K, M, Sigma_A)

    #UtilsC.maximization_L(
        #Y, m_A, m_H, L, P, XX.astype(np.int32), J, D, M, Ndrift, N)
    #PL = np.dot(P, L)
    #y_tilde = Y - PL
    #if estimateBeta:
        #logger.info("estimating beta")
        #for m in xrange(0, M):
            #Beta[m] = UtilsC.maximization_beta(
                #beta, q_Z[m, :, :].astype(np.float64),
                #Z_tilde[m, :, :].astype(np.float64), J, K,
                #neighboursIndexes.astype(np.int32), gamma, maxNeighbours,
                #MaxItGrad, gradientStep)
            #logger.info("End estimating beta")
        #logger.info(Beta)
    #UtilsC.maximization_sigma_noise(Gamma, PL, sigma_epsilone, Sigma_H, Y, m_A,
                                    #m_H, Sigma_A, XX.astype(np.int32), J, D,
                                    #M, N)

    #t02 = time.time()
    #cTime += [t02 - t1]
    #ni += 2
    #logger.info("Crit_AH before while loop: %f", Crit_AH)
    #if (Crit_AH > Thresh):
        #while ((Crit_AH > Thresh) and (ni < NitMax)):
            #logger.info("{:^80}".format("Iteration n°"+str(ni+1)))
            #UtilsC.expectation_A(q_Z, mu_M, sigma_M, PL, sigma_epsilone, Gamma,
                                 #Sigma_H, Y, y_tilde, m_A, m_H, Sigma_A,
                                 #XX.astype(np.int32), J, D, M, N, K)
            #UtilsC.expectation_H(XGamma, Q_barnCond, sigma_epsilone, Gamma, R,
                                 #Sigma_H, Y, y_tilde, m_A, m_H, Sigma_A,
                                 #XX.astype(np.int32), J, D, M, N, scale, sigmaH)
            #m_H[0] = 0
            #m_H[-1] = 0

            #DIFF = np.reshape(m_A - m_A1, (M * J))
            #Crit_A = (np.linalg.norm(DIFF) /
                      #np.linalg.norm(np.reshape(m_A1, (M * J)))) ** 2
            #m_A1[:, :] = m_A[:, :]
            #cA += [Crit_A]

            #Crit_H = (np.linalg.norm(m_H - m_H1) /
                      #(np.linalg.norm(m_H1) + eps)) ** 2
            #cH += [Crit_H]
            #m_H1[:] = m_H[:]
            #for d in xrange(0, D):
                #AH[:, :, d] = m_A[:, :] * m_H[d]
            #DIFF = np.reshape(AH - AH1, (M * J * D))
            #Crit_AH = (np.linalg.norm(DIFF) /
                       #(np.linalg.norm(np.reshape(AH1, (M * J * D))) + eps)) ** 2
            #logger.info("Crit_AH inside while loop: %f", Crit_AH)
            #cAH += [Crit_AH]
            #AH1[:, :, :] = AH[:, :, :]

            #UtilsC.expectation_Z(Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M,
                                 #q_Z, neighboursIndexes.astype(np.int32), M,
                                 #J, K, maxNeighbours)
            #DIFF = np.reshape(q_Z - q_Z1, (M * K * J))
            #Crit_Z = (np.linalg.norm(DIFF) /
                      #(np.linalg.norm(np.reshape(q_Z1, (M*K*J))) + eps)) ** 2
            #cZ += [Crit_Z]
            #q_Z1[:, :, :] = q_Z[:, :, :]

            #if estimateSigmaH:
                #logger.info("M sigma_H step ...")
                #if gamma_h > 0:
                    #sigmaH = vt.maximization_sigmaH_prior(
                        #D, Sigma_H, R, m_H, gamma_h)
                #else:
                    #sigmaH = vt.maximization_sigmaH(D, Sigma_H, R, m_H)
                #logger.info('sigmaH = %s', str(sigmaH))

            #mu_M, sigma_M = vt.maximization_mu_sigma(
                #mu_M, sigma_M, q_Z, m_A, K, M, Sigma_A)

            #UtilsC.maximization_L(
                #Y, m_A, m_H, L, P, XX.astype(np.int32), J, D, M, Ndrift, N)
            #PL = np.dot(P, L)
            #y_tilde = Y - PL
            #if estimateBeta:
                #logger.info("estimating beta")
                #for m in xrange(0, M):
                    #Beta[m] = UtilsC.maximization_beta(
                        #beta, q_Z[m, :, :].astype(np.float64),
                        #Z_tilde[m, :, :].astype(np.float64), J, K,
                        #neighboursIndexes.astype(np.int32), gamma,
                        #maxNeighbours, MaxItGrad, gradientStep)
                #logger.info("End estimating beta")
                #logger.info(Beta)
            #UtilsC.maximization_sigma_noise(
                #Gamma, PL, sigma_epsilone, Sigma_H, Y, m_A, m_H, Sigma_A,
                #XX.astype(np.int32), J, D, M, N)

            #ni += 1
            #t02 = time.time()
            #cTime += [t02 - t1]
    t2 = time.time()

    if PLOT:
        font = {'size': 15}
        matplotlib.rc('font', **font)
        #plt.savefig('./HRF_Iter_CompMod.png')
        #plt.hold(False)
        plt.figure(1)
        plt.plot(cAH, 'lightblue')
        plt.hold(True)
        # plt.plot(cFE[1:-1], 'm')
        #plt.hold(False)
        plt.legend(('CAH'))
        plt.grid(True)
        plt.savefig('./Crit_CompMod.png')
        # plt.figure(3)
        # plt.plot(FreeEnergyArray)
        # plt.grid(True)
        # plt.savefig('./FreeEnergy_CompMod.png')

        # plt.figure(4)
        # for m in xrange(M):
        #     plt.plot(SUM_q_Z_array[m])
        #     plt.hold(True)
        # plt.hold(False)
        # plt.savefig('./Sum_q_Z_Iter_CompMod.png')

        # plt.figure(5)
        # for m in xrange(M):
        #     plt.plot(mu1_array[m])
        #     plt.hold(True)
        # plt.hold(False)
        # plt.savefig('./mu1_Iter_CompMod.png')

        # plt.figure(6)
        # plt.plot(h_norm_array)
        # plt.savefig('./HRF_Norm_CompMod.png')

        #Data_save = xndarray(h_norm_array, ['Iteration'])
        #Data_save.save('./HRF_Norm_Comp.nii')

    CompTime = t2 - t1
    cTimeMean = CompTime / ni

    Norm = np.linalg.norm(m_H)
    m_H /= Norm
    Sigma_H /= Norm ** 2
    sigmaH /= Norm ** 2
    m_A *= Norm
    Sigma_A *= Norm ** 2
    mu_M *= Norm
    sigma_M *= Norm ** 2
    sigma_M = np.sqrt(np.sqrt(sigma_M))

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if computeContrast:
        if len(contrasts) > 0:
            logger.info('Compute contrasts ...')
            nrls_conds = dict([(str(cn), m_A[:, ic])
                               for ic, cn in enumerate(condition_names)])
            n = 0
            for cname in contrasts:
                #------------ contrasts ------------#
                contrast_expr = AExpr(contrasts[cname], **nrls_conds)
                contrast_expr.check()
                contrast = contrast_expr.evaluate()
                CONTRAST[:, n] = contrast
                #------------ contrasts ------------#

                #------------ variance -------------#
                ContrastCoef = np.zeros(M, dtype=float)
                ind_conds0 = {}
                for m in xrange(0, M):
                    ind_conds0[condition_names[m]] = 0.0
                for m in xrange(0, M):
                    ind_conds = ind_conds0.copy()
                    ind_conds[condition_names[m]] = 1.0
                    ContrastCoef[m] = eval(contrasts[cname], ind_conds)
                ActiveContrasts = (ContrastCoef != 0) * np.ones(M, dtype=float)
                # print ContrastCoef
                # print ActiveContrasts
                AC = ActiveContrasts * ContrastCoef
                for j in xrange(0, J):
                    S_tmp = Sigma_A[:, :, j]
                    CONTRASTVAR[j, n] = np.dot(np.dot(AC, S_tmp), AC)
                #------------ variance -------------#
                n += 1
                logger.info('Done contrasts computing.')
        #+++++++++++++++++++++++  calculate contrast maps and variance  +++++++++++++++++++++++#

    logger.info("Nb iterations to reach criterion: %d", ni)
    logger.info("Computational time = %s min %s s", str(
        np.int(CompTime // 60)), str(np.int(CompTime % 60)))
    logger.info('mu_M: %s', mu_M)
    logger.info('sigma_M: %s', sigma_M)
    logger.info("sigma_H = %s", str(sigmaH))
    logger.info("Beta = %s", str(Beta))

    StimulusInducedSignal = vt.computeFit(m_H, m_A, X, J, N)
    SNR = 20 * \
        np.log(
            np.linalg.norm(Y) / np.linalg.norm(Y - StimulusInducedSignal - PL))
    SNR /= np.log(10.)
    logger.info('SNR comp = %f', SNR)
    # ,FreeEnergyArray
    return (ni, m_A, m_H, q_Z, sigma_epsilone, mu_M, sigma_M, Beta, L, PL,
            CONTRAST, CONTRASTVAR, cA[2:], cH[2:], cZ[2:], cAH[2:], cTime[2:],
            cTimeMean, Sigma_A, StimulusInducedSignal)
