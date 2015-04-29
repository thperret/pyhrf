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


def jde_vem_bold(graph, bold_data, onsets, hrf_duration, nb_classes, tr, beta,
                 dt, scale=1, estimate_sigma_h=True, sigma_h=0.05, it_max=-1,
                 it_min=0, estimate_beta=True, plot=False, contrasts=None,
                 compute_contrasts=False, gamma_h=0, seed=6537546):
    """This is the main function that compute the VEM analysis on BOLD data.

    Parameters
    ----------
    graph : TODO
        TODO
    bold_data : ndarray, shape (nb_scans, nb_voxels)
        TODO
    onsets : dict
        dictionnary of onsets
    hrf_duration : float
        hrf total time duration
    nb_classes : TODO
        TODO
    tr : float
        time of repetition
    beta : TODO
        TODO
    dt : float
        hrf temporal precision
    scale : float, optional
        scale factor for datas ? TODO: check
    estimate_sigma_h : bool, optional
        toggle estimation of sigma H
    sigma_h : float, optional
        initial or fixed value of sigma H
    it_max : int, optional
        maximal computed iteration number
    it_min : int, optional
        minimal computed iteration number
    estimate_beta : bool, optional
        toggle the estimation of Beta
    plot : bool, optional
        if True, plot some images of some variables (TODO: describe, or better,
        remove)
    contrasts : list, optional
        list of contrasts to compute
    compute_contrasts : bool, optional
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
            - drift_coeffs : TODO
            - drift : TODO
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

    if it_max < 0:
        it_max = 100
    gamma = 7.5
    gradient_step = 0.003
    it_max_grad = 200
    thresh = 1e-5

    # Initialize sizes vectors
    hrf_len = np.int(np.ceil(hrf_duration / dt)) + 1
    nb_onsets = len(onsets)
    nb_scans = bold_data.shape[0]
    nb_voxels = bold_data.shape[1]
    condition_names = []

    max_neighbours = max([len(nl) for nl in graph])
    neighbours_indexes = np.zeros((nb_voxels, max_neighbours), dtype=np.int32)
    neighbours_indexes -= 1
    for i in xrange(nb_voxels):
        neighbours_indexes[i, :len(graph[i])] = graph[i]

    X = OrderedDict([])
    for condition, onset in onsets.iteritems():
        X[condition] = vt.compute_mat_X_2(nb_scans, tr, hrf_len, dt, onset)
        condition_names += [condition]
    XX = np.zeros((nb_onsets, nb_scans, hrf_len), dtype=np.int32)
    for nc, condition in enumerate(onsets.iterkeys()):
        XX[nc, :, :] = X[condition]

    order = 2
    D2 = vt.buildFiniteDiffMatrix(order, hrf_len)
    R = np.dot(D2, D2) / pow(dt, 2 * order)
    invR = np.linalg.inv(R)
    Det_invR = np.linalg.det(invR)

    Gamma = np.identity(nb_scans)
    Det_Gamma = np.linalg.det(Gamma)

    Crit_H = 1
    Crit_Z = 1
    Crit_A = 1
    Crit_AH = 1
    AH = np.zeros((nb_voxels, nb_onsets, hrf_len), dtype=np.float64)
    AH1 = np.zeros((nb_voxels, nb_onsets, hrf_len), dtype=np.float64)
    Crit_FreeEnergy = 1
    cTime = []
    cA = []
    cH = []
    cZ = []
    cAH = []

    CONTRAST = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    CONTRASTVAR = np.zeros((nb_voxels, len(contrasts)), dtype=np.float64)
    Q_barnCond = np.zeros((nb_onsets, nb_onsets, hrf_len, hrf_len), dtype=np.float64)
    XGamma = np.zeros((nb_onsets, hrf_len, nb_scans), dtype=np.float64)
    m1 = 0
    for k1 in X:  # Loop over the nb_onsets conditions
        m2 = 0
        for k2 in X:
            Q_barnCond[m1, m2, :, :] = np.dot(
                np.dot(X[k1].transpose(), Gamma), X[k2])
            m2 += 1
        XGamma[m1, :, :] = np.dot(X[k1].transpose(), Gamma)
        m1 += 1

    sigma_epsilone = np.ones(nb_voxels)

    logger.info(
        "Labels are initialized by setting active probabilities to ones ...")
    q_Z = np.zeros((nb_onsets, nb_classes, nb_voxels), dtype=np.float64)
    q_Z[:, 1, :] = 1

    q_Z1 = np.zeros((nb_onsets, nb_classes, nb_voxels), dtype=np.float64)
    Z_tilde = q_Z.copy()

    _, m_h = getCanoHRF(hrf_duration, dt)  # TODO: check
    m_h = m_h[:hrf_len]
    m_H = np.array(m_h).astype(np.float64)
    m_H1 = np.array(m_h)
    sigmaH1 = sigma_h
    Sigma_H = np.ones((hrf_len, hrf_len), dtype=np.float64)

    Beta = beta * np.ones((nb_onsets), dtype=np.float64)
    drift_basis = vt.PolyMat(nb_scans, 4, tr)
    drift_coeffs = vt.polyFit(bold_data, tr, 4, drift_basis)
    drift = np.dot(drift_basis, drift_coeffs)
    y_tilde = bold_data - drift
    Ndrift = drift_coeffs.shape[0]

    sigma_M = np.ones((nb_onsets, nb_classes), dtype=np.float64)
    sigma_M[:, 0] = 0.5
    sigma_M[:, 1] = 0.6
    mu_M = np.zeros((nb_onsets, nb_classes), dtype=np.float64)
    for k in xrange(1, nb_classes):
        mu_M[:, k] = 1  # InitMean
    Sigma_A = np.zeros((nb_onsets, nb_onsets, nb_voxels), np.float64)
    for j in xrange(0, nb_voxels):
        Sigma_A[:, :, j] = 0.01 * np.identity(nb_onsets)
    m_A = np.zeros((nb_voxels, nb_onsets), dtype=np.float64)
    m_A1 = np.zeros((nb_voxels, nb_onsets), dtype=np.float64)
    for j in xrange(0, nb_voxels):
        for m in xrange(0, nb_onsets):
            for k in xrange(0, nb_classes):
                m_A[j, m] += np.random.normal(
                    mu_M[m, k], np.sqrt(sigma_M[m, k])) * q_Z[m, k, j]
    m_A1 = m_A

    t1 = time.time()

    ni = 0
    while (ni < it_min + 1) or (Crit_AH > thresh and ni < it_max):

        logger.info("{:-^80}".format(" Iteration n°"+str(ni+1)+" "))

        logger.info("Expectation A step...")
        logger.debug("Before: m_A = %s, Sigma_A = %s", m_A, Sigma_A)
        UtilsC.expectation_A(q_Z, mu_M, sigma_M, drift, sigma_epsilone, Gamma,
                             Sigma_H, bold_data, y_tilde, m_A, m_H, Sigma_A,
                             XX.astype(np.int32), nb_voxels, hrf_len, nb_onsets, nb_scans, nb_classes)
        logger.debug("After: m_A = %s, Sigma_A = %s", m_A, Sigma_A)
        logger.info("Expectation H step...")
        logger.debug("Before: m_H = %s, Sigma_H = %s", m_H, Sigma_H)
        UtilsC.expectation_H(XGamma, Q_barnCond, sigma_epsilone, Gamma, R,
                             Sigma_H, bold_data, y_tilde, m_A, m_H, Sigma_A,
                             XX.astype(np.int32), nb_voxels, hrf_len, nb_onsets, nb_scans, scale, sigma_h)
        logger.debug("Before: m_H = %s, Sigma_H = %s", m_H, Sigma_H)
        m_H[0] = 0
        m_H[-1] = 0

        DIFF = np.reshape(m_A - m_A1, (nb_onsets * nb_voxels))
        Crit_A = (np.linalg.norm(DIFF) /
                  np.linalg.norm(np.reshape(m_A1, (nb_onsets * nb_voxels)))) ** 2
        cA += [Crit_A]
        m_A1[:, :] = m_A[:, :]

        Crit_H = (np.linalg.norm(m_H - m_H1) / np.linalg.norm(m_H1)) ** 2
        cH += [Crit_H]
        m_H1[:] = m_H[:]

        for d in xrange(0, hrf_len):
            AH[:, :, d] = m_A[:, :] * m_H[d]
        DIFF = np.reshape(AH - AH1, (nb_onsets * nb_voxels * hrf_len))
        Crit_AH = (np.linalg.norm(
            DIFF) / (np.linalg.norm(np.reshape(AH1, (nb_onsets * nb_voxels * hrf_len))) + eps)) ** 2
        logger.info("Convergence criteria: %f (Threshold = %f)",
                    Crit_AH, thresh)
        cAH += [Crit_AH]
        AH1[:, :, :] = AH[:, :, :]

        logger.info("Expectation Z step...")
        logger.debug("Before: Z_tilde = %s, q_Z = %s", Z_tilde, q_Z)
        UtilsC.expectation_Z(Sigma_A, m_A, sigma_M, Beta, Z_tilde, mu_M, q_Z,
                             neighbours_indexes.astype(np.int32), nb_onsets, nb_voxels, nb_classes,
                             max_neighbours)
        logger.debug("After: Z_tilde = %s, q_Z = %s", Z_tilde, q_Z)

        DIFF = np.reshape(q_Z - q_Z1, (nb_onsets * nb_classes * nb_voxels))
        Crit_Z = (np.linalg.norm(DIFF) /
                  (np.linalg.norm(np.reshape(q_Z1, (nb_onsets * nb_classes * nb_voxels))) + eps)) ** 2
        cZ += [Crit_Z]
        q_Z1[:, :, :] = q_Z[:, :, :]

        if estimate_sigma_h:
            logger.info("Maximization sigma_H step...")
            logger.debug("Before: Sigma_H = %s", Sigma_H)
            if gamma_h > 0:
                sigma_h = vt.maximization_sigmaH_prior(
                    hrf_len, Sigma_H, R, m_H, gamma_h)
            else:
                sigma_h = vt.maximization_sigmaH(hrf_len, Sigma_H, R, m_H)
            logger.debug("After: Sigma_H = %s", Sigma_H)

        logger.info("Maximization (mu,sigma) step...")
        logger.debug("Before: mu_M = %s, sigma_M = %s", mu_M, sigma_M)
        mu_M, sigma_M = vt.maximization_mu_sigma(mu_M, sigma_M, q_Z, m_A, nb_classes,
                                                 nb_onsets, Sigma_A)
        logger.debug("After: mu_M = %s, sigma_M = %s", mu_M, sigma_M)

        logger.info("Maximization L step...")
        logger.debug("Before: drift_coeffs = %s", drift_coeffs)
        UtilsC.maximization_L(bold_data, m_A, m_H, drift_coeffs, drift_basis, XX.astype(np.int32), nb_voxels, hrf_len, nb_onsets,
                              Ndrift, nb_scans)
        logger.debug("After: drift_coeffs = %s", drift_coeffs)

        drift = np.dot(drift_basis, drift_coeffs)
        y_tilde = bold_data - drift
        if estimate_beta:
            logger.info("Maximization beta step...")
            for m in xrange(0, nb_onsets):
                Beta[m] = UtilsC.maximization_beta(
                    Beta[m], q_Z[m, :, :].astype(np.float64),
                    Z_tilde[m, :, :].astype(np.float64), nb_voxels, nb_classes,
                    neighbours_indexes.astype(np.int32), gamma, max_neighbours,
                    it_max_grad, gradient_step)
            logger.debug("Beta = %s", str(Beta))

        logger.info("Maximization sigma noise step...")
        UtilsC.maximization_sigma_noise(Gamma, drift, sigma_epsilone, Sigma_H, bold_data,
                                        m_A, m_H, Sigma_A, XX.astype(np.int32),
                                        nb_voxels, hrf_len, nb_onsets, nb_scans)

        ni += 1
        t02 = time.time()
        cTime += [t02 - t1]

    t2 = time.time()

    if plot:
        font = {'size': 15}
        matplotlib.rc('font', **font)
        plt.figure(1)
        plt.plot(cAH[1:], 'lightblue')
        plt.hold(True)
        plt.legend(('CAH'))
        plt.grid(True)
        plt.savefig('./Crit_CompMod.png')

    CompTime = t2 - t1
    cTimeMean = CompTime / ni

    Norm = np.linalg.norm(m_H)
    m_H /= Norm
    Sigma_H /= Norm ** 2
    sigma_h /= Norm ** 2
    m_A *= Norm
    Sigma_A *= Norm ** 2
    mu_M *= Norm
    sigma_M *= Norm ** 2
    sigma_M = np.sqrt(np.sqrt(sigma_M))

    #+++++++++++++++++++++++  calculate contrast maps and variance +++++++++++++++++++++++#
    if compute_contrasts:
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
                ContrastCoef = np.zeros(nb_onsets, dtype=float)
                ind_conds0 = {}
                for m in xrange(0, nb_onsets):
                    ind_conds0[condition_names[m]] = 0.0
                for m in xrange(0, nb_onsets):
                    ind_conds = ind_conds0.copy()
                    ind_conds[condition_names[m]] = 1.0
                    ContrastCoef[m] = eval(contrasts[cname], ind_conds)
                ActiveContrasts = (ContrastCoef != 0) * np.ones(nb_onsets, dtype=float)
                # print ContrastCoef
                # print ActiveContrasts
                AC = ActiveContrasts * ContrastCoef
                for j in xrange(0, nb_voxels):
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
    logger.info("sigma_H = %s", str(sigma_h))
    logger.info("Beta = %s", str(Beta))

    StimulusInducedSignal = vt.computeFit(m_H, m_A, X, nb_voxels, nb_scans)
    SNR = 20 * np.log(
        np.linalg.norm(bold_data) / np.linalg.norm(bold_data - StimulusInducedSignal - drift))
    SNR /= np.log(10.)
    logger.info('SNR comp = %f', SNR)
    # ,FreeEnergyArray
    return (ni, m_A, m_H, q_Z, sigma_epsilone, mu_M, sigma_M, Beta, drift_coeffs, drift,
            CONTRAST, CONTRASTVAR, cA[2:], cH[2:], cZ[2:], cAH[2:], cTime[2:],
            cTimeMean, Sigma_A, StimulusInducedSignal)
