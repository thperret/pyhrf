# -*- coding: utf-8 -*-

import logging

from time import time
from collections import defaultdict

import numpy
import numpy as np
import scipy.stats

from numpy import (array, int32, float32, fix, sqrt, eye, dot, linalg, kron,
                   newaxis, log, array2string)
from numpy import zeros

import pyhrf

from pyhrf.tools import array_summary
from pyhrf import xmlio
from pyhrf.tools import format_duration
from pyhrf.ndarray import xndarray, stack_cuboids
from pyhrf.boldsynth.hrf import getCanoHRF


logger = logging.getLogger(__name__)


def init_dict():
    return defaultdict(list)


class RFIREstim(xmlio.XmlInitable):
    """
    Class handling the estimation of HRFs from fMRI data.
    Analysis is voxel-wise and can
    be multissession (heteroscedastic noise and session dependent drift).
    Simultaneous analysis of several conditions is treated.
    One HRF is considered at each voxel.
    """

    parametersComments = {
        'hrf_nb_coeffs': 'Number of values in the discrete HRF. '
        'Discretization is homogeneous HRF time length is then: '
        ' nb_hrf_coeffs * hrf_dt ',
        'hrf_dt': 'Required HRF temporal resolution',
        'drift_type': 'Basis type in the drift model. Either "cosine" or "poly"',
    }

    if pyhrf.__usemode__ == pyhrf.ENDUSER:
        default_stop_crit1 = 0.0001
        default_stop_crit2 = 0.00001
        default_nb_its = 500
        parametersToShow = ['hrf_nb_coeffs', 'hrf_dt', 'drift_type',
                            'nb_iterations']
    else:
        default_stop_crit1 = 0.005
        default_stop_crit2 = 0.0005
        default_nb_its = None

    def __init__(self, hrf_nb_coeffs=42, hrf_dt=0.6, drift_type='cosine',
                 stop_crit1=default_stop_crit1, stop_crit2=default_stop_crit2,
                 nb_its_max=5, nb_iterations=default_nb_its, nb_its_min=1,
                 average_bold=False, taum=0.01, lambda_reg=100., fixed_taum=False,
                 discarded_scan_indexes=None, output_fit=False):
        """
           'discarded_scan_indexes' : None if no subsampling done, else give position that were removed after temporal subsampling as a 2d numpy array

        """

        xmlio.XmlInitable.__init__(self)

        self.K = hrf_nb_coeffs
        self.OrthoBtype = drift_type
        self.DeltaT = hrf_dt
        self.nbItMax = nb_its_max
        self.nbItMin = nb_its_min
        self.nbIt = nb_iterations
        self.emStop1 = stop_crit1
        self.emStop2 = stop_crit2

        self.save_history = False

        self.compute_pct_change = False

        self.avg_bold = False

        self.fixed_taum = fixed_taum
        self.taum_init = taum
        self.lambda_reg = lambda_reg

        self.pos_removed = discarded_scan_indexes or np.array(([0]))
        self.output_fit = output_fit

    def linkToData(self, data):

        self.M = data.nbConditions
        dp = data.paradigm
        self.condition_names = dp.stimOnsets.keys()
        self.onsets = dp.stimOnsets.values()
        self.LengthOnsets = [dp.stimDurations[n] for n in dp.stimOnsets.keys()]

        self.nbVoxels = data.bold.shape[1]
        if not self.avg_bold:
            self.bold = data.bold
        else:
            self.bold = data.bold.mean(1)[:, np.newaxis]
            logger.info('BOLD is averaged -> shape: %s', str(self.bold.shape))

        self.sscans = data.sessionsScans
        self.nbSessions = data.nbSessions
        self.I = self.nbSessions
        self.ImagesNb = sum([len(ss) for ss in self.sscans])
        self.TR = data.tr

        self.Ni = array([len(ss) for ss in self.sscans])
        self.OnsetList = [[o[i] for o in self.onsets]
                          for i in xrange(self.nbSessions)]
        self.Qi = zeros(self.nbSessions, dtype=int) + 2

        self.history = defaultdict(init_dict)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        """
        function to launch the analysis
        """
        logger.info('Starting voxel-wise HRF estimation ...')
        logger.info('nvox=%d, ncond=%d, nscans=%d', self.nbVoxels, self.M,
                    self.ImagesNb)
        # initialization of the matrices that will store all voxel resuls
        logger.info("Init storage ...")
        self.InitStorageMat()
        logger.info("Compute onset matrix ...")
        self.Compute_onset_matrix3()

        self.stop_iterations = np.zeros(self.bold.shape[1], dtype=int)

        # voxelwise analysis. This loop handles the currently analyzed voxels.
        for POI in xrange(self.bold.shape[1]):  # POI = point of interest
            t0 = time()
            logger.info("Point %s / %s", str(POI), str(self.nbVoxels))
            self.ReadPointOfInterestData(POI)

            # initialize with zeros or ones all matrix and vector used in
            # the class
            logger.info("Init matrix and vectors ...")
            self.InitMatrixAndVectors(POI)

            # compute onset matrix

            # compute low frequency basis
            logger.info('build low freq mat ...')
            self.buildLowFreqMat()

            # precompute usefull data
            logger.info('Compute inv R ...')
            self.Compute_INV_R_and_R_and_DET_R()

            # solve find response functions and hyperparameters
            logger.info('EM solver ...')
            self.EM_solver(POI)

            # store current results in matrices initialized in 'InitStoringMat'
            logger.info('Store res ...')
            self.StoreRes(POI)
            logger.info("Done in %s", format_duration(time() - t0))

        self.clean_memory()

        logger.info('Nb of iterations to reach stop crit: %s',
                    array_summary(self.stop_iterations))

    def clean_memory(self):
        """ Clean all objects that are useless for outputs
        """
        pass
        # del self.Sigma
        # del self.X
        # del self.R
        # del self.InvSigma
        # del self.InvR
        # del self.h

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def InitStorageMat(self):
        """
        initialization of the matrices that will store all voxel resuls
        requires:
            input signals must have been read (in ReadRealSignal)
        """
        npos = self.bold.shape[1]
        self.Pvalues = -0.1 * numpy.ones((self.M, npos), dtype=float)
        self.ResponseFunctionsEvaluated = numpy.zeros(
            (self.M, self.K + 1, npos), dtype=np.float32)
        if self.compute_pct_change:
            self.ResponseFunctionsEvaluated_PctChgt = numpy.zeros(
                (self.M, self.K + 1, npos), dtype=np.float32)
        # sqrt of the diagonal of the covariance matrix
        self.StdValEvaluated = numpy.zeros(
            (self.M, self.K + 1, npos), dtype=np.float32)
        if self.compute_pct_change:
            # sqrt of the diagonal of the covariance matrix
            self.StdValEvaluated_PctChgt = numpy.zeros(
                (self.M, self.K + 1, npos), dtype=np.float32)
        self.SignalEvaluated = numpy.zeros(
            (self.ImagesNb, npos), dtype=np.float32)

        # orthonormal basis ('P') initialization: self.P[i][n,q] -> n^th value
        # of the q^th function of the orthonormal basis for session i
        self.P = range(self.I)
        for i in xrange(self.I):
            self.P[i] = numpy.zeros((self.Ni[i], self.Qi[i]), dtype=np.float32)

        # drifts coefs ('l') initialization: self.l[i][q] -> q^th coefficient for the the orthormal basis for session i
        # self.l = range(self.I)
        # for i in xrange(self.I):
        #     self.l[i] = numpy.zeros((self.Qi[i]),dtype=float)

        self.l = numpy.zeros((self.I, self.Qi[i], npos), dtype=np.float32)

        # inverse of R
        self.InvR = numpy.zeros((self.K - 1, self.K - 1), dtype=np.float32)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ReadPointOfInterestData(self, POI):
        """
        Initialize the parameters for a voxel analysis. The voxel ID is 'POI' in 'ConsideredCoord'
        initialized in 'ReadRealSignal'
        requires:
            input signals must have been read (in ReadRealSignal)
        """
        # initialization

        #self.Qi = []
        self.y = []

        # analysis paradigm parameters linked to the sessions...
        # the point of interest POI is considered on all sessions
        for i in xrange(self.nbSessions):
            #self.OnsetList.append([o[i] for o in self.onsets])
            self.y.append(self.bold[self.sscans[i], POI])
            # self.Qi.append((int)(2))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def InitMatrixAndVectors(self, POI):
        """
        initialize to zeros: X, y, P, l, h, InvR, Sigma
        initialize to ones: TauM, rb (<-scalar)
        requires:
            I / Ni / K /  M / Qi
        """

        logger.info('InitMatrixAndVectors ...')

        # HRFs (h) initialization: self.h[m][k] -> k^{th} HRF estimation value
        # for the condition m  (oversampled time)
        # Warning: since the first and the last values of the HRFs = 0,
        # the nb of HRFs coefs allocated is K-1
        self.h = numpy.zeros((self.M, self.K - 1), dtype=np.float32)
        hcano = getCanoHRF(dt=self.DeltaT,
                           duration=self.K * self.DeltaT)[1]
        self.h[:] = hcano[1:self.K]

        logger.info('self.h:')
        logger.info(self.h)

        for i in xrange(self.I):
            self.P[i][:, :] = 0.

        # drifts coefs ('l') initialization: self.l[i][q] -> q^th coefficient for the the orthormal basis for session i
        # for i in xrange(self.I):
        #    self.l[i,:] = 0.
        self.l[:, :, POI] = 0.

        # inverse of R
        self.InvR[:, :] = 0.

        # Sigma -> block matix => self.Sigma[m*SBS:(m+1)*SBS,n*SBS:(n+1)*SBS]]
        # -> (m,n)^th block of Sigma (variance a posteriori)
        SBS = self.K - 1  # Sigma Block Size
        self.Sigma = numpy.zeros(
            (self.M * SBS, self.M * SBS), dtype=np.float32)
        self.InvSigma = numpy.zeros(
            (self.M * SBS, self.M * SBS), dtype=np.float32)

        # TauM initialization: TauM[m] -> specific dynamics for condition m
        self.TauM = numpy.zeros(self.M, dtype=np.float32) + self.taum_init
        self.OldTauM = numpy.ones(self.M, dtype=np.float32) / 100.

        # rb initialization: self.rb[i] -> noise variance for i^th session
        self.rb = numpy.ones(self.I, dtype=np.float32)
        self.Old_rb = numpy.ones(self.I, dtype=np.float32)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def Compute_INV_R_and_R_and_DET_R(self):
        """
        both computes self.InvR and self.DetR
        requires:
            * K-1
            * InvR initialized
        """

        # for compactness
        SBS = self.K - 1

        # computes the 2nd order derivative matrix in the finite differences
        # senses
        D2 = -2 * eye(SBS).astype(np.float32)

        for i in xrange(SBS):
            if i > 0:
                D2[i, i - 1] = 1
            if i < SBS - 1:
                D2[i, i + 1] = 1

        # removes the conditions h[0]=0. and h[-1]=0.
        # D2[0,0]=1.  # !!!
        # D2[0,1]=-2.  # !!!
        # D2[0,2]=1.  # !!!
        # D2[-1,-1]=1.  # !!!
        # D2[-1,-2]=-2.  # !!!
        # D2[-1,-3]=1.  # !!!

        # removes the a priori
        # D2=eye(SBS)  # !!!

        # Computes InvR
        self.InvR = dot(D2.T, D2)

        # Compute R
        self.R = linalg.inv(self.InvR)

        # computes det(InvR)
        det_InvR = linalg.det(self.InvR)

        # computes det(R)
        self.DetR = 1. / det_InvR

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def Compute_onset_matrix3(self):
        """
        computes the onset matrix. Each stimulus onset is considered over
        a period of LengthOnsets seconds if (LengthOnsets>DetlaT) and a
        time step otherwise.
        requires:
            * X initialized
            * OnsetList
            * TR
            * DeltaT
            * K
            * LengthOnsets

        where 'self.X[i][m,n,k]' is such that:
            * session i (\in 0:I-1)
            * condition m (\in 0:M-1)
            * data nb n (\in 0:Ni[i]-1)
            * hrf coef nb k (\in 0:K-2)
        """
        # Onset matrix ('X') initialization: self.X[i][m,n,k] -> binary value of the onset matrix for session i / condition m / data nb n (real time) / hrf coef nb k (oversampled time)
        # Warning: since the first and the last values of the HRFs = 0, the nb
        # of HRFs coefs allocated is K-1
        self.X = []

        # if subsampling done, change Ni to compute design matrix X (gives the
        # real nb of scans in normal case)
        self.pos_removed = np.array(self.pos_removed)
        # print 'pos removed:', self.pos_removed, type(self.pos_removed)
        # print 'Ni:', self.Ni
        if numpy.size(self.pos_removed) > 1:
            nb_removed = numpy.size(self.pos_removed)
            print 'nb of removed pos:', nb_removed
            # print self.sscans
            self.Ni = array([len(ss) + nb_removed for ss in self.sscans])
            # print self.Ni

        for i in xrange(self.I):
            self.X.append(
                numpy.zeros((self.M, self.Ni[i], self.K - 1), dtype=np.float32))

        # print np.array(self.X).shape, self.Ni[0]
        # tests consistency between 'OnsetList' and the size of 'X'
        I = len(self.X)
        M = self.X[0].shape[0]
        K = self.X[0].shape[2] + 1
        Ni = numpy.zeros(I).astype(int32)
        for i in xrange(I):
            Ni[i] = self.X[i].shape[1]

        if K != self.K or I != self.I or M != self.M or sum(abs(Ni - self.Ni)) > 0.01:
            print 'OnsetList is not consitent with X'

        # computes the onset matrix X
        # BLOCK DESIGNED STIMULI
        if (self.LengthOnsets[0][0][0] > self.DeltaT):
            for i in xrange(I):
                for m in xrange(self.X[i].shape[0]):
                    for j in xrange(self.OnsetList[i][m].shape[0]):
                        # j^th onset of [session i / parameter m] is considered
                        CurrentOnset = self.OnsetList[i][m][j]
                        for n in xrange(Ni[i]):
                            for k in xrange(K - 1):
                                if CurrentOnset <= (n * self.TR - (k + 1.0) * self.DeltaT + 0.001) and CurrentOnset > ((n - 1.) * self.TR - (k + 1.) * self.DeltaT - self.LengthOnsets[m][i][j]):
                                    self.X[i][m][n, k] = (
                                        self.DeltaT / self.TR)

        else:  # EVENT DESIGNED STIMULI
            for i in xrange(I):
                for m in xrange(self.X[i].shape[0]):
                    for j in xrange(self.OnsetList[i][m].shape[0]):
                        # j^th onset of [session i / parameter m] is considered
                        CurrentOnset = self.OnsetList[i][m][j]
                        # print 'onset:', CurrentOnset, CurrentOnset.shape
                        for n in xrange(Ni[i]):
                            for k in xrange(K - 1):
                                if CurrentOnset <= (n * self.TR - (k) * self.DeltaT + 0.001) and CurrentOnset > ((n) * self.TR - (k + 1.) * self.DeltaT):
                                    self.X[i][m][n, k] = 1.
                                    # print 'cond ok'
                                    # print '(n)*self.TR-(k+1.)*self.DeltaT:', (n)*self.TR-(k+1.)*self.DeltaT
                                    # print 'n*self.TR-(k)*self.DeltaT+0.001:',
                                    # n*self.TR-(k)*self.DeltaT+0.001

        # Then remove the lines in X corresponding to removed positions
        # assuming same nb of scans per session and samed removed positions
        # over sessions
        if numpy.size(self.pos_removed) > 1:
            pos_kept = np.arange(self.Ni[0])
            # put 0 where positions were removed!
            pos_kept[self.pos_removed] = -1
            pos_kept_final = np.array(
                [pos_kept[i] for i in xrange(self.Ni[0]) if pos_kept[i] > -1])
            print 'pos kept:', pos_kept_final
            X_true = np.array(self.X)[:, :, pos_kept_final, :]
            print 'X_true:', X_true, X_true.shape
            print 'self.X:', self.X, np.array(self.X).shape

            self.X = X_true

            # put Ni again at its true value   with the actual nb of scans
            # after subsampling
            self.Ni = array([len(ss) for ss in self.sscans])

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def buildLowFreqMat(self):
        """
        build the low frequency basis matrix P
        requires:
            * self.OrthoBtype
            * self.Qi
            * self.TR
            * self.Ni
            * self.I
        """
        for i in xrange(self.I):
            if self.OrthoBtype == 'cosine':
                self.P[i] = self.buildCosMat(self.Qi[i], self.TR, self.Ni[i])
            else:
                self.P[i] = self.buildPolyMat(self.Qi[i], self.TR, self.Ni[i])
                # the output columns number is not always as expected
                self.Qi[i] = self.P[i].shape[1]

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def buildPolyMat(self, fctNb, tr, ny):
        """
        build a polynomial low frequency basis in P (adapted from samplerbase.py)
        requires:
            * fctNb: columns number in the current session
            * tr: the time resolution of the BOLD data (in second)
            * ny: number of data for the current session
        problems:
            * there may have no constant column in the orthogonal matrix (the algorithm suppose there is one such column)
            * the columns number is not always as expected
        """
        paramLFD = fctNb - \
            1  # order of the orthonormal basis for the drift component for the current session
        regressors = tr * numpy.arange(0, ny)
        timePower = numpy.arange(0, paramLFD + 1, dtype=int)
        regMat = numpy.zeros((len(regressors), paramLFD + 1), dtype=np.float32)
        for v in xrange(paramLFD + 1):
            regMat[:, v] = regressors[:]
        tPowerMat = numpy.matlib.repmat(timePower, ny, 1)
        lfdMat = numpy.power(regMat, tPowerMat)
        lfdMat = numpy.array(scipy.linalg.orth(lfdMat))
        return lfdMat

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def buildCosMat(self, fctNb, tr, ny):
        """
        build a cosine low frequency basis in P (adapted from samplerbase.py)
        requires:
            * fctNb: columns number in the current session
            * tr: the time resolution of the BOLD data (in second)
            * ny: number of data for the current session
        """
        paramLFD = fix(2 * (ny * tr) / (fctNb - 1.)
                       )  # order of the orthonormal basis for the drift component for the current session / +1 stands for the mean/cst regressor
        n = numpy.arange(0, ny)
        lfdMat = numpy.zeros((ny, fctNb), dtype=np.float32)
        # lfdMat[:,0] is actually lfdMat[:,0]' with matlab norms
        lfdMat[:, 0] = numpy.ones((1, ny), dtype=np.float32) / sqrt(ny)
        samples = 1 + numpy.arange(fctNb - 1, dtype=int)

        for k in samples:
            lfdMat[:, k] = numpy.sqrt(2. / ny) * \
                numpy.cos(numpy.pi * (2. * n + 1.) * k / (2. * ny))
        return lfdMat

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def CptSigma(self):
        """
        Computes the Sigma at a given iteration
        requires:
            * InvR
            * TauM
            * rb
            * X
            * M
        remark:
            self.Sigma[m*SBS:(m+1)*SBS,n*SBS:(n+1)*SBS]] -> (m,n)^th block of Sigma in session i
        """
        # 0 ) for compactness
        SBS = self.K - 1

        # 1 ) computes InvSigma
        self.InvSigma[:, :] = 0.

        for m in xrange(self.M):
            if not self.fixed_taum:
                self.InvSigma[
                    m * SBS:(m + 1) * SBS, m * SBS:(m + 1) * SBS] = self.InvR / self.TauM[m]
            else:
                self.InvSigma[
                    m * SBS:(m + 1) * SBS, m * SBS:(m + 1) * SBS] = self.InvR * self.lambda_reg

        for i in xrange(self.I):
            for m in xrange(self.M):
                for n in xrange(self.M):
                    if not self.fixed_taum:
                        self.InvSigma[
                            m * SBS:(m + 1) * SBS, n * SBS:(n + 1) * SBS] += dot(self.X[i][m].T, self.X[i][n]) / self.rb[i]
                    else:
                        self.InvSigma[
                            m * SBS:(m + 1) * SBS, n * SBS:(n + 1) * SBS] += dot(self.X[i][m].T, self.X[i][n])

        # 2 ) computes Sigma
        # TODO: use system solving instead of matrix inversion
        self.Sigma = linalg.inv(self.InvSigma)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def CptFctQ(self, CptType):
        """
        Computes the function Q(\Theta',\tilde{\Theta};y) at a given iteration
        requires:
            * All parameters and hyperparameters
            * Sigma
            * InvR
            * CptType = 'K_Km1' or 'K_K'
        """
        # 0 ) Initialisation
        SBS = self.K - 1
        result = 0.

        # 1 ) first part -> \sum_{i=1}^{I} Q_{Y_i|H}
        for i in xrange(self.I):
            result += self.rb[i] * ((float)(self.Ni[i]))

        # 2 ) second part -> Q_{H}
        # 2.1 ) First term
        for m in xrange(self.M):
            result -= ((float)(self.K - 1)) * log(self.TauM[m]) / 2.

        # 2.2 ) second term
        for m in xrange(self.M):
            # 2.2.1 ) ...
            result -= self.TauM[m] * \
                dot(self.h[m], dot(self.R, self.h[m])) / 2.

            # 2.2.2 ) ...
            if self.OrthoBtype == 'cosine':
                result -= ((self.OldTauM[m] * self.TauM[m]
                            * dot(self.R, self.R)).trace()) / 2
            else:
                result -= ((self.TauM[m] * self.TauM[m]
                            * dot(self.R, self.R)).trace()) / 2

        # 2.3 ) third term
        result -= ((float)(self.M) / 2.) * log(self.DetR)

        return result

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def StoreRes(self, POI):
        """
        Store results computed in the voxel defined in POI
        requires:
            * the estimation at this voxel must have been performed
        """

        for m in xrange(self.M):
            meanLoc = abs(self.bold[:, POI].mean())
            logger.info('Final HRF cond %d, POI %d:', m, POI)
            logger.info(self.h[m])

            # store evaluated response function (percentage of signal change)
            if self.compute_pct_change:
                self.ResponseFunctionsEvaluated_PctChgt[
                    m, 1:self.K, POI] = self.h[m] * 100 / meanLoc

            # store evaluated response function (without normalization)
            self.ResponseFunctionsEvaluated[m, 1:self.K, POI] = self.h[m]

            # store the P-value
            SBS = self.K - 1
            chi2 = dot(self.h[m], dot(
                self.InvSigma[m * SBS:(m + 1) * SBS, m * SBS:(m + 1) * SBS], self.h[m]))
            Pvalue = 1 - scipy.stats.chi2.cdf(chi2, self.h[m].shape[0])
            self.Pvalues[m, POI] = Pvalue

            # store the sqrt of the diagonal of the covariance matrix
            # (percentage of signal change)
            errors = numpy.zeros(self.K + 1, dtype=np.float32)
            for i in xrange(self.K - 1):
                errors[i + 1] = sqrt(self.Sigma[m * SBS + i][m * SBS + i])
            if self.compute_pct_change:
                self.StdValEvaluated_PctChgt[
                    m, :, POI] = errors * 100 / meanLoc
            self.StdValEvaluated[m, :, POI] = errors

        if self.compute_fit(POI).shape < self.SignalEvaluated[:, POI].shape:
            # there are more than one session
            # print 'More than one session'
            self.SignalEvaluated[
                0:self.compute_fit(POI).shape[0], POI] = self.compute_fit(POI)
        else:
            # only one session
            self.SignalEvaluated[:, POI] = self.compute_fit(POI)

    def compute_fit(self, POI):
        # store the signal evaluated in the first session
        # (drift + convolution between evluated HRF and Onset matrix)
        # print 'self.P[0].shape:', self.P[0].shape
        # fit calculated only for the first session
        EvalSignal = dot(self.P[0], self.l[0, :, POI])
        # print 'EvalSignal.shape:', EvalSignal.shape
        for m in xrange(self.M):
            EvalSignal += dot(self.X[0][m], self.h[m])

        return EvalSignal

    def getOutputs(self):
        outputs = {}

        npos = self.bold.shape[1]

        if self.avg_bold:
            pvals = np.repeat(self.Pvalues, self.nbVoxels, axis=1)

        # outputs['pvalue'] = xndarray(pvals.astype(np.float32),
            # axes_names=['condition','voxel'],
            # axes_domains={'condition':self.condition_names})

        hrf_time_axis = np.arange(self.ResponseFunctionsEvaluated.shape[1]) * \
            self.DeltaT
        if self.compute_pct_change:
            hrf_pct_change = self.ResponseFunctionsEvaluated_PctChgt.astype(
                float32)
            if self.avg_bold:
                hrf_pct_change = np.repeat(hrf_pct_change, self.nbVoxels,
                                           axis=2)
            chpc = xndarray(hrf_pct_change, axes_names=['condition', 'time', 'voxel'],
                            axes_domains={'condition': self.condition_names,
                                          'time': hrf_time_axis})
        # hrf_pct_change_errors = self.StdValEvaluated_PctChgt.astype(float32)
        # chpc_errors = xndarray(hrf_pct_change_errors,
        #                      axes_names=['condition','time','voxel'])

        # c = stack_cuboids([chpc, chpc_errors], axis='error',
        #                   domain=['value','error'], axis_pos='last')

        # outputs['ehrf_prct_bold_change'] = c
            outputs['ehrf_prct_bold_change'] = chpc

        rfe = self.ResponseFunctionsEvaluated
        if self.avg_bold:
            rfe = np.repeat(rfe, self.nbVoxels, axis=2)

        ch = xndarray(rfe.astype(float32),
                      axes_names=['condition', 'time', 'voxel'],
                      axes_domains={'condition': self.condition_names,
                                    'time': hrf_time_axis})

        ch_errors = xndarray(self.StdValEvaluated.astype(float32),
                             axes_names=['condition', 'time', 'voxel'])
        outputs['ehrf_error'] = ch_errors

        # c = stack_cuboids([ch, ch_errors], axis='error',
        #                   domain=['value','error'], axis_pos='last')

        # outputs['ehrf'] = c
        if 0:
            print 'ehrf:',
            print ch.descrip()
            print 'ehrf for cond 0', ch.data[0, :, :].shape
            for ih in xrange(ch.data.shape[2]):
                print array2string(ch.data[0, :, ih])

        outputs['ehrf'] = ch

        ad = {'condition': self.condition_names}
        outputs['ehrf_norm'] = xndarray((rfe ** 2).sum(1) ** .5,
                                        axes_names=['condition', 'voxel'],
                                        axes_domains=ad)

        se = self.SignalEvaluated
        if self.avg_bold:
            se = np.repeat(se, self.nbVoxels, axis=1)

        if self.output_fit:
            cfit = xndarray(se.astype(float32),
                            axes_names=['time', 'voxel'])
            bold = self.bold
            if self.avg_bold:
                bold = np.repeat(bold, self.nbVoxels, axis=1)

            cbold = xndarray(self.bold.astype(np.float32),
                             axes_names=['time', 'voxel'])

            outputs['fit'] = stack_cuboids([cfit, cbold], axis="type",
                                           domain=['fit', 'bold'])

            fit_error = sqrt(
                (self.SignalEvaluated - self.bold) ** 2).astype(float32)
            if self.avg_bold:
                fit_error = np.repeat(fit_error, self.nbVoxels, axis=1)
                outputs['fit_error'] = xndarray(fit_error,
                                                axes_names=['time', 'voxel'])

        # save matrix X
        outputs['matX'] = xndarray(self.X[0].astype(float32),
                                   axes_names=['cond', 'time', 'P'],
                                   value_label='value')

        # print 'self.P[0].transpose().shape:', self.P[0].transpose().shape
        # print 'self.bold.shape:', self.bold.shape
        #l = dot(self.P[0].transpose(), self.bold)

        #fu = (self.SignalEvaluated - dot(self.P[0],l)).astype(float32)
        # if self.avg_bold:
        #fu = np.repeat(fu, self.nbVoxels, axis=1)
        #outputs['fit_undrift'] = xndarray(fu, axes_names=['time','voxel'])

        outputs['drift'] = xndarray(dot(self.P[0], self.l[0]).astype(float32),
                                    axes_names=['time', 'voxel'])

        rmse = sqrt(
            (self.SignalEvaluated - self.bold) ** 2).mean(0).astype(float32)
        if self.avg_bold:
            rmse = np.repeat(rmse, self.nbVoxels, axis=0)
        outputs['rmse'] = xndarray(rmse, axes_names=['voxel'])

        if self.save_history:
            h = self.ResponseFunctionsEvaluated
            print 'h.shape:', h.shape
            sh = (self.nbIt,) + h.shape
            print 'sh:', sh
            h_history = np.zeros(sh, dtype=np.float32)
            for POI in xrange(npos):
                # print 'h_history[:,:,:,POI]:', h_history[:,:,:,POI].shape
                # print 'np.array(self.history["h"][POI]):',
                # np.array(self.history['h'][POI]).shape
                h_history[:, :, :, POI] = np.array(self.history['h'][POI])
            if self.avg_bold:
                h_history = np.repeat(h_history, self.nbVoxels, axis=3)
            outputs['ehrf_history'] = xndarray(h_history,
                                               axes_names=['iteration', 'condition',
                                                           'time', 'voxel'])

        return outputs

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # def SaveStoredResults(self):
    #     """
    #     Save the results stored in the storage matrices
    #     requires:
    #         * the whole estimation must have been performed
    #     """
    #     return
    #     #save results
    #     for m in xrange(self.M):
    #         if self.M>1:
    #             CondID="_Cond" + str(m+1)
    #         else:
    #             CondID=""

    #         PtSaveName= self.OutputID + CondID + "_pvalue.nii"
    #         writeImageWithPynifti(self.Pvalues[m,:],PtSaveName)

    #         PtSaveName= self.OutputID + CondID + "_EvalHRF_PrctChanges.nii"
    #         writeImageWithPynifti(self.ResponseFunctionsEvaluated_PctChgt[m,:],PtSaveName)

    #         PtSaveName= self.OutputID + CondID + "_EvalHRF.nii"
    #         writeImageWithPynifti(self.ResponseFunctionsEvaluated[m,:],PtSaveName)

    #         PtSaveName= self.OutputID + CondID + "_StdValEval_PrctChanges.nii"
    #         writeImageWithPynifti(self.StdValEvaluated_PctChgt[m,:],PtSaveName)

    #         PtSaveName= self.OutputID + CondID + "_StdValEval.nii"
    #         writeImageWithPynifti(self.StdValEvaluated[m,:],PtSaveName)

    #     PtSaveName= self.OutputID + "_OriginalSignal.nii"
    #     writeImageWithPynifti(self.bold[0][:],PtSaveName)

    #     PtSaveName= self.OutputID + "_EstimatedSignal.nii"
    #     writeImageWithPynifti(self.SignalEvaluated[:],PtSaveName)

    def cpt_XSigmaX(self, tempTerm2i, SBS, i):
        for m in xrange(self.M):
            for n in xrange(self.M):
                tempTerm2i += dot(self.X[i][m],
                                  dot(self.Sigma[m * SBS:(m + 1) * SBS,
                                                 n * SBS:(n + 1) * SBS],
                                      self.X[i][n].T))
                # nb_cond**2*(nscans*nb_hrf_coeffs**2 + nscans**2*nb_hrf_coeffs)
                # 10**2 * (125 * 40**2 + 125**2 * 40) ~ 8e7

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def EM_solver(self, POI):
        """
        EM_solver
        requires:
            * everything in the class is supposed initialized
        """
        # initialization
        SBS = self.K - 1
        self.CptSigma()
        FctQ_K_K = 1000000000.0
        iteration = 1
        StopTest1 = 1.0
        StopTest2 = 1.0

        BigVector = numpy.zeros(self.Sigma.shape[1], dtype=float)
        templi = [numpy.zeros(self.Ni[i], dtype=float) for i in xrange(self.I)]
        tempTerm2 = [numpy.zeros((self.X[i][0].shape[0], self.X[i][0].shape[0]),
                                 dtype=np.float32) for i in xrange(self.I)]
        delta_h = 1.
        self.epsilon = 1e-4

        # Main loop.
        def stop_crit(it, StopTest1, StopTest2):
            if self.nbIt is not None:
                return it < self.nbIt
            else:
                return it < self.nbItMin or ((it < self.nbItMax) and
                                             ((StopTest1 > self.emStop1) or
                                              (StopTest2 > self.emStop2)))

        def stop_crit_fixed_taum(it):
            if self.nbIt is not None:
                return it < self.nbIt
            else:
                return it < self.nbItMin or ((it < self.nbItMax) and
                                             delta_h > self.epsilon)

        if self.save_history:
            # Store initial values
            z = np.zeros((self.M, 1))
            self.history['h'][POI].append(np.hstack((z, self.h, z)))
            self.history['l'][POI].append(self.l[:, :, POI].copy())
            self.history['rb'][POI].append(self.rb.copy())
            self.history['tau'][POI].append(self.TauM.copy())
            self.history['fit'][POI].append(self.compute_fit(POI))

        if not self.fixed_taum:
            while stop_crit(iteration, StopTest1, StopTest2):

                if 0:
                    print 'Iteration:', iteration
                    print 'nbItMax:', self.nbItMax
                    print 'nbItMin:', self.nbItMin
                # 1 ) estimation of \hat{h^{MAP}}

                # compute_h_MAP(self.y, self.P, ...)

                BigVector[:] = 0.
                for i in xrange(self.I):
                    temp = self.y[i] - dot(self.P[i], self.l[i, :, POI])
                    for m in xrange(self.M):
                        BigVector[m * SBS:(m + 1) * SBS] += dot(self.X[i][m].T, temp) / \
                            self.rb[i]  # nb_sess*nb_cond*nscans*nb_hrf_coeffs
                        # 1*10*125*40 = 5e4

                # (nb_cond*nb_hrf_coeffs)**3
                BigVector2 = dot(self.Sigma, BigVector)
                #(10*40)**3 = 7e6

                for m in xrange(self.M):
                    logger.info('it %d, h^MAP cond %d:', iteration, m)
                    self.h[m, :] = BigVector2[m * SBS:(m + 1) * SBS]
                    logger.info(self.h[m, :])

                if self.save_history:
                    z = np.zeros((self.M, 1))
                    self.history['h'][POI].append(np.hstack((z, self.h, z)))

                # 2 ) estimation of \hat{l_i}
                for i in xrange(self.I):
                    templi[i][:] = 0.
                    for m in xrange(self.M):
                        templi[i] += dot(self.X[i][m], self.h[m])
                        # nb_sess * nb_cond * nscans * nb_hrf_coeffs ~ 5e4
                    self.l[i, :, POI] = dot(self.P[i].T, self.y[i] - templi[i])
                    # nb_sess * n_preg * nscans

                if self.save_history:
                    self.history['l'][POI].append(self.l[:, :, POI].copy())

                # 3 ) estimation of \hat{rb}...
                for i in xrange(self.I):
                    # 3.1 ) ...first term
                    temp = self.y[i] - dot(self.P[i], self.l[i, :, POI])
                    for m in xrange(self.M):
                        temp -= dot(self.X[i][m], self.h[m])

                    term1 = dot(temp, temp)

                    # 3.2 ) ...second term
                    tempTerm2[i][:] = 0.
                    #compute_XSigmaX(tempTerm2[i], SBS, i)
                    self.cpt_XSigmaX(tempTerm2[i], SBS, i)
                    term2 = tempTerm2[i].trace()

                    # 3.3 ) ...total
                    self.Old_rb[i] = self.rb[i]
                    self.rb[i] = (term1 + term2) / ((float)(self.Ni[i]))

                if self.save_history:
                    self.history['rb'][POI].append(self.rb.copy())

                # ... if rb is the same for all sessions, uncomment these lines...
                # temp=0.
                # for i in xrange(self.I):
                #    temp=self.rb[i]*((float)(self.Ni[i]))
                #
                # self.rb[:]=temp/((float)(self.Ni[:].sum()))

                # 4 ) reestimation of sigma
                self.CptSigma()

                # 5 ) estimation of \TauM
                for m in xrange(self.M):
                    self.OldTauM[m] = self.TauM[m].copy()
                    temp = dot(kron(self.h[m][:, newaxis], self.h[m]) +
                               self.Sigma[
                                   m * SBS:(m + 1) * SBS, m * SBS:(m + 1) * SBS],
                               self.InvR)
                    self.TauM[m] = (temp.trace()) / (self.K - 1)

                if self.save_history:
                    self.history['tau'][POI].append(self.TauM.copy())

                if self.save_history:
                    self.history['fit'][POI].append(self.compute_fit(POI))

                # 6 ) Q function estimation at the current iteration and
                # previous
                FctQ_Km1_Km1 = FctQ_K_K
                FctQ_K_K = self.CptFctQ('K_K')
                FctQ_K_Km1 = self.CptFctQ('K_Km1')

                # 7 ) Stopping criterion
                iteration = iteration + 1

                StopTest1 = linalg.norm(FctQ_K_Km1 - FctQ_Km1_Km1) / \
                    linalg.norm(FctQ_K_Km1)

                StopTest2 = 0.0
                for i in xrange(self.I):
                    tempCalc = sqrt(linalg.norm(self.TauM - self.OldTauM) ** 2.0 +
                                    (self.Old_rb[i] - self.rb[i]) ** 2.0) / \
                        sqrt(linalg.norm(self.TauM) ** 2.0 +
                             self.rb[i] ** 2.0)
                    if StopTest2 < tempCalc:
                        StopTest2 = tempCalc
        else:
            h_prev = np.ones(self.M * SBS)

            while stop_crit_fixed_taum(iteration):

                if 0:
                    print 'Iteration:', iteration
                    print 'nbItMax:', self.nbItMax
                    print 'nbItMin:', self.nbItMin
                # 1 ) estimation of \hat{h^{MAP}}

                # compute_h_MAP(self.y, self.P, ...)

                BigVector[:] = 0.
                for i in xrange(self.I):
                    temp = self.y[i] - dot(self.P[i], self.l[i, :, POI])
                    for m in xrange(self.M):
                        BigVector[
                            m * SBS:(m + 1) * SBS] += dot(self.X[i][m].T, temp)
                        # nb_sess*nb_cond*nscans*nb_hrf_coeffs
                        # 1*10*125*40 = 5e4

                # (nb_cond*nb_hrf_coeffs)**3
                BigVector2 = dot(self.Sigma, BigVector)
                #(10*40)**3 = 7e6

                for m in xrange(self.M):
                    logger.info('it %d, h^MAP cond %d:', iteration, m)
                    self.h[m, :] = BigVector2[m * SBS:(m + 1) * SBS]
                    logger.info(self.h[m, :])

                if self.save_history:
                    z = np.zeros((self.M, 1))
                    self.history['h'][POI].append(np.hstack((z, self.h, z)))

                # 2 ) estimation of \hat{l_i}
                for i in xrange(self.I):
                    templi[i][:] = 0.
                    for m in xrange(self.M):
                        templi[i] += dot(self.X[i][m], self.h[m])
                    self.l[i, :, POI] = dot(self.P[i].T, self.y[i] - templi[i])

                if self.save_history:
                    self.history['l'][POI].append(self.l[:, :, POI].copy())

                if self.save_history:
                    self.history['fit'][POI].append(self.compute_fit(POI))

                # 7 ) Stopping criterion
                iteration = iteration + 1

                delta_h = ((h_prev - BigVector2) ** 2).sum() ** .5
                h_prev = BigVector2.copy()

        self.stop_iterations[POI] = iteration

        logger.info(
            "iteration: %s -> delta_h=%s", str(iteration), str(delta_h))


def rfir(func_data, fir_duration=42, fir_dt=.6, nb_its_max=100,
         nb_its_min=5, fixed_taum=False, lambda_reg=100.):
    """
    Fit a Regularized FIR on functional data *func_data*:
    - multisession voxel-based fwd model: y = \sum Xh + Pl + b
    - heteroscedastic noise
    - session dependent drift coefficients
    - one HRF per condition
    - solved by Expectation-Minimization (EM) (iterative scheme)

    Reference: "Unsupervised robust non-parametric estimation of the hemodynamic
    response function for any fMRI experiment." Ciuciu, J.-B. Poline,
    G. Marrelec, J. Idier, Ch. Pallier, and H. Benali.
    IEEE Trans. Med. Imag., 22(10):1235-1251, Oct. 2003.

    Args:
        *func_data* (pyhrf.core.FmriData)
        *fir_duration* (float): FIR duration in seconds
        *fir_dt* (float): FIR temporal resolution
        *fixed_taum* (bool): enable faster (drafter) RFIR version where
                             the HRF variance hyper-parameter is fixed.
        *lambda_reg* (float): amount of temporal regularization for the HRF.
                              Only used if *fixed_taum* is true.
        *nb_its_min*: minimum number of iterations for the EM
        *nb_its_max*: maximum number of iterations for the EM

    Returns: dict of xndarray instances

        The returned dict contains:
        {"":
         "":
        }


    """
    rfir_estimator = RFIREstim(hrf_nb_coeffs=int(np.round(fir_duration / fir_dt)),
                               hrf_dt=fir_dt, nb_its_max=nb_its_max,
                               nb_its_min=nb_its_min, fixed_taum=fixed_taum,
                               lambda_reg=lambda_reg)
    rfir_estimator.linkToData(func_data)
    rfir_estimator.run()
    outputs = rfir_estimator.getOutputs()
    to_return = {'fir': outputs["ehrf"], 'fir_error': outputs["ehrf_error"],
                 'drift': outputs["drift"]}
    if outputs.has_key('fit'):
        to_return['fit'] = outputs['fits']
    return to_return
