# -*- coding: utf-8 -*-

import logging

import numpy as np

from pyhrf import xmlio
from pyhrf.jde.samplerbase import *
from pyhrf.jde.nrl.bigaussian import NRLSamplerWithRelVar as WNS


logger = logging.getLogger(__name__)


class W_Drift_Sampler(xmlio.XmlInitable, GibbsSamplerVariable):

    # other class attributes
    L_CI = 0
    L_CA = 1
    CLASSES = np.array([L_CI, L_CA], dtype=int)
    CLASS_NAMES = ['inactiv', 'activ']

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None, pr_sigmoid_slope=1., pr_sigmoid_thresh=0.):
        """
        pr_sigmoid_slope = tau1
        pr_sigmoid_thresh = tau2
        """
        # TODO : comment
        xmlio.XmlInitable.__init__(self)

        GibbsSamplerVariable.__init__(self, 'W', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value)
        self.t1 = pr_sigmoid_slope
        self.t2 = pr_sigmoid_thresh

    def initObservables(self):
        # print 'initObservables'
        logger.info('WSampler.initObservables ...')
        GibbsSamplerVariable.initObservables(self)

    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.ny = self.dataInput.ny
        self.nbVoxels = self.dataInput.nbVoxels


    def checkAndSetInitValue(self, variables):


        logger.info('WSampler.checkAndSetInitW ...')
        # Generate default W if necessary :
        # if self.useTrueVal:
        #self.currentValue = np.ones(self.nbConditions, dtype=int)
        # if self.useTrueVal:
        # if self.trueW is not None:
        # TODO : take only common conditions
        #self.currentValue = self.trueW.copy()
        # else:
        #raise Exception('True W have to be used but none defined.')

        # if no initial W specified
        if self.currentValue is None or not self.sampleFlag:
            logger.info('W are not initialized '
                        '-> suppose that all conditions are relevant')
            self.currentValue = np.ones(self.nbConditions, dtype=int)

        logger.debug('init W :')
        logger.debug(self.currentValue)

    def saveCurrentValue(self, it):
        GibbsSamplerVariable.saveCurrentValue(self, it)

    def updateObsersables(self):
        GibbsSamplerVariable.updateObsersables(self)

    def saveObservables(self, it):
        GibbsSamplerVariable.saveObservables(self, it)

    def computemoyq(self, cardClassCA, nbVoxels):
        '''
        Compute mean of labels in  ROI
        '''
        moyq = np.divide(cardClassCA, (float)(nbVoxels))
        # print 'moyq=',moyq

        return moyq

    def computeProbW1(self, gj, gTgj, rb, t1, t2, mCAj, vCIj, vCAj, j, cardClassCAj):
        '''
        ProbW1 is the probability that condition is relevant
        It is a vecteur on length nbcond
        '''

        logger.info('Sampling W - cond %d ...', j)

        val = -t1 * (cardClassCAj - t2)

        tmp = (- 0.5 * cardClassCAj) * np.log(vCIj / vCAj)

        result1 = 0.
        result2 = 0.
        for i in self.voxIdx[self.L_CA][j]:
            valeur1 = (self.nrls[j, i]) ** 2 / (2. * vCIj)
            valeur2 = (self.nrls[j, i] - mCAj) ** 2 / (2. * vCAj)
            result1 += (valeur1 - valeur2)

        y = self.dataInput.varMBY
        matPl = self.samplerEngine.get_variable('drift').matPl
        StimIndSign = np.zeros((y.shape), dtype=float)
        for i in xrange(self.nbConditions):
            if i != j:
                StimIndSign += self.currentValue[i] * self.nrls[i, :] * np.tile(
                    self.varXh[:, i], (self.nbVoxels, 1)).transpose()
        ej = y - StimIndSign - matPl

        for i in xrange(self.nbVoxels):
            valeur1 = ((self.nrls[j, i]) ** 2) * gTgj
            valeur2 = 2. * self.nrls[j, i] * np.dot(ej[:, i].transpose(), gj)
            result2 += (1. / rb[i]) * (valeur1 - valeur2)

        val0 = val + tmp
        val1 = result1 - 0.5 * result2

        # print 'w =',self.currentValue

        Maximum = val0
        if (val1 > Maximum):
            Maximum = val1

        tmp2 = 1. + (np.exp(val0 - Maximum) / np.exp(val1 - Maximum))

        print 'Cond ', j, ',  val0 =', val0, ',     val1 =', val1
        # print 'cond =',j,',     v0 =',vCIj,',   v1 =',vCAj,',   m1 =',mCAj

        return 1. / tmp2

    def sampleNextInternal(self, variables):

        snrls = self.get_variable('nrl')
        self.cardClass = snrls.cardClass
        # print 'CardClass WSampler =',self.cardClass
        self.voxIdx = snrls.voxIdx
        self.nrls = snrls.currentValue

        sIMixtP = self.get_variable('mixt_params')
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        sHrf = self.get_variable('hrf')
        self.varXh = sHrf.varXh
        h = sHrf.currentValue
        rb = self.get_variable('noise_var').currentValue

        varXh = sHrf.varXh
        ProbW1 = np.zeros(self.nbConditions, dtype=float)
        ProbW0 = np.zeros(self.nbConditions, dtype=float)
        self.WSamples = np.random.rand(self.nbConditions)

        gTg = np.diag(np.dot(varXh.transpose(), varXh))

        g = varXh
        cardClassCA = self.cardClass[self.L_CA, :]

        print '***************************************************************************'
        for j in xrange(self.nbConditions):
            ProbW1[j] = self.computeProbW1(g[:, j], gTg[j], rb, self.t1, self.t2, mean[
                                           self.L_CA, j], var[self.L_CI, j], var[self.L_CA, j], j, cardClassCA[j])
            ProbW0[j] = 1. - ProbW1[j]
            self.currentValue[j] = 0
            if(self.WSamples[j] <= ProbW1[j]):
                self.currentValue[j] = 1
            snrls.computeVarYTildeOptWithRelVar(self.varXh, self.currentValue)
        print 'W =', self.currentValue

    def threshold_W(self, meanW, thresh):
        print 'meanW =', meanW
        destW = np.zeros(self.nbConditions, dtype=int)
        for j in xrange(self.nbConditions):
            if meanW[j] >= thresh:
                destW[j] = 1
        return destW

    def finalizeSampling(self):

        GibbsSamplerVariable.finalizeSampling(self)
        #self.finalW = self.threshold_W(self.mean, 0.8722)
        self.finalW = self.threshold_W(self.mean, 0.5)
        print 'final w : ', self.finalW

    def getOutputs(self):

        outputs = GibbsSamplerVariable.getOutputs(self)
        cn = self.dataInput.cNames

        axes_names = ['condition']
        axes_domains = {'condition': cn}

        outputs['w_pm_thresh'] = xndarray(self.finalW, axes_names=axes_names,
                                          axes_domains=axes_domains)

        # outputs['w_pm'] = xndarray(self.mean, axes_names=axes_names,
        # axes_domains=axes_domains)

        return outputs


class WSampler(xmlio.XmlInitable, GibbsSamplerVariable):

    # old parameters specifications :
    # P_TAU1 = 'SlotOfSigmoid'
    # P_TAU2 = 'ThreshPointOfSigmoid'

    # other class attributes
    L_CI = 0
    L_CA = 1
    CLASSES = np.array([L_CI, L_CA], dtype=int)
    CLASS_NAMES = ['inactiv', 'activ']

    def __init__(self, do_sampling=True, use_true_value=False,
                 val_ini=None, pr_sigmoid_slope=1., pr_sigmoid_thresh=0.):
        """
        pr_sigmoid_slope = tau1
        pr_sigmoid_thresh = tau2
        """
        # TODO : comment
        xmlio.XmlInitable.__init__(self)

        GibbsSamplerVariable.__init__(self, 'W', valIni=val_ini,
                                      sampleFlag=do_sampling,
                                      useTrueValue=use_true_value)
        self.t1 = pr_sigmoid_slope
        self.t2 = pr_sigmoid_thresh

    def initObservables(self):
        # print 'initObservables'
        logger.info('WSampler.initObservables ...')
        GibbsSamplerVariable.initObservables(self)

    def linkToData(self, dataInput):

        self.dataInput = dataInput
        self.nbConditions = self.dataInput.nbConditions
        self.ny = self.dataInput.ny
        self.nbVoxels = self.dataInput.nbVoxels
        self.matXQ = self.dataInput.matXQ
        # if dataInput.simulData is not None:
        #self.trueW = dataInput.simulData.w
        # else:
        #self.trueW = None

    def checkAndSetInitValue(self, variables):

        # Generate default W if necessary :
        # if self.useTrueVal:
            #self.currentValue = np.ones(self.nbConditions, dtype=int)
        # if self.useTrueVal:
            # if self.trueW is not None:
                # TODO : take only common conditions
                #self.currentValue = self.trueW.copy()
            # else:
                #raise Exception('True W have to be used but none defined.')

        if self.currentValue is None:  # if no initial W specified
            logger.info('W are not initialized '
                        '-> suppose that all conditions are relevant')
            self.currentValue = np.ones(self.nbConditions, dtype=int)

        logger.debug('init W :')
        logger.debug(self.currentValue)

    def saveCurrentValue(self, it):
        GibbsSamplerVariable.saveCurrentValue(self, it)

    def updateObsersables(self):
        GibbsSamplerVariable.updateObsersables(self)

    def saveObservables(self, it):
        GibbsSamplerVariable.saveObservables(self, it)

    def computemoyq(self, cardClassCA, nbVoxels):
        '''
        Compute mean of labels in  ROI
        '''
        moyq = np.divide(cardClassCA, (float)(nbVoxels))

        return moyq

    def computeProbW1(self, Qgj, gTQgj, rb, moyqj, t1, t2, mCAj, vCIj, vCAj, j, cardClassCAj):
        '''
        ProbW1 is the probability that condition is relevant
        It is a vecteur on length nbcond
        '''
        logger.info('Sampling W - cond %d ...', j)

        val = -t1 * (cardClassCAj - t2)

        tmp = (- 0.5 * cardClassCAj) * np.log(vCIj / vCAj)

        result1 = 0.
        result2 = 0.
        for i in self.voxIdx[self.L_CA][j]:
            valeur1 = (self.nrls[j, i]) ** 2 / (2. * vCIj)
            valeur2 = (self.nrls[j, i] - mCAj) ** 2 / (2. * vCAj)
            result1 += (valeur1 - valeur2)

        y = self.dataInput.varMBY
        StimIndSign = np.zeros((y.shape), dtype=float)
        for i in xrange(self.nbConditions):
            if i != j:
                StimIndSign += self.currentValue[i] * self.nrls[i, :] * np.tile(
                    self.varXh[:, i], (self.nbVoxels, 1)).transpose()
        ej = y - StimIndSign

        for i in xrange(self.nbVoxels):
            valeur1 = ((self.nrls[j, i]) ** 2) * gTQgj
            valeur2 = 2. * self.nrls[j, i] * np.dot(ej[:, i].transpose(), Qgj)
            result2 += (1. / rb[i]) * (valeur1 - valeur2)

        val0 = val + tmp
        val1 = result1 - 0.5 * result2

        print 'w =', self.currentValue

        Maximum = val0
        if (val1 > Maximum):
            Maximum = val1

        tmp2 = 1. + np.exp(val0 - Maximum) / np.exp(val1 - Maximum)

        return 1. / tmp2

    def computeVarXhtQ(self, h, matXQ):
        for j in xrange(self.nbConditions):
            self.varXhtQ[j, :] = np.dot(h.transpose(), matXQ[j, :, :])

    def sampleNextInternal(self, variables):

        nrl = self.get_variable('nrl')
        self.cardClass = nrl.cardClass
        # print 'CardClass WSampler =',self.cardClass
        self.voxIdx = nrl.voxIdx
        self.nrls = nrl.currentValue

        self.varYtilde = nrl.varYtilde

        sIMixtP = self.get_variable('mixt_params')
        var = sIMixtP.getCurrentVars()
        mean = sIMixtP.getCurrentMeans()
        sHrf = self.get_variable('hrf')
        self.varXh = sHrf.varXh
        h = sHrf.currentValue
        rb = self.get_variable('noise_var').currentValue

        self.varXhtQ = np.zeros((self.nbConditions, self.ny), dtype=float)
        ProbW1 = np.zeros(self.nbConditions, dtype=float)
        ProbW0 = np.zeros(self.nbConditions, dtype=float)
        self.WSamples = np.random.rand(self.nbConditions)

        self.computeVarXhtQ(h, self.matXQ)
        logger.debug('varXhtQ %s :', str(self.varXhtQ.shape))
        logger.debug(self.varXhtQ)

        gTQg = np.diag(np.dot(self.varXhtQ, self.varXh))

        # Qg = QXh = (XhtQ)^t where XhtQ = h^t X^t Q and Q^t = Q
        Qg = self.varXhtQ.transpose()
        cardClassCA = self.cardClass[self.L_CA, :]
        moyq = self.computemoyq(cardClassCA, self.nbVoxels)

        for j in xrange(self.nbConditions):
            ProbW1[j] = self.computeProbW1(Qg[:, j], gTQg[j], rb, moyq[j], self.t1, self.t2, mean[
                                           self.L_CA, j], var[self.L_CI, j], var[self.L_CA, j], j, cardClassCA[j])
            ProbW0[j] = 1. - ProbW1[j]
            self.currentValue[j] = 0
            if(self.WSamples[j] <= ProbW1[j]):
                self.currentValue[j] = 1

    def threshold_W(self, meanW, thresh):
        print 'meanW =', meanW
        destW = np.zeros(self.nbConditions, dtype=int)
        for j in xrange(self.nbConditions):
            if meanW[j] >= thresh:
                destW[j] = 1
        return destW

    def finalizeSampling(self):
        #self.finalW = self.threshold_W(self.mean, 0.8722)
        GibbsSamplerVariable.finalizeSampling(self)
        self.finalW = self.threshold_W(self.mean, 0.5)
        print 'final w : ', self.finalW

    def getOutputs(self):

        outputs = GibbsSamplerVariable.getOutputs(self)
        cn = self.dataInput.cNames

        axes_names = ['condition']
        axes_domains = {'condition': cn}

        outputs['w_pm_thresh'] = xndarray(self.finalW, axes_names=axes_names,
                                          axes_domains=axes_domains)

        # outputs['w_pm'] = xndarray(self.mean, axes_names=axes_names,
        # axes_domains=axes_domains)

        return outputs
