# -*- coding: utf-8 -*-

import logging

from time import time

import numpy as np

import pyhrf

from pyhrf.ndarray import xndarray
from pyhrf.vbjde.vem_asl_constrained import (Main_vbjde_c_constrained,
                                             Main_vbjde_constrained)
from pyhrf.vbjde.vem_tools_asl import roc_curve
from pyhrf.xmlio import XmlInitable
from pyhrf.tools import format_duration
from pyhrf.ui.jde import JDEAnalyser


logger = logging.getLogger(__name__)


def change_dim(labels):
    '''
    Change labels dimension from
    (ncond, nclass, nvox)
    to
    (nclass, ncond, nvox)
    '''
    ncond = labels.shape[0]
    nclass = labels.shape[1]
    nvox = labels.shape[2]
    newlabels = np.zeros((nclass, ncond, nvox))
    for cond in xrange(ncond):
        for clas in xrange(nclass):
            for vox in xrange(nvox):
                newlabels[clas][cond][vox] = labels[cond][clas][vox]

    return newlabels


class JDEVEMAnalyser(JDEAnalyser):

    parametersComments = {
        'dt': 'time resolution of the estimated HRF in seconds',
        'hrfDuration': 'duration of the HRF in seconds',
        'fast': 'running fast VEM with C extensions',
        'nbClasses': 'number of classes for the response levels',
        'PLOT': 'plotting flag for convergence curves',
        'nItMax': 'maximum iteration number',
        'nItMin': 'minimum iteration number',
        'scale': 'flag for the scaling factor applied to the data fidelity '
        'term during m_h step.\n'
        'If scale=False then do nothing, else divide '
        'the data fidelity term by the number of voxels',
        'beta': 'initial value of spatial Potts regularization parameter',
        'simulation': 'indicates whether the run corresponds to a simulation \
                        example or not',
        'estimateSigmaH': 'estimate or not the HRF variance',
        'estimateSigmaG': 'estimate or not the PRF variance',
        'estimateH': 'estimate or not the HRF',
        'estimateG': 'estimate or not the PRF',
        'sigmaH': 'Initial HRF variance',
        'sigmaG': 'Initial PRF variance',
        'estimateBeta': 'estimate or not the Potts spatial regularization '
        'parameter',
        'estimateLA': 'Explicit drift and perfusion baseline estimation',
        'estimateLabels': 'estimate or not the Labels',
        'estimateMixtParam': 'estimate or not the mixture parameters',
        'InitVar': 'Initiale value of active and inactive gaussian variances',
        'InitMean': 'Initiale value of active gaussian means',
        'constrained': 'adding constrains: positivity and norm = 1 ',
    }

    parametersToShow = ['dt', 'hrfDuration', 'nItMax', 'nItMin',
                        'estimateSigmaH', 'estimateSigmaG', 'estimateH',
                        'estimateG', 'estimateBeta',
                        'estimateLabels', 'estimateLA',
                        'estimateMixtParam', 'InitVar', 'InitMean',
                        'scale', 'nbClasses', 'fast', 'PLOT',
                        'sigmaH', 'sigmaG']

    def __init__(self, hrfDuration=25., dt=.6, fast=True, constrained=False,
                 nbClasses=2, PLOT=False, nItMax=100, nItMin=1, scale=False,
                 beta=1.0, simulation=None, fmri_data=None,
                 estimateH=True, estimateG=True, estimateSigmaH=True,
                 estimateSigmaG=True, sigmaH=0.0001, sigmaG=0.0001,
                 estimateLabels=True, estimateMixtParam=True,
                 InitVar=0.5, InitMean=2.0, estimateA=True, estimateC=True,
                 estimateBeta=True, estimateNoise=True, estimateLA=True):

        XmlInitable.__init__(self)
        JDEAnalyser.__init__(self, outputPrefix='jde_vem_asl_')

        # Important thing : all parameters must have default values
        self.hrfDuration = hrfDuration
        self.dt = dt
        self.fast = fast
        self.constrained = constrained
        self.nbClasses = nbClasses
        self.PLOT = PLOT
        self.nItMax = nItMax
        self.nItMin = nItMin
        self.scale = scale
        self.beta = beta
        self.simulation = simulation
        self.fmri_data = fmri_data
        self.estimateH = estimateH
        self.estimateG = estimateG
        self.estimateSigmaH = estimateSigmaH
        self.estimateSigmaG = estimateSigmaG
        self.sigmaH = sigmaH
        self.sigmaG = sigmaG
        self.estimateLabels = estimateLabels
        self.estimateMixtParam = estimateMixtParam
        self.InitVar = InitVar
        self.InitMean = InitMean
        self.estimateA = estimateA
        self.estimateC = estimateC
        self.estimateBeta = estimateBeta
        self.estimateNoise = estimateNoise
        self.estimateLA = estimateLA

        logger.info("VEM analyzer:")
        logger.info(" - estimate sigma H: %s", str(self.estimateSigmaH))
        logger.info(" - init sigma H: %f", self.sigmaH)
        logger.info(" - estimate drift and perfusion baseline: %s",
                    str(self.estimateLA))

    def analyse_roi(self, roiData):
        # roiData is of type FmriRoiData, see pyhrf.core.FmriRoiData
        # roiData.bold : numpy array of shape
        # BOLD has shape (nscans, nvoxels)

        # roiData.graph #list of neighbours
        data = roiData.bold
        Onsets = roiData.get_joined_onsets()
        TR = roiData.tr
        # K = 2                      # number of classes
        beta = self.beta
        scale = 1                   # roiData.nbVoxels
        nvox = roiData.get_nb_vox_in_mask()
        if self.scale:
            scale = nvox
        rid = roiData.get_roi_id()
        logger.info("JDE VEM - roi %d, nvox=%d, nconds=%d, nItMax=%d", rid,
                    nvox, len(Onsets), self.nItMax)

        #self.contrasts.pop('dummy_example', None)
        cNames = roiData.paradigm.get_stimulus_names()
        graph = roiData.get_graph()
        idx_tag1 = roiData.get_extra_data('asl_first_tag_scan_idx', 0)

        t_start = time()

        logger.info("fast VEM with drift estimation and a constraint")
        print roiData.simulation

        if self.fast:
            NbIter, nrls, estimated_hrf, \
                labels, noiseVar, mu_k, sigma_k, \
                Beta, L, PL, \
                cA, cH, cZ, cAH, cTime, cTimeMean, \
                Sigma_nrls, StimuIndSignal,\
                FreeEnergy = Main_vbjde_c_constrained(graph, data, Onsets,
                                                      self.hrfDuration, self.nbClasses, TR,
                                                      beta, self.dt, scale,
                                                      estimateSigmaH=self.estimateSigmaH,
                                                      estimateSigmaG=self.estimateSigmaG,
                                                      sigmaH=self.sigmaH, sigmaG=self.sigmaG,
                                                      NitMax=self.nItMax, NitMin=self.nItMin,
                                                      estimateBeta=self.estimateBeta,
                                                      PLOT=self.PLOT, idx_first_tag=idx_tag1,
                                                      simulation=self.roiData.simulation,
                                                      estimateH=self.estimateH,
                                                      estimateG=self.estimateG,
                                                      estimateA=self.estimateA,
                                                      estimateC=self.estimateC,
                                                      estimateZ=self.estimateLabels,
                                                      estimateNoise=self.estimateNoise,
                                                      estimateMP=self.estimateMixtParam,
                                                      estimateLA=self.estimateLA)
        else:
            NbIter, brls, estimated_brf, prls, estimated_prf, \
                labels, noiseVar, cZ, cTime, cTimeMean, \
                mu_Ma, sigma_Ma, mu_Mc, sigma_Mc, Beta, L, PL, \
                Sigma_brls, Sigma_prls, \
                StimulusInducedSignal = Main_vbjde_constrained(graph, data, Onsets,
                                                               self.hrfDuration, self.nbClasses, TR,
                                                               beta, self.dt, scale=scale,
                                                               estimateSigmaH=self.estimateSigmaH,
                                                               estimateSigmaG=self.estimateSigmaG,
                                                               sigmaH=self.sigmaH, sigmaG=self.sigmaG,
                                                               NitMax=self.nItMax, NitMin=self.nItMin,
                                                               estimateBeta=self.estimateBeta,
                                                               PLOT=self.PLOT, idx_first_tag=idx_tag1,
                                                               simulation=roiData.simulation[
                                                                   0],
                                                               estimateH=self.estimateH,
                                                               estimateG=self.estimateG,
                                                               estimateA=self.estimateA,
                                                               estimateC=self.estimateC,
                                                               estimateZ=self.estimateLabels,
                                                               estimateNoise=self.estimateNoise,
                                                               estimateMP=self.estimateMixtParam,
                                                               estimateLA=self.estimateLA)

        # Plot analysis duration
        self.analysis_duration = time() - t_start
        logger.info('JDE VEM analysis took: %s',
                    format_duration(self.analysis_duration))

        # OUTPUTS: Pack all outputs within a dict
        outputs = {}
        brf_time = np.arange(len(estimated_brf)) * self.dt
        outputs['brf'] = xndarray(estimated_brf, axes_names=['time'],
                                  axes_domains={'time': brf_time},
                                  value_label="BRF")

        domCondition = {'condition': cNames}
        outputs['brls'] = xndarray(brls.T, value_label="BRLs",
                                   axes_names=['condition', 'voxel'],
                                   axes_domains=domCondition)

        prf_time = np.arange(len(estimated_prf)) * self.dt
        outputs['prf'] = xndarray(estimated_prf, axes_names=['time'],
                                  axes_domains={'time': prf_time},
                                  value_label="PRF")

        outputs['prls'] = xndarray(prls.T, value_label="PRLs",
                                   axes_names=['condition', 'voxel'],
                                   axes_domains=domCondition)

        ad = {'condition': cNames, 'condition2': Onsets.keys()}

        outputs['Sigma_brls'] = xndarray(Sigma_brls, value_label="Sigma_BRLs",
                                         axes_names=['condition', 'condition2',
                                                     'voxel'],
                                         axes_domains=ad)

        outputs['Sigma_prls'] = xndarray(Sigma_prls, value_label="Sigma_PRLs",
                                         axes_names=['condition', 'condition2',
                                                     'voxel'],
                                         axes_domains=ad)

        outputs['NbIter'] = xndarray(np.array([NbIter]), value_label="NbIter")

        outputs['beta'] = xndarray(Beta, value_label="beta",
                                   axes_names=['condition'],
                                   axes_domains=domCondition)

        nbc, nbv = len(cNames), brls.shape[0]
        repeatedBeta = np.repeat(Beta, nbv).reshape(nbc, nbv)
        outputs['beta_mapped'] = xndarray(repeatedBeta, value_label="beta",
                                          axes_names=['condition', 'voxel'],
                                          axes_domains=domCondition)

        outputs['roi_mask'] = xndarray(np.zeros(nbv) + roiData.get_roi_id(),
                                       value_label="ROI",
                                       axes_names=['voxel'])

        mixtpB = np.zeros((roiData.nbConditions, self.nbClasses, 2))
        mixtpB[:, :, 0] = mu_Ma
        mixtpB[:, :, 1] = sigma_Ma ** 2
        mixtpP = np.zeros((roiData.nbConditions, self.nbClasses, 2))
        mixtpP[:, :, 0] = mu_Mc
        mixtpP[:, :, 1] = sigma_Mc ** 2

        an = ['condition', 'Act_class', 'component']
        ad = {'Act_class': ['inactiv', 'activ'],
              'condition': cNames,
              'component': ['mean', 'var']}
        outputs['mixt_pB'] = xndarray(mixtpB, axes_names=an, axes_domains=ad)
        outputs['mixt_pP'] = xndarray(mixtpP, axes_names=an, axes_domains=ad)
        print 'mixture params BOLD = ', mixtpB
        print 'mixture params perfusion = ', mixtpP

        outputs['labels'] = xndarray(labels, value_label="Labels",
                                     axes_names=['condition', 'class', 'voxel'])
        print outputs['labels']

        outputs['noiseVar'] = xndarray(noiseVar, value_label="noiseVar",
                                       axes_names=['voxel'])
        if self.estimateLA:
            outputs['drift_coeff'] = xndarray(L, value_label="Drift",
                                              axes_names=['coeff', 'voxel'])
            outputs['drift'] = xndarray(PL, value_label="Delta BOLD",
                                        axes_names=['time', 'voxel'])

        #######################################################################
        # CONVERGENCE
        if 1:
            axes_names = ['duration']
            outName = 'Convergence_Labels'
            ax = np.arange(self.nItMax) * cTimeMean
            ax[0:len(cTime)] = cTime
            ad = {'duration': ax}
            c = np.zeros(self.nItMax)  # -.001 #
            c[:len(cZ)] = cZ
            outputs[outName] = xndarray(c, axes_names=axes_names,
                                        axes_domains=ad,
                                        value_label='Conv_Criterion_Z')
        if 0:
            outName = 'Convergence_BRF'
            #ad = {'Conv_Criterion':np.arange(len(cH))}
            c = np.zeros(self.nItMax)   # -.001 #
            c[:len(cH)] = cH
            outputs[outName] = xndarray(c, axes_names=axes_names,
                                        axes_domains=ad,
                                        value_label='Conv_Criterion_H')
            outName = 'Convergence_BRL'
            c = np.zeros(self.nItMax)  # -.001 #
            c[:len(cA)] = cA
            #ad = {'Conv_Criterion':np.arange(len(cA))}
            outputs[outName] = xndarray(c, axes_names=axes_names,
                                        axes_domains=ad,
                                        value_label='Conv_Criterion_A')
            outName = 'Convergence_PRF'
            #ad = {'Conv_Criterion':np.arange(len(cH))}
            c = np.zeros(self.nItMax)   # -.001 #
            c[:len(cG)] = cG
            outputs[outName] = xndarray(c, axes_names=axes_names,
                                        axes_domains=ad,
                                        value_label='Conv_Criterion_G')
            outName = 'Convergence_PRL'
            c = np.zeros(self.nItMax)  # -.001 #
            c[:len(cC)] = cC
            #ad = {'Conv_Criterion':np.arange(len(cA))}
            outputs[outName] = xndarray(c, axes_names=axes_names,
                                        axes_domains=ad,
                                        value_label='Conv_Criterion_C')

        #######################################################################
        # SIMULATION

        if self.simulation:
            labels_vem_audio = roiData.simulation[0]['labels'][0]
            labels_vem_video = roiData.simulation[0]['labels'][1]
            M = labels.shape[0]
            K = labels.shape[1]
            J = labels.shape[2]
            true_labels = np.zeros((K, J))
            # print true_labels.shape
            true_labels[0, :] = labels_vem_audio.flatten()
            true_labels[1, :] = labels_vem_video.flatten()
            #true_labels[0, :] = np.reshape(labels_vem_audio, (J))
            #true_labels[1, :] = np.reshape(labels_vem_video, (J))
            newlabels = np.reshape(labels[:, 1, :], (M, J))

            #true_labels = roiData.simulation[0]['labels']
            #newlabels = labels

            se = []
            sp = []
            size = np.prod(labels.shape)
            for i in xrange(0, 2):  # (0, M):
                se0, sp0, auc = roc_curve(newlabels[i, :].tolist(),
                                          true_labels[i, :].tolist())
                se.append(se0)
                sp.append(sp0)
                size = min(size, len(sp0))
            SE = np.zeros((M, size), dtype=float)
            SP = np.zeros((M, size), dtype=float)
            for i in xrange(0, 2):  # M):
                tmp = np.array(se[i])
                SE[i, :] = tmp[0:size]
                tmp = np.array(sp[i])
                SP[i, :] = tmp[0:size]
            sensData, specData = SE, SP
            axes_names = ['1-specificity', 'condition']
            outName = 'ROC_audio'
            #ad = {'1-specificity': specData[0], 'condition': cNames}
            outputs[outName] = xndarray(sensData, axes_names=axes_names,
                                        #axes_domains=ad,
                                        value_label='sensitivity')

            m = specData[0].min()
            import matplotlib.font_manager as fm
            import matplotlib.pyplot as plt
            plt.figure(200)
            plt.plot(sensData[0], specData[0], '--', color='k', linewidth=2.0,
                     label='m=1')
            plt.hold(True)
            plt.plot(sensData[1], specData[1], color='k', linewidth=2.0,
                     label='m=2')
            # legend(('audio','video'))
            plt.xticks(color='k', size=14, fontweight='bold')
            plt.yticks(color='k', size=14, fontweight='bold')
            #xlabel('1 - Specificity',fontsize=16,fontweight='bold')
            # ylabel('Sensitivity',fontsize=16,fontweight='bold')
            prop = fm.FontProperties(size=14, weight='bold')
            plt.legend(loc=1, prop=prop)
            plt.axis([0., 1., m, 1.02])

            true_labels = roiData.simulation[0]['labels']
            true_brls = roiData.simulation[0]['brls']
            true_prls = roiData.simulation[0]['prls']
            true_brf = roiData.simulation[0]['brf'][:, 0]
            true_prf = roiData.simulation[0]['prf'][:, 0]
            true_drift = roiData.simulation[0]['drift']
            true_noise = roiData.simulation[0]['noise']

            self.finalizeEstimation(true_labels, newlabels, nvox,
                                    true_brf, estimated_brf,
                                    true_prf, estimated_prf,
                                    true_brls, brls.T,
                                    true_prls, prls.T,
                                    true_drift, PL, L, true_noise, noiseVar)

        # END SIMULATION
        #######################################################################
        d = {'parcel_size': np.array([nvox])}
        outputs['analysis_duration'] = xndarray(np.array(
                                                [self.analysis_duration]),
                                                axes_names=['parcel_size'],
                                                axes_domains=d)

        return outputs

    def finalizeEstimation(self, true_labels, labels, nvox,
                           true_brf, estimated_brf, true_prf, estimated_prf,
                           true_brls, brls, true_prls, prls, true_drift, PL, L,
                           true_noise, noise):
        msg = []
        tol = .1
        fv = PL
        tv = true_drift
        delta = np.abs((fv - tv) / np.maximum(tv, fv))
        crit = (delta > tol).any()
        if crit:
            m = "Final value of %s is not close to " \
                "true value (mean delta=%f).\n" \
                " Final value:\n %s\n True value:\n %s\n" \
                % ('drift', delta.mean(), str(fv), str(tv))
            msg.append(m)

        delta = (((estimated_prf - true_prf) ** 2).sum() /
                 (true_prf ** 2).sum()) ** .5
        tol = 0.05
        crit = (delta > tol).any()
        if crit:
            m = "Final value of %s is not close to " \
                "true value (mean delta=%f).\n" \
                " Final value:\n %s\n True value:\n %s\n" \
                % ('prf', delta.mean(), str(estimated_prf), str(true_prf))
            msg.append(m)

        delta = (((estimated_brf - true_brf) ** 2).sum() /
                 (true_brf ** 2).sum()) ** .5
        tol = 0.05
        crit = (delta > tol).any()
        if crit:
            m = "Final value of %s is not close to " \
                "true value (mean delta=%f).\n" \
                " Final value:\n %s\n True value:\n %s\n" \
                % ('brf', delta.mean(), str(estimated_brf), str(true_brf))
            msg.append(m)

        tol = .1
        # print labels
        # print true_labels
        delta = ((labels != true_labels) * np.ones(labels.shape)).sum() / nvox
        crit = (delta > tol).any()
        if crit:
            m = "Final value of %s is not close to " \
                "true value (mean delta=%f).\n" \
                " Final value:\n %s\n True value:\n %s\n" \
                % ('labels', delta.mean(), str(labels), str(true_labels))
            msg.append(m)

        delta = np.abs((brls - true_brls) / np.maximum(true_brls, brls))
        crit = (delta > tol).any()
        if crit:
            m = "Final value of %s is not close to " \
                "true value (mean delta=%f).\n" \
                " Final value:\n %s\n True value:\n %s\n" \
                % ('brls', delta.mean(), str(brls), str(true_brls))
            msg.append(m)

        delta = np.abs((prls - true_prls) / np.maximum(true_prls, prls))
        crit = (delta > tol).any()
        if crit:
            m = "Final value of %s is not close to " \
                "true value (mean delta=%f).\n" \
                " Final value:\n %s\n True value:\n %s\n" \
                % ('prls', delta.mean(), str(prls), str(true_prls))
            msg.append(m)

        true_vnoise = np.var(true_noise, 0)
        delta = np.abs((noise - true_vnoise) / np.maximum(true_vnoise, noise))
        crit = (delta > tol).any()
        if crit:
            m = "Final value of %s is not close to " \
                "true value (mean delta=%f).\n" \
                " Final value:\n %s\n True value:\n %s\n" \
                % ('noise', delta.mean(), str(noise), str(true_vnoise))
            msg.append(m)
        print "\n".join(msg)


# Function to use directly in parallel computation
def run_analysis(**params):
    # from pyhrf.ui.vb_jde_analyser import JDEVEMAnalyser
    # import pyhrf
    # pyhrf.verbose.set_verbosity(1)
    # pyhrf.logger.setLevel(logging.INFO)
    fdata = params.pop('roi_data')
    # print 'doing params:'
    # print params
    vem_analyser = JDEVEMAnalyser(**params)
    return (dict([('ROI', fdata.get_roi_id())] + params.items()),
            vem_analyser.analyse_roi(fdata))
