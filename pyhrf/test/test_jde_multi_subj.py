# -*- coding: utf-8 -*-

import os
import os.path as op
import unittest
import shutil
import logging

from copy import deepcopy

import numpy as np

import pyhrf
import pyhrf.boldsynth.scenarios as sim

from pyhrf.jde.jde_multi_sujets import BOLDGibbs_Multi_SubjSampler as BMSS
from pyhrf.jde.jde_multi_sujets import simulate_single_subject
from pyhrf.jde import jde_multi_sujets as jms
from pyhrf.ui.jde import JDEMCMCAnalyser
from pyhrf.ui.treatment import FMRITreatment
from pyhrf import Condition
from pyhrf.core import FmriData, FmriGroupData


logger = logging.getLogger(__name__)


def simulate_subjects(output_dir, snr_scenario='high_snr',
                      spatial_size='tiny', hrf_group=None, nb_subjects=15,
                      vhrf=0.1, vhrf_group=0.1):
    '''
    Simulate daata for multiple subjects (5 subjects by default)
    '''
    drift_coeff_var = 1.
    drift_amplitude = 10.

    lmap1, lmap2, lmap3 = 'random_small', 'random_small', 'random_small'

    if snr_scenario == 'low_snr':  # low snr
        vars_noise = np.zeros(nb_subjects) + 1.5
        conditions = [
            Condition(name='audio', m_act=3., v_act=.3, v_inact=.3,
                      label_map=lmap1),
            Condition(name='video', m_act=2.5, v_act=.3, v_inact=.3,
                      label_map=lmap2),
            Condition(name='damier', m_act=2, v_act=.3, v_inact=.3,
                      label_map=lmap3),
        ]
    else:  # high snr

        vars_noise = np.zeros(nb_subjects) + .2
        conditions = [
            Condition(name='audio', m_act=13., v_act=.2, v_inact=.1,
                      label_map=lmap1),
            # Condition(name='video', m_act=11.5, v_act=.2, v_inact=.1,
            # label_map=lmap2),
            # Condition(name='damier', m_act=10, v_act=.2, v_inact=.1,
            # label_map=lmap3),
        ]
    vars_hrfs = np.zeros(nb_subjects) + vhrf

    # Common variable across subjects:
    labels_vol = sim.create_labels_vol(conditions)
    labels = sim.flatten_labels_vol(labels_vol)

    # use smooth multivariate gaussian prior:
    if hrf_group is None:  # simulate according to gaussian prior
        var_hrf_group = 0.1
        hrf_group = sim.create_gsmooth_hrf(dt=0.6, hrf_var=var_hrf_group,
                                           normalize_hrf=False)
        n = (hrf_group ** 2).sum() ** .5
        hrf_group /= n
        var_hrf_group /= n ** 2

    simu_subjects = []
    simus = []

    for isubj in xrange(nb_subjects):
        if output_dir is not None:
            out_dir = op.join(output_dir, 'subject_%d' % isubj)
            if not op.exists(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = None
        s = simulate_single_subject(out_dir, conditions, vars_hrfs[isubj],
                                    labels, labels_vol, vars_noise[isubj],
                                    drift_coeff_var,
                                    drift_amplitude, hrf_group, dt=0.6, dsf=4,
                                    var_hrf_group=vhrf_group)
        if 0:
            print 'simu subj %d:' % isubj
            print 'vhs:', s['var_subject_hrf']
            print 'hg:', s['hrf_group']
            print 'vhg:', s['var_hrf_group']

        simus.append(s)
        simu_subjects.append(FmriData.from_simulation_dict(s))
    simu_subjects = FmriGroupData(simu_subjects)

    return simu_subjects


class MultiSubjTest(unittest.TestCase):

    def setUp(self):

        # pyhrf.verbose.set_verbosity(0)
        pyhrf.logger.setLevel(logging.WARNING)

        np.random.seed(8652761)

        self.simu_dir = pyhrf.get_tmp_path()

        # Parameters to setup a Sampler where all samplers are OFF and
        # set to their true values.
        # This is used by the function _test_specific_samplers,
        # which will turn on specific samplers to test.

        self.sampler_params_for_single_test = {
            'nb_iterations': 40,
            'smpl_hist_pace': -1,
            'obs_hist_pace': -1,
            # HRF by subject
            'hrf_subj': jms.HRF_Sampler(do_sampling=False,
                                        normalise=1.,
                                        use_true_value=True,
                                        zero_contraint=False,
                                        prior_type='singleHRF'),

            # HRF variance
            'hrf_var_subj': jms.HRFVarianceSubjectSampler(do_sampling=False,
                                                          use_true_value=True),
            # HRF group
            'hrf_group': jms.HRF_Group_Sampler(do_sampling=False,
                                               normalise=1.,
                                               use_true_value=True,
                                               zero_contraint=False,
                                               prior_type='singleHRF'),
            # HRF variance
            'hrf_var_group': jms.RHGroupSampler(do_sampling=False,
                                                use_true_value=True),

            # neural response levels (stimulus-induced effects) by subject
            'response_levels': jms.NRLs_Sampler(do_sampling=False,
                                                use_true_value=True),
            'labels': jms.LabelSampler(do_sampling=False,
                                       use_true_value=True),
            # drift
            'drift': jms.Drift_MultiSubj_Sampler(do_sampling=False,
                                                 use_true_value=True),
            # drift variance
            'drift_var': jms.ETASampler_MultiSubj(do_sampling=False,
                                                  use_true_value=True),
            # noise variance
            'noise_var':
            jms.NoiseVariance_Drift_MultiSubj_Sampler(do_sampling=False,
                                                      use_true_value=False),
            # weights o fthe mixture
            # parameters of the mixture
            'mixt_params': jms.MixtureParamsSampler(do_sampling=False,
                                                    use_true_value=False),
            #'alpha_subj' : Alpha_hgroup_Sampler(dict_alpha_single),
            #'alpha_var_subj' : AlphaVar_Sampler(dict_alpha_var_single),
            'check_final_value': 'none',  # print or raise
        }

    def tearDown(self):
        shutil.rmtree(self.simu_dir)

    def _test_specific_samplers(self, sampler_names, simu,
                                nb_its=None, use_true_val=None,
                                save_history=False, check_fv=None,
                                normalize_hrf=1.,
                                hrf_prior_type='singleHRF',
                                normalize_hrf_group=1.,
                                hrf_group_prior_type='singleHRF',
                                reg_hgroup=True):
        """
        Test specific samplers.
        """
        if use_true_val is None:
            use_true_val = dict((n, False) for n in sampler_names)

        logger.info('_test_specific_samplers %s ...', str(sampler_names))

        params = deepcopy(self.sampler_params_for_single_test)

        # Loop over given samplers to enable them
        for var_name in sampler_names:
            var_class = params[var_name].__class__
            use_tval = use_true_val[var_name]

            # special case for HRF -> normalization and prior type
            if var_class == jms.HRF_Sampler:
                params[var_name] = jms.HRF_Sampler(do_sampling=True,
                                                   use_true_value=use_tval,
                                                   normalise=normalize_hrf,
                                                   prior_type=hrf_prior_type,
                                                   zero_contraint=False)
            elif var_class == jms.HRF_Group_Sampler:
                ptype = hrf_group_prior_type
                nhg = normalize_hrf_group
                shg = jms.HRF_Group_Sampler(do_sampling=True,
                                            use_true_value=use_tval,
                                            normalise=nhg,
                                            prior_type=ptype,
                                            zero_contraint=False,
                                            regularise=reg_hgroup)
                params[var_name] = shg
            else:
                params[var_name] = var_class(do_sampling=True,
                                             use_true_value=use_tval)

        if nb_its is not None:
            params['nb_iterations'] = nb_its

        if save_history:
            params['smpl_hist_pace'] = 1
            params['obs_hist_pace'] = 1

        if check_fv is not None:
            params['check_final_value'] = check_fv

        sampler = BMSS(**params)

        output_dir = self.simu_dir

        analyser = JDEMCMCAnalyser(sampler=sampler, osfMax=4, dtMin=.4,
                                   dt=.6, driftParam=4, driftType='polynomial',
                                   outputPrefix='jde_mcmc_',
                                   randomSeed=5421087, pass_error=False)

        treatment = FMRITreatment(fmri_data=simu, analyser=analyser,
                                  output_dir=output_dir)

        outputs = treatment.run()
        # print 'out_dir:', output_dir
        return outputs

    def test_quick(self):
        """ Test running of JDE multi subject (do not test result accuracy) """
        simu = simulate_subjects(self.simu_dir, nb_subjects=2)
        self._test_specific_samplers(['hrf_subj'], simu, nb_its=2,
                                     check_fv=None)
