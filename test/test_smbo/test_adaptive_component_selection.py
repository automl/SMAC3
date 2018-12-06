import unittest

import numpy as np

from smac.epm.base_epm import AbstractEPM
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.optimizer.adaptive_component_selection import AdaptiveComponentSelection
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI
from smac.scenario.scenario import Scenario
from smac.utils import test_helpers


class TestAdaptiveComponentSelection(unittest.TestCase):
    def setUp(self):
        config_space = test_helpers.get_branin_config_space()
        scenario = Scenario({
            'cs': config_space,
            'run_obj': 'quality',
            'output_dir': 'data-test_smbo',
        })

        self.acs = AdaptiveComponentSelection(
            rng=np.random.RandomState(1),
            scenario=scenario,
            config_space=config_space,
        )

    def test_comp_builder(self):
        conf = {"model": "RF", "acq_func": "EI"}
        model, acqf = self.acs._component_builder(conf)

        self.assertIsInstance(acqf, EI)
        self.assertIsInstance(model, RandomForestWithInstances)

        conf = {"model": "GP", "acq_func": "EI"}
        model, acqf = self.acs._component_builder(conf)

        self.assertIsInstance(acqf, EI)
        self.assertIsInstance(model, GaussianProcessMCMC)

    def test_smbo_cs(self):
        seed = 42
        cs = self.acs._get_acm_cs()

    def test_cs_comp_builder(self):
        cs = self.acs._get_acm_cs()
        conf = cs.sample_configuration()

        model, acqf = self.acs._component_builder(conf)
        self.assertIsInstance(acqf, AbstractAcquisitionFunction)
        self.assertIsInstance(model, AbstractEPM)
