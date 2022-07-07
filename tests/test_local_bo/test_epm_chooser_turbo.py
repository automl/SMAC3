import unittest

import numpy as np

from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.configuration_chooser.turbo_chooser import TurBOChooser
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.tae import StatusType
from smac.utils import test_helpers


class TestEPMChooserTuRBO(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(
            {"cs": test_helpers.get_branin_config_space(), "run_obj": "quality", "output_dir": "data-test_epmchooser"}
        )
        self.output_dirs = []
        self.output_dirs.append(self.scenario.output_dir)

    def test_choose_next(self):
        config = self.scenario.cs.sample_configuration()
        rh = RunHistory()
        rh.add(config, 10, 10, StatusType.SUCCESS)
        smbo = SMAC4BB(
            scenario=self.scenario,
            rng=np.random.RandomState(42),
            model_type="gp",
            smbo_kwargs={"epm_chooser": TurBOChooser},
            initial_design_kwargs={"init_budget": 0},
            runhistory=rh,
        ).solver

        x = next(smbo.epm_chooser.choose_next()).get_array()
        self.assertEqual(x.shape, (2,))

        # remove the init configs
        smbo.epm_chooser.turbo.init_configs = []
        x = next(smbo.epm_chooser.choose_next()).get_array()

        self.assertEqual(x.shape, (2,))
