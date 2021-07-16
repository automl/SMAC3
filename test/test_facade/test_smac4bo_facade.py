import contextlib
import shutil
import unittest.mock

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import numpy as np
from smac.configspace import ConfigurationSpace

from smac.facade.smac_bo_facade import SMAC4BO
from smac.initial_design.sobol_design import SobolDesign
from smac.scenario.scenario import Scenario


class TestSMACFacade(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs, 'run_obj': 'quality',
                                  'output_dir': ''})
        self.output_dirs = []

    def tearDown(self):
        for i in range(20):
            with contextlib.suppress(Exception):
                dirname = 'run_1' + ('.OLD' * i)
                shutil.rmtree(dirname)
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
