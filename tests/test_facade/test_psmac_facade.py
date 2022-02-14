from contextlib import suppress
import shutil
import os
import glob
import joblib
import unittest
from unittest.mock import patch

from smac.facade.experimental.psmac_facade import PSMAC
from smac.optimizer.smbo import SMBO
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class MockSMBO(SMBO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats.start_timing()

    def run(self):  # mock call such that we don't have to test with real algorithm
        return self.config_space.sample_configuration()


class TestPSMACFacade(unittest.TestCase):

    def setUp(self):
        self.output_dirs = []
        fn = os.path.join(os.path.dirname(__file__), '../test_files/spear_hydra_test_scenario.txt')
        self.scenario = Scenario(fn)

    @patch('smac.facade.smac_ac_facade.SMBO', new=MockSMBO)
    def test_psmac(self):
        with joblib.parallel_backend('multiprocessing', n_jobs=1):
            optimizer = PSMAC(self.scenario, n_optimizers=3, n_incs=2, validate=False)
            incs = optimizer.optimize()
            self.assertEqual(len(incs), 2)
            optimizer = PSMAC(self.scenario, n_optimizers=1, n_incs=4, validate=False)
            incs = optimizer.optimize()
            self.assertEqual(len(incs), 2)
            optimizer = PSMAC(self.scenario, n_optimizers=5, n_incs=4, validate=False)
            incs = optimizer.optimize()
            self.assertEqual(len(incs), 4)

    def tearDown(self):
        hydras = glob.glob1('.', 'psmac*')
        for folder in hydras:
            shutil.rmtree(folder, ignore_errors=True)
        for i in range(20):
            with suppress(Exception):
                dirname = 'run_1' + ('.OLD' * i)
                shutil.rmtree(dirname)
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
