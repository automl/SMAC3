from contextlib import suppress
import shutil
import os
import glob
import unittest
from unittest.mock import patch

import numpy as np

from smac.facade.hydra_facade import Hydra, PSMAC
from smac.utils.io.output_writer import OutputWriter
from smac.scenario.scenario import Scenario


MOCKCALLS = 0


class MockPSMAC(PSMAC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ow = OutputWriter()
        self.ow.write_scenario_file(self.scenario)
        global MOCKCALLS
        MOCKCALLS += 1

    def optimize(self):
        return np.array(self.scenario.cs.sample_configuration(self.n_optimizers))

    def get_best_incumbents_ids(self, incs):
        cost_per_conf_v = cost_per_conf_e = {}
        val_ids = est_ids = list(range(len(incs)))
        global MOCKCALLS
        for inc in incs:
            # in successive runs will always be smaller -> hydra doesn't terminate early
            cost_per_conf_v[inc] = cost_per_conf_e[inc] = {inst: max(100 - MOCKCALLS,
                                                                     0) for inst in self.scenario.train_insts}
        if not self.validate:
            cost_per_conf_v = val_ids = None
        return cost_per_conf_v, val_ids, cost_per_conf_e, est_ids


class TestHydraFacade(unittest.TestCase):

    def setUp(self):
        self.output_dirs = []
        fn = os.path.join(os.path.dirname(__file__), '../test_files/spear_hydra_test_scenario.txt')
        self.scenario = Scenario(fn)

    @patch('smac.facade.hydra_facade.PSMAC', new=MockPSMAC)
    def test_hydra(self):
        optimizer = Hydra(self.scenario, n_iterations=3)
        portfolio = optimizer.optimize()
        self.assertEqual(len(portfolio), 3)

    @patch('smac.facade.hydra_facade.PSMAC', new=MockPSMAC)
    def test_hydra_mip(self):
        optimizer = Hydra(self.scenario, n_iterations=3, incs_per_round=2)
        portfolio = optimizer.optimize()
        self.assertEqual(len(portfolio), 6)
        
    def tearDown(self):
        hydras = glob.glob1('.', 'hydra*')
        for folder in hydras:
            shutil.rmtree(folder, ignore_errors=True)
        for i in range(20):
            with suppress(Exception):
                dirname = 'run_1' + ('.OLD' * i)
                shutil.rmtree(dirname)
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
