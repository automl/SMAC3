from contextlib import suppress
import shutil
import os
import glob
import unittest
from unittest.mock import patch

import numpy as np

from smac.facade.hydra_facade import Hydra, SMAC
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory import DataOrigin
from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario


class MockSMBO(SMBO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats.start_timing()
        MockSMBO.calls = 10

    def run(self):  # mock call such that we don't have to test with real algorithm calls
        MockSMBO.calls -= 1
        return MockSMBO.calls


class MockSMAC(SMAC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate(self, **kwargs):
        # randomly populate the runhistory
        assert kwargs['config_mode'] == 'inc', "Don't waste resources on validating irrelevant configs."
        rh = RunHistory(average_cost)
        smbo_calls = self.solver.calls
        config = self.scenario.cs.sample_configuration(1)
        rh.add(config, smbo_calls, smbo_calls, 'SAT', instance_id=str(np.random.choice(['a', 'b', 'c'])),
               seed=np.random.randint(999), origin=DataOrigin.INTERNAL)
        rh.add(config, smbo_calls, smbo_calls - 1, 'SAT', instance_id=str(np.random.choice(['a', 'b', 'c'])),
               seed=np.random.randint(999), origin=DataOrigin.INTERNAL)
        rh.add(config, smbo_calls, smbo_calls - 2, 'SAT', instance_id=str(np.random.choice(['a', 'b', 'c'])),
               seed=np.random.randint(999), origin=DataOrigin.INTERNAL)
        rh.add(config, smbo_calls, smbo_calls - 1, 'SAT', instance_id=str(np.random.choice(['a', 'b', 'c'])),
               seed=np.random.randint(999), origin=DataOrigin.INTERNAL)
        rh.add(config, smbo_calls, smbo_calls - 2, 'SAT', instance_id=str(np.random.choice(['a', 'b', 'c'])),
               seed=np.random.randint(999), origin=DataOrigin.INTERNAL)
        return rh


class TestHydraFacade(unittest.TestCase):

    def setUp(self):
        self.output_dirs = []
        print(os.path.dirname(__file__))
        fn = os.path.join(os.path.dirname(__file__), '../test_files/spear_hydra_test_scenario.txt')
        self.scenario = Scenario(fn)

    @patch('smac.facade.smac_facade.SMBO', new=MockSMBO)
    @patch('smac.facade.hydra_facade.SMAC', new=MockSMAC)
    def test_hydra(self):
        optimizer = Hydra(self.scenario, n_iterations=3)
        portfolio = optimizer.optimize()
        self.assertEqual(len(portfolio), 3)
        
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
