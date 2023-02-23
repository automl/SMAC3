import glob
import os
import shutil
import unittest
from contextlib import suppress
from unittest.mock import patch

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
        base_directory = os.path.split(__file__)[0]
        base_directory = os.path.abspath(os.path.join(base_directory, "../../tests", ".."))
        os.chdir(base_directory)
        self.output_dirs = []
        fn = "tests/test_files/spear_hydra_test_scenario.txt"
        fn = "tests/test_files/test_deterministic_scenario.txt"
        self.scenario = Scenario(fn)
        self.scenario.limit_resources = True

    @patch("smac.facade.smac_ac_facade.SMBO", new=MockSMBO)
    def test_psmac(self):
        import joblib

        from smac.facade.psmac_facade import PSMAC
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.facade.smac_bb_facade import SMAC4BB
        from smac.facade.smac_hpo_facade import SMAC4HPO
        from smac.facade.smac_mf_facade import SMAC4MF

        facades = [None, SMAC4AC, SMAC4BB, SMAC4HPO, SMAC4MF]
        n_workers_list = [1, 2, 3, 4]
        n_facades = len(facades)
        target = {"x1": 7.290709845323256, "x2": 10.285684762665337}
        for i, facade in enumerate(facades):
            for j, n_workers in enumerate(n_workers_list):
                idx = n_facades * i + j
                with self.subTest(i=idx):
                    with joblib.parallel_backend("multiprocessing", n_jobs=1):
                        optimizer = PSMAC(self.scenario, facade_class=facade, n_workers=n_workers, validate=False)
                        inc = optimizer.optimize()
                        self.assertDictEqual(target, dict(inc))

    def tearDown(self):
        hydras = glob.glob1(".", "psmac*")
        for folder in hydras:
            shutil.rmtree(folder, ignore_errors=True)
        for i in range(20):
            with suppress(Exception):
                dirname = "run_1" + (".OLD" * i)
                shutil.rmtree(dirname)
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
