import json
import logging
import os
import tempfile
import unittest.mock
from unittest.mock import patch

from smac.cli.scenario import Scenario
from smac.cli.traj_logging import TrajEntry, TrajLogger
from smac.configspace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.stats import Stats
from smac.utils._result_merging import ResultMerger

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class ResultMergerTest(unittest.TestCase):
    def setUp(self) -> None:
        base_directory = os.path.split(__file__)[0]
        base_directory = os.path.abspath(os.path.join(base_directory, "../../tests", ".."))
        os.chdir(base_directory)

    def test_init_valueerror(self):
        with self.assertRaises(ValueError):
            rm = ResultMerger()

    def test_merge(self):
        print(os.getcwd())
        outdir = "test_files/example_run"
        rundirs = [outdir] * 3
        rm = ResultMerger(rundirs=rundirs)
        rh = rm.get_runhistory()
        traj = rm.get_trajectory()
        traj_fn = os.path.join(outdir, "traj.json")
        with open(traj_fn, "r") as file:
            lines = file.readlines()
        traj_from_file = [json.loads(line) for line in lines]
        self.assertEqual(len(traj_from_file), len(traj))
