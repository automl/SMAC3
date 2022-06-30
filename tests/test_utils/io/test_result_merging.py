import json
import logging
import os
import tempfile
import unittest.mock
import json
from unittest.mock import patch

from smac.configspace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajEntry, TrajLogger

from smac.utils.io.result_merging import ResultMerger

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class ResultMergerTest(unittest.TestCase):
    def test_init_valueerror(self):
        with self.assertRaises(ValueError):
            rm = ResultMerger()

    def test_merge(self):
        outdir = "../../test_files/example_run"
        rundirs = [outdir] * 3
        rm = ResultMerger(rundirs=rundirs)
        rh = rm.get_runhistory()
        traj = rm.get_trajectory()
        traj_fn = os.path.join(outdir, "traj.json")
        with open(traj_fn, "r") as file:
            lines = file.readlines()
        traj_from_file = [json.loads(l) for l in lines]
        self.assertEqual(len(traj_from_file), len(traj))


