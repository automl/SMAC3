import contextlib
import shutil
import unittest.mock

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.cli.scenario import Scenario
from smac.configspace import ConfigurationSpace
from smac.model.random_model import RandomModel
from smac.facade.hyperband_facade import HB4AC
from smac.initial_design.random_configuration_design import RandomInitialDesign
from smac.intensification.hyperband import Hyperband

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestHBFacade(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({"cs": self.cs, "run_obj": "quality", "output_dir": ""})
        self.output_dirs = []

    def tearDown(self):
        for i in range(20):
            with contextlib.suppress(Exception):
                dirname = "run_1" + (".OLD" * i)
                shutil.rmtree(dirname)
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)

    def test_initializations(self):
        cs = ConfigurationSpace()
        for i in range(40):
            cs.add_hyperparameter(UniformFloatHyperparameter("x%d" % (i + 1), 0, 1))
        scenario = Scenario({"cs": cs, "run_obj": "quality"})
        hb_kwargs = {"initial_budget": 1, "max_budget": 3}
        facade = HB4AC(scenario=scenario, intensifier_kwargs=hb_kwargs)

        self.assertIsInstance(facade.solver.initial_design, RandomInitialDesign)
        self.assertIsInstance(facade.solver.epm_chooser.model, RandomModel)
        self.assertIsInstance(facade.solver.intensifier, Hyperband)
        self.assertEqual(facade.solver.intensifier.min_chall, 1)
        self.output_dirs.append(scenario.output_dir)
