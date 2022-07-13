import shutil
import unittest
from contextlib import suppress

from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.forbidden import (
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from smac.cli.scenario import Scenario
from smac.facade.boing import SMAC4BOING
from smac.optimizer.configuration_chooser.boing_chooser import BOinGChooser


def rosenbrock_2d(x):
    x0 = x["x0"]
    x1 = x["x1"]
    return 100.0 * (x1 - x0**2.0) ** 2.0 + (1 - x0) ** 2.0


class TestSMAC4BOinGFacade(unittest.TestCase):
    def setUp(self) -> None:
        cs = ConfigurationSpace()
        x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
        x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
        x2 = CategoricalHyperparameter("x2", [0, 1], default_value=0)
        x3 = UniformFloatHyperparameter("x3", -5, 10, default_value=-4)
        cs.add_hyperparameters([x0, x1, x2, x3])
        cs.add_condition(EqualsCondition(x3, x2, 0))
        cs.add_forbidden_clause(ForbiddenAndConjunction(ForbiddenInClause(x2, [0, 1]), ForbiddenEqualsClause(x0, 0.1)))
        # Scenario object
        scenario = Scenario({"run_obj": "quality", "runcount-limit": 10, "cs": cs, "deterministic": "true"})
        self.scenario = scenario
        self.output_dirs = []

    def tearDown(self):
        shutil.rmtree("run_1", ignore_errors=True)
        for i in range(20):
            with suppress(Exception):
                dirname = "run_1" + (".OLD" * i)
                shutil.rmtree(dirname)
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)

    def test_smac4boing(self):

        smac = SMAC4BOING(
            scenario=self.scenario,
            tae_runner=rosenbrock_2d,
        )
        smac.optimize()
        self.assertIsInstance(smac.solver.epm_chooser, BOinGChooser)
        self.output_dirs.append(smac.scenario.output_dir)
