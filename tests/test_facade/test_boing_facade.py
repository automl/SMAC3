import unittest

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause

from smac.facade.experimental.smac_boing_facade import SMAC4BOING
from smac.optimizer.local_bo.epm_chooser_boing import EPMChooserBOinG
from smac.scenario.scenario import Scenario


def rosenbrock_2d(x):
    x0 = x['x0']
    x1 = x['x1']
    return 100. * (x1 - x0 ** 2.) ** 2. + (1 - x0) ** 2.


class TestSMAC4BOinGFacade(unittest.TestCase):
    def test_smac4boing(self):
        cs = ConfigurationSpace()
        x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
        x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
        x2 = CategoricalHyperparameter("x2", [0, 1], default_value=0)
        x3 = UniformFloatHyperparameter("x3", -5, 10, default_value=-4)
        cs.add_hyperparameters([x0, x1, x2, x3])
        cs.add_condition(EqualsCondition(x3, x2, 0))
        cs.add_forbidden_clause(ForbiddenAndConjunction(
            ForbiddenInClause(x2, [0, 1]),
            ForbiddenEqualsClause(x0, 0.1)
        )
        )
        # Scenario object
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": 10,
                             "cs": cs,
                             "deterministic": "true"
                             })

        smac = SMAC4BOING(scenario=scenario,
                          tae_runner=rosenbrock_2d,
                          )
        smac.optimize()
        self.assertIsInstance(smac.solver.epm_chooser, EPMChooserBOinG)
