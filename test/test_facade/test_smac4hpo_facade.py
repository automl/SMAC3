import unittest
import unittest.mock

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace

from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.sobol_design import SobolDesign
from smac.scenario.scenario import Scenario


class TestSMACFacade(unittest.TestCase):

    def test_exchange_sobol_for_lhd(self):
        cs = ConfigurationSpace()
        for i in range(40):
            cs.add_hyperparameter(UniformFloatHyperparameter('x%d' % (i + 1), 0, 1))
        scenario = Scenario({'cs': cs, 'run_obj': 'quality'})
        facade = SMAC4HPO(scenario=scenario)
        self.assertIsInstance(facade.solver.initial_design, SobolDesign)
        cs.add_hyperparameter(UniformFloatHyperparameter('x41', 0, 1))
        with self.assertRaisesRegex(
                ValueError,
                'Sobol sequence" can only handle up to 40 dimensions. Please use a different initial design, such as '
                '"the Latin Hypercube design"',
        ):
            SMAC4HPO(scenario=scenario)
