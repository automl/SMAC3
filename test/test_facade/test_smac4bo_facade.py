import contextlib
import shutil
import unittest.mock

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace

from smac.facade.smac_bo_facade import SMAC4BO
from smac.initial_design.sobol_design import SobolDesign
from smac.scenario.scenario import Scenario


class TestSMACFacade(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs, 'run_obj': 'quality',
                                  'output_dir': ''})
        self.output_dirs = []

    def tearDown(self):
        for i in range(20):
            with contextlib.suppress(Exception):
                dirname = 'run_1' + ('.OLD' * i)
                shutil.rmtree(dirname)
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)

    def test_exchange_sobol_for_lhd(self):
        cs = ConfigurationSpace()
        for i in range(40):
            cs.add_hyperparameter(UniformFloatHyperparameter('x%d' % (i + 1), 0, 1))
        scenario = Scenario({'cs': cs, 'run_obj': 'quality'})
        facade = SMAC4BO(scenario=scenario)
        self.assertIsInstance(facade.solver.initial_design, SobolDesign)
        cs.add_hyperparameter(UniformFloatHyperparameter('x41', 0, 1))

        with self.assertRaisesRegex(
            ValueError,
            'Sobol sequence" can only handle up to 40 dimensions. Please use a different initial design, such as '
            '"the Latin Hypercube design"',
        ):
            SMAC4BO(scenario=scenario)
        self.output_dirs.append(scenario.output_dir)
