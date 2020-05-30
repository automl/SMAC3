import unittest
import unittest.mock

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from smac.initial_design.sobol_design import SobolDesign


class TestSobol(unittest.TestCase):

    def test_sobol(self):
        cs = ConfigurationSpace()
        for i in range(40):
            cs.add_hyperparameter(UniformFloatHyperparameter('x%d' % (i + 1), 0, 1))

        sobol_kwargs = dict(
            rng=np.random.RandomState(1),
            traj_logger=unittest.mock.Mock(),
            ta_run_limit=1000,
            configs=None,
            n_configs_x_params=None,
            max_config_fracs=0.25,
            init_budget=1,
        )
        SobolDesign(
            cs=cs,
            **sobol_kwargs
        ).select_configurations()

        cs.add_hyperparameter(UniformFloatHyperparameter('x41', 0, 1))
        with self.assertRaisesRegex(
                Exception,
                "list index out of range",
        ):
            SobolDesign(
                cs=cs,
                **sobol_kwargs
            ).select_configurations()
