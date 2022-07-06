import copy
import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import GreaterThanCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from smac.epm.gp import GaussianProcess
from smac.epm.gp.kernels import ConstantKernel, Matern, WhiteKernel
from smac.epm.util_funcs import get_types
from smac.optimizer.acquisition import TS
from smac.optimizer.subspaces.turbo_subspace import TuRBOSubSpace


class TestTurBoSubspace(unittest.TestCase):
    def setUp(self) -> None:
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameter(UniformFloatHyperparameter("x0", 0, 1, 0.5))
        self.model_local = GaussianProcess
        exp_kernel = Matern(nu=2.5)
        cov_amp = ConstantKernel(
            2.0,
        )
        noise_kernel = WhiteKernel(1e-8)
        kernel = cov_amp * exp_kernel + noise_kernel
        self.types, self.bounds = get_types(self.cs)
        self.model_local_kwargs = {"kernel": kernel}
        self.acq_local = TS
        self.ss_kwargs = dict(
            config_space=self.cs,
            bounds=self.bounds,
            hps_types=self.types,
            model_local=self.model_local,
            model_local_kwargs=self.model_local_kwargs,
        )

    def test_init(self):
        ss = TuRBOSubSpace(**self.ss_kwargs)
        self.assertEqual(len(ss.init_configs), ss.n_init)
        ss_init_configs = copy.deepcopy(ss.init_configs)
        self.assertEqual(ss.num_valid_observations, 0)

        # init configurations are poped
        for i in reversed(range(len(ss_init_configs))):
            eval_next = next(ss.generate_challengers())
            self.assertEqual(eval_next, ss_init_configs[i])

        cs_mix = ConfigurationSpace()
        cs_mix.add_hyperparameter(UniformFloatHyperparameter("x0", 0, 1, 0.5))
        cs_mix.add_hyperparameter(CategoricalHyperparameter("x1", [0, 1, 2]))

        self.assertRaisesRegex(
            ValueError,
            "Current TurBO Optimizer only supports Numerical Hyperparameters",
            TuRBOSubSpace,
            config_space=cs_mix,
            bounds=None,
            hps_types=None,
            model_local=None,
        )

        x0 = UniformFloatHyperparameter("x0", 0, 1, 0.5)
        x1 = UniformFloatHyperparameter("x1", 0, 1, 0.5)

        cs_condition = ConfigurationSpace()
        cs_condition.add_hyperparameters([x0, x1])

        cs_condition.add_condition(GreaterThanCondition(x0, x1, 0.5))
        self.assertRaisesRegex(
            ValueError,
            "Currently TurBO does not support Conditional or Forbidden Hyperparameters",
            TuRBOSubSpace,
            config_space=cs_condition,
            bounds=None,
            hps_types=None,
            model_local=None,
        )

    def test_adjust_length(self):
        ss = TuRBOSubSpace(**self.ss_kwargs)
        ss.add_new_observations(np.array([0.5]), np.array([0.5]))

        self.assertEqual(ss.num_valid_observations, 1)

        success_tol = ss.success_tol
        failure_tol = ss.failure_tol
        length = ss.length

        for i in range(success_tol):
            ss.adjust_length(0.3 - i * 0.01)
        self.assertGreater(ss.length, length)

        # make sure that length cannot be greater than length_max
        for i in range(100):
            ss.adjust_length(0.3 - i * 0.01)
        self.assertLessEqual(ss.length, ss.length_max)

        length = ss.length
        for i in range(failure_tol):
            ss.adjust_length(0.5 + i * 0.01)
        self.assertLessEqual(ss.length, length / 2)

    @unittest.mock.patch.object(GaussianProcess, "predict")
    def test_restart(self, rf_mock):
        ss = TuRBOSubSpace(**self.ss_kwargs)
        ss.add_new_observations(np.array([0.5]), np.array([0.5]))
        ss.init_configs = []

        ss.length = 0.0
        challenge = ss.generate_challengers()

        self.assertEqual(ss.length, ss.length_init)
        self.assertGreater(len(ss.init_configs), 0)

        eval_next = next(challenge)
        self.assertTrue(eval_next.origin == "TuRBO")
        self.assertEqual(rf_mock.call_count, 0)

    def test_perturb_samples(self):
        ss = TuRBOSubSpace(**self.ss_kwargs, incumbent_array=np.array([2.0]))

        prob = 0.0
        perturb_sample = ss._perturb_samples(prob, np.random.rand(ss.n_candidates, ss.n_dims))
        # make sure that no new suggestion is replaced by the incumbent
        self.assertEqual(len(np.where(perturb_sample == 2.0)[0]), 0)

        prob = 1.0
        perturb_sample = ss._perturb_samples(prob, np.random.rand(ss.n_candidates, ss.n_dims))
        self.assertEqual(len(np.where(perturb_sample == 2.0)[0]), 0)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformFloatHyperparameter("x0", 0, 1, 0.5))
        cs.add_hyperparameter(UniformFloatHyperparameter("x1", 0, 1, 0.5))
        model_local = GaussianProcess
        exp_kernel = Matern(nu=2.5)
        cov_amp = ConstantKernel(
            2.0,
        )
        noise_kernel = WhiteKernel(1e-8)
        kernel = cov_amp * exp_kernel + noise_kernel
        types, bounds = get_types(cs)
        model_local_kwargs = {"kernel": kernel}

        ss = TuRBOSubSpace(
            config_space=cs,
            bounds=bounds,
            hps_types=types,
            model_local=model_local,
            model_local_kwargs=model_local_kwargs,
            incumbent_array=np.array([2.0, 2.0]),
        )

        prob = 0.0
        perturb_sample = ss._perturb_samples(prob, np.random.rand(ss.n_candidates, ss.n_dims))

        idx_from_incumbent = np.transpose(perturb_sample == 2.0)
        self.assertTrue(np.all(np.sum(idx_from_incumbent, axis=1)) < 2)

        prob = 1.0
        perturb_sample = ss._perturb_samples(prob, np.random.rand(ss.n_candidates, ss.n_dims))

        idx_from_incumbent = np.transpose(perturb_sample == 2.0)
        self.assertEqual(len(np.where(perturb_sample == 2.0)[0]), 0)

    def test_suggestion(self):
        num_init_points = 5
        ss = TuRBOSubSpace(**self.ss_kwargs, incumbent_array=np.array([0.5]))
        ss.length = 0.1
        ss.init_configs = []
        new_data_x = np.vstack([np.random.rand(num_init_points, 1), np.array([[0.5]])])
        new_data_y = np.vstack([np.random.rand(num_init_points, 1), np.array([[-0.1]])])
        ss.add_new_observations(new_data_x, new_data_y)
        challengers = ss._generate_challengers()

        challenger_arrays = np.asarray([challenger[1].get_array() for challenger in challengers])
        # suggestions are constrained
        self.assertTrue(np.all(0.4 < challenger_arrays) and np.all(challenger_arrays < 0.6))

        challengers = ss._generate_challengers(_sorted=False)
        challenger_acq_values = np.asarray([challenger[0] for challenger in challengers])
        np.testing.assert_equal(0.0, challenger_acq_values)
