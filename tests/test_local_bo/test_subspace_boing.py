import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from gpytorch.kernels import MaternKernel, ScaleKernel

from smac.epm.gp.augmented import GloballyAugmentedLocalGP
from smac.epm.util_funcs import get_types
from smac.optimizer.acquisition import EI
from smac.optimizer.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.optimizer.subspaces import ChallengerListLocal
from smac.optimizer.subspaces.boing_subspace import BOinGSubspace


def generate_data(num_data, rs: np.random.RandomState):
    x = rs.rand(num_data, 1)
    y = rs.rand(num_data)
    return x, y


class TestBOinGSubspace(unittest.TestCase):
    def setUp(self) -> None:
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameter(UniformFloatHyperparameter("x0", 0, 1, 0.5))

        self.model_local = GloballyAugmentedLocalGP
        cont_dims = [0]
        exp_kernel = MaternKernel(2.5, ard_num_dims=1, active_dims=tuple(cont_dims)).double()

        kernel = ScaleKernel(exp_kernel)
        self.model_local_kwargs = {"kernel": kernel}
        self.types, self.bounds = get_types(self.cs)
        self.acq_local = EI
        self.ss_kwargs = dict(
            config_space=self.cs,
            bounds=self.bounds,
            hps_types=self.types,
            model_local=self.model_local,
            model_local_kwargs=self.model_local_kwargs,
        )

    def test_init(self):
        boing_ss_1 = BOinGSubspace(**self.ss_kwargs)
        self.assertEqual(boing_ss_1.model.num_inducing_points, 2)
        self.assertIsInstance(boing_ss_1.acq_optimizer_local, LocalAndSortedRandomSearch)
        self.assertEqual(boing_ss_1.acq_optimizer_local.n_sls_iterations, 10)
        self.assertEqual(boing_ss_1.acq_optimizer_local.local_search.n_steps_plateau_walk, 5)

        acq_optimiozer = LocalAndSortedRandomSearch(
            acquisition_function=None, config_space=self.cs, n_steps_plateau_walk=10, n_sls_iterations=10
        )

        boing_ss_2 = BOinGSubspace(**self.ss_kwargs, acq_optimizer_local=acq_optimiozer)
        self.assertEqual(boing_ss_2.acq_optimizer_local.n_sls_iterations, 10)
        self.assertEqual(boing_ss_2.acq_optimizer_local.local_search.n_steps_plateau_walk, 10)

    def test_generate_challangers(self):
        rs = np.random.RandomState(1)
        init_data = generate_data(10, rs)
        boing_ss = BOinGSubspace(**self.ss_kwargs, initial_data=init_data)
        challenge = boing_ss.generate_challengers()
        self.assertIsInstance(challenge, ChallengerListLocal)
        eval_next = next(challenge)
        acq_value_challenge = boing_ss.acquisition_function([eval_next])
        acq_value_init_points = boing_ss.acquisition_function._compute(init_data[0])
        for acq_init in acq_value_init_points:
            self.assertLess(acq_init, acq_value_challenge)
