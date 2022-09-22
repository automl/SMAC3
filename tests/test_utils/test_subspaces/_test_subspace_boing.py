import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from gpytorch.kernels import MaternKernel, ScaleKernel

from smac.acquisition import LocalAndSortedRandomSearch
from smac.model.gaussian_process.gpytorch_gaussian_process import (
    GloballyAugmentedLocalGaussianProcess,
)
from smac.model.utils import get_types
from smac.utils.subspaces import ChallengerListLocal
from smac.utils.subspaces.boing_subspace import BOinGSubspace


def generate_data(num_data, rs: np.random.RandomState):
    x = rs.rand(num_data, 1)
    y = rs.rand(num_data)
    return x, y


@pytest.fixture
def get_ss_kwargs():
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("x0", 0, 1, 0.5))
    model_local = GloballyAugmentedLocalGaussianProcess
    cont_dims = [0]
    exp_kernel = MaternKernel(2.5, ard_num_dims=1, active_dims=tuple(cont_dims)).double()

    kernel = ScaleKernel(exp_kernel)
    model_local_kwargs = {"kernel": kernel}
    types, bounds = get_types(configspace)
    ss_kwargs = dict(
        config_space=configspace,
        bounds=bounds,
        hps_types=types,
        model_local=model_local,
        model_local_kwargs=model_local_kwargs,
    )
    return ss_kwargs


def test_init(get_ss_kwargs):
    boing_ss_1 = BOinGSubspace(**get_ss_kwargs)
    assert boing_ss_1.model.num_inducing_points == 2
    assert isinstance(boing_ss_1.acq_optimizer_local, LocalAndSortedRandomSearch)
    assert boing_ss_1.acq_optimizer_local.local_search_iterations == 10
    assert boing_ss_1.acq_optimizer_local.local_search.n_steps_plateau_walk == 5

    acq_optimiozer = LocalAndSortedRandomSearch(
        acquisition_function=None,
        configspace=get_ss_kwargs["config_space"],
        n_steps_plateau_walk=10,
        local_search_iterations=10,
    )

    boing_ss_2 = BOinGSubspace(**get_ss_kwargs, acq_optimizer_local=acq_optimiozer)
    assert boing_ss_2.acq_optimizer_local.local_search_iterations == 10
    assert boing_ss_2.acq_optimizer_local.local_search.n_steps_plateau_walk == 10


def test_generate_challangers(get_ss_kwargs):
    rs = np.random.RandomState(1)
    init_data = generate_data(10, rs)
    boing_ss = BOinGSubspace(**get_ss_kwargs, initial_data=init_data)
    challenge = boing_ss.generate_challengers()
    assert isinstance(challenge, ChallengerListLocal)
    eval_next = next(challenge)
    acq_value_challenge = boing_ss.acquisition_function([eval_next])
    acq_value_init_points = boing_ss.acquisition_function._compute(init_data[0])
    for acq_init in acq_value_init_points:
        assert acq_init < acq_value_challenge
