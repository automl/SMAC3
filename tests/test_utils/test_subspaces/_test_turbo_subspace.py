import copy
import pytest
import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import GreaterThanCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.model.gaussian_process.kernels import ConstantKernel, MaternKernel, WhiteKernel
from smac.model.utils import get_types
from smac.utils.subspaces.turbo_subspace import TuRBOSubSpace


@pytest.fixture
def get_ss_kwargs():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter("x0", 0, 1, 0.5))
    model_local = GaussianProcess
    exp_kernel = MaternKernel(nu=2.5)
    cov_amp = ConstantKernel(
        2.0,
    )
    noise_kernel = WhiteKernel(1e-8)
    kernel = cov_amp * exp_kernel + noise_kernel
    types, bounds = get_types(cs)
    model_local_kwargs = {"kernel": kernel}
    ss_kwargs = dict(
        config_space=cs,
        bounds=bounds,
        hps_types=types,
        model_local=model_local,
        model_local_kwargs=model_local_kwargs,
    )
    return ss_kwargs


def test_init(get_ss_kwargs):
    ss = TuRBOSubSpace(**get_ss_kwargs)
    assert len(ss.init_configs) == ss.n_init
    ss_init_configs = copy.deepcopy(ss.init_configs)
    assert ss.num_valid_observations == 0

    # init configurations are poped
    for i in reversed(range(len(ss_init_configs))):
        eval_next = next(ss.generate_challengers())
        assert eval_next == ss_init_configs[i]

    cs_mix = ConfigurationSpace()
    cs_mix.add_hyperparameter(UniformFloatHyperparameter("x0", 0, 1, 0.5))
    cs_mix.add_hyperparameter(CategoricalHyperparameter("x1", [0, 1, 2]))

    with pytest.raises(ValueError) as excinfo:
        TuRBOSubSpace(cs_mix, bounds=None, hps_types=None, model_local=None)
        assert excinfo == "Current TurBO Optimizer only supports Numerical Hyperparameters"

    x0 = UniformFloatHyperparameter("x0", 0, 1, 0.5)
    x1 = UniformFloatHyperparameter("x1", 0, 1, 0.5)

    cs_condition = ConfigurationSpace()
    cs_condition.add_hyperparameters([x0, x1])

    cs_condition.add_condition(GreaterThanCondition(x0, x1, 0.5))

    with pytest.raises(ValueError) as excinfo:
        TuRBOSubSpace(cs_mix, bounds=None, hps_types=None, model_local=None)
        assert excinfo == "Currently TurBO does not support Conditional or Forbidden Hyperparameters"


def test_adjust_length(get_ss_kwargs):
    ss = TuRBOSubSpace(**get_ss_kwargs)
    ss.add_new_observations(np.array([0.5]), np.array([0.5]))

    assert ss.num_valid_observations == 1

    success_tol = ss.success_tol
    failure_tol = ss.failure_tol
    length = ss.length

    for i in range(success_tol):
        ss.adjust_length(0.3 - i * 0.01)
    assert ss.length > length

    # make sure that length cannot be greater than length_max
    for i in range(100):
        ss.adjust_length(0.3 - i * 0.01)
    assert ss.length <= ss.length_max

    length = ss.length
    for i in range(failure_tol):
        ss.adjust_length(0.5 + i * 0.01)
    assert ss.length <= length / 2


@unittest.mock.patch.object(GaussianProcess, "predict")
def test_restart(rf_mock, get_ss_kwargs):
    ss = TuRBOSubSpace(**get_ss_kwargs)
    ss.add_new_observations(np.array([0.5]), np.array([0.5]))
    ss.init_configs = []

    ss.length = 0.0
    challenge = ss.generate_challengers()

    assert ss.length == ss.length_init
    assert len(ss.init_configs) > 0

    eval_next = next(challenge)
    assert eval_next.origin == "TuRBO"
    assert rf_mock.call_count == 0


def test_perturb_samples(get_ss_kwargs):
    ss = TuRBOSubSpace(**get_ss_kwargs, incumbent_array=np.array([2.0]))

    prob = 0.0
    perturb_sample = ss._perturb_samples(prob, np.random.rand(ss.n_candidates, ss.n_dims))
    # make sure that no new suggestion is replaced by the incumbent
    assert len(np.where(perturb_sample == 2.0)[0]) == 0

    prob = 1.0
    perturb_sample = ss._perturb_samples(prob, np.random.rand(ss.n_candidates, ss.n_dims))
    assert len(np.where(perturb_sample == 2.0)[0]) == 0

    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter("x0", 0, 1, 0.5))
    cs.add_hyperparameter(UniformFloatHyperparameter("x1", 0, 1, 0.5))
    model_local = GaussianProcess
    exp_kernel = MaternKernel(nu=2.5)
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
    assert np.all(np.sum(idx_from_incumbent, axis=1)) < 2

    prob = 1.0
    perturb_sample = ss._perturb_samples(prob, np.random.rand(ss.n_candidates, ss.n_dims))

    idx_from_incumbent = np.transpose(perturb_sample == 2.0)
    assert len(np.where(perturb_sample == 2.0)[0]) == 0


def test_suggestion(get_ss_kwargs):
    num_init_points = 5
    ss = TuRBOSubSpace(**get_ss_kwargs, incumbent_array=np.array([0.5]))
    ss.length = 0.1
    ss.init_configs = []
    new_data_x = np.vstack([np.random.rand(num_init_points, 1), np.array([[0.5]])])
    new_data_y = np.vstack([np.random.rand(num_init_points, 1), np.array([[-0.1]])])
    ss.add_new_observations(new_data_x, new_data_y)
    challengers = ss._generate_challengers()

    challenger_arrays = np.asarray([challenger[1].get_array() for challenger in challengers])
    # suggestions are constrained
    assert np.all(0.4 < challenger_arrays) and np.all(challenger_arrays < 0.6)

    challengers = ss._generate_challengers(_sorted=False)
    challenger_acq_values = np.asarray([challenger[0] for challenger in challengers])
    np.testing.assert_equal(0.0, challenger_acq_values)
