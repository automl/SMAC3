from functools import partial

import numpy as np
import pytest
import scipy.optimize

from smac.constants import VERY_SMALL_NUMBER
from smac.model.gaussian_process.priors import (
    GammaPrior,
    HorseshoePrior,
    LogNormalPrior,
    SoftTopHatPrior,
    TophatPrior,
)

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def wrap_ln(theta, prior):
    return np.exp(prior.get_log_probability(np.log(theta)))


def wrap_lnprob(x, prior):
    return prior.get_log_probability(x[0])


def wrap_gradient(x, prior):
    return prior.get_gradient(x[0])


def test_lnprob_and_grad_scalar():
    prior = TophatPrior(
        lower_bound=np.exp(-10),
        upper_bound=np.exp(2),
        seed=1,
    )

    # Legal scalar
    for val in (-1, 0, 1):
        assert prior.get_log_probability(val) == 0
        assert prior.get_gradient(val) == 0

    # Boundary
    for val in (-10, 2):
        assert prior.get_log_probability(val) == 0
        assert prior.get_gradient(val) == 0

    # Values outside the boundary
    for val in (-10 - VERY_SMALL_NUMBER, 2 + VERY_SMALL_NUMBER, -50, 50):
        assert np.isinf(prior.get_log_probability(val))
        assert prior.get_gradient(val) == 0


def test_sample_from_prior():
    prior = TophatPrior(lower_bound=np.exp(-10), upper_bound=np.exp(2), seed=1)
    samples = prior.sample_from_prior(10)
    np.testing.assert_array_equal(samples >= -10, True)
    np.testing.assert_array_equal(samples <= 2, True)
    # Test that the rng is set
    assert pytest.approx(samples[0]) == -4.995735943569112


def test_sample_from_prior_shapes():
    rng = np.random.RandomState(1)
    lower_bound = 2 + rng.random_sample() * 50
    upper_bound = lower_bound + rng.random_sample() * 50
    prior = TophatPrior(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        seed=1,
    )
    sample = prior.sample_from_prior(1)
    assert sample.shape == (1,)
    sample = prior.sample_from_prior(2)
    assert sample.shape == (2,)
    sample = prior.sample_from_prior(10)
    assert sample.shape == (10,)

    with pytest.raises(ValueError):
        prior.sample_from_prior(0)

    with pytest.raises(ValueError):
        prior.sample_from_prior((2,))


def test_lnprob_and_grad_scalar():
    prior = HorseshoePrior(scale=1, seed=1)

    # Legal scalar
    assert prior.get_log_probability(-1) == 1.1450937952919953
    assert prior.get_gradient(-1) == -0.6089187456211098

    # Boundary
    assert np.isinf(prior._get_log_probability(0))
    assert np.isinf(prior._get_gradient(0))


def test_sample_from_prior():
    prior = HorseshoePrior(scale=1, seed=1)
    samples = prior.sample_from_prior(10)

    # Test that the rng is set
    assert pytest.approx(samples[0]) == 1.0723988839129437


def test_sample_from_prior_shapes():
    rng = np.random.RandomState(1)
    lower_bound = 2 + rng.random_sample() * 50
    upper_bound = lower_bound + rng.random_sample() * 50
    prior = TophatPrior(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        seed=1,
    )
    sample = prior.sample_from_prior(1)
    assert sample.shape == (1,)

    sample = prior.sample_from_prior(2)
    assert sample.shape == (2,)

    sample = prior.sample_from_prior(10)
    assert sample.shape == (10,)

    with pytest.raises(ValueError):
        prior.sample_from_prior(0)

    with pytest.raises(ValueError):
        prior.sample_from_prior((2,))


def test_gradient():
    for scale in (0.1, 0.5, 1.0, 2.0):
        prior = HorseshoePrior(scale=scale, seed=1)
        # The function appears to be unstable above 15
        for theta in range(-20, 15):
            if theta == 0:
                continue

            wrap_lnprob_ = partial(wrap_lnprob, prior=prior)
            wrap_gradient_ = partial(wrap_gradient, prior=prior)

            error = scipy.optimize.check_grad(
                wrap_lnprob_,
                wrap_gradient_,
                np.array([theta]),
                epsilon=1e-5,
            )
            assert pytest.approx(error) == 0


def test_lnprob_and_grad_scalar():
    prior = GammaPrior(a=0.5, scale=1 / 2, loc=0, seed=1)

    # Legal scalar
    x = -1
    assert pytest.approx(prior.get_log_probability(x)) == -0.46155023  # 7
    assert prior.get_gradient(x) == -1.2357588823428847


def test_lnprob_and_grad_array():
    prior = GammaPrior(a=0.5, scale=1 / 2, loc=0, seed=1)
    val = np.array([-1, -1])

    with pytest.raises(NotImplementedError):
        prior.get_log_probability(val)

    with pytest.raises(NotImplementedError):
        prior.get_gradient(val)


def test_gradient():
    for scale in (0.5, 1.0, 2.0):
        prior = GammaPrior(a=2, scale=scale, loc=0, seed=1)
        # The function appears to be unstable above 10
        for theta in np.arange(1e-15, 10, 0.01):
            if theta == 0:
                continue

            wrap_lnprob_ = partial(wrap_lnprob, prior=prior)
            wrap_gradient_ = partial(wrap_gradient, prior=prior)

            error = scipy.optimize.check_grad(
                wrap_lnprob_,
                wrap_gradient_,
                np.array([theta]),
                epsilon=1e-5,
            )

            assert pytest.approx(error) == 0  # , delta=5, msg=str(theta))


def test_gradient():
    for sigma in (0.5, 1.0, 2.0):
        prior = LogNormalPrior(mean=0, sigma=sigma, seed=1)
        # The function appears to be unstable above 15
        for theta in range(0, 15):
            # Gradient approximation becomes unstable when going closer to zero
            theta += 1e-2

            wrap_lnprob_ = partial(wrap_lnprob, prior=prior)
            wrap_gradient_ = partial(wrap_gradient, prior=prior)

            error = scipy.optimize.check_grad(
                wrap_lnprob_,
                wrap_gradient_,
                np.array([theta]),
                epsilon=1e-5,
            )
            assert round(error) == 0  # , delta=5, msg=theta)


def test_lnprob():
    prior = SoftTopHatPrior(
        lower_bound=np.exp(-5),
        upper_bound=np.exp(5),
        exponent=2,
        seed=1,
    )

    # Legal values
    assert prior.get_log_probability(-5) == 0
    assert prior.get_log_probability(0) == 0
    assert prior.get_log_probability(5) == 0

    # Illegal values
    assert pytest.approx(prior.get_log_probability(-5.1)) == -0.01
    assert pytest.approx(prior.get_log_probability(-6)) == -1
    assert pytest.approx(prior.get_log_probability(-7)) == -4
    assert pytest.approx(prior.get_log_probability(5.1)) == -0.01
    assert pytest.approx(prior.get_log_probability(6)) == -1
    assert pytest.approx(prior.get_log_probability(7)) == -4


def test_grad():
    prior = SoftTopHatPrior(
        lower_bound=np.exp(-5),
        upper_bound=np.exp(5),
        exponent=2,
        seed=1,
    )

    # Legal values
    assert prior.get_gradient(-5) == 0
    assert prior.get_gradient(0) == 0
    assert prior.get_gradient(5) == 0

    for theta in [-10, -7, -6, -5.1, 5.1, 6, 7, 10]:
        # Gradient approximation becomes unstable when going closer to zero
        theta += 1e-2
        grad = prior.get_gradient(theta)
        grad_vector = prior.get_gradient(theta)
        assert grad == grad_vector

        def prob(x):
            return prior.get_log_probability(x[0])

        def grad(x):
            return prior.get_gradient(x[0])

        error = scipy.optimize.check_grad(prob, grad, np.array([theta]), epsilon=1e-5)
        assert np.round(error) == 0
