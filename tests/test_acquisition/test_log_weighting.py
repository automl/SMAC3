"""Weighting a logarithmic acquisition function.

The linear form and the log form agree exactly wherever the linear one can be computed at all, and the log
form keeps working where it cannot. That pair is the whole claim, and the second half is the reason the path
exists: a belief sharp enough drives $a(x) \\cdot w(x)$ to exactly zero everywhere in floating point, and a
maximizer handed a flat zero has nothing to climb.
"""

from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac.acquisition.function.abstract_acquisition_function import AcquisitionScale
from smac.acquisition.function.expected_improvement import EI
from smac.acquisition.function.log_expected_improvement import LogEI
from smac.acquisition.function.weighted_acquisition_function import (
    WeightedAcquisitionFunction,
)
from smac.acquisition.weight.composite import CompositeWeight
from smac.acquisition.weight.prior import ConfigSpacePrior, PriorWeight
from smac.utils.configspace import create_prior_configspace_copy

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class _Model:
    """A surrogate whose prediction is the same everywhere, so only the weight moves the values."""

    def __init__(self, configspace, mean: float, variance: float = 0.01) -> None:
        self._configspace = configspace
        self._mean = mean
        self._variance = variance

    @property
    def meta(self):
        return {"name": "_Model"}

    def predict_marginalized(self, X):
        n = X.shape[0]

        return np.full((n, 1), self._mean), np.full((n, 1), self._variance)


@pytest.fixture
def configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)
    cs.add(Float("x", (0.0, 1.0)))

    return cs


X = np.linspace(0.0, 1.0, 9).reshape(-1, 1)


def _weighted(acquisition, configspace, centre=0.7, denominator=4.0, mean=0.0, eta=1.0):
    prior = ConfigSpacePrior(create_prior_configspace_copy(configspace, {"x": centre}, std_denominator=denominator))
    weighted = WeightedAcquisitionFunction(acquisition, [PriorWeight(prior)])
    weighted.update(model=_Model(configspace, mean), eta=eta, num_data=10)

    return weighted._compute(X).ravel()


def test_the_two_forms_rank_identically_where_both_are_computable(configspace):
    """A broad belief and a surrogate near the incumbent - nothing underflows, so the log of the product and
    the sum of the logs must order the candidates the same way."""
    linear = _weighted(EI(), configspace, denominator=4.0)
    log = _weighted(LogEI(), configspace, denominator=4.0)

    assert np.array_equal(np.argsort(linear), np.argsort(log))


def test_the_log_form_survives_a_belief_that_flattens_the_linear_one(configspace):
    """The regime the log path exists for, and the one a graphical prior editor produces by default: a user
    drags a narrow, confident peak, and the product underflows across the whole space."""
    linear = _weighted(EI(), configspace, denominator=50.0, mean=5.0)
    log = _weighted(LogEI(), configspace, denominator=50.0, mean=5.0)

    assert np.all(linear == 0.0), "the linear form is expected to die here; that is the point"
    assert len(np.unique(linear)) == 1, "a flat surface - the maximizer has nothing to climb"

    assert np.all(np.isfinite(log))
    assert len(np.unique(log)) > 1, "still ordered"
    assert X.ravel()[int(np.argmax(log))] == pytest.approx(0.75), "and still pointing at the belief"


def test_a_weighted_log_acquisition_function_stays_logarithmic(configspace):
    weighted = WeightedAcquisitionFunction(LogEI())

    assert weighted.value_scale is AcquisitionScale.LOG


def test_an_inactive_weight_leaves_a_log_acquisition_function_alone(configspace):
    """Identity differs by space - one multiplies, zero adds - so a weight with nothing to say must add
    nothing rather than multiply by one."""
    acquisition = LogEI()
    model = _Model(configspace, 0.0)

    bare = WeightedAcquisitionFunction(acquisition)
    bare.update(model=model, eta=1.0, num_data=10)

    plain = acquisition
    plain.update(model=model, eta=1.0, num_data=10)

    assert bare._compute(X) == pytest.approx(plain._compute(X))


# ── the combination rules under a logarithm ──────────────────────────────────


def _composite(values, combination):
    from smac.acquisition.weight.abstract_weight import AbstractAcquisitionWeight

    class _Fixed(AbstractAcquisitionWeight):
        def __init__(self, value):
            super().__init__(floor=0.0)
            self._value = value

        def _compute(self, X):
            return np.full((X.shape[0], 1), self._value)

    return CompositeWeight([_Fixed(v) for v in values], combination=combination)


@pytest.mark.parametrize("combination", ["sum", "product", "max"])
def test_each_combination_agrees_with_its_own_logarithm(combination):
    """Every rule has exactly one image under the logarithm, and `product` is the one worth pinning: a product
    of weights becomes a *sum* of logarithms, so a composite that multiplies and one that adds swap places.
    Reusing the linear branch names in log space would give every ensemble the wrong rule, silently."""
    composite = _composite([0.25, 0.5], combination)
    points = np.zeros((3, 1))

    linear = composite(points, log=False)
    log = composite(points, log=True)

    assert log == pytest.approx(np.log(linear))


def test_an_empty_composite_is_the_identity_in_both_spaces():
    composite = _composite([], "sum")
    points = np.zeros((3, 1))

    assert composite(points, log=False) == pytest.approx(np.ones((3, 1)))
    assert composite(points, log=True) == pytest.approx(np.zeros((3, 1)))
