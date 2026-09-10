from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac.multi_objective.parego import ParEGO
from smac.scenario import Scenario

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


@pytest.fixture
def scenario() -> Scenario:
    cs = ConfigurationSpace(seed=0)
    cs.add(Float("x", (0, 1)))

    return Scenario(cs, objectives=["cost1", "cost2"], seed=0)


def test_reweigh_default_resamples_every_call(scenario):
    """With the default reweigh=1, every call to update_on_iteration_start resamples theta."""
    parego = ParEGO(scenario, seed=0)

    thetas = []
    for _ in range(5):
        parego.update_on_iteration_start()
        thetas.append(parego._theta.copy())

    assert not all(np.array_equal(thetas[0], theta) for theta in thetas[1:])


def test_reweigh_keeps_previous_theta_in_between(scenario):
    """With reweigh=3, theta should only change on calls 0, 3, 6, ... (0-indexed) and stay constant in between."""
    parego = ParEGO(scenario, seed=0, reweigh=3)

    thetas = []
    for _ in range(9):
        parego.update_on_iteration_start()
        thetas.append(parego._theta.copy())

    assert np.array_equal(thetas[0], thetas[1])
    assert np.array_equal(thetas[1], thetas[2])
    assert not np.array_equal(thetas[2], thetas[3])

    assert np.array_equal(thetas[3], thetas[4])
    assert np.array_equal(thetas[4], thetas[5])
    assert not np.array_equal(thetas[5], thetas[6])


def test_reweigh_first_call_always_sets_theta(scenario):
    """theta must be set after the very first call, regardless of reweigh."""
    parego = ParEGO(scenario, seed=0, reweigh=10)
    assert parego._theta is None

    parego.update_on_iteration_start()
    assert parego._theta is not None


def test_reweigh_in_meta(scenario):
    parego = ParEGO(scenario, seed=0, reweigh=5)
    assert parego.meta["reweigh"] == 5


def test_reweigh_default_in_meta(scenario):
    parego = ParEGO(scenario, seed=0)
    assert parego.meta["reweigh"] == 1


@pytest.mark.parametrize("reweigh", [0, -1])
def test_invalid_reweigh_raises(scenario, reweigh):
    with pytest.raises(ValueError):
        ParEGO(scenario, reweigh=reweigh)
