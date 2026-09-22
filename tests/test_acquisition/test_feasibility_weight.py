from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float
from test_constrained_acquisition_function import FixedModel, _runhistory

from smac.acquisition.weight import NoDecay
from smac.acquisition.weight.feasibility import FeasibilityWeight
from smac.utils.constraints import (
    OutcomeConstraint,
    parse_constraints,
    probability_of_feasibility,
)

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


@pytest.fixture
def configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)
    cs.add(Float("x", (0.0, 1.0)))

    return cs


def _weight(constraint_model, objective_model=None):
    weight = FeasibilityWeight(
        constraints=parse_constraints(["latency <= 100"]),
        constraint_model=constraint_model,
    )
    if objective_model is not None:
        weight.model = objective_model

    return weight


def test_at_least_one_constraint_is_required():
    with pytest.raises(ValueError, match="at least one constraint"):
        FeasibilityWeight(constraints=[], constraint_model=FixedModel([0.0], [1.0]))


def test_the_value_is_the_probability_of_feasibility(configspace):
    constraint_model = FixedModel([90.0], [25.0])
    weight = _weight(constraint_model, FixedModel([0.5], [1.0]))
    weight.update(model=FixedModel([0.5], [1.0]), eta=1.0, num_data=1, runhistory=_runhistory(configspace, [50.0]))

    # The model predicts residuals now, so the bound is already folded in and the probability needs only the
    # prediction - which is the branching `_training_targets` removes.
    X = np.array([[0.2], [0.7]])
    expected = probability_of_feasibility(*constraint_model.predict_marginalized(X))

    assert weight(X) == pytest.approx(expected + 1e-12)


def test_the_weight_does_not_decay():
    """A bound states a fact about the problem, which does not become less true over time."""
    weight = _weight(FixedModel([90.0], [25.0]))

    assert isinstance(weight.decay, NoDecay)


def test_the_weight_stands_aside_until_something_has_been_observed(configspace):
    weight = _weight(FixedModel([90.0], [25.0]), FixedModel([0.5], [1.0]))

    assert weight.is_active() is False

    weight.update(model=FixedModel([0.5], [1.0]), eta=1.0, num_data=0, runhistory=_runhistory(configspace, [None]))
    assert weight.is_active() is False

    weight.update(model=FixedModel([0.5], [1.0]), eta=1.0, num_data=1, runhistory=_runhistory(configspace, [50.0]))
    assert weight.is_active() is True


def test_the_weight_takes_over_while_nothing_feasible_is_known(configspace):
    objective_model = FixedModel([0.5], [1.0])
    weight = _weight(FixedModel([90.0], [25.0]), objective_model)

    weight.update(model=objective_model, eta=1.0, num_data=1, runhistory=_runhistory(configspace, [150.0]))
    assert weight.suppresses_acquisition() is True

    weight.update(model=objective_model, eta=1.0, num_data=2, runhistory=_runhistory(configspace, [150.0, 50.0]))
    assert weight.suppresses_acquisition() is False


def test_the_incumbent_is_corrected_to_the_best_feasible_one(configspace):
    objective_model = FixedModel([0.25], [1.0])
    weight = _weight(FixedModel([50.0], [1.0]), objective_model)

    # Nothing feasible yet, so the given incumbent stands.
    weight.update(model=objective_model, eta=9.0, num_data=0, runhistory=_runhistory(configspace, [None]))
    assert weight.adjust_eta(9.0) == 9.0

    weight.update(model=objective_model, eta=9.0, num_data=1, runhistory=_runhistory(configspace, [50.0]))
    assert weight.adjust_eta(9.0) == pytest.approx(0.25)


def test_the_constraint_model_trains_on_compressed_residuals(configspace):
    """Not on the raw observations, and the two transformations are worth naming separately.

    The constraint here is ``latency <= 100``, so an observed 50 is 50 *inside* the bound and 150 is 50
    outside: residuals of -50 and +50. Expressing it that way puts the feasibility boundary at zero whichever
    direction the bound was declared in, which is what lets `probability_of_feasibility` be one formula with
    no constraint list and no per-constraint branching.

    `bilog` then compresses them symmetrically, so a wildly violating observation cannot dictate the length
    scales for a boundary that lives near zero. It is strictly increasing and maps zero to zero, so the sign -
    and therefore feasibility itself - survives untouched, which is the part this asserts.
    """
    constraint_model = FixedModel([50.0], [1.0])
    weight = _weight(constraint_model, FixedModel([0.5], [1.0]))

    weight.update(
        model=FixedModel([0.5], [1.0]), eta=1.0, num_data=2, runhistory=_runhistory(configspace, [50.0, 150.0])
    )

    assert constraint_model.trained_on is not None
    targets = sorted(constraint_model.trained_on[1][:, 0])

    assert targets == pytest.approx([-np.log1p(50.0), np.log1p(50.0)])
    assert targets[0] < 0 < targets[1], "the feasible observation stays negative, the violating one positive"


def test_the_compression_can_be_turned_off(configspace):
    """Then the residuals are fitted as measured - still residuals, just uncompressed."""
    constraint_model = FixedModel([50.0], [1.0])
    weight = FeasibilityWeight(
        constraints=[OutcomeConstraint("latency", "<=", 100.0)],
        constraint_model=constraint_model,
        transform=False,
    )

    weight.update(
        model=FixedModel([0.5], [1.0]), eta=1.0, num_data=2, runhistory=_runhistory(configspace, [50.0, 150.0])
    )

    assert sorted(constraint_model.trained_on[1][:, 0]) == pytest.approx([-50.0, 50.0])


def test_meta_describes_the_constraints():
    weight = _weight(FixedModel([50.0], [1.0]))
    meta = weight.meta

    assert meta["name"] == "FeasibilityWeight"
    assert meta["constraints"] == ["latency <= 100.0"]
    assert meta["constraint_model"]["name"] == "FixedModel"
    assert meta["decay"] == {"name": "NoDecay"}
