from __future__ import annotations

import tempfile

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float, Normal

from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario
from smac.acquisition.function import (
    EI,
    LCB,
    ConstrainedAcquisitionFunction,
    PriorAcquisitionFunction,
    WeightedAcquisitionFunction,
)
from smac.acquisition.function.weighted_acquisition_function import (
    ensure_feasibility_weight,
)
from smac.acquisition.weight import FeasibilityWeight, PriorEnsemble
from smac.utils.constraints import is_feasible, parse_constraints
from smac.acquisition.function.abstract_acquisition_function import AcquisitionScale

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


FACADES = [HyperparameterOptimizationFacade, BlackBoxFacade]


class Model:
    """Surrogate returning a constant mean and variance, independent of the input."""

    def train(self, X, Y):
        return self

    def predict_marginalized(self, X):
        return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))

    @property
    def meta(self):
        return {"name": "Model"}


def _configspace(seed: int, prior: bool = False) -> ConfigurationSpace:
    configspace = ConfigurationSpace(seed=seed)
    x0 = Float("x0", (0.0, 1.0), distribution=Normal(0.6, 0.1)) if prior else Float("x0", (0.0, 1.0))
    configspace.add(x0, Float("x1", (0.0, 1.0)))

    return configspace


def target(config, seed: int = 0):
    """Minimize x0 subject to x0 + x1 >= 1, so that the unconstrained optimum is infeasible"""
    return config["x0"], {"slack": config["x0"] + config["x1"]}


def _constraints():
    return parse_constraints(["slack >= 1.0"])


def test_a_plain_acquisition_function_keeps_being_wrapped_as_before():
    """The constraints-only case has to produce the same object, and the same metadata.

    The metadata ends up in the name of the output directory, so a change here would keep existing runs from being
    continued.
    """
    weighted = ensure_feasibility_weight(EI(), _constraints(), Model())

    assert isinstance(weighted, ConstrainedAcquisitionFunction)
    assert set(weighted.meta) == {
        "name",
        "acquisition_function",
        "constraints",
        "constraint_model",
        "feasibility_floor",
    }


def test_a_prior_and_the_constraints_end_up_in_one_wrapper():
    prior = PriorAcquisitionFunction(EI(), decay_beta=2.0)
    weighted = ensure_feasibility_weight(prior, _constraints(), Model())

    assert weighted is prior
    assert isinstance(weighted.get_weight(PriorEnsemble), PriorEnsemble)
    assert isinstance(weighted.get_weight(FeasibilityWeight), FeasibilityWeight)


def test_the_constraints_are_not_added_twice():
    weighted = WeightedAcquisitionFunction(EI())
    once = ensure_feasibility_weight(weighted, _constraints(), Model())
    twice = ensure_feasibility_weight(once, _constraints(), Model())

    assert twice is once
    assert len(twice.weights) == 1


def test_the_acquisition_values_are_shifted_exactly_once():
    """Nesting the two wrappers used to shift a confidence bound by the incumbent twice."""
    weighted = ensure_feasibility_weight(PriorAcquisitionFunction(LCB(), decay_beta=2.0), _constraints(), Model())

    assert weighted._scale is AcquisitionScale.SIGNED
    assert weighted.acquisition_function.value_scale is AcquisitionScale.SIGNED
    assert isinstance(weighted.acquisition_function, LCB)


def test_wrapping_a_wrapper_is_refused():
    """The old nesting is now a loud error rather than a silent numerical one."""
    prior = PriorAcquisitionFunction(EI(), decay_beta=2.0)

    with pytest.raises(ValueError, match="must not wrap another one"):
        ConstrainedAcquisitionFunction(
            acquisition_function=prior,
            constraints=_constraints(),
            constraint_model=Model(),
        )


@pytest.mark.parametrize("facade", FACADES)
def test_a_supplied_prior_survives_the_constraints(facade):
    """The facade adds the feasibility weight to the prior wrapper instead of wrapping it again."""
    scenario = Scenario(
        _configspace(0, prior=True),
        constraints=["slack >= 1.0"],
        n_trials=10,
        output_directory=tempfile.mkdtemp(),
    )
    inner = EI(xi=0.123)
    prior = PriorAcquisitionFunction(inner, decay_beta=1.0)

    smac = facade(scenario, target, acquisition_function=prior, overwrite=True, logging_level=60)

    assert smac._acquisition_function is prior
    assert smac._acquisition_function.acquisition_function is inner
    assert smac._acquisition_function.get_weight(FeasibilityWeight) is not None
    assert smac._acquisition_function.get_weight(PriorEnsemble) is not None
    assert smac._acquisition_maximizer._acquisition_function is smac._acquisition_function


def test_a_constrained_run_with_a_prior_still_reports_a_feasible_incumbent():
    scenario = Scenario(
        _configspace(0, prior=True),
        objectives="x0",
        constraints=["slack >= 1.0"],
        n_trials=40,
        seed=0,
        deterministic=True,
        output_directory=tempfile.mkdtemp(),
    )
    prior = PriorAcquisitionFunction(
        BlackBoxFacade.get_acquisition_function(scenario),
        decay_beta=scenario.n_trials / 10,
    )
    smac = BlackBoxFacade(scenario, target, acquisition_function=prior, overwrite=True, logging_level=60)
    incumbent = smac.optimize()

    if isinstance(incumbent, list):
        incumbent = incumbent[0]

    assert is_feasible(_constraints(), smac.runhistory.get_constraint_values(incumbent))
