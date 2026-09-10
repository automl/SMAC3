from __future__ import annotations

import tempfile

import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario
from smac.acquisition.function import EI, ConstrainedAcquisitionFunction
from smac.utils.constraints import is_feasible

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


FACADES = [HyperparameterOptimizationFacade, BlackBoxFacade]


def _configspace(seed: int) -> ConfigurationSpace:
    configspace = ConfigurationSpace(seed=seed)
    configspace.add(Float("x0", (0.0, 1.0)), Float("x1", (0.0, 1.0)))

    return configspace


def target(config, seed: int = 0):
    """Minimize x0 subject to x0 + x1 >= 1, so that the unconstrained optimum is infeasible"""
    return config["x0"], {"slack": config["x0"] + config["x1"]}


def _optimize(facade, seed: int, constraints: list[str] | None, n_trials: int = 40):
    scenario = Scenario(
        _configspace(seed),
        objectives="x0",
        constraints=constraints,
        n_trials=n_trials,
        seed=seed,
        deterministic=True,
        output_directory=tempfile.mkdtemp(),
    )
    smac = facade(scenario, target, overwrite=True, logging_level=60)
    incumbent = smac.optimize()

    if isinstance(incumbent, list):
        incumbent = incumbent[0]

    return incumbent, smac


@pytest.mark.parametrize("facade", FACADES)
def test_the_incumbent_satisfies_the_constraint(facade):
    """A constrained run reports a configuration that respects the bound.

    Optimizes a problem whose unconstrained optimum is infeasible over several seeds and checks every reported
    incumbent satisfies the constraint.
    """
    for seed in range(3):
        incumbent, _ = _optimize(facade, seed, ["slack >= 1.0"])

        assert incumbent["x0"] + incumbent["x1"] >= 1.0 - 1e-9


@pytest.mark.parametrize("facade", FACADES)
def test_the_same_problem_is_solved_infeasibly_without_the_constraint(facade):
    """The constraint is what makes the difference, not the problem being easy.

    Runs the identical problem without declaring the constraint and checks the incumbent violates the bound,
    which is what the constrained run has to avoid.
    """
    for seed in range(3):
        incumbent, _ = _optimize(facade, seed, None)

        assert incumbent["x0"] + incumbent["x1"] < 1.0


@pytest.mark.parametrize("facade", FACADES)
def test_the_acquisition_function_is_wrapped_when_constraints_are_declared(facade):
    """Declaring a constraint installs the constrained acquisition function.

    Builds a facade with and without constraints and inspects the acquisition function in each case.
    """
    constrained = Scenario(
        _configspace(0), constraints=["slack >= 1.0"], n_trials=10, output_directory=tempfile.mkdtemp()
    )
    unconstrained = Scenario(_configspace(0), n_trials=10, output_directory=tempfile.mkdtemp())

    with_constraints = facade(constrained, target, overwrite=True, logging_level=60)
    without_constraints = facade(unconstrained, target, overwrite=True, logging_level=60)

    assert isinstance(with_constraints._acquisition_function, ConstrainedAcquisitionFunction)
    assert not isinstance(without_constraints._acquisition_function, ConstrainedAcquisitionFunction)

    # The maximizer has to score the same object the facade installed
    assert with_constraints._acquisition_maximizer._acquisition_function is with_constraints._acquisition_function


@pytest.mark.parametrize("facade", FACADES)
def test_a_supplied_acquisition_function_is_wrapped_rather_than_replaced(facade):
    """A user supplied acquisition function keeps being used, inside the constraint wrapper.

    Passes an explicit acquisition function to a constrained facade and checks it ended up wrapped.
    """
    scenario = Scenario(
        _configspace(0), constraints=["slack >= 1.0"], n_trials=10, output_directory=tempfile.mkdtemp()
    )
    inner = EI(xi=0.123)

    smac = facade(scenario, target, acquisition_function=inner, overwrite=True, logging_level=60)

    assert isinstance(smac._acquisition_function, ConstrainedAcquisitionFunction)
    assert smac._acquisition_function._acquisition_function is inner


@pytest.mark.parametrize("facade", FACADES)
def test_constraint_values_are_recorded_for_every_trial(facade):
    """Each evaluated trial stores the constrained output it reported.

    Runs a short optimization and checks every successful trial carries a value for the declared output.
    """
    _, smac = _optimize(facade, 0, ["slack >= 1.0"], n_trials=10)

    values = [v.constraint_values for v in smac.runhistory._data.values()]

    assert len(values) > 0
    assert all(v is not None and "slack" in v for v in values)


def test_an_all_infeasible_run_still_reports_an_incumbent():
    """A run that never finds a feasible configuration does not fail or return nothing.

    Declares a bound the target function can never satisfy and checks an incumbent is still reported.
    """
    incumbent, smac = _optimize(HyperparameterOptimizationFacade, 0, ["slack >= 99.0"], n_trials=10)

    assert incumbent is not None
    assert not is_feasible(smac.scenario.get_constraints(), smac.runhistory.get_constraint_values(incumbent))
