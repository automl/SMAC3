from __future__ import annotations

import tempfile

import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac import BlackBoxFacade, Scenario
from smac.runhistory import TrialValue
from smac.utils.constraints import (
    extract_constraint_values,
    is_feasible,
    parse_constraints,
)

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


def _configspace(seed: int = 0) -> ConfigurationSpace:
    configspace = ConfigurationSpace(seed=seed)
    configspace.add(Float("x0", (0.0, 1.0)), Float("x1", (0.0, 1.0)))

    return configspace


def target(config, seed: int = 0):
    """Minimize x0 subject to x0 + x1 >= 1, so that the unconstrained optimum is infeasible."""
    return config["x0"], {"slack": config["x0"] + config["x1"]}


def _scenario(n_trials: int = 40, constraints: list[str] | None = None) -> Scenario:
    return Scenario(
        _configspace(),
        objectives="x0",
        constraints=constraints,
        n_trials=n_trials,
        seed=0,
        deterministic=True,
        output_directory=tempfile.mkdtemp(),
    )


def test_constraint_values_are_read_out_of_what_the_caller_reports():
    """The caller ran the target function, so nothing has separated the constrained outputs out yet."""
    scenario = _scenario(n_trials=4, constraints=["slack >= 1.0"])
    smac = BlackBoxFacade(scenario, target, overwrite=True, logging_level=60)

    info = smac.ask()
    cost, measured = target(info.config)
    smac.tell(info, TrialValue(cost=cost, time=0.1, additional_info=dict(measured)))

    assert smac.runhistory.get_constraint_values(info.config) == pytest.approx(measured)


def test_the_callers_dictionary_is_left_alone():
    """The runner owns its additional_info and may empty it; a caller's dictionary is theirs."""
    scenario = _scenario(n_trials=4, constraints=["slack >= 1.0"])
    smac = BlackBoxFacade(scenario, target, overwrite=True, logging_level=60)

    info = smac.ask()
    cost, measured = target(info.config)
    reported = {"slack": measured["slack"], "something else": 1.0}
    smac.tell(info, TrialValue(cost=cost, time=0.1, additional_info=reported))

    assert "slack" in reported, "tell must not mutate the dictionary it was handed."

    # ... but the constrained output is not duplicated into what the runhistory keeps as additional_info.
    stored = next(iter(smac.runhistory.values())).additional_info
    assert stored == {"something else": 1.0}


def test_explicit_constraint_values_win():
    scenario = _scenario(n_trials=4, constraints=["slack >= 1.0"])
    smac = BlackBoxFacade(scenario, target, overwrite=True, logging_level=60)

    info = smac.ask()
    smac.tell(
        info,
        TrialValue(cost=0.5, time=0.1, additional_info={"slack": 0.0}, constraint_values={"slack": 9.0}),
    )

    assert smac.runhistory.get_constraint_values(info.config) == {"slack": 9.0}


def test_nothing_happens_without_constraints():
    scenario = _scenario(n_trials=4)
    smac = BlackBoxFacade(scenario, target, overwrite=True, logging_level=60)

    info = smac.ask()
    smac.tell(info, TrialValue(cost=0.5, time=0.1, additional_info={"slack": 1.5}))

    stored = next(iter(smac.runhistory.values()))
    assert stored.constraint_values is None
    assert stored.additional_info == {"slack": 1.5}


def test_an_ask_and_tell_run_reaches_a_feasible_incumbent():
    """The regression test: the whole mechanism used to be inert on this path."""
    scenario = _scenario(n_trials=40, constraints=["slack >= 1.0"])
    smac = BlackBoxFacade(scenario, target, overwrite=True, logging_level=60)

    for _ in range(scenario.n_trials):
        info = smac.ask()
        cost, measured = target(info.config)
        smac.tell(info, TrialValue(cost=cost, time=0.1, additional_info=dict(measured)))

    incumbent = smac.intensifier.get_incumbent()

    assert incumbent is not None
    assert is_feasible(parse_constraints(["slack >= 1.0"]), smac.runhistory.get_constraint_values(incumbent))


def test_the_helper_reads_only_declared_names():
    constraints = parse_constraints(["slack >= 1.0"])

    assert extract_constraint_values(constraints, {"slack": 2.0, "other": 1.0}) == {"slack": 2.0}
    assert extract_constraint_values(constraints, {"other": 1.0}) == {}
    assert extract_constraint_values([], {"slack": 2.0}) is None
