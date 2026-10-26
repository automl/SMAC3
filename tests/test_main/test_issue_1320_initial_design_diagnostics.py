"""Acceptance test for issue #1320 -- early diagnostics after the initial design.

This file is written for a reviewer: it checks the issue itself, not the implementation.
Every test maps to one sentence of the issue and states which sentence it covers, so that the
issue can be verified without reading the implementation first.

    Once an initial design is collected, SMAC should use the initial design evaluations to
    conduct some early diagnostics.

    - If all configurations crash --> run function/pipeline is probably broken
    - If all perform identically --> metric or space is likely misconfigured

    Maybe we can think of more but if the two aforementioned cases arise, we can safely abort
    the HPO run early since in the first case we have no data to build the surrogate on (except
    for crashes, yielding the same penalty cost) and if all the performances are the very same
    this might be either due to a huge plateau or actually a bug. In either case we have no real
    feedback signal for the surrogate model.

Run it with::

    python -m pytest tests/test_main/test_issue_1320_initial_design_diagnostics.py -v
"""

import logging

import pytest

from ConfigSpace import Categorical, ConfigurationSpace, Float
from smac import (
    AlgorithmConfigurationFacade,
    BlackBoxFacade,
    HyperbandFacade,
    HyperparameterOptimizationFacade,
    MultiFidelityFacade,
    RandomFacade,
    Scenario,
)
from smac.facade.multi_objective_facade import MultiObjectiveFacade
from smac.runhistory import StatusType

N_TRIALS = 20


class Diagnostics:
    """Captures the diagnostics emitted during a run.

    SMAC configures its own log handlers via `logging.yml`, so `caplog` does not reliably see
    these records. Attaching to the diagnostics logger directly keeps the test independent of
    the logging configuration.
    """

    LOGGER_NAME = "smac.callback.initial_design_diagnostics_callback"

    def __init__(self):
        self.messages = []

    def __enter__(self):
        outer = self

        class _Handler(logging.Handler):
            def emit(self, record):
                outer.messages.append(record.getMessage())

        self._handler = _Handler(level=logging.WARNING)
        self._logger = logging.getLogger(self.LOGGER_NAME)
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)

    @property
    def text(self):
        return "\n".join(self.messages)

    @property
    def aborted(self):
        return "Aborting the optimization" in self.text


def _configspace(seed=0):
    cs = ConfigurationSpace(seed=seed)
    cs.add([Float("x0", (0.0, 1.0), default=0.5), Float("x1", (0.0, 1.0))])
    return cs


def _scenario(tmp_path, mode, **kwargs):
    kwargs.setdefault("n_trials", N_TRIALS)
    return Scenario(
        _configspace(),
        deterministic=True,
        seed=0,
        crash_cost=1000.0,
        initial_design_diagnostics=mode,
        output_directory=tmp_path,
        **kwargs,
    )


def _run(tmp_path, mode, target, facade=HyperparameterOptimizationFacade, n_configs=None, **kwargs):
    scenario = _scenario(tmp_path, mode, **kwargs)
    extra = {}
    if n_configs is not None:
        extra["initial_design"] = facade.get_initial_design(scenario, n_configs=n_configs)

    with Diagnostics() as diagnostics:
        smac = facade(scenario, target, overwrite=True, **extra)
        smac.optimize()

    return smac, diagnostics


# --------------------------------------------------------------------------------------------
# Target functions
# --------------------------------------------------------------------------------------------
def everything_crashes(config, seed=0):
    raise RuntimeError("the pipeline is broken")


def everything_identical(config, seed=0):
    return 0.42


def healthy(config, seed=0):
    return float(config["x0"])


# --------------------------------------------------------------------------------------------
# "If all configurations crash --> run function/pipeline is probably broken"
# --------------------------------------------------------------------------------------------
def test_all_configurations_crash_is_reported(tmp_path):
    smac, diagnostics = _run(tmp_path, "warn", everything_crashes)

    assert "configurations of the initial design failed" in diagnostics.text
    assert "probably broken" in diagnostics.text
    assert all(smac.runhistory[k].status == StatusType.CRASHED for k in smac.runhistory)


def test_all_configurations_crash_aborts_the_run(tmp_path):
    """"... we can safely abort the HPO run early ..."

    This is the German note on the issue: SMAC otherwise keeps going with the crashed configs,
    which makes no sense.
    """
    smac, diagnostics = _run(tmp_path, "abort", everything_crashes)

    assert diagnostics.aborted
    assert smac.runhistory.finished < N_TRIALS, "the remaining budget is saved"


# --------------------------------------------------------------------------------------------
# "If all perform identically --> metric or space is likely misconfigured"
# --------------------------------------------------------------------------------------------
def test_identical_performance_is_reported(tmp_path):
    smac, diagnostics = _run(tmp_path, "warn", everything_identical)

    assert "same cost" in diagnostics.text
    assert "misconfigured" in diagnostics.text
    assert len({smac.runhistory[k].cost for k in smac.runhistory}) == 1


def test_identical_performance_aborts_the_run(tmp_path):
    """"... In either case we have no real feedback signal for the surrogate model."

    Whether this is a huge plateau or a bug, there is nothing to learn from, so the run stops.
    """
    smac, diagnostics = _run(tmp_path, "abort", everything_identical)

    assert diagnostics.aborted
    assert smac.runhistory.finished < N_TRIALS


def test_near_constant_performance_is_detected(tmp_path):
    """Costs that differ only by floating point noise carry no signal either."""

    def almost_identical(config, seed=0):
        return 0.42 + float(config["x0"]) * 1e-16

    _, diagnostics = _run(tmp_path, "warn", almost_identical)

    assert "same cost" in diagnostics.text


# --------------------------------------------------------------------------------------------
# "Maybe we can think of more" -- partial failures are reported but never abort
# --------------------------------------------------------------------------------------------
def test_partial_failures_are_reported_with_counts(tmp_path):
    n_configs = 10
    failing = {3, 4, 5}
    calls = {"n": 0}

    def some_crash(config, seed=0):
        index = calls["n"]
        calls["n"] += 1
        if index in failing:
            raise RuntimeError("broken")
        return float(config["x0"]) + 0.01 * index

    smac, diagnostics = _run(tmp_path, "abort", some_crash, n_configs=n_configs, n_trials=40)

    assert f"{len(failing)} of {n_configs} initial design configurations failed." in diagnostics.text
    assert not diagnostics.aborted, "partial failures are normal and must not stop the run"
    assert smac.runhistory.finished == 40


# --------------------------------------------------------------------------------------------
# No false positives
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["off", "warn", "abort"])
def test_healthy_run_is_never_touched(tmp_path, mode):
    smac, diagnostics = _run(tmp_path, mode, healthy)

    assert diagnostics.messages == []
    assert smac.runhistory.finished == N_TRIALS


def test_default_reports_but_never_aborts(tmp_path):
    """The default is `warn`: findings are reported, but the run is never cut short.

    Only the log output differs from the historic behaviour.
    """
    scenario = Scenario(
        _configspace(), n_trials=N_TRIALS, crash_cost=1000.0, output_directory=tmp_path
    )
    assert scenario.initial_design_diagnostics == "warn"

    with Diagnostics() as diagnostics:
        smac = HyperparameterOptimizationFacade(scenario, everything_crashes, overwrite=True)
        smac.optimize()

    assert "probably broken" in diagnostics.text
    assert not diagnostics.aborted
    assert smac.runhistory.finished == N_TRIALS


def test_diagnostics_can_be_switched_off_entirely(tmp_path):
    """`off` restores the exact historic behaviour, including the absence of any message."""
    scenario = Scenario(
        _configspace(),
        n_trials=N_TRIALS,
        crash_cost=1000.0,
        initial_design_diagnostics="off",
        output_directory=tmp_path,
    )

    with Diagnostics() as diagnostics:
        smac = HyperparameterOptimizationFacade(scenario, everything_crashes, overwrite=True)
        smac.optimize()

    assert diagnostics.messages == []
    assert smac.runhistory.finished == N_TRIALS


def test_a_single_failing_configuration_does_not_abort(tmp_path):
    """Facades with a `DefaultInitialDesign` propose exactly one configuration.

    A default configuration that is invalid for the given dataset is common and recoverable,
    so one failure must not be enough to stop a run.
    """
    calls = {"n": 0}

    def only_the_first_crashes(config, seed=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("default config invalid for this dataset")
        return float(config["x0"])

    smac, diagnostics = _run(tmp_path, "abort", only_the_first_crashes, facade=RandomFacade)

    assert len(smac.intensifier._config_selector._initial_design_configs) == 1
    assert not diagnostics.aborted
    assert smac.runhistory.finished == N_TRIALS


def test_repeated_evaluations_of_one_configuration_are_not_counted_twice(tmp_path):
    """The issue counts configurations, not trials.

    With instances, a single configuration produces several trials. A configuration that fails
    on one instance but succeeds on another has not failed, and must not be reported.
    """
    instances = ["broken_instance", "i2", "i3"]

    def fails_only_on_one_instance(config, instance=None, seed=0):
        if instance == "broken_instance":
            raise RuntimeError("this instance is broken, the configuration is fine")
        return float(config["x0"])

    scenario = Scenario(
        _configspace(),
        n_trials=30,
        deterministic=False,
        seed=0,
        crash_cost=1000.0,
        instances=instances,
        instance_features={name: [float(i)] for i, name in enumerate(instances)},
        initial_design_diagnostics="abort",
        output_directory=tmp_path,
    )

    with Diagnostics() as diagnostics:
        smac = AlgorithmConfigurationFacade(
            scenario,
            fails_only_on_one_instance,
            overwrite=True,
            initial_design=HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5),
        )
        smac.optimize()

    # Every configuration has at least one successful evaluation, so nothing is reported.
    assert "configurations failed" not in diagnostics.text
    assert not diagnostics.aborted

    # Sanity check: there really were failures on the trial level.
    n_crashed = sum(1 for k in smac.runhistory if smac.runhistory[k].status != StatusType.SUCCESS)
    assert n_crashed > 0, "the setup has to actually produce failed trials"


def test_legitimate_plateau_in_a_small_discrete_space(tmp_path):
    """A real plateau is reported, but `warn` lets the run finish."""
    cs = ConfigurationSpace(seed=0)
    cs.add([Categorical("solver", ["a", "b", "c", "d"])])

    scenario = Scenario(
        cs,
        n_trials=4,
        deterministic=True,
        seed=0,
        initial_design_diagnostics="warn",
        output_directory=tmp_path,
    )

    with Diagnostics() as diagnostics:
        smac = HyperparameterOptimizationFacade(scenario, lambda config, seed=0: 1.0, overwrite=True)
        smac.optimize()

    assert "same cost" in diagnostics.text
    assert not diagnostics.aborted


# --------------------------------------------------------------------------------------------
# The diagnostics have to work for every facade
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "facade",
    [
        HyperparameterOptimizationFacade,
        BlackBoxFacade,
        RandomFacade,
        AlgorithmConfigurationFacade,
    ],
)
def test_all_configurations_crash_aborts_for_every_facade(tmp_path, facade):
    scenario = _scenario(tmp_path, "abort")

    with Diagnostics() as diagnostics:
        smac = facade(scenario, everything_crashes, overwrite=True)
        smac.optimize()

    assert diagnostics.aborted
    assert smac.runhistory.finished < N_TRIALS


@pytest.mark.parametrize("facade", [MultiFidelityFacade, HyperbandFacade])
def test_multi_fidelity_facades_are_covered(tmp_path, facade):
    def crashes_on_every_budget(config, seed=0, budget=None):
        raise RuntimeError("the pipeline is broken")

    scenario = _scenario(tmp_path, "abort", min_budget=1, max_budget=9)

    with Diagnostics() as diagnostics:
        smac = facade(scenario, crashes_on_every_budget, overwrite=True)
        smac.optimize()

    assert diagnostics.aborted
    assert smac.runhistory.finished < N_TRIALS


def test_multi_objective_is_covered(tmp_path):
    """With several objectives, each objective is checked separately."""

    def constant_multi_objective(config, seed=0):
        return {"o1": 0.5, "o2": 0.25}

    scenario = Scenario(
        _configspace(),
        n_trials=N_TRIALS,
        deterministic=True,
        seed=0,
        crash_cost=1000.0,
        objectives=["o1", "o2"],
        initial_design_diagnostics="warn",
        output_directory=tmp_path,
    )

    with Diagnostics() as diagnostics:
        smac = MultiObjectiveFacade(
            scenario,
            constant_multi_objective,
            overwrite=True,
            multi_objective_algorithm=MultiObjectiveFacade.get_multi_objective_algorithm(scenario),
        )
        smac.optimize()

    assert "same cost" in diagnostics.text
    assert "o1" in diagnostics.text and "o2" in diagnostics.text


# --------------------------------------------------------------------------------------------
# Regression: the previous safeguard was dead code
# --------------------------------------------------------------------------------------------
def test_dead_first_run_crashed_safeguard_is_gone(tmp_path):
    """`SMBO._add_results` used to check `runhistory.finished == 0` *after* `tell()`.

    `tell()` increments that counter, so the condition could never hold and the
    `FirstRunCrashedException` was never raised. Both the check and the unused exception classes
    have been removed; the diagnostics callback replaces them.
    """
    import smac.runner as runner_module

    assert not hasattr(runner_module, "FirstRunCrashedException")
    assert not hasattr(runner_module, "TargetAlgorithmAbortException")

    calls = {"n": 0}

    def first_run_crashes(config, seed=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first run crashes")
        return float(config["x0"])

    # A crashing first trial must not kill an otherwise healthy run.
    smac, diagnostics = _run(tmp_path, "abort", first_run_crashes)

    assert not diagnostics.aborted
    assert smac.runhistory.finished == N_TRIALS
