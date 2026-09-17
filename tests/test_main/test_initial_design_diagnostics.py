import logging

import numpy as np
import pytest

from ConfigSpace import ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade, Scenario
from smac.callback import InitialDesignDiagnosticsCallback
from smac.runhistory import StatusType


def _configspace(seed=0):
    cs = ConfigurationSpace(seed=seed)
    cs.add([Float("x0", (0.0, 1.0)), Float("x1", (0.0, 1.0))])
    return cs


def _scenario(tmp_path, mode, n_trials=20, seed=0, **kwargs):
    return Scenario(
        _configspace(seed),
        n_trials=n_trials,
        deterministic=True,
        seed=seed,
        crash_cost=1000.0,
        initial_design_diagnostics=mode,
        output_directory=tmp_path,
        **kwargs,
    )


class _CaptureWarnings(logging.Handler):
    """Captures SMAC log records.

    `caplog` attaches to the root logger, but SMAC configures its own handlers via
    `logging.yml`, so records do not reliably reach it. Attaching directly to the
    diagnostics logger keeps the assertions independent of the logging configuration.
    """

    LOGGER_NAME = "smac.callback.initial_design_diagnostics_callback"

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    @property
    def text(self):
        return "\n".join(self.messages)

    def __enter__(self):
        self._logger = logging.getLogger(self.LOGGER_NAME)
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self)


def _crashing(config, seed=0):
    raise RuntimeError("broken pipeline")


def _constant(config, seed=0):
    return 0.42


def _healthy(config, seed=0):
    return float(config["x0"])


def test_scenario_rejects_invalid_mode(tmp_path):
    with pytest.raises(ValueError, match="initial_design_diagnostics"):
        Scenario(
            _configspace(),
            n_trials=10,
            initial_design_diagnostics="does-not-exist",
            output_directory=tmp_path,
        )


def test_diagnostics_warn_by_default(tmp_path):
    """The default reports findings but never changes what the optimization does."""
    scenario = Scenario(_configspace(), n_trials=10, output_directory=tmp_path)
    assert scenario.initial_design_diagnostics == "warn"

    smac = HyperparameterOptimizationFacade(scenario, _healthy, overwrite=True)
    registered = [c for c in smac._callbacks if isinstance(c, InitialDesignDiagnosticsCallback)]
    assert len(registered) == 1
    assert registered[0].mode == "warn"


def test_diagnostics_can_be_switched_off(tmp_path):
    scenario = Scenario(
        _configspace(), n_trials=10, initial_design_diagnostics="off", output_directory=tmp_path
    )

    smac = HyperparameterOptimizationFacade(scenario, _healthy, overwrite=True)
    registered = [c for c in smac._callbacks if isinstance(c, InitialDesignDiagnosticsCallback)]
    assert registered == [], "the callback must not be registered when switched off"


@pytest.mark.parametrize("mode", ["warn", "abort"])
def test_callback_is_registered_when_requested(tmp_path, mode):
    scenario = _scenario(tmp_path, mode)
    smac = HyperparameterOptimizationFacade(scenario, _healthy, overwrite=True)

    registered = [c for c in smac._callbacks if isinstance(c, InitialDesignDiagnosticsCallback)]
    assert len(registered) == 1
    assert registered[0].mode == mode


def test_all_trials_crashed_is_reported(tmp_path):
    scenario = _scenario(tmp_path, "warn")

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(scenario, _crashing, overwrite=True)
        smac.optimize()

    assert "configurations of the initial design failed" in logs.text
    # Warning only: the budget is still consumed.
    assert smac.runhistory.finished == 20


def test_all_trials_crashed_aborts_in_abort_mode(tmp_path):
    scenario = _scenario(tmp_path, "abort")

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(scenario, _crashing, overwrite=True)
        smac.optimize()

    assert "Aborting the optimization" in logs.text
    # The run is stopped right after the initial design instead of consuming everything.
    assert smac.runhistory.finished < 20
    assert all(smac.runhistory[k].status == StatusType.CRASHED for k in smac.runhistory)


def test_constant_objective_is_reported(tmp_path):
    scenario = _scenario(tmp_path, "warn")

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(scenario, _constant, overwrite=True)
        smac.optimize()

    assert "same cost" in logs.text


def test_constant_objective_aborts_in_abort_mode(tmp_path):
    """A constant objective leaves the surrogate model without a feedback signal.

    Whether this is a bug or a huge plateau, there is nothing to be learned from continuing,
    so `abort` stops the run here as well.
    """
    scenario = _scenario(tmp_path, "abort")

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(scenario, _constant, overwrite=True)
        smac.optimize()

    assert "same cost" in logs.text
    assert "Aborting the optimization" in logs.text
    assert smac.runhistory.finished < 20


def test_constant_objective_does_not_abort_in_warn_mode(tmp_path):
    """`warn` reports the same finding but always lets the optimization run to the end."""
    scenario = _scenario(tmp_path, "warn")

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(scenario, _constant, overwrite=True)
        smac.optimize()

    assert "same cost" in logs.text
    assert "Aborting the optimization" not in logs.text
    assert smac.runhistory.finished == 20


@pytest.mark.parametrize("mode", ["off", "warn", "abort"])
def test_healthy_run_is_never_reported(tmp_path, mode):
    """No false positives: a well-behaved target function must not trigger anything."""
    scenario = _scenario(tmp_path, mode)

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(scenario, _healthy, overwrite=True)
        smac.optimize()

    assert "configurations of the initial design failed" not in logs.text
    assert "same cost" not in logs.text
    assert smac.runhistory.finished == 20


def test_partial_crashes_do_not_trigger(tmp_path):
    """Crashes on part of the search space are normal and must not be reported."""

    def half_broken(config, seed=0):
        if float(config["x0"]) > 0.5:
            raise RuntimeError("crash in half the space")
        return float(config["x0"])

    scenario = _scenario(tmp_path, "abort")

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(scenario, half_broken, overwrite=True)
        smac.optimize()

    assert "configurations of the initial design failed" not in logs.text
    assert smac.runhistory.finished == 20


def test_first_run_crash_no_longer_raises(tmp_path):
    """Regression test for the removed dead code in `SMBO._add_results`.

    The previous check (`runhistory.finished == 0` after `tell()`) could never be satisfied,
    because `tell()` increments the counter first. It is replaced by the diagnostics callback.
    """
    calls = {"n": 0}

    def first_run_crashes(config, seed=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first run crashes")
        return float(config["x0"])

    scenario = _scenario(tmp_path, "off", n_trials=8)
    smac = HyperparameterOptimizationFacade(scenario, first_run_crashes, overwrite=True)

    # Must not raise; the optimization simply continues.
    smac.optimize()
    assert smac.runhistory.finished == 8


def test_single_config_initial_design_does_not_abort_on_one_crash(tmp_path):
    """Regression test: facades with a `DefaultInitialDesign` propose a single configuration.

    Aborting after that one trial would be a false positive -- a default configuration that is
    invalid for the given dataset is common and recoverable. At least
    `MIN_CONFIGS_FOR_DIAGNOSIS` configurations have to be observed first.
    """
    from smac import RandomFacade

    calls = {"n": 0}

    def only_default_crashes(config, seed=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("default config invalid for this dataset")
        return float(config["x0"])

    scenario = _scenario(tmp_path, "abort", n_trials=15)

    with _CaptureWarnings() as logs:
        smac = RandomFacade(scenario, only_default_crashes, overwrite=True)
        assert len(smac.intensifier._config_selector._initial_design_configs) == 1
        smac.optimize()

    assert "Aborting the optimization" not in logs.text
    assert smac.runhistory.finished == 15


def test_abort_waits_for_the_minimum_number_of_trials(tmp_path):
    """Even with a single-configuration initial design, the diagnosis needs enough evidence."""
    from smac import RandomFacade
    from smac.callback.initial_design_diagnostics_callback import MIN_CONFIGS_FOR_DIAGNOSIS

    scenario = _scenario(tmp_path, "abort", n_trials=15)

    with _CaptureWarnings() as logs:
        smac = RandomFacade(scenario, _crashing, overwrite=True)
        smac.optimize()

    assert "Aborting the optimization" in logs.text
    assert smac.runhistory.finished >= MIN_CONFIGS_FOR_DIAGNOSIS
    assert smac.runhistory.finished < 15


def test_multi_objective_diagnostics(tmp_path):
    """The diagnostics have to work for multi-objective runs as well."""
    from smac.facade.multi_objective_facade import MultiObjectiveFacade

    def constant_multi_objective(config, seed=0):
        return {"o1": 0.5, "o2": 0.25}

    scenario = Scenario(
        _configspace(),
        n_trials=20,
        deterministic=True,
        seed=0,
        crash_cost=1000.0,
        objectives=["o1", "o2"],
        initial_design_diagnostics="warn",
        output_directory=tmp_path,
    )

    with _CaptureWarnings() as logs:
        smac = MultiObjectiveFacade(
            scenario,
            constant_multi_objective,
            overwrite=True,
            multi_objective_algorithm=MultiObjectiveFacade.get_multi_objective_algorithm(scenario),
        )
        smac.optimize()

    assert "same cost" in logs.text
    assert "o1" in logs.text and "o2" in logs.text
    # Warning only -- a plateau never aborts.
    assert smac.runhistory.finished == 20


def test_partial_failures_are_counted(tmp_path):
    """Partial failures have to be reported so that the user can react to them."""

    calls = {"n": 0}

    def half_broken(config, seed=0):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("every second trial fails")
        return float(config["x0"])

    scenario = _scenario(tmp_path, "warn")

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(scenario, half_broken, overwrite=True)
        smac.optimize()

    assert "initial design configurations failed" in logs.text
    # The run continues -- partial failures are normal.
    assert "Aborting the optimization" not in logs.text
    assert smac.runhistory.finished == 20


def test_partial_failures_report_the_correct_counts(tmp_path):
    """The message has to state how many of how many trials failed."""
    n_configs = 10
    failing = {3, 4, 5}

    calls = {"n": 0}

    def some_broken(config, seed=0):
        index = calls["n"]
        calls["n"] += 1
        if index in failing:
            raise RuntimeError("broken")
        return float(config["x0"]) + 0.01 * index

    scenario = _scenario(tmp_path, "warn", n_trials=40)

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(
            scenario,
            some_broken,
            overwrite=True,
            initial_design=HyperparameterOptimizationFacade.get_initial_design(
                scenario, n_configs=n_configs
            ),
        )
        smac.optimize()

    assert f"{len(failing)} of {n_configs} initial design configurations failed." in logs.text


def test_all_failed_does_not_also_report_a_count(tmp_path):
    """The `all failed` case has its own, more specific message."""
    scenario = _scenario(tmp_path, "warn")

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(scenario, _crashing, overwrite=True)
        smac.optimize()

    assert "configurations of the initial design failed" in logs.text
    assert "initial design configurations failed." not in logs.text


def test_healthy_run_reports_no_failure_count(tmp_path):
    """No false positives: without failures there is nothing to report."""
    scenario = _scenario(tmp_path, "warn")

    with _CaptureWarnings() as logs:
        smac = HyperparameterOptimizationFacade(scenario, _healthy, overwrite=True)
        smac.optimize()

    assert "failed" not in logs.text
