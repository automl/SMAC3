from __future__ import annotations

from ConfigSpace import Configuration

import smac
from smac import HyperparameterOptimizationFacade, Scenario
from smac.callback.callback import Callback
from smac.initial_design import DefaultInitialDesign
from smac.intensifier.intensifier import Intensifier
from smac.runhistory import TrialInfo, TrialValue


class CustomCallback(Callback):
    def __init__(self) -> None:
        self.start_counter = 0
        self.end_counter = 0
        self.iteration_start_counter = 0
        self.iteration_end_counter = 0
        self.next_configurations_start_counter = 0
        self.next_configurations_end_counter = 0
        self.ask_start_counter = 0
        self.ask_end_counter = 0
        self.tell_start_counter = 0
        self.tell_end_counter = 0

    def on_start(self, smbo: smac.main.BaseSMBO) -> None:
        self.start_counter += 1

    def on_end(self, smbo: smac.main.BaseSMBO) -> None:
        self.end_counter += 1

    def on_iteration_start(self, smbo: smac.main.BaseSMBO) -> None:
        self.iteration_start_counter += 1

    def on_iteration_end(self, smbo: smac.main.BaseSMBO) -> None:
        self.iteration_end_counter += 1

    def on_next_configurations_start(self, config_selector: smac.main.config_selector.ConfigSelector) -> None:
        self.next_configurations_start_counter += 1

    def on_next_configurations_end(
        self, config_selector: smac.main.config_selector.ConfigSelector, config: Configuration
    ) -> None:
        self.next_configurations_end_counter += 1

    def on_ask_start(self, smbo: smac.main.BaseSMBO) -> None:
        self.ask_start_counter += 1

    def on_ask_end(self, smbo: smac.main.BaseSMBO, info: TrialInfo) -> None:
        self.ask_end_counter += 1

    def on_tell_start(self, smbo: smac.main.BaseSMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        self.tell_start_counter += 1
        return None

    def on_tell_end(self, smbo: smac.main.BaseSMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        self.tell_end_counter += 1
        return None


def test_callback(rosenbrock):
    N_TRIALS = 5

    scenario = Scenario(rosenbrock.configspace, n_trials=N_TRIALS)

    # The intensify percentage indirectly influences the number of calls of configuration
    # sampling.
    intensifier = Intensifier(scenario, max_config_calls=1)
    initial_design = DefaultInitialDesign(scenario)

    smac = HyperparameterOptimizationFacade(
        scenario,
        rosenbrock.train,
        intensifier=intensifier,
        initial_design=initial_design,
        callbacks=[Callback()],
        overwrite=True,
    )
    smac.optimize()


def test_custom_callback(rosenbrock):
    N_TRIALS = 40

    scenario = Scenario(rosenbrock.configspace, n_trials=N_TRIALS)
    callback = CustomCallback()

    # The intensify percentage indirectly influences the number of calls of configuration
    # sampling.
    intensifier = Intensifier(scenario, max_config_calls=1)
    initial_design = DefaultInitialDesign(scenario)

    smac = HyperparameterOptimizationFacade(
        scenario,
        rosenbrock.train,
        intensifier=intensifier,
        initial_design=initial_design,
        callbacks=[callback],
        overwrite=True,
    )
    smac.optimize()

    # Those functions are called only once
    assert callback.start_counter == 1
    assert callback.end_counter == 1

    assert callback.ask_start_counter == 40
    assert callback.ask_end_counter == 40

    assert callback.ask_start_counter == callback.tell_start_counter
    assert callback.ask_end_counter == callback.tell_end_counter

    assert callback.iteration_start_counter == 40
    assert callback.iteration_end_counter == 40
    assert callback.iteration_start_counter == callback.iteration_end_counter

    assert callback.next_configurations_start_counter == 40
    assert callback.next_configurations_end_counter == 40
