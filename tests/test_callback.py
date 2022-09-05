from __future__ import annotations

import pytest
import smac
from smac import HyperparameterFacade, Scenario
from smac.intensification.intensification import Intensifier
from ConfigSpace import Configuration, ConfigurationSpace, Integer, Float, Categorical
from smac.callback import Callback
from smac.initial_design import DefaultInitialDesign
from smac.runhistory import TrialInfo, TrialValue, TrialInfoIntent


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

    def on_next_configurations_start(self, smbo: smac.main.BaseSMBO) -> None:
        self.next_configurations_start_counter += 1

    def on_next_configurations_end(self, smbo: smac.main.BaseSMBO, configurations: list[Configuration]) -> None:
        print(len(configurations))
        self.next_configurations_end_counter += 1

    def on_ask_start(self, smbo: smac.main.BaseSMBO) -> None:
        self.ask_start_counter += 1

    def on_ask_end(self, smbo: smac.main.BaseSMBO, intent: TrialInfoIntent, info: TrialInfo) -> None:
        self.ask_end_counter += 1

    def on_tell_start(self, smbo: smac.main.BaseSMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        self.tell_start_counter += 1
        return None

    def on_tell_end(self, smbo: smac.main.BaseSMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        self.tell_end_counter += 1
        return None


def test_callback(rosenbrock):
    N_TRIALS = 50

    scenario = Scenario(rosenbrock.configspace, n_trials=N_TRIALS)
    callback = CustomCallback()
    intensifier = Intensifier(scenario, max_config_calls=1)
    initial_design = DefaultInitialDesign(scenario)
    smac = HyperparameterFacade(
        scenario,
        rosenbrock.train,
        intensifier=intensifier,
        initial_design=initial_design,
        callbacks=[callback],
        overwrite=True,
    )
    smac.optimize()

    # Warning: All of these counters strongly depend on the initial design

    # Those functions are called only once
    assert callback.start_counter == 1
    assert callback.end_counter == 1

    # Those functions are called N_TRIALS + 1 times
    assert callback.ask_start_counter == N_TRIALS
    assert callback.ask_end_counter == N_TRIALS

    # Those functions are called N_TRIALS - 1 times
    assert callback.tell_start_counter == N_TRIALS - 1
    assert callback.tell_end_counter == N_TRIALS - 1

    # We try one more round
    assert callback.iteration_start_counter == N_TRIALS
    # but we stop because we already evaluated N_TRIALS
    assert callback.iteration_end_counter == N_TRIALS - 1

    # This is depending on the number of challengers
    assert callback.next_configurations_start_counter == 999999
    assert callback.next_configurations_end_counter == 999999
