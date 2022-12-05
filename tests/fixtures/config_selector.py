from typing import Callable

import pytest

from smac.initial_design.random_design import RandomInitialDesign
from smac.main.config_selector import ConfigSelector
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


class CustomConfigSelector(ConfigSelector):
    def __init__(
        self,
        scenario: Scenario,
        runhistory: RunHistory,
        n_initial_configs: int = 3,
        retrain_after: int = 8,
        retries: int = 8,
    ) -> None:
        super().__init__(
            scenario,
            retrain_after=retrain_after,
            retries=retries,
        )

        initial_design = RandomInitialDesign(scenario, n_configs=n_initial_configs)
        self._set_components(
            initial_design=initial_design,
            runhistory=runhistory,
            runhistory_encoder=None,  # type: ignore
            model=None,  # type: ignore
            acquisition_maximizer=None,  # type: ignore
            acquisition_function=None,  # type: ignore
            random_design=None,  # type: ignore
        )

    def __iter__(self):
        for config in self._initial_design_configs:
            self._processed_configs.append(config)
            yield config

        while True:
            config = self._scenario.configspace.sample_configuration(1)
            if config not in self._processed_configs:
                self._processed_configs.append(config)
                yield config


@pytest.fixture
def make_config_selector() -> Callable:
    def _make(
        scenario: Scenario,
        runhistory: RunHistory,
        n_initial_configs: int = 3,
        retrain_after: int = 8,
        retries: int = 8,
    ) -> CustomConfigSelector:
        return CustomConfigSelector(
            scenario=scenario,
            runhistory=runhistory,
            n_initial_configs=n_initial_configs,
            retrain_after=retrain_after,
            retries=retries,
        )

    return _make
