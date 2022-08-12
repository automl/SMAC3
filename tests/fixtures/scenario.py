from typing import Callable
from ConfigSpace import ConfigurationSpace
from smac import Scenario
import pytest


@pytest.fixture
def make_scenario() -> Callable:
    def _make(configspace: ConfigurationSpace) -> Scenario:
        return Scenario(
            configspace=configspace,
            name="test",
            output_directory="smac3_output_test",
            walltime_limit=30,
            n_trials=100,
        )

    return _make
