from typing import Callable
from ConfigSpace import ConfigurationSpace
from smac import Scenario
import pytest


@pytest.fixture
def make_scenario() -> Callable:
    def _make(configspace: ConfigurationSpace, multi_objective=False) -> Scenario:
        objectives = "cost"
        if multi_objective:
            objectives = ["cost1", "cost2"]

        return Scenario(
            configspace=configspace,
            name="test",
            output_directory="smac3_output_test",
            objectives=objectives,
            walltime_limit=30,
            n_trials=100,
        )

    return _make
