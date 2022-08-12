from typing import Callable
from ConfigSpace import ConfigurationSpace
from smac import Scenario
from smac.utils.stats import Stats
import pytest


@pytest.fixture
def make_stats() -> Callable:
    def _make(scenario: Scenario) -> Scenario:
        return Stats(scenario)

    return _make
