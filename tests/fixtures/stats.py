from typing import Callable

import pytest

from smac import Scenario
from smac.stats import Stats


@pytest.fixture
def make_stats() -> Callable:
    def _make(scenario: Scenario) -> Scenario:
        return Stats(scenario)

    return _make
