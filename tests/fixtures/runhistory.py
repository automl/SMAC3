import pytest

from smac import RunHistory


@pytest.fixture
def runhistory() -> RunHistory:
    return RunHistory()
