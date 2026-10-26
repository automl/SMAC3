import pytest

from smac import RunHistory
from smac.utils.cost_transformer import CostTransformer

@pytest.fixture
def runhistory() -> RunHistory:
    return RunHistory()

@pytest.fixture
def compute_cost():
    def _compute(runhistory, config):
        raw_costs = runhistory.get_costs(config)
        return CostTransformer.aggregate(
            raw_costs,
            method="mean",
            normalize=True,
            bounds=runhistory.objective_bounds,
            scalarize=True,
            algorithm=runhistory.multi_objective_algorithm,
        )
    return _compute
