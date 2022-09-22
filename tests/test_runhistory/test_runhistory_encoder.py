import numpy as np
import pytest

from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.runhistory.encoder import (
    RunHistoryEIPSEncoder,
    RunHistoryInverseScaledEncoder,
    RunHistoryLogEncoder,
    RunHistoryLogScaledEncoder,
    RunHistoryScaledEncoder,
    RunHistorySqrtScaledEncoder,
)
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.runner.abstract_runner import StatusType


@pytest.fixture
def configs(configspace_small):
    configs = configspace_small.sample_configuration(20)
    return (configs[16], configs[15], configs[2], configs[3])


def test_transform(runhistory, make_scenario, configspace_small, configs):
    """Test if all encoders are working."""
    scenario = make_scenario(configspace_small)

    runhistory.add(
        config=configs[0],
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
    )

    runhistory.add(
        config=configs[1],
        cost=5,
        time=4,
        status=StatusType.SUCCESS,
    )

    # Normal encoder
    encoder = RunHistoryEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X1, Y1 = encoder.transform(runhistory)

    assert Y1.tolist() == [[1.0], [5.0]]
    assert ((X1 <= 1.0) & (X1 >= 0.0)).all()

    # Log encoder
    encoder = RunHistoryLogEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X, Y = encoder.transform(runhistory)
    assert Y.tolist() != Y1.tolist()
    assert ((X <= 1.0) & (X >= 0.0)).all()

    encoder = RunHistoryLogScaledEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X, Y = encoder.transform(runhistory)
    assert Y.tolist() != Y1.tolist()
    assert ((X <= 1.0) & (X >= 0.0)).all()

    encoder = RunHistoryScaledEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X, Y = encoder.transform(runhistory)
    assert Y.tolist() != Y1.tolist()
    assert ((X <= 1.0) & (X >= 0.0)).all()

    encoder = RunHistoryInverseScaledEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X, Y = encoder.transform(runhistory)
    assert Y.tolist() != Y1.tolist()
    assert ((X <= 1.0) & (X >= 0.0)).all()

    encoder = RunHistorySqrtScaledEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X, Y = encoder.transform(runhistory)
    assert Y.tolist() != Y1.tolist()
    assert ((X <= 1.0) & (X >= 0.0)).all()

    encoder = RunHistoryEIPSEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X, Y = encoder.transform(runhistory)
    assert Y.tolist() != Y1.tolist()
    assert ((X <= 1.0) & (X >= 0.0)).all()


def test_transform_conditionals(runhistory, make_scenario, configspace_large, configs):
    configs = configspace_large.sample_configuration(20)
    scenario = make_scenario(configspace_large)

    runhistory.add(
        config=configs[0],
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
    )

    runhistory.add(
        config=configs[2],
        cost=5,
        time=4,
        status=StatusType.SUCCESS,
    )

    encoder = RunHistoryEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X, Y = encoder.transform(runhistory)

    assert Y.tolist() == [[1.0], [5.0]]
    assert np.isnan(X[0][5])
    assert not np.isnan(X[1][5])


def test_multi_objective(runhistory, make_scenario, configspace_small, configs):
    configs = configspace_small.sample_configuration(20)
    scenario = make_scenario(configspace_small, use_multi_objective=True)

    runhistory.add(
        config=configs[0],
        cost=[0.0, 100.0],
        time=5,
        status=StatusType.SUCCESS,
    )

    # Multi objective algorithm must be set
    with pytest.raises(AssertionError):
        encoder = RunHistoryEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
        _, Y = encoder.transform(runhistory)

    encoder._set_multi_objective_algorithm(MeanAggregationStrategy(scenario))
    _, Y = encoder.transform(runhistory)

    # We expect the result to be 1 because no normalization could be done yet
    assert Y.flatten() == 1.0

    runhistory.add(
        config=configs[2],
        cost=[50.0, 50.0],
        time=4,
        status=StatusType.SUCCESS,
    )

    # Now we expect something different
    _, Y = encoder.transform(runhistory)
    assert Y.tolist() == [[0.5], [0.5]]

    runhistory.add(
        config=configs[3],
        cost=[200.0, 0.0],
        time=4,
        status=StatusType.SUCCESS,
    )

    _, Y = encoder.transform(runhistory)
    assert Y.tolist() == [
        [0.5],  # (0+1)/2
        [0.375],  # (0.25+0.5) / 2
        [0.5],  # (1+0) / 2
    ]


def test_ignore(runhistory, make_scenario, configspace_small, configs):
    """Tests if only successful states are considered."""
    scenario = make_scenario(configspace_small)

    runhistory.add(
        config=configs[0],
        cost=1,
        time=1,
        status=StatusType.MEMORYOUT,
    )

    runhistory.add(
        config=configs[1],
        cost=5,
        time=4,
        status=StatusType.MEMORYOUT,
    )

    # Normal encoder
    encoder = RunHistoryEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X1, Y1 = encoder.transform(runhistory)

    assert Y1.tolist() == []

    runhistory.add(
        config=configs[3],
        cost=5,
        time=4,
        status=StatusType.SUCCESS,
    )

    X1, Y1 = encoder.transform(runhistory)

    # cost 5 should be included now
    assert Y1.tolist() == [[5.0]]


def test_budgets(runhistory, make_scenario, configspace_small, configs):
    """Tests if only successful states are considered."""
    scenario = make_scenario(configspace_small)

    runhistory.add(
        config=configs[0],
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        budget=5,
    )

    runhistory.add(
        config=configs[1],
        cost=99999999,
        time=1,
        status=StatusType.SUCCESS,
        budget=2,
    )

    runhistory.add(config=configs[1], cost=5, time=4, status=StatusType.SUCCESS, budget=2)

    # Normal encoder
    encoder = RunHistoryEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X, Y = encoder.transform(runhistory, budget_subset=[2])
    assert Y.tolist() == [[99999999]]

    X, Y = encoder.transform(runhistory, budget_subset=[5])
    assert Y.tolist() == [[1]]


def test_budgets(runhistory, make_scenario, configspace_small, configs):
    """Tests if only specific budgets are considered."""
    scenario = make_scenario(configspace_small)

    runhistory.add(
        config=configs[0],
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        budget=5,
    )

    runhistory.add(
        config=configs[1],
        cost=99999999,
        time=1,
        status=StatusType.SUCCESS,
        budget=2,
    )

    runhistory.add(config=configs[1], cost=5, time=4, status=StatusType.SUCCESS, budget=2)

    # Normal encoder
    encoder = RunHistoryEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    X, Y = encoder.transform(runhistory, budget_subset=[2])
    assert Y.tolist() == [[99999999]]

    X, Y = encoder.transform(runhistory, budget_subset=[5])
    assert Y.tolist() == [[1]]


def test_lower_budget_states(runhistory, make_scenario, configspace_small, configs):
    """Tests lower budgets based on budget subset and considered states."""
    scenario = make_scenario(configspace_small)
    encoder = RunHistoryEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])

    runhistory.add(config=configs[0], cost=1, time=1, status=StatusType.SUCCESS, budget=3)
    runhistory.add(config=configs[0], cost=2, time=2, status=StatusType.SUCCESS, budget=4)
    runhistory.add(config=configs[0], cost=3, time=4, status=StatusType.TIMEOUT, budget=5)

    # We request a higher budget but can't find it, so we expect an empty list
    X, Y = encoder.transform(runhistory, budget_subset=[500])
    assert Y.tolist() == []

    encoder = RunHistoryEncoder(
        scenario=scenario,
        considered_states=[StatusType.SUCCESS],
        lower_budget_states=[StatusType.TIMEOUT],
    )

    # We request a higher budget but can't find it but since we consider TIMEOUT for lower budget states, we should
    # receive the cost of 3
    X, Y = encoder.transform(runhistory, budget_subset=[500])
    assert Y.tolist() == [[3.0]]
