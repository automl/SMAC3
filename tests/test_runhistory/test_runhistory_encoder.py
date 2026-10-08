import numpy as np
import pytest
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter

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
    encoder.runhistory = runhistory

    # TODO: Please replace with the more general solution once ConfigSpace 1.0
    # upper = np.array([hp.upper_vectorized for hp in space.values()])
    # lower = np.array([hp.lower_vectorized for hp in space.values()])
    # -
    # Categoricals are upperbounded by their size, rest of hyperparameters are
    # upperbounded by 1.
    upper_bounds = {
        hp.name: (hp.size - 1) if isinstance(hp, CategoricalHyperparameter) else 1.0
        for hp in list(configspace_small.values())
    }
    # Need to ensure they match the order in the Configuration vectorized form
    sorted_by_indices = sorted(
        upper_bounds.items(),
        key=lambda x: configspace_small.index_of[x[0]],
    )
    upper = np.array([upper_bound for _, upper_bound in sorted_by_indices])
    lower = 0.0

    X1, Y1 = encoder.transform()

    assert Y1.tolist() == [[1.0], [5.0]]
    assert ((X1 <= upper) & (X1 >= lower)).all()

    # Log encoder
    encoder = RunHistoryLogEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    encoder.runhistory = runhistory
    X, Y = encoder.transform()
    assert Y.tolist() != Y1.tolist()
    assert ((X <= upper) & (X >= lower)).all()

    encoder = RunHistoryLogScaledEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    encoder.runhistory = runhistory
    X, Y = encoder.transform()
    assert Y.tolist() != Y1.tolist()
    assert ((X <= upper) & (X >= lower)).all()

    encoder = RunHistoryScaledEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    encoder.runhistory = runhistory
    X, Y = encoder.transform()
    assert Y.tolist() != Y1.tolist()
    assert ((X <= upper) & (X >= lower)).all()

    encoder = RunHistoryInverseScaledEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    encoder.runhistory = runhistory
    X, Y = encoder.transform()
    assert Y.tolist() != Y1.tolist()
    assert ((X <= upper) & (X >= lower)).all()

    encoder = RunHistorySqrtScaledEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    encoder.runhistory = runhistory
    X, Y = encoder.transform()
    assert Y.tolist() != Y1.tolist()
    assert ((X <= upper) & (X >= lower)).all()

    encoder = RunHistoryEIPSEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    encoder.runhistory = runhistory
    X, Y = encoder.transform()
    assert Y.tolist() != Y1.tolist()
    assert ((X <= upper) & (X >= lower)).all()

def test_transform_conditionals(runhistory, make_scenario, configspace_large):
    scenario = make_scenario(configspace_large)

    config_1 = Configuration(
        configspace_large,
        values={
            "activation": "tanh",
            "n_layer": 5,
            "n_neurons": 27,
            "solver": "lbfgs",
        },
    )
    config_2 = Configuration(
        configspace_large,
        values={
            "activation": "tanh",
            "batch_size": 47,
            "learning_rate": "adaptive",
            "learning_rate_init": 0.6673206111956781,
            "n_layer": 3,
            "n_neurons": 88,
            "solver": "sgd",
        },
    )
    runhistory.add(
        config=config_1,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
    )

    runhistory.add(
        config=config_2,
        cost=5,
        time=4,
        status=StatusType.SUCCESS,
    )

    encoder = RunHistoryEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    encoder.runhistory = runhistory
    X, Y = encoder.transform()

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
        encoder.runhistory = runhistory
        _, Y = encoder.transform()

    encoder.multi_objective_algorithm = MeanAggregationStrategy(scenario)
    encoder.runhistory = runhistory
    _, Y = encoder.transform()

    # We expect the result to be 1 because no normalization could be done yet
    assert Y.flatten() == 1.0

    runhistory.add(
        config=configs[2],
        cost=[50.0, 50.0],
        time=4,
        status=StatusType.SUCCESS,
    )

    # Now we expect something different
    _, Y = encoder.transform()
    assert Y.tolist() == [[0.5], [0.5]]

    runhistory.add(
        config=configs[3],
        cost=[200.0, 0.0],
        time=4,
        status=StatusType.SUCCESS,
    )

    _, Y = encoder.transform()
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
    encoder.runhistory = runhistory
    X1, Y1 = encoder.transform()

    assert Y1.tolist() == []

    runhistory.add(
        config=configs[3],
        cost=5,
        time=4,
        status=StatusType.SUCCESS,
    )

    X1, Y1 = encoder.transform()

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
    encoder.runhistory = runhistory
    X, Y = encoder.transform(budget_subset=[2])
    assert Y.tolist() == [[99999999]]

    X, Y = encoder.transform(budget_subset=[5])
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
    encoder.runhistory = runhistory
    X, Y = encoder.transform(budget_subset=[2])
    assert Y.tolist() == [[99999999]]

    X, Y = encoder.transform(budget_subset=[5])
    assert Y.tolist() == [[1]]


def test_lower_budget_states(runhistory, make_scenario, configspace_small, configs):
    """Tests lower budgets based on budget subset and considered states."""
    scenario = make_scenario(configspace_small)
    encoder = RunHistoryEncoder(scenario=scenario, considered_states=[StatusType.SUCCESS])
    encoder.runhistory = runhistory

    runhistory.add(config=configs[0], cost=1, time=1, status=StatusType.SUCCESS, budget=3)
    runhistory.add(config=configs[0], cost=2, time=2, status=StatusType.SUCCESS, budget=4)
    runhistory.add(config=configs[0], cost=3, time=4, status=StatusType.TIMEOUT, budget=5)

    # We request a higher budget but can't find it, so we expect an empty list
    X, Y = encoder.transform(budget_subset=[500])
    assert Y.tolist() == []

    encoder = RunHistoryEncoder(
        scenario=scenario,
        considered_states=[StatusType.SUCCESS],
        lower_budget_states=[StatusType.TIMEOUT],
    )
    encoder.runhistory = runhistory

    # We request a higher budget but can't find it but since we consider TIMEOUT for lower budget states, we should
    # receive the cost of 3
    X, Y = encoder.transform(budget_subset=[500])
    assert Y.tolist() == [[3.0]]


def _encoded(encoder_class, runhistory, make_scenario, configspace, configs):
    """An encoder of `encoder_class` that has transformed a runhistory of four costs, so its statistics are set."""
    for config, cost in zip(configs, (0.2, 1.0, 3.0, 8.0)):
        runhistory.add(config=config, cost=cost, time=1, status=StatusType.SUCCESS)

    encoder = encoder_class(scenario=make_scenario(configspace), considered_states=[StatusType.SUCCESS])
    encoder.runhistory = runhistory
    encoder.transform()

    return encoder


@pytest.mark.parametrize(
    "encoder_class",
    [
        RunHistoryEncoder,
        RunHistoryScaledEncoder,
        RunHistoryLogEncoder,
        RunHistoryLogScaledEncoder,
        RunHistoryInverseScaledEncoder,
        RunHistorySqrtScaledEncoder,
    ],
)
def test_inverse_transform_recovers_the_costs(runhistory, make_scenario, configspace_small, configs, encoder_class):
    """Mapping costs to the model's values and back gives the costs, so a model's prediction can be read in the
    objective's units."""
    encoder = _encoded(encoder_class, runhistory, make_scenario, configspace_small, configs)
    costs = np.array([[0.2], [0.5], [1.0], [3.0], [8.0]])

    recovered = encoder.inverse_transform_response_values(encoder.transform_response_values(costs.copy()))

    np.testing.assert_allclose(recovered, costs, rtol=1e-8)


def test_inverse_transform_is_increasing(runhistory, make_scenario, configspace_small, configs):
    """A value below anything the model was trained on still maps to a cost, and the order is kept - a lower
    confidence bound on a log scale can fall far below the observed costs."""
    encoder = _encoded(RunHistoryLogScaledEncoder, runhistory, make_scenario, configspace_small, configs)
    values = np.array([[-50.0], [-5.0], [-1.0], [0.0]])

    costs = encoder.inverse_transform_response_values(values)

    assert np.all(np.diff(costs[:, 0]) > 0)


def test_an_encoder_without_an_inverse_says_so(runhistory, make_scenario, configspace_small, configs):
    """An encoder whose transformation cannot be inverted answers `None` rather than a wrong cost."""
    encoder = _encoded(RunHistoryEIPSEncoder, runhistory, make_scenario, configspace_small, configs)

    assert encoder.inverse_transform_response_values(np.array([[0.5, 0.1]])) is None


def test_a_subclass_with_its_own_transform_does_not_inherit_an_inverse(
    runhistory, make_scenario, configspace_small, configs
):
    """Changing the transformation without supplying its inverse leaves the encoder without one, rather than with
    the inverse of the transformation it replaced."""

    class Squared(RunHistoryEncoder):
        def transform_response_values(self, values: np.ndarray) -> np.ndarray:
            return np.square(values)

    encoder = _encoded(Squared, runhistory, make_scenario, configspace_small, configs)

    assert encoder.inverse_transform_response_values(np.array([[4.0]])) is None
