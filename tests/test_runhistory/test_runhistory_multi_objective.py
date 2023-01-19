import pickle
import tempfile

import pytest

from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.runner.abstract_runner import StatusType
from smac.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


@pytest.fixture
def scenario(configspace_small):
    return Scenario(configspace_small, objectives=["a", "b"])


@pytest.fixture
def config1(configspace_small):
    configspace_small.seed(0)
    return configspace_small.sample_configuration()


@pytest.fixture
def config2(configspace_small):
    configspace_small.seed(1)
    return configspace_small.sample_configuration()


@pytest.fixture
def config3(configspace_small):
    configspace_small.seed(2)
    return configspace_small.sample_configuration()


def test_add_and_pickle(scenario, runhistory, config1):
    """
    Simply adding some rundata to runhistory, then pickle it.
    """
    assert runhistory.empty()
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)

    runhistory.add(
        config=config1,
        cost=[10, 20],
        time=20,
        status=StatusType.SUCCESS,
        instance=None,
        seed=None,
        starttime=100,
        endtime=120,
        additional_info=None,
    )

    runhistory.add(
        config=config1,
        cost=[4.5, 5.5],
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=12354,
        starttime=10,
        endtime=30,
        additional_info={"start_time": 10},
    )

    runhistory.add(
        config=config1,
        cost=["4.8", "5.8"],
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=12354,
        starttime=10,
        endtime=30,
        additional_info={"start_time": 10},
    )

    assert not runhistory.empty()

    tmpfile = tempfile.NamedTemporaryFile(mode="wb", delete=False)
    pickle.dump(runhistory, tmpfile, -1)
    name = tmpfile.name
    tmpfile.close()

    with open(name, "rb") as fh:
        loaded_runhistory = pickle.load(fh)  # nosec

    assert loaded_runhistory._data == runhistory._data


def test_illegal_input(scenario, runhistory, config1):
    assert runhistory.empty()
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)

    with pytest.raises(ValueError):
        runhistory.add(
            config=config1,
            cost=[4.5, 5.5, 6.5],
            time=20,
            status=StatusType.SUCCESS,
            instance=1,
            seed=12354,
            starttime=10,
            endtime=30,
            additional_info={"start_time": 10},
        )

        runhistory.add(
            config=config1,
            cost=[2.5, 5.5],
            time=20,
            status=StatusType.SUCCESS,
            instance=1,
            seed=12354,
            starttime=10,
            endtime=30,
            additional_info={"start_time": 10},
        )


def test_add_multiple_times(scenario, runhistory, config1):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)

    for i in range(5):
        runhistory.add(
            config=config1,
            cost=[i + 1, i + 2],
            time=i + 1,
            status=StatusType.SUCCESS,
            instance=None,
            seed=12345,
            additional_info=None,
        )

    assert len(runhistory._data) == 1
    assert len(runhistory.get_trials(config1, highest_observed_budget_only=True)) == 1
    assert len(runhistory._config_id_to_isk_to_budget[1]) == 1

    # We expect to get 1.0 and 2.0 because runhistory does not overwrite by default
    assert list(runhistory._data.values())[0].cost == [1.0, 2.0]


def test_full(scenario, runhistory, config1, config2, config3):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)
    runhistory.add(
        config=config1,
        cost=[50, 100],
        time=20,
        status=StatusType.SUCCESS,
    )

    # Only one value: Normalization goes to 1.0
    assert runhistory.get_cost(config1) == 1.0

    runhistory.add(
        config=config2,
        cost=[150, 50],
        time=30,
        status=StatusType.SUCCESS,
    )

    # The cost of the first config must be updated
    # We would expect [0, 1] and the normalized value would be 0.5
    assert runhistory.get_cost(config1) == 0.5

    # We would expect [1, 0] and the normalized value would be 0.5
    assert runhistory.get_cost(config2) == 0.5

    runhistory.add(
        config=config3,
        cost=[100, 0],
        time=40,
        status=StatusType.SUCCESS,
    )

    # [0, 1] -> 0.5
    assert runhistory.get_cost(config1) == 0.5

    # [1, 0.5] -> 0.75
    assert runhistory.get_cost(config2) == 0.75

    # [0.5, 0] -> 0.25
    assert runhistory.get_cost(config3) == 0.25


def test_full_update(scenario, runhistory, config1, config2):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)
    runhistory.overwrite_existing_runs = True

    runhistory.add(
        config=config1,
        cost=[10, 40],
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
    )

    runhistory.add(
        config=config1,
        cost=[0, 100],
        time=20,
        status=StatusType.SUCCESS,
        instance=2,
        seed=2,
    )

    runhistory.add(
        config=config2,
        cost=[10, 40],
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
    )

    runhistory.add(
        config=config2,
        cost=[20, 80],
        time=20,
        status=StatusType.SUCCESS,
        instance=2,
        seed=2,
    )

    cost_config2 = runhistory.get_cost(config2)

    runhistory.update_costs()
    updated_cost_config2 = runhistory.get_cost(config2)

    assert cost_config2 == updated_cost_config2

    runhistory.update_costs(instances=[2])
    updated_cost_config2 = runhistory.get_cost(config2)

    assert updated_cost_config2 == pytest.approx(0.833, 0.001)


def test_incremental_update(scenario, runhistory, config1):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)

    runhistory.add(
        config=config1,
        cost=[10, 100],
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
    )

    assert runhistory.get_cost(config1) == 1.0

    runhistory.add(
        config=config1,
        cost=[20, 50],
        time=20,
        status=StatusType.SUCCESS,
        instance=2,
        seed=1,
    )

    # We don't except moving average of 0.75 here because
    # the costs should always be updated.
    assert runhistory.get_cost(config1) == 0.5

    runhistory.add(
        config=config1,
        cost=[0, 100],
        time=20,
        status=StatusType.SUCCESS,
        instance=3,
        seed=1,
    )

    assert runhistory.get_cost(config1) == pytest.approx(0.583, 0.001)


def test_multiple_budgets(scenario, runhistory, config1):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)

    runhistory.add(
        config=config1,
        cost=[10, 50],
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    assert runhistory.get_cost(config1) == 1.0

    # Only the higher budget gets included in the config cost
    # However, we expect that the bounds are changed
    runhistory.add(
        config=config1,
        cost=[20, 25],
        time=25,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=5,
    )

    assert runhistory.get_cost(config1) == 0.5


def test_get_configs_per_budget(scenario, runhistory, config1, config2, config3):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)
    runhistory.add(
        config=config1,
        cost=[10, 20],
        time=10,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    runhistory.add(
        config=config2,
        cost=[20, 30],
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    runhistory.add(
        config=config3,
        cost=[30, 40],
        time=30,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=3,
    )

    configs = runhistory.get_configs_per_budget([1])
    assert configs == [config1, config2]


def test_objective_bounds(scenario, runhistory, config1, config2, config3):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)

    runhistory.add(
        config=config1,
        cost=[10, 50],
        time=5,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    runhistory.add(
        config=config2,
        cost=[5, 100],
        time=10,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    runhistory.add(
        config=config3,
        cost=[7.5, 150],
        time=15,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    assert runhistory._objective_bounds[0] == (5, 10)
    assert runhistory._objective_bounds[1] == (50, 150)


def test_objective_bounds2(scenario, runhistory, config1, config2, config3):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)
    runhistory.add(
        config=config1,
        cost=10,
        time=5,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    runhistory.add(
        config=config2,
        cost=5,
        time=10,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    runhistory.add(
        config=config3,
        cost=7.5,
        time=15,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    assert runhistory._objective_bounds[0] == (5, 10)


def test_bounds_on_crash(scenario, runhistory, config1, config2, config3):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)

    runhistory.add(
        config=config1,
        cost=[10, 50],
        time=5,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    runhistory.add(
        config=config2,
        cost=[100, 100],
        time=10,
        status=StatusType.CRASHED,
        instance=1,
        seed=1,
        budget=1,
    )

    runhistory.add(
        config=config3,
        cost=[0, 150],
        time=15,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    assert runhistory._objective_bounds[0] == (0, 10)
    assert runhistory._objective_bounds[1] == (50, 150)


def test_instances(scenario, runhistory, config1, config2):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)

    runhistory.add(
        config=config1,
        cost=[0, 10],
        time=5,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=0,
    )

    runhistory.add(
        config=config1,
        cost=[50, 20],
        time=10,
        status=StatusType.SUCCESS,
        instance=2,
        seed=1,
        budget=0,
    )

    runhistory.add(
        config=config1,
        cost=[75, 20],
        time=10,
        status=StatusType.SUCCESS,
        instance=3,
        seed=1,
        budget=0,
    )

    runhistory.add(
        config=config2,
        cost=[100, 30],
        time=15,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=0,
    )

    runhistory.add(
        config=config2,
        cost=[0, 30],
        time=15,
        status=StatusType.SUCCESS,
        instance=2,
        seed=1,
        budget=0,
    )

    assert runhistory._objective_bounds[0] == (0, 100)
    assert runhistory._objective_bounds[1] == (10, 30)

    # Average cost returns us the cost of the latest budget
    assert runhistory.get_cost(config1) == 0.375
    assert runhistory.get_cost(config2) == 0.75


def test_budgets(scenario, runhistory, config1, config2):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)

    runhistory.add(
        config=config1,
        cost=[0, 50],
        time=5,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=5,
    )

    runhistory.add(
        config=config1,
        cost=[40, 100],
        time=10,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=15,
    )

    # SMAC does not overwrite by default
    runhistory.add(
        config=config1,
        cost=[502342352, 23425234],
        time=11,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=15,
    )

    runhistory.add(
        config=config2,
        cost=[0, 150],
        time=15,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=5,
    )

    assert runhistory._objective_bounds[0] == (0, 40)
    assert runhistory._objective_bounds[1] == (50, 150)

    # Average cost returns us the cost of the latest budget
    assert runhistory.get_cost(config1) == 0.75
    assert runhistory.average_cost(config1), [40.0, 100.0]

    assert runhistory.get_cost(config2) == 0.5
    assert runhistory.average_cost(config2) == [0, 150]


def test_objective_weights(scenario, runhistory, config1, config2):
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario)
    runhistory.add(
        config=config1,
        cost=[0, 10],
        time=5,
        status=StatusType.SUCCESS,
    )

    runhistory.add(
        config=config2,
        cost=[100, 0],
        time=15,
        status=StatusType.SUCCESS,
    )

    assert runhistory._objective_bounds[0] == (0, 100)
    assert runhistory._objective_bounds[1] == (0, 10)

    # Average cost returns us 0.5
    assert runhistory.get_cost(config1) == 0.5

    # If we change the weights/mo algorithm now, we expect a higher value in the second cost
    runhistory.multi_objective_algorithm = MeanAggregationStrategy(scenario, objective_weights=[1, 2])
    assert round(runhistory.get_cost(config1), 2) == 0.67
