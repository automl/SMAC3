from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np
import pytest
from ConfigSpace import Configuration

from smac.runhistory.runhistory import RunHistory, TrialKey
from smac.runner.abstract_runner import StatusType
from smac.utils.cost_transformer import CostTransformer

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


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


def test_add_and_pickle(runhistory, config1):
    """
    simply adding some rundata to runhistory, then pickle it
    """
    assert runhistory.empty()

    runhistory.add(
        config=config1,
        cost=10,
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
        cost=10,
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
        loaded_runhistory = pickle.load(fh)

    assert loaded_runhistory._data == runhistory._data


def test_illegal_input(runhistory):
    with pytest.raises(TypeError, match="Configuration must not be None."):
        runhistory.add(config=None, cost=1.23, time=2.34, status=StatusType.SUCCESS)

    with pytest.raises(
        TypeError,
        match="Configuration is not of type Configuration, but .*str.*",
    ):
        runhistory.add(config="abc", cost=1.23, time=2.34, status=StatusType.SUCCESS)


def test_add_multiple_times(runhistory, config1):
    for i in range(5):
        runhistory.add(
            config=config1,
            cost=i + 1,
            time=i + 1,
            status=StatusType.SUCCESS,
            instance=None,
            seed=12345,
            additional_info=None,
            budget=0,
        )

    assert len(runhistory._data) == 1
    assert len(runhistory.get_trials(config1, highest_observed_budget_only=True)) == 1
    assert len(runhistory._config_id_to_isk_to_budget[1]) == 1
    assert list(runhistory._data.values())[0].cost == 1


def test_get_config_runs(runhistory, config1, config2):
    """
    get some config runs from runhistory
    """
    # Return max observed budget only
    runhistory.add(
        config=config1,
        cost=10,
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )
    runhistory.add(
        config=config1,
        cost=10,
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=2,
    )

    # Why should this not happen?
    # with pytest.raises(ValueError, "This should not happen!"):
    #    runhistory.add(
    #        config=config1,
    #        cost=10,
    #        time=20,
    #        status=StatusType.SUCCESS,
    #        instance=2,
    #        seed=2,
    #        budget=1,
    #    )

    runhistory.add(
        config=config2,
        cost=10,
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    ist = runhistory.get_trials(config=config1, highest_observed_budget_only=True)

    # assert len(ist) == 2
    assert len(ist) == 1

    assert ist[0].instance == 1
    # assert ist[1].instance == 2
    assert ist[0].budget == 2
    # assert ist[1].budget == 1


def test_get_config_runs2(runhistory, config1, config2):
    """
    get some config runs from runhistory (multiple budgets (highest_observed_budget_only=False))
    """
    runhistory.add(
        config=config1,
        cost=5,
        time=10,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )
    runhistory.add(
        config=config1,
        cost=10,
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=2,
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
        config=config2,
        cost=10,
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=2,
    )

    ist = runhistory.get_trials(config=config1, highest_observed_budget_only=False)

    assert len(ist) == 2
    assert ist[0].instance == 1
    assert ist[0].budget == 1
    assert ist[1].budget == 2


def test_cost_transformer():
    raw_costs = [10.0, 20.0, 30.0]

    assert CostTransformer.mean(raw_costs) == pytest.approx(20.0)
    assert CostTransformer.sum(raw_costs) == pytest.approx(60.0)
    assert CostTransformer.min(raw_costs) == pytest.approx(10.0)


def test_incremental_update(runhistory, config1, compute_cost):
    runhistory.add(
        config=config1,
        cost=10,
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
    )

    assert compute_cost(runhistory, config1) == 10

    runhistory.add(
        config=config1,
        cost=20,
        time=20,
        status=StatusType.SUCCESS,
        instance=2,
        seed=1,
    )

    assert compute_cost(runhistory, config1) == 15


def test_multiple_budgets(runhistory, config1, compute_cost):
    runhistory.add(
        config=config1,
        cost=10,
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    assert compute_cost(runhistory, config1) == 10

    # only the higher budget gets included in the config cost
    runhistory.add(
        config=config1,
        cost=20,
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=2,
    )

    assert compute_cost(runhistory, config1) == 20
    raw_costs = runhistory.get_costs(config1, highest_observed_budget_only=False)
    assert CostTransformer.min(raw_costs) == 10


def test_get_configs_per_budget(runhistory, config1, config2, config3):
    runhistory.add(
        config=config1,
        cost=10,
        time=10,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    runhistory.add(
        config=config2,
        cost=20,
        time=20,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=1,
    )

    runhistory.add(
        config=config3,
        cost=30,
        time=30,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        budget=3,
    )

    assert runhistory.get_configs_per_budget([1]) == [config1, config2]


def test_json_origin(configspace_small, config1):
    for i, origin in enumerate(["test_origin", None]):
        config1.origin = origin
        runhistory = RunHistory()
        runhistory.add(
            config=config1,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance=1,
            seed=1,
        )

        path = f"tests/test_files/test_json_origin_{i}.json"
        runhistory.save(path)
        runhistory.load(path, configspace_small)

        assert runhistory.get_configs()[0].origin == origin

        os.remove(path)


def add_item(
    runhistory,
    config,
    cost: float = 0.0,
    time: float = 1.0,
    seed: int = 0,
    instance: str | None = None,
    budget: float | None = None,
    status=StatusType.SUCCESS,
) -> TrialKey:
    """No easy way to generate a key before hand."""
    runhistory.add(
        config=config,
        cost=cost,
        time=time,
        instance=instance,
        seed=seed,
        budget=budget,
        status=status,
    )

    return TrialKey(
        config_id=runhistory._n_id,  # What's used internally during `add`
        instance=instance,
        seed=seed,
        budget=budget,
    )


def test_contains(runhistory, config1):
    """Test that keys added are contained `in` and not if not added"""
    k = add_item(runhistory, config1)
    assert k in runhistory

    new_rh = RunHistory()
    assert k not in new_rh


def test_getting(runhistory, config1, config2):
    """Test that rh[k] will return the correct value"""
    k = add_item(runhistory, config1, cost=1.0)
    k2 = add_item(runhistory, config2, cost=2.0)

    v = runhistory[k]
    assert v.cost == 1.0

    v2 = runhistory[k2]
    assert v2.cost == 2.0


def test_len(runhistory, config1):
    """Test that adding items will increase the length monotonically"""
    assert len(runhistory) == 0

    n_items = 5

    for i in range(n_items):
        assert len(runhistory) == i

        add_item(runhistory, config1, budget=i)

        assert len(runhistory) == i + 1

    assert len(runhistory) == n_items


def test_iter(runhistory, config1):
    """Test that iter goes in the order of insertion and has consitent length
    with the runhistory's advertised length and it's internal `data`
    """
    params = [
        {"instance": "a", "cost": 1.0},
        {"instance": "b", "cost": 2.0},
        {"instance": "c", "cost": 3.0},
        {"instance": "d", "cost": 4.0},
    ]

    for p in params:
        add_item(runhistory, config1, **p)

    expected_id_order = [p["instance"] for p in params]
    assert [k.instance for k in iter(runhistory)] == expected_id_order

    assert len(list(iter(runhistory))) == len(runhistory)
    assert len(list(iter(runhistory))) == len(runhistory._data)


def test_items(runhistory, config1):
    """Test that items goes in correct insertion order and returns key values
    as expected.
    """
    params = [
        {"instance": "a", "cost": 1.0},
        {"instance": "b", "cost": 2.0},
        {"instance": "c", "cost": 3.0},
        {"instance": "d", "cost": 4.0},
    ]

    for p in params:
        add_item(runhistory, config1, **p)

    for (k, v), expected in zip(runhistory.items(), params):
        assert k.instance == expected["instance"]
        assert v.cost == expected["cost"]


def test_unpack(runhistory, config1):
    """Test that unpacking maintains order and returns key values as expected"""
    params = [
        {"instance": "a", "cost": 1.0},
        {"instance": "b", "cost": 2.0},
        {"instance": "c", "cost": 3.0},
        {"instance": "d", "cost": 4.0},
    ]

    for p in params:
        add_item(runhistory, config1, **p)

    unpacked = {**runhistory}

    for (k, v), expected in zip(unpacked.items(), params):
        assert k.instance == expected["instance"]
        assert v.cost == expected["cost"]


def test_numpy_and_native_types_are_the_same_config(runhistory, configspace_small):
    """Regression test for #1241.

    ``Configuration.__hash__`` hashes ``repr(config)`` while ``__eq__`` compares values, so two
    equal configurations hash differently as soon as one of them carries numpy scalars. The
    runhistory used to store both of them, and the config selector eventually ran out of new
    configurations.
    """
    native = Configuration(configspace_small, values={"a": 5, "b": 1e-2, "c": "dog"})
    numpyish = Configuration(
        configspace_small,
        values={"a": np.int64(5), "b": np.float64(1e-2), "c": np.str_("dog")},
    )

    # This is the upstream inconsistency the fix has to work around.
    assert native == numpyish
    assert hash(native) != hash(numpyish)

    runhistory.add(config=native, cost=1.0, time=1.0, status=StatusType.SUCCESS, seed=0)
    runhistory.add(config=numpyish, cost=1.0, time=1.0, status=StatusType.SUCCESS, seed=0)

    assert len(runhistory.config_ids) == 1
    assert runhistory.finished == 1


def test_config_can_be_looked_up_with_numpy_types(runhistory, configspace_small):
    """Regression test for #1241: adding a configuration and looking it up stay consistent.

    Normalising only on write would make the runhistory forget the very configuration it was
    just given, because the lookup would still hash the original numpy-typed object.
    """
    numpyish = Configuration(
        configspace_small,
        values={"a": np.int64(7), "b": np.float64(1e-3), "c": np.str_("cat")},
    )

    runhistory.add(config=numpyish, cost=2.0, time=1.0, status=StatusType.SUCCESS, seed=0)

    assert runhistory.has_config(numpyish)
    assert runhistory.get_config_id(numpyish) == numpyish.config_id
    assert len(runhistory.get_trials(numpyish)) == 1
    assert len(runhistory.get_instance_seed_budget_keys(numpyish)) == 1
    assert runhistory.get_costs(numpyish) == [2.0]

    # The very same configuration expressed with built-in types has to resolve to the same entry.
    native = Configuration(configspace_small, values={"a": 7, "b": 1e-3, "c": "cat"})
    assert runhistory.get_config_id(native) == runhistory.get_config_id(numpyish)
