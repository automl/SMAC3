from pathlib import Path

import pytest
from ConfigSpace import ConfigurationSpace

from smac import Scenario


@pytest.fixture
def configspace() -> ConfigurationSpace:
    return ConfigurationSpace({"a": (0, 5)})


@pytest.fixture
def scenario1(configspace: ConfigurationSpace) -> Scenario:
    return Scenario(configspace, output_directory=Path("smac3_output_test"), n_trials=50)


@pytest.fixture
def scenario2(configspace: ConfigurationSpace) -> Scenario:
    return Scenario(configspace, output_directory=Path("smac3_output_test"), n_trials=50)


@pytest.fixture
def scenario3(configspace: ConfigurationSpace) -> Scenario:
    return Scenario(
        configspace,
        name="test_scenario",
        output_directory=Path("smac3_output_test"),
        n_trials=20,
        seed=5,
    )


@pytest.fixture
def scenario4(configspace: ConfigurationSpace) -> Scenario:
    return Scenario(configspace, objectives=["test", "blub"])


@pytest.fixture
def scenario5(configspace: ConfigurationSpace) -> Scenario:
    return Scenario(
        configspace,
        name="test_scenario",
        output_directory=Path("smac3_output_test"),
        n_trials=100,
        seed=5,
        instances=["i1", "i2", "i3"],
        instance_features={"i1": [1, 2, 3], "i2": [4, 5, 6], "i3": [7, 8, 9]},
    )


@pytest.fixture
def scenario6(configspace: ConfigurationSpace) -> Scenario:
    return Scenario(
        configspace,
        name="test_scenario",
        output_directory=Path("smac3_output_test"),
        instances=["i1", "i2", "i3"],
        instance_features={"i1": [1, 2, 3], "i2": [4, 5], "i3": [7, 8, 9]},
    )


@pytest.fixture
def scenario7(configspace: ConfigurationSpace) -> Scenario:
    return Scenario(
        configspace,
        name="test_scenario",
        output_directory=Path("smac3_output_test"),
        instances=["i1", "i2", "i3"],
        instance_features={"blub": [1, 2, 3], "i2": [4, 5, 6], "i3": [7, 8, 9]},
    )


def test_comparison(scenario1: Scenario, scenario2: Scenario, scenario3: Scenario) -> None:
    assert scenario1 == scenario2
    assert scenario1 != scenario3


def test_directory(scenario3: Scenario) -> None:
    assert str(scenario3.output_directory) == "smac3_output_test/test_scenario/5"


def test_frozen(scenario1: Scenario) -> None:
    with pytest.raises(Exception):
        scenario1.deterministic = False


def test_objectives(scenario3: Scenario, scenario4: Scenario) -> None:
    assert scenario3.count_objectives() == 1
    assert scenario4.count_objectives() == 2


def test_save_load(scenario1: Scenario, scenario3: Scenario) -> None:
    # This should fail because we don't know the name of the scenario as meta data are not defined either
    with pytest.raises(RuntimeError):
        scenario1.save()

    # If we set meta data, it should work
    meta = {"test": {"test": "test"}}
    scenario1._set_meta(meta)
    scenario1.save()

    # We reload the scenario again and it should be the same as before
    reloaded_scenario = Scenario.load(scenario1.output_directory)
    assert scenario1 == reloaded_scenario

    # Do it one more time with scenario 3
    scenario3.save()
    reloaded_scenario = Scenario.load(scenario3.output_directory)
    assert scenario3 == reloaded_scenario


def test_instances(scenario5: Scenario, scenario6: Scenario, scenario7: Scenario) -> None:
    assert scenario5.count_instance_features() == 3

    with pytest.raises(RuntimeError, match="Instances must have the same number of features"):
        scenario6.count_instance_features()

    with pytest.raises(RuntimeError, match="Instance blub is not specified"):
        scenario7.count_instance_features()
