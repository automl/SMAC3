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


def test_constraints_default_to_none(scenario1: Scenario) -> None:
    """An unconstrained scenario reports no constraints.

    Reads the field and both accessors on a scenario built without the argument.
    """
    assert scenario1.constraints is None
    assert scenario1.count_constraints() == 0
    assert scenario1.get_constraints() == []


def test_constraints_are_parsed_on_access(configspace: ConfigurationSpace) -> None:
    """Declared expressions become constraint objects while staying strings on the scenario.

    Builds a scenario with two expressions and compares the stored field against the parsed accessor.
    """
    scenario = Scenario(configspace, constraints=["latency <= 100", "accuracy >= 0.9"])

    assert scenario.constraints == ["latency <= 100", "accuracy >= 0.9"]
    assert scenario.count_constraints() == 2
    assert [c.name for c in scenario.get_constraints()] == ["latency", "accuracy"]
    assert [c.bound for c in scenario.get_constraints()] == [100.0, 0.9]


def test_constraints_do_not_count_as_objectives(configspace: ConfigurationSpace) -> None:
    """Constraining an output does not turn the run multi-objective.

    Checks count_objectives on a single-objective scenario that declares two constraints.
    """
    scenario = Scenario(configspace, objectives="error", constraints=["latency <= 100", "memory <= 4096"])

    assert scenario.count_objectives() == 1


def test_malformed_constraint_is_rejected_on_construction(configspace: ConfigurationSpace) -> None:
    """A scenario refuses to be built around an unparseable constraint.

    Passes an expression with an unsupported operator and expects construction to fail.
    """
    with pytest.raises(ValueError, match="Could not parse"):
        Scenario(configspace, constraints=["latency < 100"])


def test_an_output_cannot_be_both_objective_and_constraint(configspace: ConfigurationSpace) -> None:
    """The same name cannot be optimized and bounded at once.

    Declares a constraint on the objective's own name and expects a ValueError.
    """
    with pytest.raises(ValueError, match="both an objective and a constraint"):
        Scenario(configspace, objectives="error", constraints=["error <= 1.0"])

    with pytest.raises(ValueError, match="both an objective and a constraint"):
        Scenario(configspace, objectives=["error", "time"], constraints=["time <= 1.0"])


def test_constraints_survive_a_save_and_load(configspace: ConfigurationSpace) -> None:
    """Constraints round trip through the scenario file unchanged.

    Saves a constrained scenario, reloads it, and compares the expressions and the whole scenario.
    """
    scenario = Scenario(
        configspace,
        name="test_constraint_scenario",
        output_directory=Path("smac3_output_test"),
        constraints=["latency <= 100", "accuracy >= 0.9"],
    )
    scenario._set_meta({"test": "meta"})
    scenario.save()

    reloaded = Scenario.load(scenario.output_directory)

    assert reloaded.constraints == ["latency <= 100", "accuracy >= 0.9"]
    assert reloaded.get_constraints() == scenario.get_constraints()
    assert reloaded == scenario
