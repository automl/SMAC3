import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace

from smac.runhistory import StatusType
from smac.runner.target_function_script_runner import TargetFunctionScriptRunner


@pytest.fixture
def configspace():
    return ConfigurationSpace({"x0": (0, 1000)}, seed=0)


def test_success(configspace, make_scenario):
    script = "tests/test_runner/files/success.sh"
    scenario = make_scenario(configspace, use_instances=True)
    runner = TargetFunctionScriptRunner(script, scenario, required_arguments=["seed", "instance"])

    config = configspace.get_default_configuration()
    status, cost, runtime, additional_info = runner.run(config, instance=scenario.instances[0], seed=0)

    assert status == StatusType.SUCCESS
    assert cost == config["x0"]
    assert additional_info == {"additional_info": "blub"}


def test_success_multi_objective(configspace, make_scenario):
    script = "tests/test_runner/files/success_multi_objective.sh"
    scenario = make_scenario(configspace, use_instances=True, use_multi_objective=True)
    runner = TargetFunctionScriptRunner(script, scenario, required_arguments=["seed", "instance"])

    config = configspace.get_default_configuration()
    status, cost, runtime, additional_info = runner.run(config, instance=scenario.instances[0], seed=0)

    assert status == StatusType.SUCCESS
    assert cost == [config["x0"], config["x0"]]
    assert additional_info == {}


def test_exit(configspace, make_scenario):
    script = "tests/test_runner/files/exit.sh"
    scenario = make_scenario(configspace, use_instances=True)
    runner = TargetFunctionScriptRunner(script, scenario, required_arguments=["seed", "instance"])

    config = configspace.get_default_configuration()
    status, cost, runtime, additional_info = runner.run(config, instance=scenario.instances[0], seed=0)

    assert status == StatusType.CRASHED
    assert "error" in additional_info


def test_crashed(configspace, make_scenario):
    script = "tests/test_runner/files/crashed.sh"
    scenario = make_scenario(configspace, use_instances=True)
    runner = TargetFunctionScriptRunner(script, scenario, required_arguments=["seed", "instance"])

    config = configspace.get_default_configuration()
    status, cost, runtime, additional_info = runner.run(config, instance=scenario.instances[0], seed=0)

    assert status == StatusType.CRASHED
    assert cost == np.inf


def test_python(configspace, make_scenario):
    script = "tests/test_runner/files/python.py"
    scenario = make_scenario(configspace, use_instances=True)
    runner = TargetFunctionScriptRunner(script, scenario, required_arguments=["seed", "instance"])

    config = configspace.get_default_configuration()
    status, cost, runtime, additional_info = runner.run(config, instance=scenario.instances[0], seed=0)

    assert status == StatusType.SUCCESS
    assert cost == config["x0"]
