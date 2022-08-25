import time
from smac.runhistory import RunInfo, RunValue
from smac.runner.runner import StatusType
from smac.runner.serial_runner import SerialRunner
from smac.runner.target_algorithm_runner import TargetAlgorithmRunner

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def target(x, seed, instance):
    return x**2, {"key": seed, "instance": instance}


def target_delayed(x, seed, instance):
    time.sleep(1)
    return x**2, {"key": seed, "instance": instance}


def target_failed(x, seed, instance):
    raise RuntimeError("Failed.")


def target_dummy(config, seed, instance, budget):
    return seed


def target_multi_objective1(config, seed, instance, budget):
    return [seed, seed]


def target_multi_objective2(config, seed, instance, budget):
    return {"cost1": seed, "cost2": seed}


def test_run(configspace_small, make_stats, make_scenario):
    """Makes sure that we are able to run a configuration and
    return the expected values/types"""
    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = TargetAlgorithmRunner(target_algorithm=target, scenario=scenario, stats=stats)
    assert isinstance(runner, SerialRunner)

    run_info = RunInfo(
        config=2,
        instance="test",
        seed=0,
        budget=0.0,
    )

    # submit runs! then get the value
    runner.submit_run(run_info)
    result = next(runner.iter_results(), None)

    assert result is not None

    run_info, run_value = result

    assert run_value.cost == 4
    assert run_value.status == StatusType.SUCCESS


def test_serial_runs(configspace_small, make_stats, make_scenario):
    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = TargetAlgorithmRunner(target_algorithm=target_delayed, scenario=scenario, stats=stats)
    assert isinstance(runner, SerialRunner)

    run_info = RunInfo(
        config=2,
        instance="test",
        seed=0,
        budget=0.0,
    )
    runner.submit_run(run_info)

    run_info = RunInfo(
        config=3,
        instance="test",
        seed=0,
        budget=0.0,
    )
    runner.submit_run(run_info)

    run_values = list(runner.iter_results())
    assert len(run_values) == 2

    # To make sure runs launched serially, we just make sure that the end time of
    # a run is later than the other
    # Results are returned in left to right
    assert int(run_values[1][1].endtime) <= int(run_values[0][1].starttime)

    # No wait time in serial runs!
    start = time.time()
    runner.wait()

    # The run takes a second, so 0.5 is sufficient
    assert time.time() - start < 0.5


def test_fail(configspace_small, make_stats, make_scenario):
    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = TargetAlgorithmRunner(target_algorithm=target_failed, scenario=scenario, stats=stats)
    assert isinstance(runner, SerialRunner)

    run_info = RunInfo(
        config=2,
        instance="test",
        seed=0,
        budget=0.0,
    )
    runner.submit_run(run_info)
    run_info, run_value = next(runner.iter_results())

    # Make sure the traceback message is included
    assert "traceback" in run_value.additional_info
    assert "RuntimeError" in run_value.additional_info["traceback"]


def test_call(configspace_small, make_stats, make_scenario):
    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = TargetAlgorithmRunner(target_algorithm=target_dummy, scenario=scenario, stats=stats)
    assert isinstance(runner, SerialRunner)

    config = configspace_small.get_default_configuration()

    SEED = 2345
    status, cost, _, _ = runner.run(
        config=config,
        instance=None,
        seed=SEED,
        budget=None,
    )

    assert cost == SEED
    assert status == StatusType.SUCCESS


def test_multi_objective(configspace_small, make_stats, make_scenario):
    scenario = make_scenario(configspace_small, use_multi_objective=True)
    stats = make_stats(scenario)

    # We always expect a list of costs (although a dict is returned).
    # Internally, target algorithm runner maps the dict to a list of costs in the right order.
    for target in [target_multi_objective1, target_multi_objective2]:

        runner = TargetAlgorithmRunner(target_algorithm=target, scenario=scenario, stats=stats)
        assert isinstance(runner, SerialRunner)

        config = configspace_small.get_default_configuration()

        SEED = 2345
        status, cost, _, _ = runner.run(
            config=config,
            instance=None,
            seed=SEED,
            budget=None,
        )

        assert isinstance(cost, list)
        assert cost == [SEED, SEED]
        assert status == StatusType.SUCCESS
