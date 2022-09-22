import pytest

from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.runner.abstract_runner import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


@pytest.fixture
def intensifier(make_scenario, configspace_small):
    scenario = make_scenario(configspace_small)
    return AbstractIntensifier(scenario=scenario)


@pytest.fixture
def configs(configspace_small):
    return configspace_small.sample_configuration(3)


def test_compare_configs_no_joint_set(intensifier, runhistory, configs):
    for i in range(2):
        runhistory.add(
            config=configs[0],
            cost=2,
            time=2,
            status=StatusType.SUCCESS,
            instance=1,
            seed=i,
            additional_info=None,
        )

    for i in range(2, 5):
        runhistory.add(
            config=configs[1],
            cost=1,
            time=1,
            status=StatusType.SUCCESS,
            instance=1,
            seed=i,
            additional_info=None,
        )

    # The sets for the incumbent are completely disjoint.
    conf = intensifier._compare_configs(incumbent=configs[0], challenger=configs[1], runhistory=runhistory)
    assert conf is None

    # The incumbent has still one instance-seed pair left on which the
    # challenger was not run yet.
    runhistory.add(
        config=configs[1],
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance=1,
        seed=1,
        additional_info=None,
    )
    conf = intensifier._compare_configs(incumbent=configs[0], challenger=configs[1], runhistory=runhistory)
    assert conf is None


def test_compare_configs_chall(intensifier, runhistory, configs):
    """
    Challenger is better.
    """
    runhistory.add(
        config=configs[0],
        cost=1,
        time=2,
        status=StatusType.SUCCESS,
        instance=1,
        seed=None,
        additional_info=None,
    )

    runhistory.add(
        config=configs[1],
        cost=0,
        time=1,
        status=StatusType.SUCCESS,
        instance=1,
        seed=None,
        additional_info=None,
    )

    conf = intensifier._compare_configs(
        incumbent=configs[0],
        challenger=configs[1],
        runhistory=runhistory,
        log_trajectory=False,
    )

    # Challenger has enough runs and is better
    assert conf == configs[1]


def test_compare_configs_inc(intensifier, runhistory, configs):
    """
    Incumbent is better
    """

    runhistory.add(
        config=configs[0],
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance=1,
        seed=None,
        additional_info=None,
    )

    runhistory.add(
        config=configs[1],
        cost=2,
        time=2,
        status=StatusType.SUCCESS,
        instance=1,
        seed=None,
        additional_info=None,
    )

    conf = intensifier._compare_configs(incumbent=configs[0], challenger=configs[1], runhistory=runhistory)

    # Challenger worse than inc
    assert conf == configs[0]


def test_compare_configs_unknow(intensifier, runhistory, configs):
    """
    Challenger is better but has less runs;
    -> no decision (None)
    """

    runhistory.add(
        config=configs[0],
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance=1,
        seed=None,
        additional_info=None,
    )

    runhistory.add(
        config=configs[0],
        cost=1,
        time=2,
        status=StatusType.SUCCESS,
        instance=2,
        seed=None,
        additional_info=None,
    )

    runhistory.add(
        config=configs[0],
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance=2,
        seed=None,
        additional_info=None,
    )

    conf = intensifier._compare_configs(incumbent=configs[0], challenger=configs[1], runhistory=runhistory)

    # Challenger worse than inc
    assert conf is None
