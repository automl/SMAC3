import pytest

import numpy as np

from smac.runhistory import TrialInfoIntent
from smac.intensification.simple_intensifier import SimpleIntensifier
from smac.runhistory import TrialInfo, TrialValue
from smac.runner.runner import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


@pytest.fixture
def intensifier(make_scenario, make_stats, configspace_small):
    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)
    intensifier = SimpleIntensifier(scenario=scenario)
    intensifier.stats = stats

    return intensifier


@pytest.fixture
def configs(configspace_small):
    return configspace_small.sample_configuration(2)


def test_get_next_run(intensifier, runhistory, configs):
    """
    Makes sure that sampling a configuration returns a valid
    configuration.
    """
    intent, run_info = intensifier.get_next_run(
        challengers=[configs[0]],
        incumbent=None,
        runhistory=runhistory,
        n_workers=1,
        ask=None,
    )
    assert intent == TrialInfoIntent.RUN

    # @KEggensperger: Why is it instance 1 here?
    # run_info2 = RunInfo(
    #    config=configs[0],
    #    instance=1,
    #    seed=0,
    #    budget=0.0,
    # )

    assert run_info.config == configs[0]
    assert run_info.budget == 0.0
    assert run_info.instance is None
    assert run_info.seed != 0  # Random seed because of deterministic false


def test_get_next_run_waits_if_no_workers(intensifier, runhistory, configs):
    """
    In the case all workers are busy, we wait so that we do
    not saturate the process with configurations that will not
    finish in time
    """
    intent, run_info = intensifier.get_next_run(
        challengers=[configs[0], configs[1]],
        incumbent=None,
        runhistory=runhistory,
        n_workers=1,
        ask=None,
    )

    # Same as above
    # run_info2 = RunInfo(
    #    config=configs[0],
    #    instance=1,
    #    seed=0,
    #    budget=0.0,
    # )

    # We can get the configuration 1
    assert intent == TrialInfoIntent.RUN
    assert run_info.config == configs[0]
    assert run_info.budget == 0.0
    assert run_info.instance is None
    assert run_info.seed != 0

    # We should not get configuration 2
    # As there is just 1 worker
    intent, run_info = intensifier.get_next_run(
        challengers=[configs[1]],
        incumbent=None,
        runhistory=runhistory,
        n_workers=1,
        ask=None,
    )
    assert intent == TrialInfoIntent.WAIT

    run_info2 = TrialInfo(
        config=None,
        instance=None,
        seed=0,
        budget=0.0,
    )

    assert run_info == run_info2


def test_process_results(intensifier, runhistory, configs):
    """
    Makes sure that we can process the results of a completed
    configuration
    """
    intent, run_info = intensifier.get_next_run(
        challengers=[configs[0], configs[1]],
        incumbent=None,
        runhistory=runhistory,
        n_workers=1,
        ask=None,
    )

    run_value = TrialValue(
        cost=1,
        time=0.5,
        status=StatusType.SUCCESS,
        starttime=1,
        endtime=2,
        additional_info=None,
    )

    runhistory.add(
        config=run_info.config,
        cost=1,
        time=0.5,
        status=StatusType.SUCCESS,
        instance=run_info.instance,
        seed=run_info.seed,
        additional_info=None,
    )

    incumbent, inc_perf = intensifier.process_results(
        run_info=run_info,
        run_value=run_value,
        incumbent=None,
        runhistory=runhistory,
        time_bound=np.inf,
    )

    assert incumbent == run_info.config
    assert inc_perf == 1
