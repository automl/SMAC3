import numpy as np
import pytest

from smac.intensification.simple_intensifier import SimpleIntensifier
from smac.runhistory import TrialInfo, TrialInfoIntent, TrialValue
from smac.runner.abstract_runner import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


@pytest.fixture
def intensifier(make_scenario, make_stats, configspace_small):
    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)
    intensifier = SimpleIntensifier(scenario=scenario)
    intensifier._stats = stats

    return intensifier


@pytest.fixture
def configs(configspace_small):
    return configspace_small.sample_configuration(2)


def test_get_next_run(intensifier, runhistory, configs):
    """
    Makes sure that sampling a configuration returns a valid
    configuration.
    """
    intent, trial_info = intensifier.get_next_run(
        challengers=[configs[0]],
        incumbent=None,
        runhistory=runhistory,
        n_workers=1,
        get_next_configurations=None,
    )
    assert intent == TrialInfoIntent.RUN

    # @KEggensperger: Why is it instance 1 here?
    # trial_info2 = RunInfo(
    #    config=configs[0],
    #    instance=1,
    #    seed=0,
    #    budget=0.0,
    # )

    assert trial_info.config == configs[0]
    assert trial_info.budget is None
    assert trial_info.instance is None
    assert trial_info.seed != 0  # Random seed because of deterministic false


def test_get_next_run_waits_if_no_workers(intensifier, runhistory, configs):
    """
    In the case all workers are busy, we wait so that we do
    not saturate the process with configurations that will not
    finish in time
    """
    intent, trial_info = intensifier.get_next_run(
        challengers=[configs[0], configs[1]],
        incumbent=None,
        runhistory=runhistory,
        n_workers=1,
        get_next_configurations=None,
    )

    # Same as above
    # trial_info2 = RunInfo(
    #    config=configs[0],
    #    instance=1,
    #    seed=0,
    #    budget=0.0,
    # )

    # We can get the configuration 1
    assert intent == TrialInfoIntent.RUN
    assert trial_info.config == configs[0]
    assert trial_info.budget is None
    assert trial_info.instance is None
    assert trial_info.seed != 0

    # We should not get configuration 2
    # As there is just 1 worker
    intent, trial_info = intensifier.get_next_run(
        challengers=[configs[1]],
        incumbent=None,
        runhistory=runhistory,
        n_workers=1,
        get_next_configurations=None,
    )
    assert intent == TrialInfoIntent.WAIT

    trial_info2 = TrialInfo(
        config=None,
        instance=None,
        seed=None,
        budget=None,
    )

    assert trial_info == trial_info2


def test_process_results(intensifier, runhistory, configs):
    """
    Makes sure that we can process the results of a completed
    configuration
    """
    intent, trial_info = intensifier.get_next_run(
        challengers=[configs[0], configs[1]],
        incumbent=None,
        runhistory=runhistory,
        n_workers=1,
        get_next_configurations=None,
    )

    trial_value = TrialValue(
        cost=1,
        time=0.5,
        status=StatusType.SUCCESS,
        starttime=1,
        endtime=2,
        additional_info=None,
    )

    runhistory.add(
        config=trial_info.config,
        cost=1,
        time=0.5,
        status=StatusType.SUCCESS,
        instance=trial_info.instance,
        seed=trial_info.seed,
        additional_info=None,
    )

    incumbent, inc_perf = intensifier.process_results(
        trial_info=trial_info,
        trial_value=trial_value,
        incumbent=None,
        runhistory=runhistory,
        time_bound=np.inf,
    )

    assert incumbent == trial_info.config
    assert inc_perf == 1
