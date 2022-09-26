import unittest
from unittest import mock

from smac.intensifier.abstract_parallel_intensifier import AbstractParallelIntensifier
from smac.runhistory import TrialInfo, TrialInfoIntent, TrialValue
from smac.runner.abstract_runner import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def mock_ranker(sh):
    return sh._stage, len(sh.run_tracker)


def test_sort_instances_by_stage(make_scenario, make_stats, configspace_small, runhistory):
    """Ensures that we prioritize the more advanced stage iteration"""
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3, deterministic=True)
    stats = make_stats(scenario)
    scheduler = AbstractParallelIntensifier(scenario=scenario)
    scheduler._stats = stats

    scheduler._get_intensifier_ranking = mock_ranker

    def add_sh_mock(stage, config_inst_pairs):
        sh = mock.Mock()
        sh.run_tracker = []
        for i in range(config_inst_pairs):
            sh.run_tracker.append((i, i, i))
        sh._stage = stage
        return sh

    # Add more SH to make testing interesting
    instances = {}
    instances[0] = add_sh_mock(stage=1, config_inst_pairs=6)
    instances[1] = add_sh_mock(stage=1, config_inst_pairs=2)

    # We only have two configurations in the same stage.
    # In this case, we want to prioritize the one with more launched runs
    # that is zero
    assert list(scheduler._sort_instances_by_stage(instances)) == [0, 1]

    # One more instance comparison to be supper safe
    instances[2] = add_sh_mock(stage=1, config_inst_pairs=7)
    assert list(scheduler._sort_instances_by_stage(instances)) == [2, 0, 1]

    # Not let us add a more advanced stage run
    instances[3] = add_sh_mock(stage=2, config_inst_pairs=1)
    assert list(scheduler._sort_instances_by_stage(instances)) == [3, 2, 0, 1]

    # Make 1 the oldest stage
    instances[1] = add_sh_mock(stage=4, config_inst_pairs=1)
    assert list(scheduler._sort_instances_by_stage(instances)) == [1, 3, 2, 0]

    # Add a new run that's empty
    instances[4] = add_sh_mock(stage=0, config_inst_pairs=0)
    assert list(scheduler._sort_instances_by_stage(instances)) == [1, 3, 2, 0, 4]

    # Make 4 stage 4 but with not as many instances as 1
    instances[4] = add_sh_mock(stage=4, config_inst_pairs=0)
    assert list(scheduler._sort_instances_by_stage(instances)) == [1, 4, 3, 2, 0]

    # And lastly 0 -> stage 4
    instances[0] = add_sh_mock(stage=4, config_inst_pairs=0)
    assert list(scheduler._sort_instances_by_stage(instances)) == [1, 0, 4, 3, 2]


def test_process_results(make_scenario, make_stats, configspace_small, runhistory):
    """Ensures that the results are processed by the pertinent intensifer,
    based on the source id"""
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3, deterministic=True)
    stats = make_stats(scenario)
    scheduler = AbstractParallelIntensifier(scenario=scenario)
    scheduler._stats = stats

    scheduler._intensifier_instances = {
        0: mock.Mock(),
        1: mock.Mock(),
        2: mock.Mock(),
    }

    trial_info = TrialInfo(
        config=None,
        instance=0,
        seed=0,
        budget=0.0,
        source=2,
    )

    trial_value = TrialValue(cost=1, time=0.5, status=StatusType.SUCCESS, starttime=1, endtime=2, additional_info={})

    scheduler.process_results(
        trial_info=trial_info, trial_value=trial_value, incumbent=None, runhistory=None, time_bound=None
    )
    assert scheduler._intensifier_instances[0].process_results.call_args is None
    assert scheduler._intensifier_instances[1].process_results.call_args is None
    assert scheduler._intensifier_instances[2].process_results.call_args[1]["trial_info"] == trial_info


def test_get_next_run_wait(make_scenario, make_stats, configspace_small, runhistory):
    """Makes sure we wait if all intensifiers are busy, and no new instance got added.
    This test the case that number of workers are equal to number of instances
    """
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3, deterministic=True)
    stats = make_stats(scenario)
    scheduler = AbstractParallelIntensifier(scenario=scenario)
    scheduler._stats = stats

    scheduler._get_intensifier_ranking = mock_ranker
    scheduler._intensifier_instances = {0: mock.Mock()}
    scheduler._intensifier_instances[0].get_next_run.return_value = (TrialInfoIntent.WAIT, None)
    scheduler._intensifier_instances[0]._stage = 0
    scheduler._intensifier_instances[0].run_tracker = ()

    with unittest.mock.patch(
        "smac.intensifier.abstract_parallel_intensifier.AbstractParallelIntensifier._add_new_instance"
    ) as add_new_instance:
        add_new_instance.return_value = False
        intent, trial_info = scheduler.get_next_run(
            challengers=None,
            incumbent=None,
            get_next_configurations=None,
            runhistory=None,
            repeat_configs=False,
            n_workers=1,
        )
        assert intent == TrialInfoIntent.WAIT


def test_get_next_run_add_instance(make_scenario, make_stats, configspace_small, runhistory):
    """Makes sure we add an instance only when all other instances are waiting,
    This happens when n_workers greater than the number of instances
    """
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3, deterministic=True)
    stats = make_stats(scenario)

    with unittest.mock.patch(
        "smac.intensifier.abstract_parallel_intensifier.AbstractParallelIntensifier._add_new_instance"
    ) as add_new_instance:
        scheduler = AbstractParallelIntensifier(scenario=scenario)
        scheduler._stats = stats

        def instance_added(args):
            source = len(scheduler._intensifier_instances)
            scheduler._intensifier_instances[source] = mock.Mock()
            scheduler._intensifier_instances[source].get_next_run.return_value = (
                TrialInfoIntent.RUN,
                None,
            )
            return True

        add_new_instance.side_effect = instance_added
        scheduler._get_intensifier_ranking = mock_ranker
        scheduler._intensifier_instances = {0: mock.Mock()}
        scheduler._intensifier_instances[0].get_next_run.return_value = (
            TrialInfoIntent.WAIT,
            None,
        )
        scheduler._intensifier_instances[0]._stage = 0
        scheduler._intensifier_instances[0].run_tracker = ()

        intent, trial_info = scheduler.get_next_run(
            challengers=None,
            incumbent=None,
            get_next_configurations=None,
            runhistory=None,
            repeat_configs=False,
            n_workers=1,
        )
        assert len(scheduler._intensifier_instances) == 2
        assert intent == TrialInfoIntent.RUN
