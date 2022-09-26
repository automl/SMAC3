import numpy as np
import pytest

from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade
from smac.intensifier.intensifier import Intensifier, IntensifierStage
from smac.runhistory import RunHistory, TrialInfo, TrialInfoIntent
from smac.runner.abstract_runner import StatusType
from smac.runner.target_function_runner import TargetFunctionRunner
from smac.stats import Stats


def evaluate_challenger(
    trial_info: TrialInfo,
    target_function: TargetFunctionRunner,
    stats: Stats,
    runhistory: RunHistory,
    force_update: bool = False,
):
    """
    Wrapper over challenger evaluation.

    SMBO objects handles the run history, but to keep same testing functionality this function is a small
    wrapper to launch the taf and add it to the history.
    """
    # Evaluating configuration
    trial_info, result = target_function.run_wrapper(trial_info=trial_info)

    stats._target_function_walltime_used += float(result.time)
    stats._finished += 1

    runhistory.add(
        config=trial_info.config,
        cost=result.cost,
        time=result.time,
        status=result.status,
        instance=trial_info.instance,
        seed=trial_info.seed,
        budget=trial_info.budget,
        force_update=force_update,
    )
    stats._n_configs = len(runhistory.config_ids)

    return result


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def test_race_challenger_1(make_scenario, make_stats, configspace_small, runhistory):
    """
    Makes sure that a racing configuration with better performance, is selected as incumbent.
    """

    def target(x, seed=0):
        return (x["a"] + 1) / 1000.0

    scenario = make_scenario(configspace_small, use_instances=True)
    stats = make_stats(scenario)
    intensifier = Intensifier(scenario=scenario)
    intensifier._stats = stats
    target_function = TargetFunctionRunner(target, scenario, stats, required_arguments=["seed"])
    configs = configspace_small.sample_configuration(3)

    assert intensifier._stage == IntensifierStage.RUN_FIRST_CONFIG

    runhistory.add(
        config=configs[0],
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=55,
        additional_info=None,
    )

    intensifier._N = 1
    incumbent, instance, seed = intensifier._get_next_racer(
        challenger=configs[1],
        incumbent=configs[0],
        runhistory=runhistory,
    )

    assert intensifier._stage == IntensifierStage.RUN_FIRST_CONFIG
    assert incumbent == configs[0]  # Must be the first config since incumbent is only passed through
    assert instance == "i1"
    assert seed == 55

    # Here we evaluate configs[1] and add it to the runhistory.
    # Cost is around ~7.1 on the sampled config.
    trial_info = TrialInfo(config=configs[1], instance=instance, seed=seed, budget=0.0)
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)

    incumbent, _ = intensifier.process_results(
        trial_info=trial_info,
        trial_value=trial_value,
        incumbent=configs[1],
        runhistory=runhistory,
        time_bound=np.inf,
    )

    assert intensifier._stage == IntensifierStage.RUN_INCUMBENT
    assert incumbent == configs[1]  # Well, again it's basically passed down
    assert intensifier._num_challenger_run == 1


def test_race_challenger_large(make_scenario, make_stats, configspace_small, runhistory):
    """
    Makes sure that a racing configuration with better performance, is selected as incumbent.
    """

    def target(x, seed=0, instance=None):
        return 1

    scenario = make_scenario(configspace_small, use_instances=True, deterministic=True)
    stats = make_stats(scenario)
    intensifier = Intensifier(scenario=scenario)
    intensifier._stats = stats
    target_function = TargetFunctionRunner(target, scenario, stats, required_arguments=["seed", "instance"])
    configs = configspace_small.sample_configuration(3)

    for i in range(3):
        runhistory.add(
            config=configs[0],
            cost=i + 1,
            time=1,
            status=StatusType.SUCCESS,
            instance=f"i{i+1}",
            seed=12345,
            additional_info=None,
        )

    intensifier._stage = IntensifierStage.RUN_CHALLENGER

    # Tie on first instances and then challenger should always win
    # and be returned as inc
    while True:
        if intensifier._continue_challenger:
            config = intensifier._current_challenger
        else:
            config, _ = intensifier.get_next_challenger(
                challengers=[configs[1], configs[2]], get_next_configurations=None
            )

        incumbent, instance, seed = intensifier._get_next_racer(
            challenger=config, incumbent=configs[0], runhistory=runhistory
        )

        trial_info = TrialInfo(config=config, instance=instance, seed=seed)
        trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)

        incumbent, _ = intensifier.process_results(
            trial_info=trial_info,
            trial_value=trial_value,
            incumbent=configs[0],
            runhistory=runhistory,
            time_bound=np.inf,
        )

        # stop when challenger evaluation is over
        if not intensifier._stage == IntensifierStage.RUN_CHALLENGER:
            break

    assert incumbent == configs[1]
    assert runhistory.get_cost(configs[1]) == 1

    # Get data for config2 to check that the correct run was performed
    runs = runhistory.get_trials(configs[1], only_max_observed_budget=True)
    assert len(runs) == 3
    assert intensifier._num_trials == 3
    assert intensifier._num_challenger_run == 3


def test_race_challenger_large_blocked_seed(make_scenario, make_stats, configspace_small, runhistory):
    """
    Test _race_challenger whether seeds are blocked for challenger runs.
    """

    def target(x, seed=0):
        return 1

    scenario = make_scenario(configspace_small, use_instances=True, deterministic=False)
    stats = make_stats(scenario)
    intensifier = Intensifier(scenario=scenario)
    intensifier._stats = stats
    target_function = TargetFunctionRunner(target, scenario, stats, required_arguments=["seed"])
    configs = configspace_small.sample_configuration(3)

    for i in range(3):
        runhistory.add(
            config=configs[0],
            cost=i + 1,
            time=1,
            status=StatusType.SUCCESS,
            instance=f"i{i+1}",
            seed=i,
            additional_info=None,
        )

    intensifier._stage = IntensifierStage.RUN_CHALLENGER

    # Tie on first instances and then challenger should always win and be returned as inc
    while True:
        if intensifier._continue_challenger:
            config = intensifier._current_challenger
        else:
            config, _ = intensifier.get_next_challenger(
                challengers=[configs[1], configs[2]], get_next_configurations=None
            )

        incumbent, instance, seed = intensifier._get_next_racer(
            challenger=config, incumbent=configs[0], runhistory=runhistory
        )

        trial_info = TrialInfo(config=config, instance=instance, seed=seed, budget=None)
        trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)

        incumbent, _ = intensifier.process_results(
            trial_info=trial_info,
            trial_value=trial_value,
            incumbent=configs[0],
            runhistory=runhistory,
            time_bound=np.inf,
        )

        # Stop when challenger evaluation is over
        if not intensifier._stage == IntensifierStage.RUN_CHALLENGER:
            break

    assert incumbent == configs[1]
    assert runhistory.get_cost(configs[1]) == 1

    # Get data for config2 to check that the correct run was performed
    runs = runhistory.get_trials(configs[1], only_max_observed_budget=True)
    assert len(runs) == 3

    seeds = sorted([r.seed for r in runs])
    assert list(range(3)) == seeds

    assert intensifier._num_trials == 3
    assert intensifier._num_challenger_run == 3


def test_add_incumbent(make_scenario, make_stats, configspace_small, runhistory):
    """
    Test _race_challenger whether seeds are blocked for challenger runs.
    """

    def target(x, seed=0):
        return (x["a"] + 1) / 1000.0

    scenario = make_scenario(configspace_small, use_instances=True, deterministic=True)
    stats = make_stats(scenario)
    intensifier = Intensifier(scenario=scenario)
    intensifier._stats = stats
    target_function = TargetFunctionRunner(target, scenario, stats, required_arguments=["seed"])
    configs = configspace_small.sample_configuration(3)
    pending_instances = intensifier._get_pending_instances(incumbent=configs[0], runhistory=runhistory)
    assert len(pending_instances) == 3
    instance, seed = intensifier._get_next_instance(pending_instances=pending_instances)

    trial_info = TrialInfo(config=configs[0], instance=instance, seed=seed, budget=0.0)
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    intensifier._stage = IntensifierStage.PROCESS_FIRST_CONFIG_RUN
    inc, perf = intensifier.process_results(
        trial_info=trial_info,
        trial_value=trial_value,
        incumbent=configs[0],
        runhistory=runhistory,
        time_bound=np.inf,
    )

    assert len(runhistory._data) == 1

    # Since we assume deterministic=1,
    # the second call should not add any more runs
    # given only one instance
    # So the returned seed/instance is None so that a new
    # run to be triggered is not launched
    pending_instances = intensifier._get_pending_instances(incumbent=configs[0], runhistory=runhistory)
    # Make sure that the list is empty (or the first instance removed), and hence no new call
    # of incumbent will be triggered
    assert len(pending_instances) == 2

    # The following two tests evaluate to zero because _next_iteration is triggered by _add_inc_run
    # as it is the first evaluation of this intensifier
    # After the above incumbent run, the stage is
    # IntensifierStage.RUN_CHALLENGER. Change it to test next iteration
    intensifier._stage = IntensifierStage.PROCESS_FIRST_CONFIG_RUN
    intensifier.process_results(
        trial_info=trial_info,
        trial_value=trial_value,
        incumbent=None,
        runhistory=runhistory,
        time_bound=np.inf,
    )
    assert intensifier._num_challenger_run == 0


def test_add_incumbent_non_deterministic(make_scenario, make_stats, configspace_small, runhistory):
    """
    Test _add_inc_run().
    """

    def target(x, seed=0):
        return (x["a"] + 1) / 1000.0

    scenario = make_scenario(configspace_small, use_instances=True, deterministic=False)
    stats = make_stats(scenario)
    intensifier = Intensifier(scenario=scenario)
    intensifier._stats = stats
    target_function = TargetFunctionRunner(target, scenario, stats, required_arguments=["seed"])
    configs = configspace_small.sample_configuration(3)

    instance, seed = intensifier._get_next_instance(
        pending_instances=intensifier._get_pending_instances(incumbent=configs[0], runhistory=runhistory)
    )
    trial_info = TrialInfo(config=configs[0], instance=instance, seed=seed, budget=0.0)
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    intensifier.process_results(
        trial_info=trial_info,
        incumbent=configs[0],
        runhistory=runhistory,
        time_bound=np.inf,
        trial_value=trial_value,
    )
    assert len(runhistory._data) == 1

    instance, seed = intensifier._get_next_instance(
        pending_instances=intensifier._get_pending_instances(incumbent=configs[0], runhistory=runhistory)
    )
    trial_info = TrialInfo(config=configs[0], instance=instance, seed=seed, budget=0.0)
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    intensifier.process_results(
        trial_info=trial_info,
        incumbent=configs[0],
        runhistory=runhistory,
        time_bound=np.inf,
        trial_value=trial_value,
    )
    assert len(runhistory._data) == 2
    runs = runhistory.get_trials(config=configs[0], only_max_observed_budget=True)

    # Exactly one run on each instance
    assert "i2" in [runs[0].instance, runs[1].instance]
    assert "i3" in [runs[0].instance, runs[1].instance]

    instance, seed = intensifier._get_next_instance(
        pending_instances=intensifier._get_pending_instances(
            incumbent=configs[0],
            runhistory=runhistory,
        )
    )
    trial_info = TrialInfo(config=configs[0], instance=instance, seed=seed, budget=0.0)
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    intensifier.process_results(
        trial_info=trial_info,
        incumbent=configs[0],
        runhistory=runhistory,
        time_bound=np.inf,
        trial_value=trial_value,
    )
    assert len(runhistory._data) == 3

    # The number of runs performed should be 3
    # No Next iteration call as an incumbent is provided
    assert intensifier._num_trials == 2
    assert intensifier._num_challenger_run == 0


def test_get_next_challenger(make_scenario, make_stats, configspace_small):
    """
    Test get_next_challenger().
    """

    def target(x, seed=0):
        return (x["a"] + 1) / 1000.0

    scenario = make_scenario(configspace_small, use_instances=True, deterministic=True)
    stats = make_stats(scenario)
    intensifier = Intensifier(scenario=scenario)
    intensifier._stats = stats
    configs = configspace_small.sample_configuration(3)

    intensifier._stage = IntensifierStage.RUN_CHALLENGER

    # get a new challenger to evaluate
    config, new = intensifier.get_next_challenger(challengers=[configs[0], configs[1]], get_next_configurations=None)

    assert config == configs[0] == intensifier._current_challenger
    assert intensifier._challenger_id == 1
    assert intensifier._N == 1
    assert new

    # when already evaluating a challenger, return the same challenger
    intensifier._to_run = [(1, 1, 0)]
    config, new = intensifier.get_next_challenger(challengers=[configs[1]], get_next_configurations=None)
    assert config == configs[0] == intensifier._current_challenger
    assert intensifier._challenger_id, 1
    assert not new


def test_generate_challenger(make_scenario, make_stats, configspace_small, runhistory):
    """
    Test generate_challenger()
    """

    def target(x, seed=0, instance=None):
        return (x["a"] + 1) / 1000.0

    scenario = make_scenario(configspace_small, use_instances=True, deterministic=True)
    stats = make_stats(scenario)
    intensifier = Intensifier(scenario=scenario)
    intensifier._stats = stats
    configs = configspace_small.sample_configuration(3)

    gen = intensifier._generate_challengers(challengers=[configs[0], configs[1]], get_next_configurations=None)

    assert next(gen) == configs[0]
    assert next(gen) == configs[1]
    with pytest.raises(StopIteration):
        next(gen)

    smac = AlgorithmConfigurationFacade(scenario, target, overwrite=True)
    gen = intensifier._generate_challengers(challengers=None, get_next_configurations=smac.get_next_configurations)

    assert next(gen).get_dictionary() == {"a": 5489, "b": 0.007257005721594277, "c": "dog"}

    with pytest.raises(StopIteration):
        next(gen)

    # when both are none, raise error
    with pytest.raises(ValueError, match="No .* provided"):
        intensifier._generate_challengers(challengers=None, get_next_configurations=None)


def test_evaluate_challenger_1(make_scenario, make_stats, configspace_small, runhistory):
    """
    Test evaluate_challenger() - a complete intensification run without a `always_race_against` configuration.
    """

    def target(x, seed=0):
        return 2 * x["a"] + x["b"]

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=1, deterministic=False)
    stats = make_stats(scenario)
    intensifier = Intensifier(scenario=scenario, race_against=None, min_challenger=2)
    intensifier._stats = stats
    target_function = TargetFunctionRunner(target, scenario, stats, required_arguments=["seed"])
    configs = configspace_small.sample_configuration(20)

    config0 = configs[16]
    config1 = configs[15]
    config2 = configs[2]

    assert intensifier._n_iters == 0
    assert intensifier._stage == IntensifierStage.RUN_FIRST_CONFIG

    # intensification iteration #1
    # run first config as incumbent if incumbent is None
    intent, trial_info = intensifier.get_next_run(
        challengers=[config2],
        incumbent=None,
        runhistory=runhistory,
        get_next_configurations=None,
    )
    assert trial_info.config == config2
    assert intensifier._stage == IntensifierStage.PROCESS_FIRST_CONFIG_RUN

    # eval config 2 (=first run)
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    inc, perf = intensifier.process_results(
        trial_info=trial_info,
        trial_value=trial_value,
        incumbent=None,
        runhistory=runhistory,
        time_bound=np.inf,
    )
    assert inc == config2
    assert intensifier._stage == IntensifierStage.RUN_INCUMBENT
    assert stats.incumbent_changed == 1
    assert intensifier._n_iters == 1  # 1 intensification run complete!

    # Regular intensification begins - run incumbent
    # Normally a challenger will be given, which in this case is the incumbent
    # But no more instances are available. So to prevent cicles
    # where No iteration happens, provide the challengers
    intent, trial_info = intensifier.get_next_run(
        challengers=[
            config1,
            config0,
        ],  # since incumbent is run, no configs required
        incumbent=inc,
        runhistory=runhistory,
        get_next_configurations=None,
    )

    # no new TA runs as there are no more instances to run
    assert inc == config2
    assert stats.incumbent_changed == 1
    assert len(runhistory.get_trials(config2, only_max_observed_budget=True)) == 1

    assert intensifier._stage == IntensifierStage.PROCESS_INCUMBENT_RUN

    # run challenger now that the incumbent has been executed
    # So this call happen above, to save one iteration
    # assert intensifier._stage == IntensifierStage.RUN_CHALLENGER
    assert trial_info.config == config2
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    inc, perf = intensifier.process_results(
        trial_info=trial_info,
        trial_value=trial_value,
        incumbent=inc,
        runhistory=runhistory,
        time_bound=np.inf,
    )

    # challenger has a better performance, so incumbent has changed
    assert inc == config2
    assert stats.incumbent_changed == 1
    assert intensifier._stage == IntensifierStage.RUN_CHALLENGER
    assert not intensifier._continue_challenger
    assert intensifier._n_iters == 1  # iteration continues as `min_chall` condition is not met

    # intensification continues running incumbent again in same iteration...
    # run incumbent
    # Same here, No further instance-seed pairs for incumbent available
    # so above call gets the new config to run
    assert trial_info.config == config2

    # There is a transition from:
    # IntensifierStage.RUN_FIRST_CONFIG-> IntensifierStage.RUN_INCUMBENT
    # Because after the first run, incumbent is run.
    # Nevertheless, there is now a transition:
    # IntensifierStage.RUN_INCUMBENT->IntensifierStage.RUN_CHALLENGER
    # because in add_inc_run, there are more available instance pairs
    # FROM: IntensifierStage.RUN_INCUMBENT TO: IntensifierStage.RUN_INCUMBENT WHY: no more to run
    # if all <instance, seed> have been run, compare challenger performance
    # assert intensifier._stage == IntensifierStage.RUN_CHALLENGER
    assert intensifier._stage == IntensifierStage.RUN_CHALLENGER

    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    inc, perf = intensifier.process_results(
        trial_info=trial_info,
        trial_value=trial_value,
        incumbent=inc,
        runhistory=runhistory,
        time_bound=np.inf,
    )

    # run challenger
    intent, trial_info = intensifier.get_next_run(
        challengers=None,  # don't need a new list here as old one is cont'd
        incumbent=inc,
        runhistory=runhistory,
        get_next_configurations=None,
    )
    assert trial_info.config == config2
    assert intensifier._stage == IntensifierStage.PROCESS_INCUMBENT_RUN
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    inc, perf = intensifier.process_results(
        trial_info=trial_info,
        incumbent=inc,
        runhistory=runhistory,
        time_bound=np.inf,
        trial_value=trial_value,
    )

    assert inc == config2
    assert stats.incumbent_changed == 1
    assert intensifier._stage == IntensifierStage.RUN_CHALLENGER
    assert intensifier._n_iters == 1  # 2 intensification run complete!

    # No configs should be left at the end
    with pytest.raises(StopIteration):
        next(intensifier._configs_to_run)

    assert len(runhistory.get_trials(config0, only_max_observed_budget=True)) == 0
    assert len(runhistory.get_trials(config1, only_max_observed_budget=True)) == 0
    assert len(runhistory.get_trials(config2, only_max_observed_budget=True)) == 3


def test_evaluate_challenger_2(make_scenario, make_stats, configspace_small, runhistory):
    """
    Test evaluate_challenger for a resumed SMAC run (first run with incumbent)
    """

    def target(x, seed=0):
        return 2 * x["a"] + x["b"]

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=1, deterministic=True)
    stats = make_stats(scenario)
    intensifier = Intensifier(scenario=scenario, race_against=None)
    intensifier._stats = stats
    target_function = TargetFunctionRunner(target, scenario, stats, required_arguments=["seed"])
    configs = configspace_small.sample_configuration(20)

    config0 = configs[16]
    config1 = configs[15]

    assert intensifier._n_iters == 0
    assert intensifier._stage == IntensifierStage.RUN_FIRST_CONFIG

    # adding run for incumbent configuration
    runhistory.add(
        config=config0,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance=1,
        seed=None,
        additional_info=None,
    )

    # intensification - incumbent will be run, but not as RUN_FIRST_CONFIG stage
    intent_, trial_info = intensifier.get_next_run(
        challengers=[config1],
        incumbent=config0,
        runhistory=runhistory,
        get_next_configurations=None,
    )
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    inc, perf = intensifier.process_results(
        trial_info=trial_info,
        incumbent=config0,
        runhistory=runhistory,
        time_bound=np.inf,
        trial_value=trial_value,
    )

    assert intensifier._stage == IntensifierStage.RUN_CHALLENGER
    assert len(runhistory.get_trials(config0, only_max_observed_budget=True)) == 2


def test_no_new_intensification_wo_challenger_run(make_scenario, make_stats, configspace_small, runhistory):
    """
    This test ensures that no new iteration is started if no challenger run was conducted.
    """

    def target(x, seed=0):
        return 2 * x["a"] + x["b"]

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=1, deterministic=True)
    stats = make_stats(scenario)
    intensifier = Intensifier(scenario=scenario, race_against=None, min_challenger=1)
    intensifier._stats = stats
    target_function = TargetFunctionRunner(target, scenario, stats, required_arguments=["seed"])
    configs = configspace_small.sample_configuration(20)

    config0 = configs[16]
    config1 = configs[15]
    config2 = configs[2]

    assert intensifier._n_iters == 0
    assert intensifier._stage == IntensifierStage.RUN_FIRST_CONFIG

    intent, trial_info = intensifier.get_next_run(
        challengers=[config2],
        incumbent=None,
        runhistory=runhistory,
        get_next_configurations=None,
    )
    assert trial_info.config == config2
    assert intensifier._stage == IntensifierStage.PROCESS_FIRST_CONFIG_RUN
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    inc, perf = intensifier.process_results(
        trial_info=trial_info,
        incumbent=None,
        runhistory=runhistory,
        time_bound=np.inf,
        trial_value=trial_value,
    )
    assert inc == config2
    assert intensifier._stage == IntensifierStage.RUN_INCUMBENT
    assert intensifier._n_iters == 1  # 1 intensification run complete!

    # Regular intensification begins - run incumbent

    # No further instance-seed pairs for incumbent available
    # Here None challenger is suggested. Code jumps to next iteration
    # This causes a transition from RUN_INCUMBENT->RUN_CHALLENGER
    # But then, the next configuration to run is the incumbent
    # We don't rerun the incumbent (see message):
    # Challenger was the same as the current incumbent; Skipping challenger
    # Then, we try to get more challengers, but below all challengers
    # Provided are config3, the incumbent which means nothing more to run
    intent, trial_info = intensifier.get_next_run(
        challengers=[config2],  # since incumbent is run, no configs required
        incumbent=inc,
        runhistory=runhistory,
        get_next_configurations=None,
    )

    assert trial_info.config is None
    assert intensifier._stage == IntensifierStage.RUN_CHALLENGER

    intensifier._next_iteration()

    # Add a configuration, then try to execute it afterwards
    assert intensifier._n_iters == 2

    runhistory.add(
        config=config0,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=0,
        additional_info=None,
    )
    intensifier._stage = IntensifierStage.RUN_CHALLENGER

    # In the upcoming get next run, the stage is RUN_CHALLENGER
    # so the intensifier tries to run config1. Nevertheless,
    # there are no further instances for this configuration available.
    # In this scenario, the intensifier produces a SKIP intent as an indication
    # that a new iteration must be initiated, and for code simplicity,
    # relies on a new call to get_next_run to yield more configurations
    intent, trial_info = intensifier.get_next_run(
        challengers=[config0], incumbent=inc, runhistory=runhistory, get_next_configurations=None
    )
    assert intent == TrialInfoIntent.SKIP

    # This doesn't return a config because the array of configs is exhausted
    intensifier._stage = IntensifierStage.RUN_CHALLENGER
    config, _ = intensifier.get_next_challenger(challengers=None, get_next_configurations=None)
    assert config is None
    # This finally gives a runable configuration
    intent, trial_info = intensifier.get_next_run(
        challengers=[config1], incumbent=inc, runhistory=runhistory, get_next_configurations=None
    )
    trial_value = evaluate_challenger(trial_info, target_function, stats, runhistory)
    inc, perf = intensifier.process_results(
        trial_info=trial_info,
        incumbent=inc,
        runhistory=runhistory,
        time_bound=np.inf,
        trial_value=trial_value,
    )
    # 4 Iterations due to the proactive runs
    # of get next challenger
    assert intensifier._n_iters == 3
    assert intensifier._num_challenger_run == 1
