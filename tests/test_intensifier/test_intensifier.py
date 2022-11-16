import pytest
from smac.intensifier.intensifier import Intensifier
from smac.main.config_selector import ConfigSelector
from smac.runhistory.dataclasses import TrialInfo
from smac.runhistory.enumerations import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.initial_design.random_design import RandomInitialDesign


class CustomConfigSelector(ConfigSelector):
    def __init__(self, scenario: Scenario, runhistory: RunHistory, n_initial_configs: int = 3) -> None:
        initial_design = RandomInitialDesign(scenario, n_configs=n_initial_configs)
        super().__init__(
            scenario,
            initial_design=initial_design,
            runhistory=runhistory,
            runhistory_encoder=None,  # type: ignore
            model=None,  # type: ignore
            acquisition_maximizer=None,  # type: ignore
            acquisition_function=None,  # type: ignore
            random_design=None,  # type: ignore
            n=8,
        )

    def __iter__(self):

        for config in self._initial_design_configs:
            self._processed_configs.append(config)
            yield config

        while True:
            config = self._scenario.configspace.sample_configuration(1)
            if config not in self._processed_configs:
                self._processed_configs.append(config)
                yield config


def test_trials_of_interest(make_scenario, configspace_small):
    """Tests whether the trials of interests are as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=10, seed=0)
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

    trials = intensifier.get_trials_of_interest(None)
    assert len(trials) == 10
    assert trials[0].seed == trials[1].seed == trials[2].seed
    assert trials[2].seed != trials[3].seed
    assert trials[0].instance != trials[1].instance != trials[2].instance
    assert trials[0].instance == trials[3].instance

    # Test validation
    val_trials = intensifier.get_trials_of_interest(None, validate=True, seed=8888)
    assert len(val_trials) == 10
    assert val_trials[0].seed == val_trials[1].seed == val_trials[2].seed
    assert val_trials[2].seed != val_trials[3].seed
    assert val_trials[0].instance != val_trials[1].instance != val_trials[2].instance
    assert val_trials[0].instance == val_trials[3].instance

    # We expect different seeds for the validation runs
    assert trials[0].seed != val_trials[0].seed
    assert trials[3].seed != val_trials[3].seed

    # But the same instances
    assert trials[0].instance == val_trials[0].instance
    assert trials[3].instance == val_trials[3].instance


def test_next_trials(make_scenario, configspace_small):
    """Tests whether the next trials are as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=9, seed=0)
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

    # If we don't specify N we expect max config call trials here
    trials = intensifier._get_next_trials(None)
    assert len(trials) == 9

    trials = intensifier._get_next_trials(None, N=3)
    assert len(trials) == 3

    trials = intensifier._get_next_trials(None, N=141)
    assert len(trials) == 9

    # Now we want to test if the shuffling works as expected
    # Since we group the trials by seed, we expect the first 3 trials to have the same seed
    seed_changes = 0
    previous_seed = None
    for trial in trials:
        if trial.seed != previous_seed:
            seed_changes += 1
            previous_seed = trial.seed

    assert seed_changes == 3

    # Next, we want to check if evaluated trials are removed
    config = configspace_small.get_default_configuration()
    runhistory.add(config=config, cost=0.5, time=0.0, instance=trials[0].instance, seed=trials[0].seed)

    trials = intensifier._get_next_trials(config)
    assert len(trials) == 8

    # And if we add a running trial?
    runhistory.add(
        config=config,
        cost=0.5,
        time=0.0,
        instance=trials[0].instance,
        seed=trials[0].seed,
        status=StatusType.RUNNING,
    )

    trials = intensifier._get_next_trials(config)
    assert len(trials) == 7

    # The only thing missing is ``from_instances`` now
    isbk = trials[0].get_instance_seed_budget_key()
    isbk2 = trials[1].get_instance_seed_budget_key()

    trials = intensifier._get_next_trials(config, from_instances=[isbk])
    assert len(trials) == 1
    assert trials[0].instance == isbk.instance
    assert trials[0].seed == isbk.seed

    # Add trial as running
    runhistory.add(
        config=config,
        cost=0.5,
        time=0.0,
        instance=isbk.instance,
        seed=isbk.seed,
        status=StatusType.RUNNING,
    )


def test_intensifier(make_scenario, configspace_small):
    """Tests whether the generator returns trials as expected."""
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory, n_initial_configs=1)

    gen = iter(intensifier)

    # Because queue is empty and we don't have any incumbents yet, we expect the first trial
    # to be the config from the random initial design
    trial = next(gen)
    runhistory.add_running_trial(trial)  # We have to mark it as running manually
    intensifier.update_incumbents(trial.config)
    assert intensifier.config_selector._initial_design_configs[0] == trial.config

    # In the next step, (config, N*2) is added to the queue but the intensifier realizes
    # that the previous trial is still running
    # Therefore, we expect to sample a new configuration as there are still no incumbents available
    # (not evaluated yet)
    trial2 = next(gen)
    runhistory.add_running_trial(trial2)
    intensifier.update_incumbents(trial2.config)
    assert intensifier.config_selector._processed_configs[1] == trial2.config

    # Let's mark the first trial as finished
    # The config should become an incumbent now.
    runhistory.add(config=trial.config, cost=10, time=0.0, instance=trial.instance, seed=trial.seed, force_update=True)
    intensifier.update_incumbents(trial.config)
    assert intensifier.get_incumbent() == trial.config

    # Since everything in the queue is running, we start to intensify the incumbent
    trial3 = next(gen)
    runhistory.add_running_trial(trial3)
    intensifier.update_incumbents(trial3.config)
    assert trial3.config == trial.config

    # And we expect a new config again (after incumbent intensification)
    trial4 = next(gen)
    runhistory.add_running_trial(trial4)
    intensifier.update_incumbents(trial4.config)
    assert intensifier.config_selector._processed_configs[2] == trial4.config


def test_intensifier_with_filled_runhistory():
    pass


def test_intensifier_multiple_workers():
    pass


# def test_missing_and_evaluated_trials(make_scenario, configspace_small):
#     """Tests which trials are missing/evaluated based on the runhistory."""

#     scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
#     runhistory = RunHistory()
#     intensifier = Intensifier(scenario=scenario, max_config_calls=3)
#     intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

#     config = configspace_small.get_default_configuration()

#     N = 3
#     for i in range(N):
#         missing_trials = intensifier._get_missing_trials(config, N=3, seed=0)
#         evaluated_trials = intensifier._get_evaluated_trials(config, N=3, seed=0)
#         assert len(missing_trials) == N - i
#         assert len(evaluated_trials) == i

#         # Now we add the first trial
#         if len(missing_trials) == 0:
#             break

#         runhistory.add(
#             config=config, cost=0.5, time=0.0, instance=missing_trials[0].instance, seed=missing_trials[0].seed
#         )


# def test_trials_of_interest_with_filled_runhistory(make_scenario, configspace_small):
#     """Whether the seed of the user is used."""
#     seed = 5

#     scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)

#     # It is important to pass the full runhistory to the configselector otherwise not all seeds/instances are registered
#     runhistory = RunHistory()
#     runhistory.add(config=configspace_small.get_default_configuration(), cost=0.5, time=0.0, instance="i2", seed=seed)

#     intensifier = Intensifier(scenario=scenario, max_config_calls=3)
#     intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

#     gen = iter(intensifier)
#     trial = next(gen)
#     assert trial.seed == seed


# def test_intensification_only(make_scenario, configspace_small):
#     """General behaviour if we only do intensification."""

#     runhistory = RunHistory()
#     scenario = make_scenario(
#         configspace_small,
#         use_instances=True,
#         n_instances=3,
#     )
#     intensifier = Intensifier(
#         scenario=scenario,
#         intensify_percentage=1.0,  # Only intensify here
#         max_config_calls=3,
#     )
#     intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

#     # Get our generator
#     gen = iter(intensifier)

#     # First iteration: We expect the default configuration with different instances
#     for _ in range(3):
#         trial = next(gen)
#         assert trial.config == configspace_small.get_default_configuration()
#         # We should also mark them as run; otherwise we would always get the same trials
#         runhistory.add_running_trial(trial)

#     # Now we would expect a random configuration from the runhistory... but there are no other configs yet!
#     # If we have a smaller intensifier rate, we would actually end up sampling from the config selector
#     with pytest.raises(StopIteration):
#         next(gen)


# def test_new_configs_and_intensification(make_scenario, configspace_small):
#     """General behaviour if we incorporate config selector."""

#     runhistory = RunHistory()
#     scenario = make_scenario(
#         configspace_small,
#         use_instances=True,
#         n_instances=3,
#     )
#     intensifier = Intensifier(
#         scenario=scenario,
#         intensify_percentage=0.0,  # Only intensify here
#         max_config_calls=3,
#     )
#     intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

#     # Get our generator
#     gen = iter(intensifier)

#     # First iteration: We expect the default configuration with different instances
#     for _ in range(3):
#         trial = next(gen)
#         assert trial.config == configspace_small.get_default_configuration()
#         # We should also mark them as run; otherwise we would always get the same trials
#         runhistory.add_running_trial(trial)

#     # Now we expect ONE trial with next config from the config selector.
#     trial2 = next(gen)
#     runhistory.add_running_trial(trial2)

#     # The last trial is added to the pending queue because it waits for the results
#     # In the meantime, we sample the next trial with another new config from the config selector
#     trial3 = next(gen)
#     runhistory.add_running_trial(trial3)

#     # Just check if we have the same seed here
#     assert trial2.seed == trial3.seed

#     # Since we always get a new config new, it goes on like this ...
#     # HOWEVER, we set the intensification percentage to 1.0 now and see if we intensify existing configs
#     intensifier._intensify_percentage = 1.0

#     # First, trial3 goes to the pending queu again
#     # However, in the next iteration, we go into intensification now and the first item in the pending queue is chosen
#     # Nevertheless, we should get into an infinity loop as we don't report any results
#     with pytest.raises(StopIteration):
#         next(gen)


# def test_intensify_one_config(make_scenario, configspace_small):
#     """General behaviour if we incorporate config selector."""

#     # Let's pass a configuration here
#     runhistory = RunHistory()
#     incumbent = configspace_small.sample_configuration(1)
#     runhistory.add(config=incumbent, cost=1.0, time=0.0, instance="i1", seed=5)

#     scenario = make_scenario(
#         configspace_small,
#         use_instances=True,
#         n_instances=3,
#     )
#     intensifier = Intensifier(
#         scenario=scenario,
#         intensify_percentage=0.0,
#         max_config_calls=3,
#     )
#     intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

#     # Get our generator
#     gen = iter(intensifier)

#     # We expect two incumbent trials as one is already evaluated
#     for _ in range(2):
#         trial = next(gen)
#         assert trial.config == incumbent
#         # We should also mark them as run; otherwise we would always get the same trials
#         runhistory.add(config=trial.config, cost=1.0, time=0.0, instance=trial.instance, seed=trial.seed)

#     # Let's get one next trial with config from the config selector (intensify percentage is set to 0) and directly add
#     # it to the runhistory
#     trial = next(gen)
#     runhistory.add(config=trial.config, cost=0.5, time=0.0, instance=trial.instance, seed=trial.seed)

#     # We want to intensify now so we have to change the percentage internally
#     intensifier._intensify_percentage = 1.0

#     # We should get a new trial with the same seed/config as before but different instance
#     # and we do it two more times!
#     for _ in range(2):
#         trial2 = next(gen)
#         runhistory.add(config=trial2.config, cost=0.5, time=0.0, instance=trial2.instance, seed=trial2.seed)
#         assert trial.seed == trial2.seed
#         assert trial.config == trial2.config
#         assert trial.instance != trial2.instance

#     # If we would do it one more time again, we would run into a stop iteration because all trials of the config
#     # has been evaluated.
#     # However, the incumbent should also have changed by now.
#     with pytest.raises(StopIteration):
#         next(gen)

#     runhistory.get_incumbent() == trial.config

#     # And now the first config we added should be rejected because it is worse than the new config and hence
#     # is never evaluated again
#     assert incumbent in intensifier._rejected
