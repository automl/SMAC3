import pytest
from smac.intensifier.intensifier import Intensifier
from smac.config_selector.config_selector import ConfigSelector
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.initial_design.random_design import RandomInitialDesign


class CustomConfigSelector(ConfigSelector):
    def __init__(self, scenario: Scenario, runhistory: RunHistory) -> None:
        initial_design = RandomInitialDesign(scenario, n_configs=3)
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
            if config not in self._processed_configs:
                self._processed_configs.append(config)
                yield config


def test_trials_of_interest(make_scenario, configspace_small):
    """Tests whether the trials of interests are as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=3)
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

    # We expect all trials to have the same seed but different instances
    trials = intensifier._get_trials_of_interest(None, N=3, seed=0)
    assert len(trials) == 3
    assert trials[0].seed == trials[1].seed == trials[2].seed
    assert trials[0].instance != trials[1].instance != trials[2].instance

    # Now we request more trials
    trials = intensifier._get_trials_of_interest(None, N=2000, seed=0)
    assert len(trials) == 3  # We still get three trials because of max config calls

    # Now we request less trials
    trials = intensifier._get_trials_of_interest(None, N=2, seed=0)
    assert len(trials) == 2

    # Note: If we have higher max config calls, we can not expect the same seeds because of the shuffling


def test_missing_and_evaluated_trials(make_scenario, configspace_small):
    """Tests which trials are missing/evaluated based on the runhistory."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=3)
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

    config = configspace_small.get_default_configuration()

    N = 3
    for i in range(N):
        missing_trials = intensifier._get_missing_trials(config, N=3, seed=0)
        evaluated_trials = intensifier._get_evaluated_trials(config, N=3, seed=0)
        assert len(missing_trials) == N - i
        assert len(evaluated_trials) == i

        # Now we add the first trial
        if len(missing_trials) == 0:
            break

        runhistory.add(
            config=config, cost=0.5, time=0.0, instance=missing_trials[0].instance, seed=missing_trials[0].seed
        )


def test_trials_of_interest_with_filled_runhistory(make_scenario, configspace_small):
    """Whether the seed of the user is used."""
    seed = 5

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)

    # It is important to pass the full runhistory to the configselector otherwise not all seeds/instances are registered
    runhistory = RunHistory()
    runhistory.add(config=configspace_small.get_default_configuration(), cost=0.5, time=0.0, instance="i2", seed=seed)

    intensifier = Intensifier(scenario=scenario, max_config_calls=3)
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

    gen = iter(intensifier)
    trial = next(gen)
    assert trial.seed == seed


def test_intensification_only(make_scenario, configspace_small):
    """General behaviour if we only do intensification."""

    runhistory = RunHistory()
    scenario = make_scenario(
        configspace_small,
        use_instances=True,
        n_instances=3,
    )
    intensifier = Intensifier(
        scenario=scenario,
        intensify_percentage=1.0,  # Only intensify here
        max_config_calls=3,
    )
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

    # Get our generator
    gen = iter(intensifier)

    # First iteration: We expect the default configuration with different instances
    for _ in range(3):
        trial = next(gen)
        assert trial.config == configspace_small.get_default_configuration()
        # We should also mark them as run; otherwise we would always get the same trials
        runhistory.add_running_trial(trial)

    # Now we would expect a random configuration from the runhistory... but there are no other configs yet!
    # If we have a smaller intensifier rate, we would actually end up sampling from the config selector
    with pytest.raises(StopIteration):
        next(gen)


def test_new_configs_and_intensification(make_scenario, configspace_small):
    """General behaviour if we incorporate config selector."""

    runhistory = RunHistory()
    scenario = make_scenario(
        configspace_small,
        use_instances=True,
        n_instances=3,
    )
    intensifier = Intensifier(
        scenario=scenario,
        intensify_percentage=0.0,  # Only intensify here
        max_config_calls=3,
    )
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

    # Get our generator
    gen = iter(intensifier)

    # First iteration: We expect the default configuration with different instances
    for _ in range(3):
        trial = next(gen)
        assert trial.config == configspace_small.get_default_configuration()
        # We should also mark them as run; otherwise we would always get the same trials
        runhistory.add_running_trial(trial)

    # Now we expect ONE trial with next config from the config selector.
    trial2 = next(gen)
    runhistory.add_running_trial(trial2)

    # The last trial is added to the pending queue because it waits for the results
    # In the meantime, we sample the next trial with another new config from the config selector
    trial3 = next(gen)
    runhistory.add_running_trial(trial3)

    # Just check if we have the same seed here
    assert trial2.seed == trial3.seed

    # Since we always get a new config new, it goes on like this ...
    # HOWEVER, we set the intensification percentage to 1.0 now and see if we intensify existing configs
    intensifier._intensify_percentage = 1.0

    # First, trial3 goes to the pending queu again
    # However, in the next iteration, we go into intensification now and the first item in the pending queue is chosen
    # Nevertheless, we should get into an infinity loop as we don't report any results
    with pytest.raises(StopIteration):
        next(gen)


def test_intensify_one_config(make_scenario, configspace_small):
    """General behaviour if we incorporate config selector."""

    # Let's pass a configuration here
    runhistory = RunHistory()
    incumbent = configspace_small.sample_configuration(1)
    runhistory.add(config=incumbent, cost=1.0, time=0.0, instance="i1", seed=5)

    scenario = make_scenario(
        configspace_small,
        use_instances=True,
        n_instances=3,
    )
    intensifier = Intensifier(
        scenario=scenario,
        intensify_percentage=0.0,
        max_config_calls=3,
    )
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

    # Get our generator
    gen = iter(intensifier)

    # We expect two incumbent trials as one is already evaluated
    for _ in range(2):
        trial = next(gen)
        assert trial.config == incumbent
        # We should also mark them as run; otherwise we would always get the same trials
        runhistory.add(config=trial.config, cost=1.0, time=0.0, instance=trial.instance, seed=trial.seed)

    # Let's get one next trial with config from the config selector (intensify percentage is set to 0) and directly add
    # it to the runhistory
    trial = next(gen)
    runhistory.add(config=trial.config, cost=0.5, time=0.0, instance=trial.instance, seed=trial.seed)

    # We want to intensify now so we have to change the percentage internally
    intensifier._intensify_percentage = 1.0

    # We should get a new trial with the same seed/config as before but different instance
    # and we do it two more times!
    for _ in range(2):
        trial2 = next(gen)
        runhistory.add(config=trial2.config, cost=0.5, time=0.0, instance=trial2.instance, seed=trial2.seed)
        assert trial.seed == trial2.seed
        assert trial.config == trial2.config
        assert trial.instance != trial2.instance

    # If we would do it one more time again, we would run into a stop iteration because all trials of the config
    # has been evaluated.
    # However, the incumbent should also have changed by now.
    with pytest.raises(StopIteration):
        next(gen)

    runhistory.get_incumbent() == trial.config

    # And now the first config we added should be rejected because it is worse than the new config and hence
    # is never evaluated again
    assert incumbent in intensifier._rejected
