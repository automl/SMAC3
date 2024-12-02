from smac.initial_design.random_design import RandomInitialDesign
from smac.intensifier.intensifier import Intensifier
from smac.main.config_selector import ConfigSelector
from smac.runhistory import TrialInfo, TrialKey, TrialValue
from smac.runhistory.enumerations import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


def test_trials_of_interest(make_scenario, configspace_small, make_config_selector):
    """Tests whether the trials of interests are as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=10, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

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


def test_next_trials(make_scenario, configspace_small, make_config_selector):
    """Tests whether the next trials are as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=9, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

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
    runhistory.add(config=config, cost=0.5, time=0.0, instance=trials[0].instance,
                   seed=trials[0].seed)

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

    # The only thing missing is ``from_keys`` now
    isbk = trials[0].get_instance_seed_budget_key()

    trials = intensifier._get_next_trials(config, from_keys=[isbk])
    assert len(trials) == 1
    assert trials[0].instance == isbk.instance
    assert trials[0].seed == isbk.seed


def test_next_trials_counter(make_scenario, configspace_small, make_config_selector):
    """Tests whether the next trials are as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=9, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()
    config = configspace_small.get_default_configuration()

    # All good here
    trials = intensifier._get_next_trials(config, N=5)
    assert len(trials) == 5

    # But now we set a trial running
    runhistory.add_running_trial(trials[0])

    # Now we call the same thing again but expect 4 trials instead
    trials = intensifier._get_next_trials(config, N=5)
    assert len(trials) == 4

    # More interesting case: Use ``from_keys`` too
    instances = []
    for trial in trials:
        instances.append(trial.get_instance_seed_budget_key())

    # We have three instances now ...
    del instances[0]

    # ... but request 8
    trials = intensifier._get_next_trials(config, N=8, from_keys=instances)
    assert len(trials) == 3

    # ... but request 2
    trials = intensifier._get_next_trials(config, N=2, from_keys=instances)
    assert len(trials) == 2


def test_intensifier(make_scenario, configspace_small, make_config_selector):
    """Tests whether the generator returns trials as expected."""
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory

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
    runhistory.add(config=trial.config, cost=10, time=0.0, instance=trial.instance, seed=trial.seed,
                   force_update=True)
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


def test_intensifier_with_capping(make_scenario, configspace_small, make_config_selector):
    """Test whether adaptive capping works with the intensifier."""
    SLACK = 1.2
    RUNTIME_CUTOFF = 10
    scenario: Scenario = make_scenario(
        configspace_small,
        use_instances=True,
        n_instances=10,
        max_budget=10,
        deterministic=True
    )
    runhistory = RunHistory()
    intensifier = Intensifier(
        scenario=scenario,
        max_config_calls=6,
        seed=0,
        runtime_cutoff=RUNTIME_CUTOFF,
        adaptive_capping_slackfactor=SLACK
    )
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=0)
    intensifier.runhistory = runhistory
    cb = intensifier.get_callback()
    incumbent = configspace_small.get_default_configuration()

    budget1, budget2 = 3, 2

    # This part will check that the RHCallback works as intended
    class SMBO:
        def __init__(self, intensifier):
            self.intensifier = intensifier

    _, info, value = cb.on_tell_start(
        SMBO(intensifier),
        info=TrialInfo(
            config=incumbent,
            instance=scenario.instances[0],
            budget=budget1, # this is what we need to overwrite with the cb!
            seed=8,
        ),
        value=TrialValue(
            cost=budget1, # this is the variable deciding over the incumbent!
            time=0,
            status=StatusType.SUCCESS,
            starttime=0,
            endtime=0,
            additional_info={},
        )
    )
    assert info.budget == 0 # we successfully overwrote the budget
    runhistory.add(
        config=info.config,
        cost=value.cost,
        time=value.time,
        status=value.status,
        instance=info.instance,
        seed=info.seed,
        budget=info.budget,
        starttime=value.starttime,
        endtime=value.endtime,
        additional_info=value.additional_info,
        force_update=True,  # Important to overwrite the status RUNNING
    )
    #     config=incumbent,
    #     cost=budget1,
    #     time=0.0,
    #     instance=scenario.instances[0],
    #     seed=8,
    #     budget=0, # RUNTIME_CUTOFF, # for update_incumbents isb key comparison, we need to blank
    #     # the budget on tell. In SMBO this is done using the RunHistoryCallback.on_tell_start!
    #     status=StatusType.SUCCESS,
    # )
    intensifier.update_incumbents(incumbent)
    assert incumbent == intensifier.get_incumbent()

    # Intensifier with known incumbent (due to updated rh)
    gen = iter(intensifier)

    # first an intensification of the incumbent is triggered; i.e. one new instance for the
    # incumbent is selected:
    inc = next(gen)

    assert inc.config == incumbent
    assert inc.budget == RUNTIME_CUTOFF - budget1  # no slack here, because first incumbent!

    runhistory.add(
        config=inc.config,
        cost=budget2,
        budget=0,  # inc.budget, # from now on we will continue just explicitly setting the budget
        seed=inc.seed,
        status=StatusType.SUCCESS,
        time=0.0,
        instance=inc.instance
    )
    intensifier.update_incumbents(inc.config)

    # let's sample a challenger:
    challenger = next(gen)
    assert challenger.config != incumbent
    assert challenger.seed == 8
    assert challenger.instance == scenario.instances[0], \
        (f"instance of challenger must be evaluated on the first subset of instances"
         f"the incumbent was evaluated on.")

    # The budget for the challenger is calculated as follows:
    # The instance subsets (/tiers) the challenger is evaluated on is 2**tier
    # where tier is the intensification level of the challenger.
    # I.e. in the first round, the challenger is evaluated on the first instance subset i1
    # if it survives, it will be evaluated on the second instance subset i1, i2, then i1, i2, i3,
    # i4, and so on. The cutoff time is always the same as the incumbent's sum of runtimes on the
    # instance subset minus the runtime the challenger spent on the instance subset so far.
    # Here budget1 is the runtime the incumbent spent on the first instance subset (i1).
    # Notice, that we don't use budget2 here, because i2 is not part of the instance subset of the
    # challenger.
    assert challenger.budget == (budget1) * SLACK - 0  # challenger has not been evaluated yet

    runhistory.add(
        config=challenger.config,
        cost=budget1,  # challenger is on par with incumbent
        budget=0,
        seed=challenger.seed,
        status=StatusType.SUCCESS,
        time=0.0,
        instance=challenger.instance
    )
    intensifier.update_incumbents(challenger.config)
    assert inc.config == intensifier.get_incumbent()  # no incumbent change

    # Since this configuration was on-par with the incumbent, we can further intensify this
    # challenger configuration on a new instance
    config2 = next(gen)
    assert config2.config == challenger.config
    # notice, that the inc.instance is the last instance the incumbent was evaluated on and is in
    # the in the second instance subset of the challenger
    assert config2.instance in inc.instance, \
        "challenger should be intensified on the second instance subset, because it survived the first"
    assert config2.instance != challenger.instance
    assert config2.budget == min(RUNTIME_CUTOFF, (budget1 + budget2) * SLACK - budget1)
    assert inc.config == intensifier.get_incumbents(sort_by="num_trials")[0]  # no incumbent change

    runhistory.add(
        config=config2.config,
        cost=budget2 - 2,  # we beat the incumbent here!
        budget=0,
        seed=config2.seed,
        status=StatusType.SUCCESS,
        time=0.0,
        instance=config2.instance
    )
    # we accept config2 as the new incumbent, because it is evaluated on the same instances as the
    # incumbent and has a lower cost (-2).
    intensifier.update_incumbents(config2.config)
    assert config2.config == intensifier.get_incumbent()  # incumbent changed!!!

    # given the incumbent change, we need to intensify the new incumbent on
    # the next instance subset (i.e. intensify it with two more instances)
    config3 = next(gen)

    assert config3.config == config2.config
    seen_instances = [isb.instance for isb in runhistory.get_instance_seed_budget_keys(inc.config)]
    assert config3.instance not in seen_instances, \
        "config3 should be evaluated on a new instance subset"

    # we have a new incumbent, so its budget is immediately set to the cutoff
    assert config3.budget == RUNTIME_CUTOFF - (budget1 + budget2 - 2)
    budget3 = 2
    runhistory.add(
        config=config3.config,
        cost=budget3,
        budget=0,
        seed=config3.seed,
        status=StatusType.SUCCESS,
        time=0.0,
        instance=config3.instance
    )

    intensifier.update_incumbents(config3.config)
    assert config3.config == intensifier.get_incumbents(sort_by="num_trials")[0]  # no incumbent
    # change

    # intensifier._queue sollte die config nicht mehr halten
    # TODO how do i test a reject in config? wouldn't the config be able to recover, once we have access
    #  to a new instance; provided we didn't already spent all our budget yet.


def test_intensifier_with_filled_runhistory(make_scenario, configspace_small, make_config_selector):
    """Tests whether entries from the runhistory are incorporated."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory
    config = configspace_small.get_default_configuration()
    config2 = configspace_small.sample_configuration(1)

    # Add some entries to the runhistory
    runhistory.add(
        config=config,
        cost=0.5,
        time=0.0,
        instance=scenario.instances[1],
        seed=8,
        status=StatusType.RUNNING,
    )

    runhistory.add(
        config=config2,
        cost=0.5,
        time=0.0,
        instance=scenario.instances[1],
        seed=59,
        status=StatusType.SUCCESS,
    )

    # Now we post init
    intensifier.__post_init__()

    assert 8 in intensifier._tf_seeds
    assert 59 in intensifier._tf_seeds

    gen = iter(intensifier)
    trial = next(gen)
    assert trial.seed == 8 or trial.seed == 59

    # Make sure the configs are inside
    assert (config, 1) in intensifier._queue

    # Config 2 is not in the queue anymore because it is the only config which was successful
    # Therefore, it is the incumbent and not a challenger anymore
    assert (config2, 1) not in intensifier._queue
    assert config2 in intensifier.get_incumbents()
