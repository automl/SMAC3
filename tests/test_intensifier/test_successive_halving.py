import pytest

from smac.initial_design.random_design import RandomInitialDesign
from smac.intensifier.successive_halving import SuccessiveHalving
from smac.main.config_selector import ConfigSelector
from smac.runhistory.dataclasses import InstanceSeedBudgetKey
from smac.runhistory.enumerations import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


def test_initialization_fails(make_scenario, configspace_small):
    """Tests whether SH is initialized correctly."""
    # Error: Max budget must be higher than min budget
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, min_budget=10, max_budget=6)
    intensifier = SuccessiveHalving(scenario=scenario)
    runhistory = RunHistory()

    with pytest.raises(ValueError):
        intensifier.runhistory = runhistory
        intensifier.__post_init__()

    # Error: Must be integers
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, min_budget=3.0, max_budget=6.0)
    intensifier = SuccessiveHalving(scenario=scenario)
    runhistory = RunHistory()

    with pytest.raises(ValueError):
        intensifier.runhistory = runhistory
        intensifier.__post_init__()

    # Error: Need budgets defined
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=None, max_budget=None)
    intensifier = SuccessiveHalving(scenario=scenario)
    runhistory = RunHistory()

    with pytest.raises(ValueError):
        intensifier.runhistory = runhistory
        intensifier.__post_init__()

    # Error: Min budget must be higher than 0
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=-1, max_budget=50)
    intensifier = SuccessiveHalving(scenario=scenario)
    runhistory = RunHistory()

    with pytest.raises(ValueError):
        intensifier.runhistory = runhistory
        intensifier.__post_init__()


def test_initialization_with_instances(make_scenario, configspace_small, make_config_selector):
    """Tests whether SH is initialized correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, n_instances=3, min_budget=1, max_budget=6)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario,
        eta=2,
        n_seeds=2,
        instance_seed_order=None,
        incumbent_selection="any_budget",
    )
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

    assert intensifier._eta == 2
    assert intensifier._min_budget == 1
    assert intensifier._max_budget == 6
    assert intensifier._n_configs_in_stage[0] == [4, 2, 1]
    assert intensifier._budgets_in_stage[0] == [1.5, 3.0, 6.0]
    assert len(intensifier.get_instance_seed_keys_of_interest()) == 6


def test_initialization_with_instances_fail(make_scenario, configspace_small, make_config_selector):
    """Tests whether SH is initialized correctly."""
    scenario: Scenario = make_scenario(
        configspace_small, use_instances=True, n_instances=3, min_budget=1, max_budget=10
    )
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=3, n_seeds=2, instance_seed_order=None, incumbent_selection="any_budget"
    )
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)

    # As soon as we set the runhistory, we should get into trouble because we only may have max budget of 6
    # (3 instances * 2 seeds)
    with pytest.raises(ValueError):
        intensifier.runhistory = runhistory
        intensifier.__post_init__()


def test_initialization_with_budgets(make_scenario, configspace_small, make_config_selector):
    """Tests whether SH is initialized correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=6)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=2, n_seeds=1, instance_seed_order=None, incumbent_selection="any_budget"
    )
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

    assert intensifier._eta == 2
    assert intensifier._min_budget == 1
    assert intensifier._max_budget == 6
    assert intensifier._n_configs_in_stage[0] == [4, 2, 1]
    assert intensifier._budgets_in_stage[0] == [1.5, 3.0, 6.0]
    assert len(intensifier.get_instance_seed_keys_of_interest()) == 1


def test_initialization_with_budgets_fail(make_scenario, configspace_small, make_config_selector):
    """Tests whether SH is initialized correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=6)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=2, n_seeds=2, instance_seed_order=None, incumbent_selection="any_budget"
    )
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    with pytest.raises(ValueError):
        intensifier.runhistory = runhistory
        intensifier.__post_init__()


def test_incumbents_any_budget(make_scenario, configspace_small, make_config_selector):
    """Tests whether the incumbent is updated correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=3)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=3, n_seeds=1, instance_seed_order=None, incumbent_selection="any_budget"
    )
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory
    config = configspace_small.get_default_configuration()
    config2 = configspace_small.sample_configuration(1)

    # Add some entries to the runhistory
    runhistory.add(config=config, cost=0.5, time=0.0, seed=8, budget=1, status=StatusType.SUCCESS)
    intensifier.update_incumbents(config)
    assert intensifier.get_incumbent() == config
    assert intensifier.get_instance_seed_budget_keys(config) == [
        InstanceSeedBudgetKey(instance=None, seed=8, budget=1.0)
    ]

    runhistory.add(config=config2, cost=0.3, time=0.0, seed=8, budget=1, status=StatusType.SUCCESS)
    intensifier.update_incumbents(config2)
    assert intensifier.get_incumbent() == config2
    assert intensifier.get_instance_seed_budget_keys(config2) == [
        InstanceSeedBudgetKey(instance=None, seed=8, budget=1.0)
    ]

    # Let's see what happens if we add another budget
    # We expect that all instance-seed-budget keys are used for the incumbent
    runhistory.add(config=config, cost=-9999, time=0.0, seed=8, budget=2, status=StatusType.SUCCESS)
    intensifier.update_incumbents(config)
    assert intensifier.get_incumbent() == config
    assert intensifier.get_instance_seed_budget_keys(config) == [
        InstanceSeedBudgetKey(instance=None, seed=8, budget=1.0),
        InstanceSeedBudgetKey(instance=None, seed=8, budget=2.0),
    ]

    # If we want to compare the isb keys, we expect only one because giving the fact that config is already evaluated
    # on at least one budget. If we have evaluated on at least one budget, we are good to go.
    assert intensifier.get_instance_seed_budget_keys(config, compare=True) == [
        InstanceSeedBudgetKey(instance=None, seed=8, budget=None)
    ]


def test_incumbents_highest_observed_budget(make_scenario, configspace_small, make_config_selector):
    """Tests whether the incumbent is updated correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=3)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=3, n_seeds=1, instance_seed_order=None, incumbent_selection="highest_observed_budget"
    )
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()
    config = configspace_small.get_default_configuration()
    config2 = configspace_small.sample_configuration(1)

    # Add some entries to the runhistory
    runhistory.add(config=config, cost=0.5, time=0.0, seed=8, budget=1, status=StatusType.SUCCESS)
    intensifier.update_incumbents(config)
    assert intensifier.get_incumbent() == config
    assert intensifier.get_instance_seed_budget_keys(config) == [
        InstanceSeedBudgetKey(instance=None, seed=8, budget=1.0)
    ]

    # TODO: We actually want to prioritze configs evaluated on higher budgets
    # Not working yet
    runhistory.add(config=config2, cost=0.5, time=0.0, seed=8, budget=1, status=StatusType.SUCCESS)
    intensifier.update_incumbents(config2)
    assert intensifier.get_incumbent() == config
    assert intensifier.get_instance_seed_budget_keys(config2) == [
        InstanceSeedBudgetKey(instance=None, seed=8, budget=1.0)
    ]

    # If we evaluate on a higher budget, we don't want to use the lower budget anymore
    runhistory.add(config=config2, cost=0.0, time=0.0, seed=8, budget=2, status=StatusType.SUCCESS)
    intensifier.update_incumbents(config2)
    assert intensifier.get_incumbent() == config2
    assert intensifier.get_instance_seed_budget_keys(config2) == [
        # We don't want InstanceSeedBudgetKey(instance=None, seed=8, budget=1.0) anymore
        # That means, we only calculate the cost based on the highest observed budget
        InstanceSeedBudgetKey(instance=None, seed=8, budget=2.0),
    ]


def test_incumbents_highest_budget(make_scenario, configspace_small, make_config_selector):
    """Tests whether the incumbent is updated correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=3)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=3, n_seeds=1, instance_seed_order=None, incumbent_selection="highest_budget"
    )
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()
    config = configspace_small.get_default_configuration()
    config2 = configspace_small.sample_configuration(1)

    # Add some entries to the runhistory
    runhistory.add(config=config, cost=0.5, time=0.0, seed=8, budget=1, status=StatusType.SUCCESS)
    intensifier.update_incumbents(config)
    assert intensifier.get_incumbent() is None
    assert intensifier.get_instance_seed_budget_keys(config) == [
        # No isb keys here because we only want to evaluate the highest ones
        # InstanceSeedBudgetKey(instance=None, seed=8, budget=1.0)
    ]

    runhistory.add(config=config2, cost=0.5, time=0.0, seed=8, budget=3, status=StatusType.SUCCESS)
    intensifier.update_incumbents(config2)
    # Incumbent changes because it is evaluated on the highest budget
    assert intensifier.get_incumbent() == config2
    assert intensifier.get_instance_seed_budget_keys(config2) == [
        InstanceSeedBudgetKey(instance=None, seed=8, budget=3.0)
    ]


def test_state(make_scenario, configspace_small, make_config_selector):
    """Tests whether the tracker is saved and loaded correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=3)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(scenario=scenario)
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory

    gen = iter(intensifier)

    # Add some configs to the tracker
    for _ in range(10):
        trial = next(gen)
        runhistory.add_running_trial(trial)  # We have to mark it as running manually
        intensifier.update_incumbents(trial.config)

    # Save current state as old state
    old_state = intensifier.get_state()
    assert len(intensifier.get_state()["tracker"]) > 0

    # Now we reset the intensifier
    intensifier._tracker = {}
    assert len(intensifier.get_state()["tracker"]) == 0

    # Next, we set the state again
    intensifier.set_state(old_state)
    new_state = intensifier.get_state()

    assert old_state == new_state


def test_trials_of_interest(make_scenario, configspace_small):
    """Tests whether we get the right trials of interests."""
    # Without instances
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=3)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(scenario=scenario)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

    # We expect to get only one trial of interest but with the highest budget (because real-value based)
    trials = intensifier.get_trials_of_interest(None)
    assert len(trials) == 1
    assert trials[0].budget == 3

    ################
    # With instances
    scenario: Scenario = make_scenario(
        configspace_small, use_instances=True, n_instances=5, min_budget=1, max_budget=10
    )
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(scenario=scenario, n_seeds=10)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

    # We receive 50 trials because n_instances*n_seeds
    # We need to receive all of them because of shuffling
    trials = intensifier.get_trials_of_interest(None)
    assert len(trials) == 5 * 10
    assert trials[0].budget is None


def test_with_filled_runhistory(make_scenario, configspace_small, make_config_selector):
    """Tests whether the tracker is updated when providing a filled runhistory"""
    # Without instances
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, min_budget=1, max_budget=3)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(scenario=scenario)
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)

    # Add some entries to the runhistory
    configs = configspace_small.sample_configuration(5)
    for config in configs:
        runhistory.add(
            config=config,
            cost=0.5,
            time=0.0,
            instance=scenario.instances[1],
            seed=8,
            status=StatusType.SUCCESS,
        )
    assert len(configs) == 5

    # Now we set the runhistory
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

    assert 8 in intensifier._tf_seeds

    gen = iter(intensifier)
    next(gen)

    # The tracker has to be updated now, meaning that all configs should have been added to
    # the first stage
    # Since we have a max budget of 3, we expect to have these configs batched
    assert len(intensifier._tracker[(0, 0)][0][1]) == 3

    # We would expect 2 here but SH is always filling up empty spaces so that we naturally get 3
    assert len(intensifier._tracker[(0, 0)][1][1]) == 3


def test_promoting(make_scenario, configspace_small, make_config_selector):
    """Tests whether an evaluated batch is promoted correctly."""
    max_budget = 3
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=max_budget)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(scenario=scenario)
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

    n_configs = intensifier._n_configs_in_stage
    budgets = intensifier._budgets_in_stage

    assert n_configs[0] == [3, 1]
    assert budgets[0] == [1.0, 3.0]

    gen = iter(intensifier)
    for i in range(n_configs[0][0]):
        trial = next(gen)
        assert trial.budget == budgets[0][0]
        runhistory.add(
            config=trial.config,
            cost=i,
            time=0.0,
            instance=trial.instance,
            seed=trial.seed,
            budget=trial.budget,
            status=StatusType.SUCCESS,
        )

        assert (0, 1) not in intensifier._tracker

    # Now we should trigger the next stage
    trial = next(gen)
    runhistory.add_running_trial(trial)
    assert (0, 1) in intensifier._tracker

    # There should be only one batch
    assert len(intensifier._tracker[(0, 1)]) == 1

    # There should be only one configuration inside this batch
    assert len(intensifier._tracker[(0, 1)][0][1]) == n_configs[0][1]

    # And the next trial should have the highest budget
    assert trial.budget == budgets[0][1]

    # If we would call next(gen) again we expect a new batch in the first stage
    first_stage_trial = next(gen)
    assert first_stage_trial.budget == budgets[0][0]

    # Now we mark the previous trial as finished
    runhistory.add(
        config=trial.config,
        cost=i,
        time=0.0,
        instance=trial.instance,
        seed=trial.seed,
        budget=trial.budget,
    )

    # If we call next(gen) again, we expect no entries in the second stage anymore
    assert len(intensifier._tracker[(0, 1)]) == 0
    assert (0, 2) not in intensifier._tracker

    # However, the first batch should still be there
    assert len(intensifier._tracker[(0, 0)]) == 1
    assert len(intensifier._tracker[(0, 0)][0][1]) == n_configs[0][0]
