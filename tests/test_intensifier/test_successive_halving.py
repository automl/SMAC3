import pytest
from smac.initial_design.random_design import RandomInitialDesign
from smac.intensifier.successive_halving import SuccessiveHalving
from smac.main.config_selector import ConfigSelector
from smac.runhistory.dataclasses import InstanceSeedBudgetKey
from smac.runhistory.enumerations import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


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


def test_initialization_with_instances(make_scenario, configspace_small):
    """Tests whether SH is initialized correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, n_instances=3, min_budget=1, max_budget=6)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario,
        eta=2,
        n_seeds=2,
        instance_order=None,
        incumbent_selection="any_budget",
    )
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory

    assert intensifier._eta == 2
    assert intensifier._min_budget == 1
    assert intensifier._max_budget == 6
    assert intensifier._n_configs_in_stage == [4, 2, 1]
    assert intensifier._budgets_in_stage == [1.5, 3.0, 6.0]
    assert len(intensifier.get_instance_seed_keys_of_interest()) == 6


def test_initialization_with_instances_fail(make_scenario, configspace_small):
    """Tests whether SH is initialized correctly."""
    scenario: Scenario = make_scenario(
        configspace_small, use_instances=True, n_instances=3, min_budget=1, max_budget=10
    )
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=3, n_seeds=2, instance_order=None, incumbent_selection="any_budget"
    )
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory, n_initial_configs=1)

    # As soon as we set the runhistory, we should get into trouble because we only may have max budget of 6
    # (3 instances * 2 seeds)
    with pytest.raises(ValueError):
        intensifier.runhistory = runhistory


def test_initialization_with_budgets(make_scenario, configspace_small):
    """Tests whether SH is initialized correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=6)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=2, n_seeds=1, instance_order=None, incumbent_selection="any_budget"
    )
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory

    assert intensifier._eta == 2
    assert intensifier._min_budget == 1
    assert intensifier._max_budget == 6
    assert intensifier._n_configs_in_stage == [4, 2, 1]
    assert intensifier._budgets_in_stage == [1.5, 3.0, 6.0]
    assert len(intensifier.get_instance_seed_keys_of_interest()) == 1


def test_initialization_with_budgets_fail(make_scenario, configspace_small):
    """Tests whether SH is initialized correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=6)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=2, n_seeds=2, instance_order=None, incumbent_selection="any_budget"
    )
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory, n_initial_configs=1)
    with pytest.raises(ValueError):
        intensifier.runhistory = runhistory


def test_incumbents_any_budget(make_scenario, configspace_small):
    """Tests whether the incumbent is updated correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=3)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=3, n_seeds=1, instance_order=None, incumbent_selection="any_budget"
    )
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory, n_initial_configs=1)
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


def test_incumbents_highest_observed_budget(make_scenario, configspace_small):
    """Tests whether the incumbent is updated correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=3)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=3, n_seeds=1, instance_order=None, incumbent_selection="highest_observed_budget"
    )
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory, n_initial_configs=1)
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


def test_incumbents_highest_budget(make_scenario, configspace_small):
    """Tests whether the incumbent is updated correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=3)
    runhistory = RunHistory()
    intensifier = SuccessiveHalving(
        scenario=scenario, eta=3, n_seeds=1, instance_order=None, incumbent_selection="highest_budget"
    )
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory
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
