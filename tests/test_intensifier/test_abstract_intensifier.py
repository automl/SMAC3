import random

import pytest

from smac.initial_design.random_design import RandomInitialDesign
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.intensifier.intensifier import Intensifier
from smac.main.config_selector import ConfigSelector
from smac.runhistory.enumerations import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


def test_setting_runhistory(make_scenario, configspace_small, make_config_selector):
    """Tests information from runhistory are used."""
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)

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

    intensifier.runhistory = runhistory
    intensifier.__post_init__()
    assert intensifier._tf_seeds == [8, 59]


def test_incumbent_selection_single_objective(make_scenario, configspace_small, make_config_selector):
    """Tests whether the incumbents are updated as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=10, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory)
    intensifier.runhistory = runhistory

    config = configspace_small.get_default_configuration()
    config2 = configspace_small.sample_configuration(1)

    runhistory.add(config=config, cost=50, time=0.0, instance=scenario.instances[0], seed=999)
    intensifier.update_incumbents(config)
    assert intensifier.get_incumbent() == config

    # If we add the same config with another instance, nothing should change
    runhistory.add(config=config, cost=50, time=0.0, instance=scenario.instances[1], seed=999)
    intensifier.update_incumbents(config)
    assert intensifier.get_incumbent() == config

    # Now we add another config, however, it is only evaluated on the first instance
    # So the incumbent should not have changed
    runhistory.add(config=config2, cost=40, time=0.0, instance=scenario.instances[0], seed=999)
    intensifier.update_incumbents(config2)
    assert intensifier.get_incumbent() == config

    # Now we add the next trial, then it should change
    runhistory.add(config=config2, cost=40, time=0.0, instance=scenario.instances[1], seed=999)
    intensifier.update_incumbents(config2)
    assert intensifier.get_incumbent() == config2


def test_incumbent_selection_multi_objective(make_scenario, configspace_small, make_config_selector):
    """Tests whether the incumbents are updated as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3, use_multi_objective=True)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=10, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory)
    intensifier.runhistory = runhistory

    config = configspace_small.get_default_configuration()
    config2 = configspace_small.sample_configuration(1)

    runhistory.add(config=config, cost=[50, 10], time=0.0, instance=scenario.instances[0], seed=999)
    intensifier.update_incumbents(config)
    assert intensifier.get_incumbents() == [config]

    runhistory.add(config=config2, cost=[10, 50], time=0.0, instance=scenario.instances[0], seed=999)
    intensifier.update_incumbents(config2)
    assert intensifier.get_incumbents() == [config, config2]

    # Add another trial for the first config; nothing should happen because it can not be compared
    runhistory.add(config=config, cost=[50, 10], time=0.0, instance=scenario.instances[1], seed=999)
    intensifier.update_incumbents(config)
    assert intensifier.get_incumbents() == [config, config2]

    # However, if we add another trial for the second config, it should be removed if it's really bad
    runhistory.add(config=config2, cost=[500, 500], time=0.0, instance=scenario.instances[1], seed=999)
    intensifier.update_incumbents(config2)
    assert intensifier.get_incumbents() == [config]


def test_config_rejection_single_objective(configspace_small, make_scenario):
    """ Tests whether configs are rejected properly if they are worse than the incumbent. """
    scenario = make_scenario(configspace_small, use_instances=False)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario)
    intensifier.runhistory = runhistory

    configs = configspace_small.sample_configuration(3)

    runhistory.add(config=configs[0],
                   cost=5,
                   time=0.0,
                   seed=0,
                   status=StatusType.SUCCESS,
                   force_update=True)
    intensifier.update_incumbents(configs[0])

    assert intensifier._rejected_config_ids == []

    # add config that yielded better results, updating incumbent and sending prior incumbent to rejected
    runhistory.add(config=configs[1],
                   cost=1,
                   time=0.0,
                   seed=0,
                   status=StatusType.SUCCESS,
                   force_update=True)
    intensifier.update_incumbents(config=configs[1])
    
    assert intensifier._rejected_config_ids == [1]

    # add config that is no better should thus go to rejected
    runhistory.add(config=configs[2],
                   cost=1,
                   time=0.0,
                   seed=0,
                   status=StatusType.SUCCESS,
                   force_update=True)
    intensifier.update_incumbents(config=configs[2])
    
    assert intensifier._rejected_config_ids == [1, 3]


def test_incumbent_differences(make_scenario, configspace_small):
    pass


def test_save_and_load(make_scenario, configspace_small, make_config_selector):
    """Tests whether entries from the runhistory are incorporated."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory
    gen = iter(intensifier)

    for i in range(50):
        trial = next(gen)
        cost = random.random()
        runhistory.add(
            config=trial.config,
            cost=cost,
            time=0.0,
            instance=trial.instance,
            seed=trial.seed,
            status=StatusType.SUCCESS,
            force_update=True,
        )
        intensifier.update_incumbents(trial.config)

    filename = "smac3_output_test/test_intensifier/intensifier.json"
    intensifier.save(filename)

    old_incumbents_changed = intensifier._incumbents_changed
    old_trajectory = intensifier._trajectory
    assert old_incumbents_changed > 0

    intensifier.load(filename)

    assert intensifier._incumbents_changed == old_incumbents_changed
    assert intensifier._trajectory == old_trajectory


def test_pareto_front1(make_scenario, configspace_small):
    """Tests whether the configs are in the incumbents."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = AbstractIntensifier(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.runhistory = runhistory
    config1 = configspace_small.sample_configuration(1)
    config2 = configspace_small.sample_configuration(1)

    runhistory.add(
        config=config1,
        cost=[0, 10],
        time=5,
        status=StatusType.SUCCESS,
    )
    intensifier.update_incumbents(config1)

    runhistory.add(
        config=config2,
        cost=[100, 0],
        time=15,
        status=StatusType.SUCCESS,
    )
    intensifier.update_incumbents(config2)

    incumbents = intensifier.get_incumbents()
    assert config1 in incumbents and config2 in incumbents


def test_pareto_front2(make_scenario, configspace_small):
    """Tests whether the configs are in the incumbents."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = AbstractIntensifier(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.runhistory = runhistory
    config1 = configspace_small.sample_configuration(1)
    config2 = configspace_small.sample_configuration(1)

    runhistory.add(
        config=config1,
        cost=[0, 10],
        time=5,
        status=StatusType.SUCCESS,
    )
    intensifier.update_incumbents(config1)

    runhistory.add(
        config=config2,
        cost=[0, 15],
        time=15,
        status=StatusType.SUCCESS,
    )
    intensifier.update_incumbents(config2)

    incumbents = intensifier.get_incumbents()
    assert config1 in incumbents and config2 not in incumbents


def test_pareto_front3(make_scenario, configspace_small):
    """Tests whether the configs are in the incumbents."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = AbstractIntensifier(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.runhistory = runhistory
    config1 = configspace_small.sample_configuration(1)
    config2 = configspace_small.sample_configuration(1)
    config3 = configspace_small.sample_configuration(1)

    runhistory.add(
        config=config1,
        cost=[10, 15],
        time=5,
        status=StatusType.SUCCESS,
    )
    intensifier.update_incumbents(config1)

    runhistory.add(
        config=config2,
        cost=[10, 10],
        time=15,
        status=StatusType.SUCCESS,
    )
    intensifier.update_incumbents(config2)

    incumbents = intensifier.get_incumbents()
    assert config2 in incumbents and config1 not in incumbents

    runhistory.add(
        config=config3,
        cost=[5, 15],
        time=15,
        status=StatusType.SUCCESS,
    )
    intensifier.update_incumbents(config3)

    incumbents = intensifier.get_incumbents()
    assert len(incumbents) == 2
    assert config2 in incumbents
    assert config3 in incumbents
