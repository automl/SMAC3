import pytest
from smac.intensifier.intensifier import Intensifier
from smac.main.config_selector import ConfigSelector
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


def test_setting_runhistory(make_scenario, configspace_small):
    """Tests information from runhistory are used."""
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory, n_initial_configs=1)

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
    assert intensifier._tf_seeds == [8, 59]


def test_incumbent_selection_single_objective(make_scenario, configspace_small):
    """Tests whether the incumbents are updated as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=10, seed=0)
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)
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


def test_incumbent_selection_multi_objective(make_scenario, configspace_small):
    """Tests whether the incumbents are updated as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3, use_multi_objective=True)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario, max_config_calls=10, seed=0)
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)
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


def test_incumbent_differences(make_scenario, configspace_small):
    pass
