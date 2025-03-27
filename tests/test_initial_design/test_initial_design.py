import pytest

from smac.initial_design import AbstractInitialDesign
from smac.initial_design.default_design import DefaultInitialDesign

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def test_single_default_config_design(make_scenario, configspace_small):
    dc = DefaultInitialDesign(
        scenario=make_scenario(configspace_small),
        n_configs=10,
    )

    # should return only the default config
    configs = dc.select_configurations()
    assert len(configs) == 1
    assert configs[0]["a"] == 1
    assert configs[0]["b"] == 1e-1
    assert configs[0]["c"] == "cat"


def test_multi_config_design(make_scenario, configspace_small):
    scenario = make_scenario(configspace_small)
    configs = configspace_small.sample_configuration(5)

    dc = AbstractInitialDesign(
        scenario=scenario,
        n_configs=0,
        additional_configs=configs,
    )

    # Selects multiple initial configurations to run.
    # Since the configs were passed to initial design (and n_configs == 0), it should return the same.
    init_configs = dc.select_configurations()
    assert len(init_configs) == len(configs)
    assert init_configs == configs


def test_config_numbers(make_scenario, configspace_small):
    n_configs = 5
    n_configs_per_hyperparameter = 10

    scenario = make_scenario(configspace_small)
    configs = configspace_small.sample_configuration(n_configs)

    n_hps = len(configspace_small.get_hyperparameters())

    dc = AbstractInitialDesign(
        scenario=scenario,
        n_configs=15,
        max_ratio=1.0,
    )

    assert dc._n_configs == 15

    dc = AbstractInitialDesign(
        scenario=scenario,
        n_configs_per_hyperparameter=n_configs_per_hyperparameter,
        additional_configs=configs,
        max_ratio=1.0,
    )

    assert dc._n_configs == n_hps * n_configs_per_hyperparameter

    # If we have max ratio then we expect less
    dc = AbstractInitialDesign(
        scenario=scenario,
        n_configs_per_hyperparameter=1234523,
        # additional_configs=configs,
        max_ratio=0.5,
    )

    assert dc._n_configs == int(scenario.n_trials / 2)

    # If we have max ratio then we expect less
    dc = AbstractInitialDesign(
        scenario=scenario,
        n_configs_per_hyperparameter=1234523,
        additional_configs=configs,
        max_ratio=0.5,
    )

    assert dc._n_configs == int(scenario.n_trials / 2)

    dc = AbstractInitialDesign(
        scenario=scenario,
        n_configs_per_hyperparameter=n_configs_per_hyperparameter,
        max_ratio=1.0,
    )

    assert dc._n_configs == n_hps * n_configs_per_hyperparameter

    # We can't have more initial configs than
    with pytest.raises(ValueError):
        dc = AbstractInitialDesign(
            scenario=scenario,
            n_configs=32351235,
            # If we add additional configs then we should get a value although n_configs is cut by max ratio
            additional_configs=configs,
            max_ratio=1.0,
        )

    # We need to specify at least `n_configs`, `configs` or `n_configs_per_hyperparameter`
    with pytest.raises(ValueError):
        dc = AbstractInitialDesign(
            scenario=scenario,
            n_configs_per_hyperparameter=None,
        )


def test_select_configurations(make_scenario, configspace_small):
    scenario = make_scenario(configspace_small)

    dc = AbstractInitialDesign(
        scenario=scenario,
        n_configs=15,
    )

    # We expect empty list here
    with pytest.raises(NotImplementedError):
        dc.select_configurations()


def test_include_default_config(make_scenario, configspace_small):
    scenario = make_scenario(configspace_small, use_default_config=True)

    dc = AbstractInitialDesign(
        scenario=scenario,
        n_configs=15,
    )

    # if use_default_config is True, then the default config should be included in the additional_configs
    default_config = scenario.configspace.get_default_configuration()
    assert default_config in dc._additional_configs
