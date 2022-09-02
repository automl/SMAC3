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
        n_configs=10,  # Will be ignored
        configs=configs,
    )

    # Selects multiple initial configurations to run.
    # Since the configs were passed to initial design, it should return the same.
    init_configs = dc.select_configurations()
    assert len(init_configs) == 5
    assert init_configs == configs


def test_config_numbers(make_scenario, configspace_small):
    scenario = make_scenario(configspace_small)
    configs = configspace_small.sample_configuration(5)

    dc = AbstractInitialDesign(
        scenario=scenario,
        n_configs=15,
    )

    assert dc.n_configs == 15

    dc = AbstractInitialDesign(
        scenario=scenario,
        configs=configs,
    )

    assert dc.n_configs == 5

    dc = AbstractInitialDesign(
        scenario=scenario,
        n_configs_per_hyperparameter=5,
    )

    assert dc.n_configs == len(configspace_small.get_hyperparameters()) * 5

    # We can't have more initial configs than
    with pytest.raises(ValueError):
        dc = AbstractInitialDesign(
            scenario=scenario,
            n_configs=200,
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
