import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac.initial_design.sobol_design import SobolInitialDesign

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def test_sobol_design(make_scenario, configspace_large):
    scenario = make_scenario(configspace_large)

    initial_design = SobolInitialDesign(
        scenario=scenario,
        n_configs=54,
        max_ratio=1,
    )

    configs = initial_design.select_configurations()
    assert initial_design._n_configs == 54
    assert len(configs) == 54


def test_max_hyperparameters(make_scenario):
    cs = ConfigurationSpace()
    hyperparameters = [Float("x%d" % (i + 1), (0, 1)) for i in range(21202)]
    cs.add_hyperparameters(hyperparameters)

    scenario = make_scenario(cs)

    with pytest.raises(ValueError):
        initial_design = SobolInitialDesign(
            scenario=scenario,
            n_configs=5,
        )
        initial_design.select_configurations()
