from smac.initial_design.latin_hypercube_design import LatinHypercubeInitialDesign

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def test_latin_hypercube_design(make_scenario, configspace_large):
    scenario = make_scenario(configspace_large)

    initial_design = LatinHypercubeInitialDesign(
        scenario=scenario,
        n_configs=54,
        max_ratio=1,
    )

    configs = initial_design.select_configurations()
    assert len(configs) == 54
