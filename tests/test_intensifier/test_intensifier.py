from smac.intensifier.intensifier import Intensifier
from smac.config_selector.config_selector import ConfigSelector
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.initial_design.random_design import RandomInitialDesign


class CustomConfigSelector(ConfigSelector):
    def __init__(self, scenario: Scenario, runhistory: RunHistory) -> None:
        initial_design = RandomInitialDesign(scenario)
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


def test_intensifier(make_scenario, configspace_small):
    """General behaviour."""

    scenario = make_scenario(configspace_small, use_instances=True)
    runhistory = RunHistory()
    intensifier = Intensifier(scenario=scenario)
    intensifier.config_selector = CustomConfigSelector(scenario, runhistory)

    iterator = iter(intensifier)

    # trials = []
    # for i in iterator:
    #    trials.append(i)

    # Next is different than iterating it
    # trial1 = next(iterator)
    # trial2 = next(iterator)
    # trial3 = next(iterator)


# - Checks if runhistory instance/seed is prioriized
# - Make sure same seed is used etc.
