import pytest

from smac import BlackBoxFacade, Scenario

# from smac.facade import Facade

from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterFacade, BlackBoxFacade, Scenario
from smac.initial_design import LatinHypercubeInitialDesign


@pytest.fixture
def target_algorithm():
    class Rosenbrock2D:
        def configspace(self) -> ConfigurationSpace:
            cs = ConfigurationSpace(seed=0)
            x0 = Float("x0", (-5, 10), default=-3)
            x1 = Float("x1", (-5, 10), default=-4)
            cs.add_hyperparameters([x0, x1])

            return cs

        def train(self, config: Configuration) -> float:
            x1 = config["x0"]
            x2 = config["x1"]

            cost = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
            return cost

    return Rosenbrock2D()


def test_continue_after_depleted_budget(target_algorithm):
    """
    Run optimize until budget depletion, then instantiate the new facade and expect it
    to terminate immediately
    """
    scenario = Scenario(target_algorithm.configspace(), n_trials=5)
    smac = BlackBoxFacade(scenario, target_algorithm.train)
    _ = smac.optimize()

    scenario = Scenario(target_algorithm.configspace(), n_trials=5)
    smac1 = BlackBoxFacade(
        scenario=scenario,
        target_algorithm=target_algorithm.train,
    )

    # check instant termination because of budget depletion
    smac1.optimize()
    assert smac1.runhistory == smac.runhistory


def test_continue_run(target_algorithm):
    """Run facade. terminate using end of iteration callback prematurely. Instantiate facade
    and continue until the budget is actually completed."""
    scenario = Scenario(target_algorithm.configspace(), n_trials=7)
    smac = HyperparameterFacade(
        scenario, target_algorithm.train, initial_design=LatinHypercubeInitialDesign(scenario, n_configs=3)
    )
    _ = smac.optimize()

    scenario = Scenario(target_algorithm.configspace(), n_trials=8)
    smac1 = HyperparameterFacade(
        scenario=scenario,
        target_algorithm=target_algorithm.train,
        initial_design=LatinHypercubeInitialDesign(scenario, n_configs=3),
    )

    # check continuation is loading the proper value
    smac1.optimize()

    for k, k1 in zip(smac.runhistory.data.keys(), smac1.runhistory.data.keys()):
        assert k == k1

    assert len(smac1.runhistory.data) == len(smac.runhistory.data) + 1


def test_continuation_state_same(target_algorithm):
    """Ensure that if you continue from a checkpoint of your model, you actully
    produce the same state as if you had run it immediately.

    Using:
    callback smbo object access -- safe as internal variable in the callback
    what you want to check"""
    pass
