import pytest

from smac import (
    AlgorithmConfigurationFacade,
    BlackBoxFacade,
    HyperbandFacade,
    HPOFacade,
    MultiFidelityFacade,
    RandomFacade,
    Scenario,
)
from smac.acquisition.functions.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizers.abstract_acqusition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.callback import Callback
from smac.initial_design import LatinHypercubeInitialDesign
from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.model.abstract_model import AbstractModel
from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.runhistory.encoder.abstract_encoder import AbstractRunHistoryEncoder


class CustomCallback(Callback):
    def __init__(self):
        self.counter = 0

    def on_iteration_end(self, _):
        self.counter += 1


def test_facades(rosenbrock):

    for facade in [
        BlackBoxFacade,
        AlgorithmConfigurationFacade,
        HPOFacade,
        RandomFacade,
        MultiFidelityFacade,
        HyperbandFacade,
    ]:
        scenario = Scenario(rosenbrock.configspace, n_trials=20, min_budget=5, max_budget=50)
        smac = facade(scenario, rosenbrock.train, overwrite=True)
        smac.optimize()
        assert smac.stats.finished > 0

        # Also try it with instances
        scenario = Scenario(
            rosenbrock.configspace,
            n_trials=20,
            min_budget=1,
            max_budget=3,
            instances=["i1", "i2", "i3"],
        )
        smac = facade(scenario, rosenbrock.train, overwrite=True)
        smac.optimize()
        assert smac.stats.finished > 0

        # And check components here
        assert isinstance(facade.get_model(scenario), AbstractModel)
        assert isinstance(facade.get_acquisition_function(scenario), AbstractAcquisitionFunction)
        assert isinstance(facade.get_acquisition_maximizer(scenario), AbstractAcquisitionMaximizer)
        assert isinstance(facade.get_intensifier(scenario), AbstractIntensifier)
        assert isinstance(facade.get_initial_design(scenario), AbstractInitialDesign)
        assert isinstance(facade.get_random_design(scenario), AbstractRandomDesign)
        assert isinstance(facade.get_runhistory_encoder(scenario), AbstractRunHistoryEncoder)
        assert isinstance(facade.get_multi_objective_algorithm(scenario), AbstractMultiObjectiveAlgorithm)


def test_random_facade(rosenbrock):

    for facade in [RandomFacade, HyperbandFacade]:
        facade = RandomFacade

        scenario = Scenario(rosenbrock.configspace, n_trials=200, min_budget=5, max_budget=50)
        smac = facade(scenario, rosenbrock.train, overwrite=True)
        smac.optimize()

        configs = smac.runhistory.get_configs()
        configs[0].origin == "Default"
        assert all([c.origin == "Random Search" for c in configs[1:]])


def test_continue_after_depleted_budget(rosenbrock):
    """
    Run optimize until budget depletion, then instantiate the new facade and expect it
    to terminate immediately.
    """
    custom_callback1 = CustomCallback()
    scenario = Scenario(rosenbrock.configspace, n_trials=5)
    smac1 = BlackBoxFacade(scenario, rosenbrock.train, callbacks=[custom_callback1], overwrite=True)
    _ = smac1.optimize()

    custom_callback2 = CustomCallback()
    scenario = Scenario(rosenbrock.configspace, n_trials=5)
    smac2 = BlackBoxFacade(scenario, rosenbrock.train, callbacks=[custom_callback2])

    # This should terminate immediately
    smac2.optimize()

    # Stats object should be filled now
    assert smac2.stats.get_incumbent() == smac1.stats.get_incumbent()
    assert smac1.runhistory == smac2.runhistory

    # We expect different counter because the callback should not be called (since immediately termination)
    assert custom_callback1.counter != custom_callback2.counter
    assert custom_callback2.counter == 0


def test_continue_run(rosenbrock):
    """Run facade. terminate using end of iteration callback prematurely. Instantiate facade
    and continue until the budget is actually completed."""
    scenario = Scenario(rosenbrock.configspace, n_trials=7)
    smac = HPOFacade(
        scenario,
        rosenbrock.train,
        initial_design=LatinHypercubeInitialDesign(scenario, n_configs=3),
    )
    _ = smac.optimize()

    scenario = Scenario(rosenbrock.configspace, n_trials=8)
    smac1 = HPOFacade(
        scenario=scenario,
        target_function=rosenbrock.train,
        initial_design=LatinHypercubeInitialDesign(scenario, n_configs=3),
    )

    # check continuation is loading the proper value
    smac1.optimize()

    for k, k1 in zip(smac.runhistory._data.keys(), smac1.runhistory._data.keys()):
        assert k == k1

    assert len(smac1.runhistory._data) == len(smac.runhistory._data) + 1


def test_continuation_state_same(rosenbrock):
    """Ensure that if you continue from a checkpoint of your model, you actully
    produce the same state as if you had run it immediately.

    Using:
    callback smbo object access -- safe as internal variable in the callback
    what you want to check"""
    pass
