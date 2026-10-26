"""
Related to issue #1324 regarding refactors, including the moving overwrite from
Facade to Scenario.

Setting overwrite in Facade is deprecated since 2.4.2.
This test checks that it still works in Facade, but emits a warning about it,
and the test should be removed in future versions when overwrite is removed from Facade.
"""

import inspect
import warnings

import pytest

from smac import AlgorithmConfigurationFacade as ACFacade
from smac import BlackBoxFacade as BBFacade
from smac import HyperbandFacade as HBFacade
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import MultiFidelityFacade as MFFacade
from smac import RandomFacade as RFacade
from smac import MultiObjectiveFacade as MOFacade
from smac.scenario import Scenario


FACADES = [BBFacade, HPOFacade, MFFacade, RFacade, HBFacade, ACFacade, MOFacade]


@pytest.mark.parametrize("facade", FACADES)
def test_facade_accepts_overwrite_argument(rosenbrock, facade):
    assert "overwrite" in inspect.signature(facade.__init__).parameters, (
        "Facade should accept an overwrite argument. If it was removed, this test should be removed as well."
    )


@pytest.mark.parametrize("facade", FACADES)
def test_facade_overwrite_default(rosenbrock, facade):
    scenario = Scenario(rosenbrock.configspace, overwrite=False, n_trials=5)
    smac = facade(scenario, rosenbrock.train)
    assert smac._scenario.overwrite is False, "Default Facade.overwrite should match Scenario.overwrite."

    scenario = Scenario(rosenbrock.configspace, overwrite=True, n_trials=5)
    smac = facade(scenario, rosenbrock.train)
    assert smac._scenario.overwrite is True, "Default Facade.overwrite should match Scenario.overwrite."


@pytest.mark.parametrize("facade", FACADES)
def test_facade_overwrite_deprecated(rosenbrock, facade):
    scenario = Scenario(rosenbrock.configspace, n_trials=5)
    with pytest.warns(FutureWarning, match="overwrite"):
        smac = facade(scenario, rosenbrock.train, overwrite=False)
    assert smac._scenario.overwrite is False


@pytest.mark.parametrize("facade", FACADES)
def test_facade_no_warning_when_overwrite_not_set(rosenbrock, facade):
    scenario = Scenario(rosenbrock.configspace, overwrite=False, n_trials=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        smac = facade(scenario, rosenbrock.train)
    assert smac._scenario.overwrite is False
    assert smac._scenario is scenario, "Scenario should not be copied when overwrite is not set in the Facade."


@pytest.mark.parametrize("facade", FACADES)
def test_facades_overwrite_takes_precedence(rosenbrock, facade):
    scenario = Scenario(rosenbrock.configspace, overwrite=True, n_trials=5)
    with pytest.warns(FutureWarning):
        smac = facade(scenario, rosenbrock.train, overwrite=False)
    assert smac._scenario.overwrite is False, "Facade overwrite should take precedence over Scenario overwrite."
    assert scenario.overwrite is True, (
        "Scenario overwrite should remain True, only Facade's internal Scenario should change."
    )


@pytest.mark.parametrize("facade", FACADES)
def test_continue_same_scenario_via_facade(rosenbrock, facade):
    """
    Similar to test_continue tests.
    """

    # Optimize from scratch
    scenario = Scenario(rosenbrock.configspace, n_trials=10, min_budget=1, max_budget=10)
    with pytest.warns(FutureWarning):
        smac = facade(scenario, rosenbrock.train, overwrite=True)
    incumbent1 = smac.optimize()

    # Continue finished optimization
    scenario = Scenario(rosenbrock.configspace, n_trials=10, min_budget=1, max_budget=10)
    with pytest.warns(FutureWarning):
        smac2 = facade(scenario, rosenbrock.train, overwrite=False)
    incumbent2 = smac2.optimize()

    # Should be the same because it is just loaded from previous
    assert incumbent1 == incumbent2


@pytest.mark.parametrize("facade", FACADES)
def test_overwrite_default_is_true(rosenbrock, facade):
    """
    Check that overwrite default is True in Scenario, and that it is not changed by Facade.
    """

    scenario = Scenario(rosenbrock.configspace, n_trials=5)
    assert scenario.overwrite is True, "Default Scenario.overwrite should be True."

    smac = facade(scenario, rosenbrock.train)
    assert smac._scenario.overwrite is True, "Default Facade.overwrite should match Scenario.overwrite."
