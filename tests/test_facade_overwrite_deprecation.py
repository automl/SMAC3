"""
Related to issue #1343 regarding refactors: `overwrite` in the Facade has been replaced by `continue_run`
in the Scenario (with the opposite meaning).

Setting overwrite in Facade is deprecated since 2.4.2.
These tests check that it still works in Facade (as `continue_run = not overwrite`), but emits a warning about it.
The tests should be removed in future versions when overwrite is removed from Facade.
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
def test_facade_accepts_overwrite_argument(facade):
    assert "overwrite" in inspect.signature(facade.__init__).parameters, (
        "Facade should accept an overwrite argument. If it was removed, this test should be removed as well."
    )


@pytest.mark.parametrize("facade", FACADES)
def test_scenario_continue_run_default_is_false(rosenbrock, facade):
    """Default is to overwrite (continue_run=False), and the Facade does not change it."""
    scenario = Scenario(rosenbrock.configspace, n_trials=5)
    assert scenario.continue_run is False

    smac = facade(scenario, rosenbrock.train)
    assert smac._scenario.continue_run is False


@pytest.mark.parametrize("continue_run", [True, False])
@pytest.mark.parametrize("facade", FACADES)
def test_facade_no_warning_when_overwrite_not_set(rosenbrock, facade, continue_run):
    scenario = Scenario(rosenbrock.configspace, continue_run=continue_run, n_trials=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        smac = facade(scenario, rosenbrock.train)
    assert smac._scenario.continue_run is continue_run, "Facade should keep Scenario.continue_run."
    assert smac._scenario is scenario, "Scenario should not be copied when overwrite is not set in the Facade."


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("facade", FACADES)
def test_facade_overwrite_deprecated(rosenbrock, facade, overwrite):
    scenario = Scenario(rosenbrock.configspace, n_trials=5)
    with pytest.warns(FutureWarning, match="continue_run"):
        smac = facade(scenario, rosenbrock.train, overwrite=overwrite)
    assert smac._scenario.continue_run is (not overwrite), "Facade should set continue_run = not overwrite."


@pytest.mark.parametrize("facade", FACADES)
def test_facade_overwrite_overrides_scenario_continue_run(rosenbrock, facade):
    scenario = Scenario(rosenbrock.configspace, continue_run=False, n_trials=5)
    with pytest.warns(FutureWarning, match="overrides"):
        smac = facade(scenario, rosenbrock.train, overwrite=False)
    assert smac._scenario.continue_run is True, "Facade overwrite should override Scenario.continue_run."
    assert scenario.continue_run is True, "The user's Scenario should also be changed."


@pytest.mark.parametrize("facade", FACADES)
def test_continue_same_scenario_via_facade(rosenbrock, facade):
    """
    Similar to test_continue tests, but setting the behavior via the deprecated Facade argument.
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
def test_continue_run_finds_run_started_without_it(rosenbrock, facade):
    """The experiment hash must not depend on continue_run, so continuing finds the earlier run."""
    scenario = Scenario(rosenbrock.configspace, continue_run=False, n_trials=10, min_budget=1, max_budget=10)
    smac = facade(scenario, rosenbrock.train)
    incumbent1 = smac.optimize()

    scenario2 = Scenario(rosenbrock.configspace, continue_run=True, n_trials=10, min_budget=1, max_budget=10)
    assert scenario2.experiment_name is None
    smac2 = facade(scenario2, rosenbrock.train)
    assert smac2._scenario.experiment_name == smac._scenario.experiment_name
    assert smac2.optimize() == incumbent1


def test_scenario_set_continue_run(rosenbrock):
    """Setting continue_run must not change the scenario's identity (equality and experiment name)."""
    scenario = Scenario(rosenbrock.configspace, n_trials=5)
    other = Scenario(rosenbrock.configspace, n_trials=5)
    assert scenario.continue_run is False

    scenario.set_continue_run(True)
    assert scenario.continue_run is True
    assert scenario == other, "continue_run is not part of the scenario's definition."

    scenario.set_continue_run(False)
    assert scenario.continue_run is False
