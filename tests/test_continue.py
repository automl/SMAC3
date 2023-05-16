from __future__ import annotations

import os
import shutil
from io import StringIO

import pytest

from smac import AlgorithmConfigurationFacade as ACFacade
from smac import BlackBoxFacade as BBFacade
from smac import HyperbandFacade as HBFacade
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import MultiFidelityFacade as MFFacade
from smac import RandomFacade as RFacade
from smac import Scenario
from smac.callback.callback import Callback
from smac.runhistory.dataclasses import TrialInfo, TrialValue

FACADES = [BBFacade, HPOFacade, MFFacade, RFacade, HBFacade, ACFacade]


@pytest.mark.parametrize("facade", FACADES)
def test_continue_same_scenario(rosenbrock, facade):
    # We did not optimize in the first run
    scenario = Scenario(rosenbrock.configspace, n_trials=10, min_budget=1, max_budget=10)
    smac = facade(scenario, rosenbrock.train, overwrite=True)

    scenario = Scenario(rosenbrock.configspace, n_trials=10, min_budget=1, max_budget=10)
    smac = facade(scenario, rosenbrock.train)
    smac.optimize()

    ##############

    scenario = Scenario(rosenbrock.configspace, n_trials=10, min_budget=1, max_budget=10)
    smac = facade(scenario, rosenbrock.train, overwrite=True)
    incumbent1 = smac.optimize()

    scenario = Scenario(rosenbrock.configspace, n_trials=10, min_budget=1, max_budget=10)
    smac2 = facade(scenario, rosenbrock.train)
    incumbent2 = smac2.optimize()

    # We expect that the old state is just reloaded
    # Since in the first optimization, we already finished, we should have the same incumbent
    assert incumbent1 == incumbent2


@pytest.mark.parametrize("facade", FACADES)
def test_continue_different_scenario(rosenbrock, monkeypatch, facade):
    """Tests whether we can continue a run with a different scenario but using the same name."""
    # Overwrite completely
    number_inputs = StringIO("1\n")
    monkeypatch.setattr("sys.stdin", number_inputs)
    scenario = Scenario(rosenbrock.configspace, name="blub1", n_trials=10, min_budget=1, max_budget=10)
    smac = facade(scenario, rosenbrock.train, overwrite=True)
    smac.optimize()

    scenario = Scenario(rosenbrock.configspace, name="blub1", n_trials=11, min_budget=1, max_budget=10)
    smac = facade(scenario, rosenbrock.train)

    # Keep old run
    try:
        shutil.rmtree("smac3_output/blub2")
        shutil.rmtree("smac3_output/blub2-old")
    except FileNotFoundError:
        pass

    number_inputs = StringIO("2\n")
    monkeypatch.setattr("sys.stdin", number_inputs)
    scenario = Scenario(rosenbrock.configspace, name="blub2", n_trials=10, min_budget=1, max_budget=10)
    smac = facade(scenario, rosenbrock.train, overwrite=True)
    smac.optimize()
    scenario = Scenario(rosenbrock.configspace, name="blub2", n_trials=11, min_budget=1, max_budget=10)
    smac = facade(scenario, rosenbrock.train)
    assert os.path.isdir("smac3_output/blub2-old")


@pytest.mark.parametrize("facade", FACADES)
def test_continue_when_trials_stopped(rosenbrock, facade):
    class CustomCallbackTrials(Callback):
        def on_tell_end(self, smbo, info: TrialInfo, value: TrialValue) -> bool | None:
            if smbo.runhistory.finished == 20:
                return False

            return

    # We did not optimize in the first run
    scenario = Scenario(rosenbrock.configspace, n_trials=25, min_budget=1, max_budget=10)
    smac = facade(scenario, rosenbrock.train, callbacks=[CustomCallbackTrials()], overwrite=True)
    smac.optimize()

    # Because of the callback, we stop after evaluating 20 trials
    assert smac.runhistory.finished == 20

    # Now we want to continue the run
    scenario = Scenario(rosenbrock.configspace, n_trials=25, min_budget=1, max_budget=10)
    smac2 = facade(scenario, rosenbrock.train)

    # Let's see if we restored the runhistory correctly
    assert smac2.runhistory.finished == 20

    smac2.optimize()
    assert smac2.runhistory.finished == 25


@pytest.mark.parametrize("facade", FACADES)
def test_continue_when_walltime_stopped(rosenbrock, facade):
    class CustomCallbackWalltime(Callback):
        def __init__(self, limit: float):
            self._limit = limit

        def on_tell_end(self, smbo, info: TrialInfo, value: TrialValue) -> bool | None:
            if smbo.used_walltime > self._limit:
                return False

            return

    LIMIT = 1
    FINAL_LIMIT = 8

    # Same thing now but with walltime limit
    scenario = Scenario(rosenbrock.configspace, n_trials=99999, walltime_limit=FINAL_LIMIT, min_budget=1, max_budget=10)
    smac = facade(scenario, rosenbrock.train, callbacks=[CustomCallbackWalltime(LIMIT)], overwrite=True)
    smac.optimize()
    assert smac._optimizer.used_walltime > LIMIT and smac._optimizer.used_walltime < FINAL_LIMIT

    # Now we want to continue the run
    scenario = Scenario(rosenbrock.configspace, n_trials=99999, walltime_limit=FINAL_LIMIT, min_budget=1, max_budget=10)
    smac2 = facade(scenario, rosenbrock.train, logging_level=0)

    # Let's see if we restored the runhistory correctly; used walltime should be roughly the same
    # However, since some more things happen in the background, it might be slightly different
    assert pytest.approx(smac2._optimizer.used_walltime, 0.5) == smac._optimizer.used_walltime
    assert smac2.runhistory.finished == smac.runhistory.finished

    smac2.optimize()
    assert smac2.runhistory.finished > smac.runhistory.finished
    assert smac2._optimizer.used_walltime > FINAL_LIMIT
