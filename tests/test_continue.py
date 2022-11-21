import os
import shutil
from io import StringIO

import pytest

from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario
from smac.callback import Callback
from smac.runhistory.dataclasses import TrialInfo, TrialValue


def test_continue_same_scenario(rosenbrock):
    for facade in [BlackBoxFacade]:
        # We did not optimize in the first run
        scenario = Scenario(rosenbrock.configspace, n_trials=10)
        smac = facade(scenario, rosenbrock.train, overwrite=True)

        scenario = Scenario(rosenbrock.configspace, n_trials=10)
        smac = facade(scenario, rosenbrock.train)
        smac.optimize()

        ##############

        scenario = Scenario(rosenbrock.configspace, n_trials=10)
        smac = facade(scenario, rosenbrock.train, overwrite=True)
        incumbent1 = smac.optimize()

        scenario = Scenario(rosenbrock.configspace, n_trials=10)
        smac2 = facade(scenario, rosenbrock.train)
        incumbent2 = smac2.optimize()

        # We expect that the old state is just reloaded
        # Since in the first optimization, we already finished, we should have the same incumbent
        assert incumbent1 == incumbent2


def test_continue_different_scenario(rosenbrock, monkeypatch):
    """Tests whether we can continue a run with a different scenario but using the same name."""
    for facade in [BlackBoxFacade, HyperparameterOptimizationFacade]:
        # Overwrite completely
        number_inputs = StringIO("1\n")
        monkeypatch.setattr("sys.stdin", number_inputs)
        scenario = Scenario(rosenbrock.configspace, name="blub1", n_trials=10)
        smac = facade(scenario, rosenbrock.train, overwrite=True)
        smac.optimize()

        scenario = Scenario(rosenbrock.configspace, name="blub1", n_trials=11)
        smac = facade(scenario, rosenbrock.train)

        # Keep old run
        try:
            shutil.rmtree("smac3_output/blub2")
            shutil.rmtree("smac3_output/blub2-old")
        except FileNotFoundError:
            pass

        number_inputs = StringIO("2\n")
        monkeypatch.setattr("sys.stdin", number_inputs)
        scenario = Scenario(rosenbrock.configspace, name="blub2", n_trials=10)
        smac = facade(scenario, rosenbrock.train, overwrite=True)
        smac.optimize()
        scenario = Scenario(rosenbrock.configspace, name="blub2", n_trials=11)
        smac = facade(scenario, rosenbrock.train)
        assert os.path.isdir("smac3_output/blub2-old")


def test_continue_when_trials_stopped(rosenbrock):
    class CustomCallbackTrials(Callback):
        def on_tell_end(self, smbo, info: TrialInfo, value: TrialValue) -> bool | None:
            if smbo.runhistory.finished == 20:
                return False

            return

    for facade in [BlackBoxFacade]:
        # We did not optimize in the first run
        scenario = Scenario(rosenbrock.configspace, n_trials=25)
        smac = facade(scenario, rosenbrock.train, callbacks=[CustomCallbackTrials()], overwrite=True)
        smac.optimize()

        # Because of the callback, we stop after evaluating 20 trials
        assert smac.runhistory.finished == 20

        # Now we want to continue the run
        scenario = Scenario(rosenbrock.configspace, n_trials=25)
        smac2 = facade(scenario, rosenbrock.train)

        # Let's see if we restored the runhistory correctly
        assert smac2.runhistory.finished == 20

        smac2.optimize()
        assert smac2.runhistory.finished == 25


def test_continue_when_walltime_stopped(rosenbrock):
    class CustomCallbackWalltime(Callback):
        def __init__(self, limit: float):
            self._limit = limit

        def on_tell_end(self, smbo, info: TrialInfo, value: TrialValue) -> bool | None:
            if smbo.used_walltime > self._limit:
                return False

            return

    for facade in [BlackBoxFacade]:
        LIMIT = 1
        FINAL_LIMIT = 5

        # Same thing now but with walltime limit
        scenario = Scenario(rosenbrock.configspace, n_trials=99999, walltime_limit=FINAL_LIMIT)
        smac = facade(scenario, rosenbrock.train, callbacks=[CustomCallbackWalltime(LIMIT)], overwrite=True)
        smac.optimize()
        assert smac._optimizer.used_walltime > LIMIT and smac._optimizer.used_walltime < FINAL_LIMIT

        # Now we want to continue the run
        scenario = Scenario(rosenbrock.configspace, n_trials=99999, walltime_limit=FINAL_LIMIT)
        smac2 = facade(scenario, rosenbrock.train)

        # Let's see if we restored the runhistory correctly; used walltime should be roughly the same
        # However, since some more things happen in the background, it might be slightly different
        assert pytest.approx(smac2._optimizer.used_walltime, 0.1) == smac._optimizer.used_walltime
        assert smac2.runhistory.finished == smac.runhistory.finished

        smac2.optimize()
        assert smac2.runhistory.finished > smac.runhistory.finished
        assert smac2._optimizer.used_walltime > FINAL_LIMIT
