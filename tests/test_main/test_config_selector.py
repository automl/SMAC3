from unittest import mock

import numpy as np
import pytest

from smac.main.exceptions import ConfigurationSpaceExhaustedException
from ConfigSpace import ConfigurationSpace, Categorical
from smac import HyperparameterOptimizationFacade, Scenario
from smac.main.config_selector import ConfigSelector


def test_exhausted_configspace():
    cs = ConfigurationSpace()
    # The configuration space contains only 3 possible configurations.
    cs.add(Categorical("x", [1, 2, 3]))

    def objective_function(x, seed):
        return x["x"] ** 2
    
    scenario = Scenario(
        configspace=cs,
        n_trials=10,
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        objective_function,
        overwrite=True,
    )

    smac.optimize()

    # SMAC should stop gracefully after evaluating all 3 configurations.
    assert len(smac.runhistory.get_configs()) == 3

def test_get_x_best_predicts_configurations_in_one_batch():
    scenario = Scenario(ConfigurationSpace())
    selector = ConfigSelector(scenario)
    X = np.array([[0.3, 1.0], [0.1, 2.0], [0.1, 3.0]])

    model = mock.Mock()
    model.predict_marginalized.return_value = (
        np.array([[0.3], [0.1], [0.1]]),
        np.ones((3, 1)),
    )
    selector._model = model

    x_best, best_observation = selector._get_x_best(X)

    model.predict_marginalized.assert_called_once()
    np.testing.assert_array_equal(model.predict_marginalized.call_args.args[0], X)
    np.testing.assert_array_equal(x_best, X[1])
    assert best_observation == pytest.approx(0.1)

def test_q_ei_strategy_does_not_estimate_running_configs():
    selector = ConfigSelector(Scenario(ConfigurationSpace()), batch_sampling_estimation_strategy="q_ei")
    assert selector.estimate_running_config_costs(np.zeros((2, 1)), np.zeros((3, 1)), "q_ei") is None


def test_q_ei_passes_running_configs_as_pending_points():
    from smac import BlackBoxFacade
    from smac.acquisition.function import QExpectedImprovement
    from smac.acquisition.maximizer import RandomSearch

    cs = ConfigurationSpace({"x": (-5.0, 5.0)}, seed=0)
    scenario = Scenario(cs, n_trials=12, n_workers=2, seed=0, deterministic=True)
    acq = QExpectedImprovement(n_samples=16)
    pending = []
    original_update = acq._update

    def spy(**kwargs):
        pending.append(kwargs.get("X_pending"))
        original_update(**kwargs)

    acq._update = spy
    smac = BlackBoxFacade(
        scenario,
        lambda config, seed=0: config["x"] ** 2,
        acquisition_function=acq,
        acquisition_maximizer=RandomSearch(cs, acq, challengers=50),
        config_selector=BlackBoxFacade.get_config_selector(scenario, batch_sampling_estimation_strategy="q_ei"),
        overwrite=True,
        logging_level=50,
    )
    smac.optimize()

    assert len(smac.runhistory) == 12
    assert len(pending) > 0
    assert all(p is not None and p.ndim == 2 for p in pending)
