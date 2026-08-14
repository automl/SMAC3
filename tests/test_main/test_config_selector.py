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