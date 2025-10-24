import pytest

from smac.main.exceptions import ConfigurationSpaceExhaustedException
from ConfigSpace import ConfigurationSpace, Categorical
from smac import HyperparameterOptimizationFacade, Scenario


def test_exhausted_configspace():
    cs = ConfigurationSpace()
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

    with pytest.raises(ConfigurationSpaceExhaustedException):
        smac.optimize()
