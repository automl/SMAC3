import pytest
from smac import HyperparameterFacade, Scenario, MultiFidelityFacade


def test_termination_cost_threshold(rosenbrock):
    termination_cost_threshold = 100
    scenario = Scenario(rosenbrock.configspace, n_trials=200, termination_cost_threshold=termination_cost_threshold)
    smac = HyperparameterFacade(
        scenario,
        rosenbrock.train,
        intensifier=HyperparameterFacade.get_intensifier(scenario, max_config_calls=1),
        overwrite=True,
    )
    i = smac.optimize()

    counter = 0
    config = None
    for k, v in smac.runhistory.items():
        if v.cost < termination_cost_threshold:
            config = smac.runhistory.get_config(k.config_id)
            counter += 1

    # We expect only one cost below termination_cost_threshold
    assert config == i
    assert counter == 1
    assert smac.validate(i) < termination_cost_threshold


def test_termination_cost_threshold_with_fidelities(rosenbrock):
    max_budget = 9
    termination_cost_threshold = 100
    scenario = Scenario(
        rosenbrock.configspace,
        n_trials=200,
        min_budget=1,
        max_budget=max_budget,
        termination_cost_threshold=termination_cost_threshold,
    )
    smac = MultiFidelityFacade(
        scenario,
        rosenbrock.train,
        overwrite=True,
    )
    i = smac.optimize()

    counter = 0
    config = None
    for k, v in smac.runhistory.items():
        if v.cost < termination_cost_threshold and k.budget == max_budget:
            config = smac.runhistory.get_config(k.config_id)
            counter += 1

    # We expect only one cost below termination_cost_threshold
    assert config == i
    assert counter == 1
    assert smac.validate(i) < termination_cost_threshold
