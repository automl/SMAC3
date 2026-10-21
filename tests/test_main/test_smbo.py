from unittest.mock import patch

import pytest

from smac import HyperparameterOptimizationFacade, MultiFidelityFacade, Scenario
from smac.main import smbo
from smac.runhistory.dataclasses import TrialInfo, TrialValue


def test_termination_cost_threshold(rosenbrock):
    termination_cost_threshold = 100
    scenario = Scenario(
        rosenbrock.configspace,
        n_trials=200,
        termination_cost_threshold=termination_cost_threshold,
    )
    smac = HyperparameterOptimizationFacade(
        scenario,
        rosenbrock.train,
        intensifier=HyperparameterOptimizationFacade.get_intensifier(
            scenario, max_config_calls=1
        ),
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


def test_termination_cost_threshold_with_fidelities(rosenbrock, compute_cost):
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
    for c in smac.runhistory.get_configs():
        if compute_cost(smac.runhistory, c) < termination_cost_threshold:
            config = c
            counter += 1

    # We expect only one cost below termination_cost_threshold
    assert config == i
    assert counter == 1
    assert smac.validate(i) < termination_cost_threshold


def test_smbo_reset(rosenbrock):
    scenario = Scenario(rosenbrock.configspace, n_trials=3)
    smac = HyperparameterOptimizationFacade(
        scenario,
        rosenbrock.train,
        intensifier=HyperparameterOptimizationFacade.get_intensifier(
            scenario, max_config_calls=1
        ),
        overwrite=True,
    )

    smac.optimize()
    runhistory_len_before_reset = len(smac.runhistory)
    assert runhistory_len_before_reset == 3

    smac.optimizer.reset()

    assert smac.optimizer._used_target_function_walltime == 0.0
    assert smac.optimizer._used_target_function_cputime == 0.0
    assert len(smac.runhistory) == 0

    smac.optimize()
    runhistory_len_after_reset = len(smac.runhistory)
    assert runhistory_len_after_reset == 3


def test_tell_without_seed(rosenbrock):
    scenario = Scenario(
        rosenbrock.configspace,
        n_trials=100,
        min_budget=1,
        max_budget=9,
    )
    smac = MultiFidelityFacade(
        scenario,
        rosenbrock.train,
        overwrite=True,
    )
    info = smac.ask()

    # We expect an error because the trial can not be matched without seed
    with pytest.raises(ValueError, match="seed"):
        smac.tell(TrialInfo(config=info.config, budget=info.budget), TrialValue(cost=1.0))

    assert len(smac.runhistory.get_trials(info.config)) == 0


def test_tell_with_same_seed(rosenbrock):
    scenario = Scenario(
        rosenbrock.configspace,
        n_trials=100,
        min_budget=1,
        max_budget=9,
    )
    smac = MultiFidelityFacade(
        scenario,
        rosenbrock.train,
        overwrite=True,
    )

    # The first bracket evaluates 9 configs on budget 1 and promotes 3 of them to budget 3
    with patch.object(smbo.logger, "warning") as warning:
        for _ in range(30):
            info = smac.ask()
            trial = TrialInfo(config=info.config, seed=info.seed, budget=info.budget)
            smac.tell(trial, TrialValue(cost=rosenbrock.train(info.config)))

    counter = 0
    for c in smac.runhistory.get_configs():
        budgets = [t.budget for t in smac.runhistory.get_trials(c, highest_observed_budget_only=False)]
        if 1 in budgets and 3 in budgets:
            counter += 1

    # We expect promoted configs and no trials stuck in RUNNING
    assert counter > 0
    assert len(smac.runhistory.get_running_trials()) == 0
    warning.assert_not_called()


def test_tell_with_unknown_seed(rosenbrock):
    scenario = Scenario(
        rosenbrock.configspace,
        n_trials=100,
        min_budget=1,
        max_budget=9,
    )
    smac = MultiFidelityFacade(
        scenario,
        rosenbrock.train,
        overwrite=True,
    )
    info = smac.ask()
    unknown_seed = info.seed + 1

    with patch.object(smbo.logger, "warning") as warning:
        smac.tell(TrialInfo(config=info.config, seed=unknown_seed, budget=info.budget), TrialValue(cost=1.0))

    # We expect a warning because the trial can not be matched with an unknown seed
    warning.assert_called_once()
    assert f"Passed seed {unknown_seed}" in warning.call_args.args[0]
