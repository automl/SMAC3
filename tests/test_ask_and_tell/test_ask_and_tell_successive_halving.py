from __future__ import annotations

import pytest

from smac import MultiFidelityFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


@pytest.fixture
def make_facade(digits_dataset, make_sgd) -> MultiFidelityFacade:
    def create(deterministic: bool = True, use_instances: bool = False, n_seeds: int = 1) -> MultiFidelityFacade:
        model = make_sgd(digits_dataset)

        instances_kwargs = {}
        if use_instances:
            instances_kwargs = {
                "instances": digits_dataset.get_instances(),
                "instance_features": digits_dataset.get_instance_features(),
            }

        scenario = Scenario(
            model.configspace,
            deterministic=deterministic,
            n_trials=15,  # We want to try max 5000 different configurations
            min_budget=1,  # Use min one instance
            max_budget=45,  # Use max 45 instances (if we have a lot of instances we could constraint it)
            **instances_kwargs,
        )

        # Create our SMAC object and pass the scenario and the train method
        smac = MultiFidelityFacade(
            scenario,
            model.train,
            initial_design=MultiFidelityFacade.get_initial_design(scenario, n_configs=2, max_ratio=1),
            intensifier=MultiFidelityFacade.get_intensifier(scenario, n_seeds=n_seeds),
            logging_level=0,
            overwrite=True,
        )

        return model, smac

    return create


# --------------------------------------------------------------
# Test tell without ask
# --------------------------------------------------------------


def test_tell_without_ask_instances(make_facade):
    """Tests whether tell works without ask. In the case of ``intensifier`` it should work."""
    model, smac = make_facade(deterministic=False, use_instances=True)
    N = 10
    seed = 95
    instance = smac.scenario.instances[0]

    # We can provide SMAC with custom configurations first
    for config in model.configspace.sample_configuration(N):
        cost = model.train(config, seed=seed, instance=instance)

        trial_info = TrialInfo(config, seed=seed, instance=instance)
        trial_value = TrialValue(cost=cost, time=0.5)
        smac.tell(trial_info, trial_value)

        # Instance is not supported
        with pytest.raises(ValueError):
            trial_info = TrialInfo(config, seed=seed, instance=None)
            trial_value = TrialValue(cost=cost, time=0.5)
            smac.tell(trial_info, trial_value)

    # We submitted N configurations and finished all of them
    assert smac.runhistory.finished == N
    assert smac.runhistory.submitted == N

    smac.optimize()

    # After optimization we expect to have +10 finished
    assert smac.runhistory.finished == smac._scenario.n_trials
    assert smac.runhistory.submitted == smac._scenario.n_trials

    # We expect SMAC to use the same seed if configs with a seed were passed
    for k in smac.runhistory.keys():
        assert k.seed == 95


def test_tell_without_ask_budgets(make_facade):
    """Tests whether tell works without ask. In the case of ``intensifier`` it should work."""
    N = 10
    model, smac = make_facade(deterministic=False, use_instances=False, n_seeds=1)
    seeds = [i + 90 for i in range(N)]
    budget = 2

    # We can provide SMAC with custom configurations first
    # Since we use N different seeds, the incumbent will only be updated after teeling the first trial
    for seed, config in zip(seeds, model.configspace.sample_configuration(N)):
        cost = model.train(config, seed=seed, budget=budget)

        trial_info = TrialInfo(config, seed=seed, budget=budget)
        trial_value = TrialValue(cost=cost, time=0.5)
        smac.tell(trial_info, trial_value)

    # We submitted N configurations and finished all of them
    assert smac.runhistory.finished == N
    assert smac.runhistory.submitted == N

    smac.optimize()

    # After optimization we expect to have +10 finished
    assert smac.runhistory.finished == smac._scenario.n_trials
    assert smac.runhistory.submitted == smac._scenario.n_trials

    # We expect SMAC to use the same seed if configs with a seed were passed
    for k in smac.runhistory.keys():
        assert k.seed in seeds


# --------------------------------------------------------------
# Test ask and tell
# --------------------------------------------------------------


def test_ask_and_tell(make_facade):
    """Tests whether ask followed by a tell works."""
    model, smac = make_facade(deterministic=False, use_instances=True)

    for _ in range(smac._scenario.n_trials):
        trial_info = smac.ask()

        cost = model.train(trial_info.config, seed=trial_info.seed, instance=trial_info.instance)
        trial_value = TrialValue(cost=cost, time=0.5)

        smac.tell(trial_info, trial_value)

    assert smac.intensifier.get_incumbent() == smac.optimize()


def test_ask_and_tell_after_optimization(make_facade):
    """Tests whether ask followed by a tell works after the optimization."""
    model, smac = make_facade(deterministic=False, use_instances=True)
    smac.optimize()
    trials = len(smac.runhistory)

    for _ in range(10):
        trial_info = smac.ask()

        cost = model.train(trial_info.config, seed=trial_info.seed, instance=trial_info.instance)
        trial_value = TrialValue(cost=cost, time=0.5)

        smac.tell(trial_info, trial_value)

    # We should have more entries in the runhistory now
    assert trials < len(smac.runhistory)


# --------------------------------------------------------------
# Test multiple asks successively
# --------------------------------------------------------------


def test_multiple_asks_successively(make_facade):
    model, smac = make_facade(deterministic=True, use_instances=True)

    info = []
    for _ in range(50):
        trial_info = smac.ask()

        # Make sure the trials are different
        assert trial_info not in info
        info += [trial_info]
