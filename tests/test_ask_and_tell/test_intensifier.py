from __future__ import annotations

import pytest

from smac import HPOFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


@pytest.fixture
def make_facade(digits_dataset, make_sgd) -> HPOFacade:
    def create(deterministic=True, use_instances=False) -> HPOFacade:
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
        smac = HPOFacade(
            scenario,
            model.train,
            initial_design=HPOFacade.get_initial_design(scenario, n_configs=2, max_ratio=1),
            intensifier=HPOFacade.get_intensifier(scenario, max_config_calls=5, intensify_percentage=0.0),
            logging_level=0,
            overwrite=True,
        )

        return model, smac

    return create


# --------------------------------------------------------------
# Test target function arguments
# --------------------------------------------------------------


def test_target_function_arguments_deterministic(make_facade):
    """Tests whether we get the right arguments."""
    model, smac = make_facade(deterministic=True)

    # The intensifier should not use any budgets
    budgets = smac.get_target_function_budgets()
    assert budgets == [None]

    # The intensifier should not use any instances
    instances = smac.get_target_function_instances()
    assert instances == [None]

    # However, we expect ONE seeds (deterministic), although we have max_config_calls=5
    seeds = smac.get_target_function_seeds()
    assert seeds == [0]


def test_target_function_arguments_deterministic_instances(make_facade):
    """Tests whether we get the right arguments."""
    model, smac = make_facade(deterministic=True, use_instances=True)

    # The intensifier should not use any budgets
    budgets = smac.get_target_function_budgets()
    assert budgets == [None]

    # The intensifier should not use any instances
    instances = smac.get_target_function_instances()
    assert len(instances) == 45

    # However, we expect ONE seeds (deterministic), although we have max_config_calls=5
    seeds = smac.get_target_function_seeds()
    assert seeds == [0]


def test_target_function_arguments_non_deterministic(make_facade):
    """Tests whether we get the right arguments."""
    model, smac = make_facade(deterministic=False)

    # The intensifier should not use any budgets
    budgets = smac.get_target_function_budgets()
    assert budgets == [None]

    # The intensifier should not use any instances
    instances = smac.get_target_function_instances()
    assert instances == [None]

    # However, we expect ONE seeds (deterministic), although we have max_config_calls=5
    seeds = smac.get_target_function_seeds()
    assert len(seeds) == 5


def test_target_function_arguments_non_deterministic_instances(make_facade):
    """Tests whether we get the right arguments."""
    model, smac = make_facade(deterministic=False, use_instances=True)

    # The intensifier should not use any budgets
    budgets = smac.get_target_function_budgets()
    assert budgets == [None]

    # The intensifier should not use any instances
    instances = smac.get_target_function_instances()
    assert len(instances) == 45

    # However, we expect ONE seeds (deterministic), although we have max_config_calls=5
    seeds = smac.get_target_function_seeds()
    assert len(seeds) == 5


# --------------------------------------------------------------
# Test tell without ask
# --------------------------------------------------------------


def test_tell_without_ask(make_facade):
    """Tests whether tell works without ask. In the case of ``intensifier`` it should work."""
    model, smac = make_facade(deterministic=False, use_instances=True)
    seed = smac.get_target_function_seeds()[0]
    instance = smac.get_target_function_instances()[0]

    # We can provide SMAC with custom configurations first
    for config in model.configspace.sample_configuration(10):
        cost = model.train(config, seed=seed, instance=instance)

        trial_info = TrialInfo(config, seed=seed, instance=instance)
        trial_value = TrialValue(cost=cost, time=0.5)

        smac.tell(trial_info, trial_value)

        # Seed is not supported
        with pytest.raises(ValueError):
            trial_info = TrialInfo(config, seed=23412351234)
            trial_value = TrialValue(cost=cost, time=0.5)

            smac.tell(trial_info, trial_value)

        # Instance is not supported
        with pytest.raises(ValueError):
            trial_info = TrialInfo(config, seed=seed, instance=None)
            trial_value = TrialValue(cost=cost, time=0.5)

            smac.tell(trial_info, trial_value)

    assert smac.stats.finished == 10
    assert smac.stats.submitted == 0  # We have 0 submittions because we don't call the ask method

    smac.optimize()

    # After optimization we expect to have +10 finished
    assert smac.stats.finished == smac._scenario.n_trials + 10
    assert smac.stats.submitted == smac._scenario.n_trials + 1  # We have one submittion which is skipped
    assert len(smac.runhistory) == smac._scenario.n_trials + 10  # However, the skipped one is not saved anymore

    # We expect SMAC to use the same seed if configs with a seed were passed
    for k in smac.runhistory.keys():
        assert k.seed in smac.get_target_function_seeds()


# --------------------------------------------------------------
# Test ask and tell
# --------------------------------------------------------------


def test_ask_and_tell(make_facade):
    """Tests whether ask followed by a tell works. We can not ensure the same results when using
    ask and tell and optimize."""
    model, smac = make_facade(deterministic=False, use_instances=True)

    for i in range(smac._scenario.n_trials):
        trial_info = smac.ask()

        cost = model.train(trial_info.config, seed=trial_info.seed, instance=trial_info.instance)
        trial_value = TrialValue(cost=cost, time=0.5)

        smac.tell(trial_info, trial_value)

    assert smac.incumbent is not None


def test_ask_and_tell_after_optimization(make_facade):
    """
    Tests whether ask followed by a tell works after the optimization.
    """
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
    """
    model, smac = make_facade(deterministic=False, use_instances=True)

    info = []
    for i in range(10):
        trial_info = smac.ask()

        # Make sure the trials are different
        assert trial_info not in info

        info += [trial_info]
    """

    # Not supported yet
    pass
