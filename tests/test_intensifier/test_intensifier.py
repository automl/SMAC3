import itertools
from functools import lru_cache
import pytest

from smac.initial_design.random_design import RandomInitialDesign
from smac.intensifier.intensifier import Intensifier
from smac.main.config_selector import ConfigSelector
from smac.runhistory import TrialInfo, TrialKey, TrialValue
from smac.runhistory.enumerations import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


##Mixins

## Intermediate decision
from smac.intensifier.mixins import intermediate_decision as idmixin
from smac.intensifier.mixins import intermediate_update as iumixin
from smac.intensifier.mixins import update_incumbent as uimixin

intermediate_decision_mixins = [
    idmixin.NewCostDominatesOldCost,
    idmixin.NewCostDominatesOldCostSkipFirst,
    idmixin.DoublingNComparison,
    idmixin.DoublingNComparisonFour,
    idmixin.Always,
    idmixin.Never,
]

intermediate_update_mixins = [
    iumixin.FullIncumbentComparison,
    iumixin.SingleIncumbentComparison,
    iumixin.ClosestIncumbentComparison,
    iumixin.NoComparison,
]

update_incumbent_mixins = [
    uimixin.NonDominatedUpdate,
    uimixin.BootstrapUpdate,
]

ALL_MIXINS = [o for o in itertools.product(intermediate_decision_mixins,
                                           intermediate_update_mixins,
                                           update_incumbent_mixins)]

def mixin_id(mixins):
    return "Intensifier" if not mixins else "".join(m.__name__ for m in mixins)

def build_intensifier_class(mixins):
    """Create a unique class for a given list of mixins."""
    # MRO: mixins first, then Intensifier
    bases = tuple(mixins) + (Intensifier,)
    name = mixin_id(mixins)
    return type(name, bases, {})

@pytest.fixture(params=ALL_MIXINS, ids=mixin_id)
def IntensifierClass(request):
    return build_intensifier_class(request.param)



def test_trials_of_interest(IntensifierClass, make_scenario, configspace_small, make_config_selector):
    """Tests whether the trials of interests are as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = IntensifierClass(scenario=scenario, max_config_calls=10, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

    trials = intensifier.get_trials_of_interest(None)
    assert len(trials) == 10
    assert trials[0].seed == trials[1].seed == trials[2].seed
    assert trials[2].seed != trials[3].seed
    assert trials[0].instance != trials[1].instance != trials[2].instance
    assert trials[0].instance == trials[3].instance

    # Test validation
    val_trials = intensifier.get_trials_of_interest(None, validate=True, seed=8888)
    assert len(val_trials) == 10
    assert val_trials[0].seed == val_trials[1].seed == val_trials[2].seed
    assert val_trials[2].seed != val_trials[3].seed
    assert val_trials[0].instance != val_trials[1].instance != val_trials[2].instance
    assert val_trials[0].instance == val_trials[3].instance

    # We expect different seeds for the validation runs
    assert trials[0].seed != val_trials[0].seed
    assert trials[3].seed != val_trials[3].seed

    # But the same instances
    assert trials[0].instance == val_trials[0].instance
    assert trials[3].instance == val_trials[3].instance


def test_next_trials(IntensifierClass, make_scenario, configspace_small, make_config_selector):
    """Tests whether the next trials are as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = IntensifierClass(scenario=scenario, max_config_calls=9, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

    # If we don't specify N we expect max config call trials here
    trials = intensifier._get_next_trials(None)
    assert len(trials) == 9

    trials = intensifier._get_next_trials(None, N=3)
    assert len(trials) == 3

    trials = intensifier._get_next_trials(None, N=141)
    assert len(trials) == 9

    # Now we want to test if the shuffling works as expected
    # Since we group the trials by seed, we expect the first 3 trials to have the same seed
    seed_changes = 0
    previous_seed = None
    for trial in trials:
        if trial.seed != previous_seed:
            seed_changes += 1
            previous_seed = trial.seed

    assert seed_changes == 3

    # Next, we want to check if evaluated trials are removed
    config = configspace_small.get_default_configuration()
    runhistory.add(config=config, cost=0.5, time=0.0, instance=trials[0].instance,
                   seed=trials[0].seed)

    trials = intensifier._get_next_trials(config)
    assert len(trials) == 8

    # And if we add a running trial?
    runhistory.add(
        config=config,
        cost=0.5,
        time=0.0,
        instance=trials[0].instance,
        seed=trials[0].seed,
        status=StatusType.RUNNING,
    )

    trials = intensifier._get_next_trials(config)
    assert len(trials) == 7

    # The only thing missing is ``from_keys`` now
    isbk = trials[0].get_instance_seed_budget_key()

    trials = intensifier._get_next_trials(config, from_keys=[isbk])
    assert len(trials) == 1
    assert trials[0].instance == isbk.instance
    assert trials[0].seed == isbk.seed


def test_next_trials_counter(IntensifierClass, make_scenario, configspace_small, make_config_selector):
    """Tests whether the next trials are as expected."""

    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = IntensifierClass(scenario=scenario, max_config_calls=9, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory)
    intensifier.runhistory = runhistory
    intensifier.__post_init__()
    config = configspace_small.get_default_configuration()

    # All good here
    trials = intensifier._get_next_trials(config, N=5)
    assert len(trials) == 5

    # But now we set a trial running
    runhistory.add_running_trial(trials[0])

    # Now we call the same thing again but expect 4 trials instead
    trials = intensifier._get_next_trials(config, N=5)
    assert len(trials) == 4

    # More interesting case: Use ``from_keys`` too
    instances = []
    for trial in trials:
        instances.append(trial.get_instance_seed_budget_key())

    # We have three instances now ...
    del instances[0]

    # ... but request 8
    trials = intensifier._get_next_trials(config, N=8, from_keys=instances)
    assert len(trials) == 3

    # ... but request 2
    trials = intensifier._get_next_trials(config, N=2, from_keys=instances)
    assert len(trials) == 2


def test_intensifier(IntensifierClass, make_scenario, configspace_small, make_config_selector):
    """Tests whether the generator returns trials as expected."""
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = IntensifierClass(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory

    gen = iter(intensifier)

    # Because queue is empty and we don't have any incumbents yet, we expect the first trial
    # to be the config from the random initial design
    trial = next(gen)
    runhistory.add_running_trial(trial)  # We have to mark it as running manually
    intensifier.update_incumbents(trial.config)
    assert intensifier.config_selector._initial_design_configs[0] == trial.config

    # In the next step, (config, N*2) is added to the queue but the intensifier realizes
    # that the previous trial is still running
    # Therefore, we expect to sample a new configuration as there are still no incumbents available
    # (not evaluated yet)
    trial2 = next(gen)
    runhistory.add_running_trial(trial2)
    intensifier.update_incumbents(trial2.config)
    assert intensifier.config_selector._processed_configs[1] == trial2.config

    # Let's mark the first trial as finished
    # The config should become an incumbent now.
    runhistory.add(
        config=trial.config,
        cost=10,
        time=0.0,
        instance=trial.instance,
        seed=trial.seed,
        force_update=True
    )
    intensifier.update_incumbents(trial.config)
    assert intensifier.get_incumbent() == trial.config

    # Since everything in the queue is running, we start to intensify the incumbent
    trial3 = next(gen)
    runhistory.add_running_trial(trial3)
    intensifier.update_incumbents(trial3.config)
    assert trial3.config == trial.config

    # And we expect a new config again (after incumbent intensification)
    trial4 = next(gen)
    runhistory.add_running_trial(trial4)
    intensifier.update_incumbents(trial4.config)
    assert intensifier.config_selector._processed_configs[2] == trial4.config


def test_intensifier_with_filled_runhistory(IntensifierClass, make_scenario, configspace_small, make_config_selector):
    """Tests whether entries from the runhistory are incorporated."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=True, n_instances=3)
    runhistory = RunHistory()
    intensifier = IntensifierClass(scenario=scenario, max_config_calls=3, seed=0)
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory
    config = configspace_small.get_default_configuration()
    config2 = configspace_small.sample_configuration()

    # Add some entries to the runhistory
    runhistory.add(
        config=config,
        cost=0.5,
        time=0.0,
        instance=scenario.instances[1],
        seed=8,
        status=StatusType.RUNNING,
    )

    runhistory.add(
        config=config2,
        cost=0.5,
        time=0.0,
        instance=scenario.instances[1],
        seed=59,
        status=StatusType.SUCCESS,
    )

    # Now we post init
    intensifier.__post_init__()

    assert 8 in intensifier._tf_seeds
    assert 59 in intensifier._tf_seeds

    gen = iter(intensifier)
    trial = next(gen)
    assert trial.seed == 8 or trial.seed == 59

    # Make sure the configs are inside
    assert (config, 1) in intensifier._queue

    # Config 2 is not in the queue anymore because it is the only config which was successful
    # Therefore, it is the incumbent and not a challenger anymore
    assert (config2, 1) not in intensifier._queue
    assert config2 in intensifier.get_incumbents()
