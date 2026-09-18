from __future__ import annotations

import tempfile

from ConfigSpace import ConfigurationSpace, Float

from smac import BlackBoxFacade, Scenario
from smac.runhistory import TrialValue

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


def _configspace() -> ConfigurationSpace:
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("x0", (0.0, 1.0)))

    return configspace


def target(config, seed: int = 0):
    return config["x0"]


def _smac(**kwargs):
    scenario = Scenario(
        _configspace(),
        n_trials=30,
        seed=0,
        deterministic=True,
        output_directory=tempfile.mkdtemp(),
    )

    return BlackBoxFacade(scenario, target, overwrite=True, logging_level=60, **kwargs)


def _warm_up(smac, config_selector):
    """Runs until the surrogate model has been trained, so the acquisition path is actually reached."""
    for _ in range(smac.scenario.n_trials):
        if config_selector._previous_entries > 0:
            return

        info = smac.ask()
        smac.tell(info, TrialValue(cost=target(info.config), time=0.1))

    raise AssertionError("The surrogate model was never trained.")


def _settle(smac):
    """Asks once more without telling, so that no new data is pending and nothing would normally happen."""
    smac.ask()


def _counting(config_selector):
    """Replaces the model and the acquisition function with counting stand-ins."""
    trains = []
    updates = []

    model = config_selector._model
    acquisition_function = config_selector._acquisition_function

    original_train = model.train
    original_update = acquisition_function.update

    def train(X, Y):
        trains.append(X.shape[0])

        return original_train(X, Y)

    def update(**kwargs):
        updates.append(kwargs["num_data"])

        return original_update(**kwargs)

    model.train = train
    acquisition_function.update = update

    return trains, updates


def test_the_acquisition_function_is_updated_without_retraining_the_model():
    smac = _smac()
    selector = smac._intensifier.config_selector

    _warm_up(smac, selector)
    _settle(smac)

    trains, updates = _counting(selector)

    # Nothing new has been reported, so neither would normally happen.
    smac.ask()
    assert trains == []

    selector.invalidate_acquisition()
    smac.ask()

    assert len(updates) == 1
    assert trains == [], "Invalidating the acquisition function must not retrain the surrogate model."


def test_invalidating_clears_the_flag_so_it_happens_once():
    smac = _smac()
    selector = smac._intensifier.config_selector

    _warm_up(smac, selector)
    _settle(smac)

    _, updates = _counting(selector)

    selector.invalidate_acquisition()
    smac.ask()
    before = len(updates)

    smac.ask()

    assert len(updates) == before


def test_the_challengers_ranked_before_the_change_are_discarded():
    """Without this the search hands out up to retrain_after configurations chosen by the old ranking."""
    smac = _smac()
    selector = smac._intensifier.config_selector

    _warm_up(smac, selector)

    # Isolate the flag from the ordinary "retrain every retrain_after configurations" rule.
    selector._counter = 0

    assert selector._check_for_retrain() is False

    selector.invalidate_acquisition(force_retrain=True)

    assert selector._check_for_retrain() is True
    assert selector._check_for_retrain() is False, "The flag has to clear, or the maximizer runs every time."


def test_the_challengers_can_be_kept():
    smac = _smac()
    selector = smac._intensifier.config_selector

    selector.invalidate_acquisition(force_retrain=False)

    assert selector._acquisition_needs_update is True
    assert selector._check_for_retrain() is False


def test_the_trial_count_matches_what_the_acquisition_function_is_told():
    """Anything anchored against the trial count has to be measured in the same units as the decay."""
    smac = _smac()
    selector = smac._intensifier.config_selector

    _warm_up(smac, selector)
    _settle(smac)

    _, updates = _counting(selector)

    expected = selector._current_num_data()
    selector.invalidate_acquisition()
    smac.ask()

    assert updates == [expected]
