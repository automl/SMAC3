from smac.initial_design.random_design import RandomInitialDesign
from smac.intensifier.hyperband import Hyperband
from smac.main.config_selector import ConfigSelector
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


def test_initialization(make_scenario, configspace_small):
    """Tests whether HB is initialized correctly."""
    scenario: Scenario = make_scenario(
        configspace_small,
        use_instances=True,
        n_instances=100,
        min_budget=1,
        max_budget=81,
    )
    runhistory = RunHistory()
    intensifier = Hyperband(
        scenario=scenario,
        eta=3,
    )
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

    # Test table 1 from https://arxiv.org/pdf/1603.06560.pdf
    assert intensifier._max_iterations[0] == 5
    assert intensifier._max_iterations[1] == 4
    assert intensifier._max_iterations[2] == 3
    assert intensifier._max_iterations[3] == 2
    assert intensifier._max_iterations[4] == 1

    assert intensifier._n_configs_in_stage[0] == [81, 27, 9, 3, 1]
    assert intensifier._n_configs_in_stage[1] == [34, 11, 3, 1]
    assert intensifier._n_configs_in_stage[2] == [15, 5, 1]
    assert intensifier._n_configs_in_stage[3] == [8, 2]
    assert intensifier._n_configs_in_stage[4] == [5]

    assert intensifier._budgets_in_stage[0] == [1, 3, 9, 27, 81]
    assert intensifier._budgets_in_stage[1] == [3, 9, 27, 81]
    assert intensifier._budgets_in_stage[2] == [9, 27, 81]
    assert intensifier._budgets_in_stage[3] == [27, 81]
    assert intensifier._budgets_in_stage[4] == [81]


def test_next_bracket(make_scenario, configspace_small):
    """Tests whether next bracket works as expected."""
    scenario: Scenario = make_scenario(
        configspace_small,
        use_instances=True,
        n_instances=100,
        min_budget=1,
        max_budget=81,
    )
    runhistory = RunHistory()
    intensifier = Hyperband(
        scenario=scenario,
        eta=3,
    )
    intensifier.runhistory = runhistory
    intensifier.__post_init__()

    for _ in range(20):
        next_bracket = intensifier._get_next_bracket()
        assert next_bracket in intensifier._budgets_in_stage


def test_state(make_scenario, configspace_small, make_config_selector):
    """Tests whether the tracker is saved and loaded correctly."""
    scenario: Scenario = make_scenario(configspace_small, use_instances=False, min_budget=1, max_budget=3)
    runhistory = RunHistory()
    intensifier = Hyperband(scenario=scenario)
    intensifier.config_selector = make_config_selector(scenario, runhistory, n_initial_configs=1)
    intensifier.runhistory = runhistory

    gen = iter(intensifier)

    # Add some configs to the tracker
    for _ in range(12):
        trial = next(gen)
        runhistory.add_running_trial(trial)  # We have to mark it as running manually
        intensifier.update_incumbents(trial.config)

    state = intensifier.get_state()
    assert state["next_bracket"] == 1
    assert len(state["tracker"]) > 0
    intensifier.reset()

    new_state = intensifier.get_state()
    assert new_state["next_bracket"] == 0
    assert len(new_state["tracker"]) == 0

    intensifier.set_state(state)
    new_state = intensifier.get_state()
    assert new_state == state
