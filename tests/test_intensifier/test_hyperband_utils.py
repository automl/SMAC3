from __future__ import annotations

from smac.intensifier.hyperband_utils import (
    determine_HB,
    determine_hyperband_for_multifidelity,
    get_n_trials_for_hyperband_multifidelity,
)


def test_determine_HB():
    min_budget = 1.0
    max_budget = 81.0
    eta = 3

    result = determine_HB(min_budget=min_budget, max_budget=max_budget, eta=eta)

    # Follow algorithm (not the table!) from https://arxiv.org/pdf/1603.06560.pdf (see https://github.com/automl/SMAC3/issues/977)
    expected_max_iterations = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}
    expected_n_configs_in_stage = {
        0: [81, 27, 9, 3, 1],
        1: [34, 11, 3, 1],
        2: [15, 5, 1],
        3: [8, 2],
        4: [5],
    }
    expected_budgets_in_stage = {
        0: [1, 3, 9, 27, 81],
        1: [3, 9, 27, 81],
        2: [9, 27, 81],
        3: [27, 81],
        4: [81],
    }
    expected_trials_used = 206
    expected_budget_used = 1902
    expected_number_of_brackets = 5

    assert result["max_iterations"] == expected_max_iterations
    assert result["n_configs_in_stage"] == expected_n_configs_in_stage
    assert result["budgets_in_stage"] == expected_budgets_in_stage
    assert result["trials_used"] == expected_trials_used
    assert result["budget_used"] == expected_budget_used
    assert result["number_of_brackets"] == expected_number_of_brackets


def test_determine_hyperband_for_multifidelity():
    total_budget = 1000.0
    min_budget = 1.0
    max_budget = 81.0
    eta = 3

    result = determine_hyperband_for_multifidelity(
        total_budget=total_budget, min_budget=min_budget, max_budget=max_budget, eta=eta
    )

    expected_n_trials = 188  # Budget not enough for one full round (would nee 1902 as total budget)

    assert result["n_trials"] == expected_n_trials
    assert result["total_budget"] == total_budget
    assert result["eta"] == eta
    assert result["min_budget"] == min_budget
    assert result["max_budget"] == max_budget


def test_get_n_trials_for_hyperband_multifidelity():
    total_budget = 1000.0
    min_budget = 1.0
    max_budget = 81.0
    eta = 3

    n_trials = get_n_trials_for_hyperband_multifidelity(
        total_budget=total_budget, min_budget=min_budget, max_budget=max_budget, eta=eta
    )

    assert n_trials == 188

if __name__=="__main__":
    test_determine_HB()
    test_determine_hyperband_for_multifidelity()
    test_get_n_trials_for_hyperband_multifidelity()