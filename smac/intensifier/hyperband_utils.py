from __future__ import annotations

from smac.intensifier.successive_halving import SuccessiveHalving
import numpy as np


def determine_HB(eta: int, min_budget: float, max_budget: float) -> dict:
    _s_max = SuccessiveHalving._get_max_iterations(eta, max_budget, min_budget)

    _max_iterations: dict[int, int] = {}
    _n_configs_in_stage: dict[int, list] = {}
    _budgets_in_stage: dict[int, list] = {}

    for i in range(_s_max + 1):
        max_iter = _s_max - i

        _budgets_in_stage[i], _n_configs_in_stage[i] = SuccessiveHalving._compute_configs_and_budgets_for_stages(
            eta, max_budget, max_iter, _s_max
        )
        _max_iterations[i] = max_iter + 1

    total_trials = np.sum([np.sum(v) for v in _n_configs_in_stage.values()])
    total_budget = np.sum([np.sum(v) for v in _budgets_in_stage.values()])

    return {
        "max_iterations": _max_iterations,
        "n_configs_in_stage": _n_configs_in_stage,
        "budgets_in_stage": _budgets_in_stage,
        "trials_used": total_trials,
        "budget_used": total_budget,
        "number_of_brackets": _s_max
    }


def determine_hyperband_for_multifidelity(
    total_budget: float, min_budget: float, max_budget: float, eta: int = 3
) -> dict:
    # Determine the HB
    hyperband_round = determine_HB(eta=eta, min_budget=min_budget, max_budget=max_budget)

    # Calculate how many HB rounds we can have
    budget_used_per_hyperband_round = hyperband_round["budget_used"]
    number_of_full_hb_rounds = int(np.floor(total_budget / budget_used_per_hyperband_round))
    remaining_budget = total_budget % budget_used_per_hyperband_round
    trials_used_per_hb_round = hyperband_round["trials_used"]
    n_configs_in_stage = hyperband_round["n_configs_in_stage"]
    budgets_in_stage = hyperband_round["budgets_in_stage"]


    remaining_trials = 0
    for stage in n_configs_in_stage.keys():
        B = budgets_in_stage[stage]
        C = n_configs_in_stage[stage]
        for b, c in zip(B, C):
            
            # How many trials are left?
            # If b * c is lower than remaining budget, we can add full c
            # otherwise we need to find out how many trials we can do with this budget
            remaining_trials += min(c, int(np.floor(remaining_budget / b)))
            # We cannot go lower than 0
            # If we are in the case of b*c > remaining_budget, we will not have any
            # budget left. We can not add full c but the number of trials that still fit
            remaining_budget = max(0, remaining_budget - b*c)

            # print(stage, b, c)
            # print("-"*20, remaining_trials, remaining_budget)


    n_trials = int(number_of_full_hb_rounds * trials_used_per_hb_round + remaining_trials)

    hyperband_info = hyperband_round
    hyperband_info["n_trials"] = n_trials
    hyperband_info["total_budget"] = total_budget
    hyperband_info["eta"] = eta
    hyperband_info["min_budget"] = min_budget
    hyperband_info["max_budget"] = max_budget

    return hyperband_info

def print_hyperband_summary(hyperband_info: dict) -> None:
    I = hyperband_info  # noqa: E741
    print("-"*30, "HYPERBAND IN MULTI-FIDELITY", "-"*30)
    print("total budget:\t\t",  I["total_budget"])
    print("total number of trials:\t", I["n_trials"])
    print("number of HB rounds:\t", I["total_budget"] / I["budget_used"])
    print()

    print("\t~~~~~~~~~~~~HYPERBAND ROUND")
    print("\teta:\t\t\t\t\t", I["eta"])
    print("\tmin budget per trial:\t\t\t", I["min_budget"])
    print("\tmax budget per trial:\t\t\t", I["max_budget"])
    print("\ttotal number of trials per HB round:\t", I["trials_used"])
    print("\tbudget used per HB round:\t\t", I["budget_used"])
    print("\tnumber of brackets:\t\t\t", I["number_of_brackets"])
    print("\tbudgets per stage:\t\t\t", I["budgets_in_stage"])
    print("\tn configs per stage:\t\t\t", I["n_configs_in_stage"])
    print("-"* (2*30 + len("HYPERBAND IN MULTI-FIDELITY") + 2))

def get_n_trials_for_hyperband_multifidelity(total_budget: float, min_budget: float, max_budget: float, eta: int = 3, print_summary: bool = True) -> int:
    hyperband_info = determine_hyperband_for_multifidelity(total_budget=total_budget, eta=eta, min_budget=min_budget, max_budget=max_budget)
    if print_summary:
        print_hyperband_summary(hyperband_info=hyperband_info)
    return hyperband_info["n_trials"]