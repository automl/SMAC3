from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration

from smac.runhistory import RunHistory
from smac.runhistory.dataclasses import InstanceSeedBudgetKey


def _get_costs(
    runhistory: RunHistory,
    configs: list[Configuration],
    config_instance_seed_budget_keys: list[list[InstanceSeedBudgetKey]],
) -> np.ndarray:
    """Returns the costs of the passed configurations.

    Parameters
    ----------
    runhistory : RunHistory
        The runhistory containing the passed configs.
    configs : list[Configuration]
        The configs for which the costs should be returned.
    config_instance_seed_budget_keys: list[list[InstanceSeedBudgetKey]]
        The instance-seed budget keys for the configs for which the costs should be returned.

    Returns
    -------
    costs : np.ndarray[n_points, n_objectives]
        Costs of the given configs.
    """
    assert len(configs) == len(config_instance_seed_budget_keys)

    # Now we get the costs for the trials of the config
    average_costs = []

    for config, isb_keys in zip(configs, config_instance_seed_budget_keys):
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each
        # configuration
        # However, we only want to consider the config trials
        # Average cost is a list of floats (one for each objective)
        average_cost = runhistory.average_cost(config, isb_keys, normalize=False)
        average_costs += [average_cost]

    # Let's work with a numpy array for efficiency
    return np.vstack(average_costs)


def calculate_pareto_front(
    runhistory: RunHistory,
    configs: list[Configuration],
    config_instance_seed_budget_keys: list[list[InstanceSeedBudgetKey]],
) -> list[Configuration]:
    """Compares the passed configurations and returns only the ones on the pareto front.

    Parameters
    ----------
    runhistory : RunHistory
        The runhistory containing the given configurations.
    configs : list[Configuration]
        The configurations from which the Pareto front should be computed.
    config_instance_seed_budget_keys: list[list[InstanceSeedBudgetKey]]
        The instance-seed budget keys for the configurations on the basis of which the Pareto front should be computed.

    Returns
    -------
    pareto_front : list[Configuration]
        The pareto front computed from the given configurations.
    """
    costs = _get_costs(runhistory, configs, config_instance_seed_budget_keys)

    # The following code is an efficient pareto front implementation
    is_efficient = np.arange(costs.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    new_incumbents = [configs[i] for i in is_efficient]
    return new_incumbents


def sort_by_crowding_distance(
    runhistory: RunHistory,
    configs: list[Configuration],
    config_instance_seed_budget_keys: list[list[InstanceSeedBudgetKey]],
) -> list[Configuration]:
    """Sorts the passed configurations by their crowding distance. Taken from
    https://github.com/anyoptimization/pymoo/blob/20abef1ade71915352217400c11ece4c2f35163e/pymoo/algorithms/nsga2.py


    Parameters
    ----------
    runhistory : RunHistory
        The runhistory containing the given configurations.
    configs : list[Configuration]
        The configurations which should be sorted.
    config_instance_seed_budget_keys: list[list[InstanceSeedBudgetKey]]
        The instance-seed budget keys for the configurations which should be sorted.

    Returns
    -------
    sorted_list : list[Configuration]
        Configurations sorted by crowding distance.
    """
    F = _get_costs(runhistory, configs, config_instance_seed_budget_keys)
    infinity = 1e14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        # distances = np.full(n_points, infinity)
        return configs
    else:
        # Sort each column and get index
        I = np.argsort(F, axis=0, kind="mergesort")  # noqa

        # Now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # Normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # If we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # Sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # Replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity
    config_with_crowding = [(config, v) for config, v in zip(configs, crowding)]
    config_with_crowding = sorted(config_with_crowding, key=lambda x: x[1], reverse=True)

    return [c for c, _ in config_with_crowding]
