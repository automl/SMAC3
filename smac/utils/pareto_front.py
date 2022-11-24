from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration

from smac.runhistory import RunHistory
from smac.runhistory.dataclasses import InstanceSeedBudgetKey


def calculate_pareto_front(
    runhistory: RunHistory,
    configs: list[Configuration],
    instances: list[InstanceSeedBudgetKey],
) -> list[Configuration]:
    """Compares the passed configurations and returns only the ones one the pareto front."""
    # Now we get the costs for the trials of the config
    average_costs = []
        
    for config in configs:
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each
        # configuration
        # However, we only want to consider the config trials
        # Average cost is a list of floats (one for each objective)
        average_cost = runhistory.average_cost(config, instances, normalize=False)
        average_costs += [average_cost]

    # Let's work with a numpy array for efficiency
    costs = np.vstack(average_costs)

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
        
    