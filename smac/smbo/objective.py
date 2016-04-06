import numpy as np


MAXINT = 2 ** 31 - 1


"""Define overall objectives.

Overall objectives are functions or callables that calculate the overall
objective of a configuration on the instances and seeds it already ran on."""


def _cost(config, inst_seeds, run_history):
    """Return array of all costs for the given config for further calculations.

    Parameters
    ----------
    config : Configuration
        configuration to calculate objective for
    inst_seeds : list
        list of tuples of instance-seeds pairs
    run_history : RunHistory
        RunHistory object from which the objective value is computed.

    Returns
    ----------
    list
    """
    try:
        id_ = run_history.config_ids[config.__repr__()]
    except KeyError:  # challenger was not running so far
        return []
    costs = []
    for i, r in inst_seeds:
        k = run_history.RunKey(id_, i, r)
        costs.append(run_history.data[k].cost)
    return costs


def total_cost(config, inst_seeds, run_history):
    """Return the total cost of a configuration.

    This is the sum of costs of all instance-seed pairs.

    Parameters
    ----------
    config : Configuration
        configuration to calculate objective for
    inst_seeds : list
        list of tuples of instance-seeds pairs
    run_history : RunHistory
        RunHistory object from which the objective value is computed.

    Returns
    ----------
    float
    """
    return np.sum(_cost(config, inst_seeds, run_history))


def average_cost(config, inst_seeds, run_history):
    """Return the average cost of a configuration.

    This is the mean of costs of all instance-seed pairs.

    Parameters
    ----------
    config : Configuration
        configuration to calculate objective for
    inst_seeds : list
        list of tuples of instance-seeds pairs
    run_history : RunHistory
        RunHistory object from which the objective value is computed.

    Returns
    ----------
    float
    """
    return np.mean(_cost(config, inst_seeds, run_history))