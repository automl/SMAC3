import numpy as np


MAXINT = 2 ** 31 - 1


"""Define overall objectives.

Overall objectives are functions or callables that calculate the overall
objective of a configuration on the instances and seeds it already ran on."""


def _runtime(config, run_history, instance_seed_pairs=None):
    """Return array of all runtimes for the given config for further calculations.

    Parameters
    ----------
    config : Configuration
        configuration to calculate objective for
    run_history : RunHistory
        RunHistory object from which the objective value is computed.
    instance_seed_pairs : list, optional (default=None)
        list of tuples of instance-seeds pairs. If None, the run_history is
        queried for all runs of the given configuration.

    Returns
    ----------
    list
    """
    try:
        id_ = run_history.config_ids[config]
    except KeyError:  # challenger was not running so far
        return []

    if instance_seed_pairs is None:
        instance_seed_pairs = run_history.get_runs_for_config(config)

    runtimes = []
    for i, r in instance_seed_pairs:
        k = run_history.RunKey(id_, i, r)
        runtimes.append(run_history.data[k].time)
    return runtimes


def total_runtime(config, run_history, instance_seed_pairs=None):
    """Return the total cost of a configuration.

    This is the sum of costs of all instance-seed pairs.

    Parameters
    ----------
    config : Configuration
        configuration to calculate objective for
    run_history : RunHistory
        RunHistory object from which the objective value is computed.
    instance_seed_pairs : list, optional (default=None)
        list of tuples of instance-seeds pairs. If None, the run_history is
        queried for all runs of the given configuration.

    Returns
    ----------
    float
    """
    return np.sum(_runtime(config, run_history, instance_seed_pairs))


def _cost(config, run_history, instance_seed_pairs=None):
    """Return array of all costs for the given config for further calculations.

    Parameters
    ----------
    config : Configuration
        configuration to calculate objective for
    run_history : RunHistory
        RunHistory object from which the objective value is computed.
    instance_seed_pairs : list, optional (default=None)
        list of tuples of instance-seeds pairs. If None, the run_history is
        queried for all runs of the given configuration.

    Returns
    ----------
    list
    """
    try:
        id_ = run_history.config_ids[config]
    except KeyError:  # challenger was not running so far
        return []
    
    if instance_seed_pairs is None:
        instance_seed_pairs = run_history.get_runs_for_config(config)

    costs = []
    for i, r in instance_seed_pairs:
        k = run_history.RunKey(id_, i, r)
        costs.append(run_history.data[k].cost)
    return costs


def average_cost(config, run_history, instance_seed_pairs=None):
    """Return the average cost of a configuration.

    This is the mean of costs of all instance-seed pairs.

    Parameters
    ----------
    config : Configuration
        configuration to calculate objective for
    run_history : RunHistory
        RunHistory object from which the objective value is computed.
    instance_seed_pairs : list, optional (default=None)
        list of tuples of instance-seeds pairs. If None, the run_history is
        queried for all runs of the given configuration.

    Returns
    ----------
    float
    """
    return np.mean(_cost(config, run_history, instance_seed_pairs))

def sum_cost(config, run_history, instance_seed_pairs=None):
    """Return the sum of costs of a configuration.

    This is the sum of costs of all instance-seed pairs.

    Parameters
    ----------
    config : Configuration
        configuration to calculate objective for
    run_history : RunHistory
        RunHistory object from which the objective value is computed.
    instance_seed_pairs : list, optional (default=None)
        list of tuples of instance-seeds pairs. If None, the run_history is
        queried for all runs of the given configuration.

    Returns
    ----------
    float
    """
    return np.sum(_cost(config, run_history, instance_seed_pairs))