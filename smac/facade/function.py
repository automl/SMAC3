from __future__ import annotations

from typing import Any, Callable, Iterable, List, Mapping, Tuple, Union

import logging

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import Configuration, ConfigurationSpace
from smac.facade.hyperparameter import HyperparameterFacade
from smac.runhistory.runhistory import RunKey
from smac.scenario import Scenario

__author__ = "Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


def fmin_smac(
    target_algorithm: Callable,
    x0: List[float],
    bounds: List[Iterable[float]],
    n_runs: int = 20,
    seed: int = -1,
    scenario_args: Mapping[str, Any] | None = None,
) -> Tuple[Configuration, Union[np.ndarray, float], HyperparameterFacade]:
    """
    Minimize a target_algorithmtion target_algorithm using the HyperparameterFacade facade

    The HyperparameterFacade is a version of SMAC with a random forest as a surrogate
    model. This target_algorithmtion is a convenience wrapper for the HyperparameterFacade class.

    Parameters
    ----------
    target_algorithm : Callable
        Function to minimize.
    x0 : List[float]
        Initial guess/default configuration.
    bounds : List[Iterable[float]]
        ``(min, max)`` pairs for each element in ``x``, defining the bound on
        that parameters.
    n_runs : int
        Maximum number of target_algorithmtion evaluations.
    seed : int = -1
        Seed to initialize random generators. If <0, use random seed.

    Returns
    -------
    x : list
        Estimated position of the minimum.
    f : Union[np.ndarray, float]
        Value of `target_algorithm` at the minimum. Depending on the scenario_args, it could be a scalar value
        (for single objective problems) or a np.ndarray (for multi objective problems).
    s : :class:`smac.facade.hyperparameter.HyperparameterFacade`
        SMAC objects which enables the user to get e.g., the trajectory and runhistory.
    """
    # Create configuration space.
    configspace = ConfigurationSpace()

    # Template for hyperparameter name, adjust zero padding.
    tmplt = "x{0:0" + str(len(str(len(bounds)))) + "d}"

    # Create hyperparameters and add to configuration space.
    for idx, (lower_bound, upper_bound) in enumerate(bounds):
        parameter = UniformFloatHyperparameter(
            name=tmplt.format(idx + 1),
            lower=lower_bound,
            upper=upper_bound,
            default_value=x0[idx],
        )
        configspace.add_hyperparameter(parameter)

    # Create scenario.
    scenario = Scenario(
        configspace=configspace,
        n_runs=n_runs,
        seed=seed,
    )

    smac = HyperparameterFacade(
        scenario=scenario,
        target_algorithm=target_algorithm,
    )

    incumbent = smac.optimize()
    config_id = smac.optimizer.runhistory.config_ids[incumbent]
    run_key = RunKey(config_id, None, 0)
    incumbent_performance = smac.optimizer.runhistory.data[run_key]
    incumbent = np.array([incumbent[tmplt.format(idx + 1)] for idx in range(len(bounds))], dtype=float)

    return incumbent, incumbent_performance.cost, smac
