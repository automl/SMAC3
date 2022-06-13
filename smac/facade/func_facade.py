from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import logging

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter  # type: ignore

from smac.configspace import Configuration, ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory import RunKey
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncArray

__author__ = "Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


def fmin_smac(
    func: Callable,
    x0: List[float],
    bounds: List[Iterable[float]],
    maxfun: int = -1,
    rng: Optional[Union[np.random.RandomState, int]] = None,
    scenario_args: Optional[Mapping[str, Any]] = None,
    tae_runner_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Configuration, Union[np.ndarray, float], SMAC4HPO]:
    """Minimize a function func using the SMAC4HPO facade (i.e., a modified version of SMAC). This
    function is a convenience wrapper for the SMAC4HPO class.

    Parameters
    ----------
    func : Callable
        Function to minimize.
    x0 : List[float]
        Initial guess/default configuration.
    bounds : List[List[float]]
        ``(min, max)`` pairs for each element in ``x``, defining the bound on
        that parameters.
    maxfun : int, optional
        Maximum number of function evaluations.
    rng : np.random.RandomState, optional
            Random number generator used by SMAC.
    scenario_args: Mapping[str,Any]
        Arguments passed to the scenario
        See smac.scenario.scenario.Scenario
    **kwargs:
        Arguments passed to the optimizer class
        See ~smac.facade.smac_facade.SMAC

    Returns
    -------
    x : list
        Estimated position of the minimum.
    f : Union[np.ndarray, float]
        Value of `func` at the minimum. Depending on the scenario_args, it could be a scalar value
        (for single objective problems) or a np.ndarray (for multi objective problems).
    s : :class:`smac.facade.smac_hpo_facade.SMAC4HPO`
        SMAC objects which enables the user to get
        e.g., the trajectory and runhistory.
    """
    # create configuration space
    cs = ConfigurationSpace()

    # Adjust zero padding
    tmplt = "x{0:0" + str(len(str(len(bounds)))) + "d}"

    for idx, (lower_bound, upper_bound) in enumerate(bounds):
        parameter = UniformFloatHyperparameter(
            name=tmplt.format(idx + 1),
            lower=lower_bound,
            upper=upper_bound,
            default_value=x0[idx],
        )
        cs.add_hyperparameter(parameter)

    # create scenario
    scenario_dict = {
        "run_obj": "quality",
        "cs": cs,
        "deterministic": True,
        "initial_incumbent": "DEFAULT",
    }

    if scenario_args is not None:
        scenario_dict.update(scenario_args)

    if maxfun > 0:
        scenario_dict["runcount_limit"] = maxfun
    scenario = Scenario(scenario_dict)

    # Handle optional tae  arguments
    if tae_runner_kwargs is not None:
        if "ta" not in tae_runner_kwargs:
            tae_runner_kwargs.update({"ta": func})
    else:
        tae_runner_kwargs = {"ta": func}

    smac = SMAC4HPO(
        scenario=scenario,
        tae_runner=ExecuteTAFuncArray,
        tae_runner_kwargs=tae_runner_kwargs,
        rng=rng,
        **kwargs,
    )

    smac.logger = logging.getLogger(smac.__module__ + "." + smac.__class__.__name__)
    incumbent = smac.optimize()
    config_id = smac.solver.runhistory.config_ids[incumbent]
    run_key = RunKey(config_id, None, 0)
    incumbent_performance = smac.solver.runhistory.data[run_key]
    incumbent = np.array([incumbent[tmplt.format(idx + 1)] for idx in range(len(bounds))], dtype=float)

    return incumbent, incumbent_performance.cost, smac
