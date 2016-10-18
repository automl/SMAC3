import logging

import numpy as np

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace
from smac.smbo.objective import average_cost
from smac.runhistory.runhistory import RunKey
from smac.tae.execute_func import ExecuteTAFuncArray

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

__author__ = "Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


def fmin_smac(func: callable,
              x0: list,
              bounds: list,
              maxfun: int=-1,
              maxtime: int=-1,
              rng: np.random.RandomState=None):
    """Minimize a function func using the SMAC algorithm.

    This method is a convenience wrapper for the SMAC class.

    Parameters
    ----------
    func : callable f(x)
        Function to minimize.
    x0 : list
        Initial guess/default configuration.
    bounds : list
        ``(min, max)`` pairs for each element in ``x``, defining the bound on
        that parameters.
    maxtime : int, optional
        Maximum runtime in seconds.
    maxfun : int, optional
        Maximum number of function evaluations.
    rng : np.random.RandomState, optional
            Random number generator used by SMAC.

    Returns
    -------
    x : list
        Estimated position of the minimum.
    f : float
        Value of `func` at the minimum.
    r : :class:`smac.runhistory.runhistory.RunHistory`
        Information on the SMAC run.
    """

    aggregate_func = average_cost

    # create configuration space
    cs = ConfigurationSpace()
    for idx, (lower_bound, upper_bound) in enumerate(bounds):
        parameter = UniformFloatHyperparameter(name="x%d" % (idx + 1),
                                               lower=lower_bound,
                                               upper=upper_bound,
                                               default=x0[idx])
        cs.add_hyperparameter(parameter)

    # Create target algorithm runner
    ta = ExecuteTAFuncArray(ta=func)

    # create scenario
    scenario_dict = {"run_obj": "quality",  # we optimize quality
                     "cs": cs,  # configuration space
                     "deterministic": "true",
                     }
    if maxfun > 0:
        scenario_dict["runcount_limit"] = maxfun
    if maxtime > 0:
        scenario_dict["wallclock_limit"] = maxtime
    scenario = Scenario(scenario_dict)

    smac = SMAC(scenario=scenario, tae_runner=ta, rng=rng)
    smac.logger = logging.getLogger("fmin_smac")
    incumbent = smac.optimize()

    config_id = smac.solver.runhistory.config_ids[incumbent]
    run_key = RunKey(config_id, None, 0)
    incumbent_performance = smac.solver.runhistory.data[run_key]
    incumbent = np.array([incumbent['x%d' % (idx + 1)]
                          for idx in range(len(bounds))], dtype=np.float)
    return incumbent, incumbent_performance.cost, \
           smac.solver.runhistory
