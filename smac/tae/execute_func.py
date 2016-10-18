import logging
import inspect

import numpy as np
import pynisher

from smac.tae.execute_ta_run import StatusType, ExecuteTARun


__author__ = "Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.2"



class AbstractTAFunc(ExecuteTARun):
    """Baseclass to execute target algorithms which are python functions.

    DO NOT USE DIRECTLY.
    """

    def __init__(self, ta, stats=None, runhistory=None, run_obj="quality",
                 par_factor=1):

        super().__init__(ta=ta, stats=stats, runhistory=runhistory,
                         run_obj=run_obj, par_factor=par_factor)
        self._supports_memory_limit = True

        signature = inspect.signature(ta).parameters
        self._accepts_seed = len(signature) > 1
        self._accepts_instance = len(signature) > 2

    def run(self, config, instance=None,
            cutoff=None,
            memory_limit=None,
            seed=12345,
            instance_specific="0"):

        """
            runs target algorithm <self.ta> with configuration <config>for at
            most <cutoff> seconds allowing it to use at most <memory_limit>
            RAM.

            Whether the target algorithm is called with the <instance> and
            <seed> depends on the subclass implementing the actual call to
            the target algorithm

            Parameters
            ----------
                config : dictionary (or similar)
                    dictionary param -> value
                instance : str
                    problem instance
                cutoff : int, optional
                    Wallclock time limit of the target algorithm. If no value is
                    provided no limit will be enforced.
                memory_limit : int, optional
                    Memory limit in MB enforced on the target algorithm If no
                    value is provided no limit will be enforced.
                seed : int
                    random seed
                instance_specific: str
                    instance specific information (e.g., domain file or solution)
            Returns
            -------
                status: enum of StatusType (int)
                    {SUCCESS, TIMEOUT, CRASHED, ABORT}
                cost: float
                    cost/regret/quality/runtime (float) (None, if not returned by TA)
                runtime: float
                    runtime (None if not returned by TA)
                additional_info: dict
                    all further additional run information
        """

        arguments = {'logger': logging.getLogger("pynisher"),
                     'wall_time_in_s': cutoff,
                     'mem_in_mb': memory_limit}

        obj = pynisher.enforce_limits(**arguments)(self.ta)

        obj_kwargs = {}
        if self._accepts_seed:
            obj_kwargs['seed'] = seed
        if self._accepts_instance:
            obj_kwargs['instance'] = instance

        rval = self._call_ta(obj, config, **obj_kwargs)

        if isinstance(rval, tuple):
            result = rval[0]
            additional_run_info = rval[1]
        else:
            result = rval
            additional_run_info = {}

        if obj.exit_status is pynisher.TimeoutException:
            status = StatusType.TIMEOUT
            cost = 1234567890
        elif obj.exit_status is pynisher.MemorylimitException:
            status = StatusType.MEMOUT
            cost = 1234567890
        elif obj.exit_status == 0 and result is not None:
            status = StatusType.SUCCESS
            cost = result
        else:
            status = StatusType.CRASHED
            cost = 1234567890  # won't be used for the model

        runtime = float(obj.wall_clock_time)

        return status, cost, runtime, additional_run_info

    def _call_ta(self, obj, config, instance, seed):
        raise NotImplementedError()


class ExecuteTAFuncDict(AbstractTAFunc):

    """Evaluate function for given configuration and resource limit.

    Passes the configuration as a dictionary to the target algorithm. The
    target algorithm needs to implement one of the following signatures:

    * ``target_algorithm(config: Configuration) -> Union[float, Tuple[float, Any]]``
    * ``target_algorithm(config: Configuration, seed: int) -> Union[float, Tuple[float, Any]]``
    * ``target_algorithm(config: Configuration, seed: int, instance: str) -> Union[float, Tuple[float, Any]]``

    The target algorithm can either return a float (the loss), or a tuple
    with the first element being a float and the second being additional run
    information.

    ExecuteTAFuncDict will use inspection to figure out the correct call to
    the target algorithm.

    Parameters
    ----------
    ta : callable
        Function (target algorithm) to be optimized.
    stats : smac.stats.stats.Stats, optional
        Stats object to collect statistics about runtime etc.
    run_obj: str, optional
        Run objective (runtime or quality)
    runhistory: RunHistory, optional
        runhistory to keep track of all runs; only used if set
    par_factor: int, optional
        Penalized average runtime factor. Only used when `run_obj='runtime'`
    """

    def _call_ta(self, obj, config, **kwargs):

        return obj(config, **kwargs)


class ExecuteTAFuncArray(AbstractTAFunc):
    """Evaluate function for given configuration and resource limit.

    Passes the configuration as an array-like to the target algorithm. The
    target algorithm needs to implement one of the following signatures:

    * ``target_algorithm(config: np.ndarray) -> Union[float, Tuple[float, Any]]``
    * ``target_algorithm(config: np.ndarray, seed: int) -> Union[float, Tuple[float, Any]]``
    * ``target_algorithm(config: np.ndarray, seed: int, instance: str) -> Union[float, Tuple[float, Any]]``

    The target algorithm can either return a float (the loss), or a tuple
    with the first element being a float and the second being additional run
    information.

    ExecuteTAFuncDict will use inspection to figure out the correct call to
    the target algorithm.

    Parameters
    ----------
    ta : callable
        Function (target algorithm) to be optimized.
    stats : smac.stats.stats.Stats, optional
        Stats object to collect statistics about runtime etc.
    run_obj: str, optional
        Run objective (runtime or quality)
    runhistory: RunHistory, optional
        runhistory to keep track of all runs; only used if set
    par_factor: int, optional
        Penalized average runtime factor. Only used when `run_obj='runtime'`
    """

    def _call_ta(self, obj, config, **kwargs):

        x = np.array([val for _, val in sorted(config.get_dictionary().items())],
                     dtype=np.float)
        return obj(x, **kwargs)
