import logging
import inspect
import math
import time

import numpy as np
import pynisher

from smac.tae.execute_ta_run import StatusType, ExecuteTARun
from smac.utils.constants import MAXINT

__author__ = "Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.2"


class AbstractTAFunc(ExecuteTARun):
    """Baseclass to execute target algorithms which are python functions.

    **Note:*** Do not use directly

    Attributes
    ----------
    memory_limit
    use_pynisher
    """

    def __init__(self, ta, stats=None, runhistory=None, run_obj:str="quality",
                 memory_limit:int=None, par_factor:int=1,
                 cost_for_crash:float=float(MAXINT),
                 abort_on_first_run_crash: bool=False,
                 use_pynisher:bool=True):

        super().__init__(ta=ta, stats=stats, runhistory=runhistory,
                         run_obj=run_obj, par_factor=par_factor,
                         cost_for_crash=cost_for_crash)
        """
        Abstract class for having a function as target algorithm

        Parameters
        ----------
        ta : callable
            Function (target algorithm) to be optimized.
        stats: Stats()
             stats object to collect statistics about runtime and so on
        runhistory: RunHistory
            runhistory to keep track of all runs; only used if set
        run_obj: str
            run objective of SMAC
        memory_limit : int, optional
            Memory limit (in MB) that will be applied to the target algorithm.
        par_factor: int
            penalization factor
        cost_for_crash : float
            cost that is used in case of crashed runs (including runs
            that returned NaN or inf)
        use_pynisher: bool
            use pynisher to limit resources; 
            if disabled
              * TA func can use as many resources 
              as it wants (time and memory) --- use with caution
              * all runs will be returned as SUCCESS if returned value is not None
            
        """

        signature = inspect.signature(ta).parameters
        self._accepts_seed = len(signature) > 1
        self._accepts_instance = len(signature) > 2

        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        self.memory_limit = memory_limit
        
        self.use_pynisher = use_pynisher

    def run(self, config, instance=None,
            cutoff=None,
            seed=12345,
            instance_specific="0"):
        """Runs target algorithm <self.ta> with configuration <config> for at
        most <cutoff> seconds, allowing it to use at most <memory_limit> RAM.

        Whether the target algorithm is called with the <instance> and
        <seed> depends on the subclass implementing the actual call to
        the target algorithm.

        Parameters
        ----------
            config : dictionary (or similar)
                Dictionary param -> value
            instance : str
                Problem instance
            cutoff : int, optional
                Wallclock time limit of the target algorithm. If no value is
                provided no limit will be enforced.
            memory_limit : int, optional
                Memory limit in MB enforced on the target algorithm If no
                value is provided no limit will be enforced.
            seed : int
                Random seed
            instance_specific: str
                Instance specific information (e.g., domain file or solution)
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
        # walltime for pynisher has to be a rounded up integer
        if cutoff is not None:
            cutoff = int(math.ceil(cutoff))
            if cutoff > 65535:
                raise ValueError("%d is outside the legal range of [0, 65535] "
                                 "for cutoff (when using pynisher, due to OS limitations)" % cutoff)

        arguments = {'logger': logging.getLogger("pynisher"),
                     'wall_time_in_s': cutoff,
                     'mem_in_mb': self.memory_limit}

        obj_kwargs = {}
        if self._accepts_seed:
            obj_kwargs['seed'] = seed
        if self._accepts_instance:
            obj_kwargs['instance'] = instance

        if self.use_pynisher:

            obj = pynisher.enforce_limits(**arguments)(self.ta)
    
            rval = self._call_ta(obj, config, **obj_kwargs)
    
            if isinstance(rval, tuple):
                result = rval[0]
                additional_run_info = rval[1]
            else:
                result = rval
                additional_run_info = {}
    
            if obj.exit_status is pynisher.TimeoutException:
                status = StatusType.TIMEOUT
                cost = self.crash_cost
            elif obj.exit_status is pynisher.MemorylimitException:
                status = StatusType.MEMOUT
                cost = self.crash_cost
            elif obj.exit_status == 0 and result is not None:
                status = StatusType.SUCCESS
                cost = result
            else:
                status = StatusType.CRASHED
                cost = self.crash_cost
        
            runtime = float(obj.wall_clock_time)
        else:
            start_time = time.time()
            result = self.ta(config, **obj_kwargs)
            
            if result is not None:
                status = StatusType.SUCCESS
                cost = result
            else:
                status = StatusType.CRASHED
                cost = self.crash_cost
            
            runtime = time.time() - start_time
            additional_run_info = {}

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
    run_obj : str, optional
        Run objective (runtime or quality)
    runhistory : RunHistory, optional
        runhistory to keep track of all runs; only used if set
    memory_limit : int, optional
        Memory limit (in MB) that will be applied to the target algorithm.
    par_factor : int, optional
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
    memory_limit : int, optional
        Memory limit (in MB) that will be applied to the target algorithm.
    par_factor: int, optional
        Penalized average runtime factor. Only used when `run_obj='runtime'`
    """

    def _call_ta(self, obj, config, **kwargs):

        x = np.array([val for _, val in sorted(config.get_dictionary().items())],
                     dtype=np.float)
        return obj(x, **kwargs)
