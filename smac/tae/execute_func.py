import inspect
import math
import time
import traceback
import typing

import numpy as np
import pynisher

from smac.configspace import Configuration
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.utils.constants import MAXINT, MAX_CUTOFF
from smac.tae.serial_runner import SerialRunner
from smac.utils.logging import PickableLoggerAdapter

__author__ = "Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.2"


class AbstractTAFunc(SerialRunner):
    """Baseclass to execute target algorithms which are python functions.

    **Note:*** Do not use directly

    Attributes
    ----------
    memory_limit
    use_pynisher
    """

    def __init__(
        self,
        ta: typing.Callable,
        stats: Stats,
        run_obj: str = "quality",
        memory_limit: typing.Optional[int] = None,
        par_factor: int = 1,
        cost_for_crash: float = float(MAXINT),
        abort_on_first_run_crash: bool = False,
        use_pynisher: bool = True,
    ):

        super().__init__(ta=ta, stats=stats,
                         run_obj=run_obj, par_factor=par_factor,
                         cost_for_crash=cost_for_crash,
                         abort_on_first_run_crash=abort_on_first_run_crash,
                         )
        """
        Abstract class for having a function as target algorithm

        Parameters
        ----------
        ta : callable
            Function (target algorithm) to be optimized.
        stats: Stats()
             stats object to collect statistics about runtime and so on
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
        self.ta = ta
        self.stats = stats
        self.run_obj = run_obj

        self.par_factor = par_factor
        self.cost_for_crash = cost_for_crash
        self.abort_on_first_run_crash = abort_on_first_run_crash

        signature = inspect.signature(ta).parameters
        self._accepts_seed = 'seed' in signature.keys()
        self._accepts_instance = 'instance' in signature.keys()
        self._accepts_budget = 'budget' in signature.keys()
        if not callable(ta):
            raise TypeError('Argument `ta` must be a callable, but is %s' % type(ta))
        self._ta = typing.cast(typing.Callable, ta)

        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))
        self.memory_limit = memory_limit

        self.use_pynisher = use_pynisher

        self.logger = PickableLoggerAdapter(
            self.__module__ + '.' + self.__class__.__name__)

    def run(self, config: Configuration,
            instance: typing.Optional[str] = None,
            cutoff: typing.Optional[float] = None,
            seed: int = 12345,
            budget: typing.Optional[float] = None,
            instance_specific: str = "0") -> typing.Tuple[StatusType, float, float, typing.Dict]:
        """Runs target algorithm <self._ta> with configuration <config> for at
        most <cutoff> seconds, allowing it to use at most <memory_limit> RAM.

        Whether the target algorithm is called with the <instance> and
        <seed> depends on the subclass implementing the actual call to
        the target algorithm.

        Parameters
        ----------
            config : Configuration, dictionary (or similar)
                Dictionary param -> value
            instance : str, optional
                Problem instance
            cutoff : float, optional
                Wallclock time limit of the target algorithm. If no value is
                provided no limit will be enforced. It is casted to integer internally.
            seed : int
                Random seed
            budget : float, optional
                A positive, real-valued number representing an arbitrary limit to the target algorithm
                Handled by the target algorithm internally
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

        obj_kwargs = {}  # type: typing.Dict[str, typing.Union[int, str, float, None]]
        if self._accepts_seed:
            obj_kwargs['seed'] = seed
        if self._accepts_instance:
            obj_kwargs['instance'] = instance
        if self._accepts_budget:
            obj_kwargs['budget'] = budget

        if self.use_pynisher:
            # walltime for pynisher has to be a rounded up integer
            if cutoff is not None:
                cutoff = int(math.ceil(cutoff))
                if cutoff > MAX_CUTOFF:
                    raise ValueError("%d is outside the legal range of [0, 65535] "
                                     "for cutoff (when using pynisher, due to OS limitations)" % cutoff)

            arguments = {
                'logger': self.logger,
                'wall_time_in_s': cutoff,
                'mem_in_mb': self.memory_limit
            }

            # call ta
            try:
                obj = pynisher.enforce_limits(**arguments)(self._ta)
                rval = self._call_ta(obj, config, obj_kwargs)
            except Exception as e:
                exception_traceback = traceback.format_exc()
                error_message = repr(e)
                additional_info = {
                    'traceback': exception_traceback,
                    'error': error_message
                }
                return StatusType.CRASHED, self.cost_for_crash, 0.0, additional_info

            if isinstance(rval, tuple):
                result = rval[0]
                additional_run_info = rval[1]
            else:
                result = rval
                additional_run_info = {}

            # get status, cost, time
            if obj.exit_status is pynisher.TimeoutException:
                status = StatusType.TIMEOUT
                cost = self.cost_for_crash
            elif obj.exit_status is pynisher.MemorylimitException:
                status = StatusType.MEMOUT
                cost = self.cost_for_crash
            elif obj.exit_status == 0 and result is not None:
                status = StatusType.SUCCESS
                cost = result
            else:
                status = StatusType.CRASHED
                cost = self.cost_for_crash

            runtime = float(obj.wall_clock_time)
        else:
            start_time = time.time()
            # call ta
            try:
                rval = self._call_ta(self._ta, config, obj_kwargs)
                if isinstance(rval, tuple):
                    result = rval[0]
                    additional_run_info = rval[1]
                else:
                    result = rval
                    additional_run_info = {}
                status = StatusType.SUCCESS
                cost = result
            except Exception as e:
                self.logger.exception(e)
                cost, result = self.cost_for_crash, self.cost_for_crash
                status = StatusType.CRASHED
                additional_run_info = {}

            runtime = time.time() - start_time

        if status == StatusType.SUCCESS and not isinstance(result, (int, float)):
            status = StatusType.CRASHED
            cost = self.cost_for_crash

        return status, cost, runtime, additional_run_info

    def _call_ta(
        self,
        obj: typing.Callable,
        config: Configuration,
        obj_kwargs: typing.Dict[str, typing.Union[int, str, float, None]],
    ) -> typing.Union[float, typing.Tuple[float, typing.Dict]]:
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
    memory_limit : int, optional
        Memory limit (in MB) that will be applied to the target algorithm.
    par_factor : int, optional
        Penalized average runtime factor. Only used when `run_obj='runtime'`
    use_pynisher: bool, optional
        use pynisher to limit resources;
    """

    def _call_ta(
        self,
        obj: typing.Callable,
        config: Configuration,
        obj_kwargs: typing.Dict[str, typing.Union[int, str, float, None]],
    ) -> typing.Union[float, typing.Tuple[float, typing.Dict]]:

        return obj(config, **obj_kwargs)


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
    memory_limit : int, optional
        Memory limit (in MB) that will be applied to the target algorithm.
    par_factor: int, optional
        Penalized average runtime factor. Only used when `run_obj='runtime'`
    """

    def _call_ta(
        self,
        obj: typing.Callable,
        config: Configuration,
        obj_kwargs: typing.Dict[str, typing.Union[int, str, float, None]],
    ) -> typing.Union[float, typing.Tuple[float, typing.Dict]]:

        x = np.array([val for _, val in sorted(config.get_dictionary().items())],
                     dtype=np.float)
        return obj(x, **obj_kwargs)
