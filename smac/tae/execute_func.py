import logging

from smac.tae.execute_ta_run import StatusType, ExecuteTARun

import pynisher

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class ExecuteTAFunc(ExecuteTARun):

    """Evaluate function for given configuration and resource limit.

    Parameters
    ----------
    ta : callable
        Function (target algorithm) to be optimized. Needs to accept at least a
        configuration and a seed. Can return a float (the loss) or a tuple
        (the loss and additional run information in a dictionary).
    stats : smac.stats.stats.Stats
        Stats object to collect statistics about runtime etc.
    run_obj: str
        Run objective (runtime or quality)
    par_factor: int
        Penalized average runtime factor. Only used when `run_obj='runtime'`
    """

    def __init__(self, ta, stats, run_obj="runtime", par_factor=1):
        super().__init__(ta, stats, run_obj, par_factor)
        self._supports_memory_limit = True

    def run(self, config, instance=None,
            cutoff=None,
            memory_limit=None,
            seed=12345,
            instance_specific="0"
            ):
        """
            runs target algorithm <self.ta> with configuration <config> on
            instance <instance> with instance specifics <specifics>
            for at most <cutoff> seconds and random seed <seed>

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

        if instance:
            rval = obj(config, instance, seed)
        else:
            rval = obj(config, seed)

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
