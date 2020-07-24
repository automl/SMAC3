import logging
import math
import time
import typing

import numpy as np

from smac.runhistory.runhistory import RunValue, RunInfo
from smac.tae.execute_ta_run import StatusType, ExecuteTARun, TAEAbortException


def execute_ta_run_wrapper(
    tae_runner: ExecuteTARun,
    run_info: RunInfo,
    logger: typing.Optional[logging.Logger] = None
) -> RunValue:
    """Wrapper around ExecuteTARun to run and check the execution of a given config file

    This function relies on the tae runner object, in order to asses whether a run
    was intended to be evaluated with a runtime goal or quality goal.

    Parameters
    ----------
        tae_runner: 'ExecuteTARun'
            An object with a run method to execute a config provided via run_info
        run_info : RunInfo
            Object that contains enough information to execute a configuration run in
            isolation.

    Returns
    -------
        RunValue:
            Contains information about the status/performance of config
    """
    start = time.time()

    if tae_runner.stats.is_budget_exhausted():
        # In case there is a budget exhausted, return par_factor time.
        # Cutoff can be None, and in such case not par factor can be taken
        if run_info.cutoff is not None:
            ehausted_cost = run_info.cutoff * tae_runner.par_factor
        else:
            ehausted_cost = run_info.cutoff
        return RunValue(
            status=StatusType.BUDGETEXHAUSTED,
            cost=ehausted_cost,
            time=0.0,
            additional_info={},
            starttime=start,
            endtime=time.time()
        )

    if run_info.cutoff is None and tae_runner.run_obj == "runtime":
        if logger:
            logger.critical(
                "For scenarios optimizing running time "
                "(run objective), a cutoff time is required, "
                "but not given to this call."
            )
        raise ValueError(
            "For scenarios optimizing running time "
            "(run objective), a cutoff time is required, "
            "but not given to this call."
        )
    cutoff = None
    if run_info.cutoff is not None:
        cutoff = int(math.ceil(run_info.cutoff))

    status, cost, runtime, additional_info = tae_runner.run(
        config=run_info.config,
        instance=run_info.instance,
        cutoff=cutoff,
        seed=run_info.seed,
        budget=run_info.budget,
        instance_specific=run_info.instance_specific
    )
    end = time.time()

    if run_info.budget == 0 and status == StatusType.DONOTADVANCE:
        raise ValueError(
            "Cannot handle DONOTADVANCE state when using intensify or SH/HB on "
            "instances."
        )

    # Catch NaN or inf.
    if (
        tae_runner.run_obj == 'runtime' and not np.isfinite(runtime)
        or tae_runner.run_obj == 'quality' and not np.isfinite(cost)
    ):
        if logger:
            logger.warning("Target Algorithm returned NaN or inf as {}. "
                           "Algorithm run is treated as CRASHED, cost "
                           "is set to {} for quality scenarios. "
                           "(Change value through \"cost_for_crash\""
                           "-option.)".format(tae_runner.run_obj,
                                              tae_runner.cost_for_crash))
        status = StatusType.CRASHED

    if status == StatusType.ABORT:
        raise TAEAbortException("Target algorithm status ABORT - SMAC will "
                                "exit. The last incumbent can be found "
                                "in the trajectory-file.")

    if tae_runner.run_obj == "runtime":
        # The following line pleases mypy - we already check for cutoff not being none above,    prior to calling
        # run. However, mypy assumes that the data type of cutoff is still Optional[int]
        assert cutoff is not None
        if runtime > tae_runner.par_factor * cutoff:
            if logger:
                logger.warning("Returned running time is larger "
                               "than {0} times the passed cutoff time. "
                               "Clamping to {0} x cutoff.".format(tae_runner.par_factor))
            runtime = cutoff * tae_runner.par_factor
            status = StatusType.TIMEOUT
        if status == StatusType.SUCCESS:
            cost = runtime
        else:
            cost = cutoff * tae_runner.par_factor
        if status == StatusType.TIMEOUT and run_info.capped:
            status = StatusType.CAPPED
            # In case of a capped run, we expect the
            # status to be:
            # status, cost, dur, res = StatusType.CAPPED, float(MAXINT), run_info.cutoff, {}
            # This is set by the TA callable
    else:
        if status == StatusType.CRASHED:
            cost = tae_runner.cost_for_crash

    if tae_runner.run_obj == "runtime":
        # The following line pleases mypy - we already check for cutoff not being none above,    prior to calling
        # run. However, mypy assumes that the data type of cutoff is still Optional[int]
        assert cutoff is not None
        if runtime > tae_runner.par_factor * cutoff:
            tae_runner.logger.warning(
                "Returned running time is larger "
                "than {0} times the passed cutoff time. "
                "Clamping to {0} x cutoff.".format(tae_runner.par_factor))
            runtime = cutoff * tae_runner.par_factor
            status = StatusType.TIMEOUT
        if status == StatusType.SUCCESS:
            cost = runtime
        else:
            cost = cutoff * tae_runner.par_factor
        if status == StatusType.TIMEOUT and run_info.capped:
            status = StatusType.CAPPED
    else:
        if status == StatusType.CRASHED:
            cost = tae_runner.cost_for_crash

    return RunValue(
        status=status,
        cost=cost,
        time=runtime,
        additional_info=additional_info,
        starttime=start,
        endtime=end
    )
