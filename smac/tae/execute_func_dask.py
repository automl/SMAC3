import typing

from smac.stats.stats import Stats
from smac.tae.dask_runner import DaskParallelRunner
from smac.tae.execute_func import ExecuteTAFuncArray
from smac.utils.constants import MAXINT


class ExecuteParallelTAFuncArray(DaskParallelRunner, ExecuteTAFuncArray):
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
        n_workers: int = 1,
    ):

        super().__init__(ta=ta, stats=stats,
                         run_obj=run_obj, par_factor=par_factor,
                         cost_for_crash=cost_for_crash,
                         abort_on_first_run_crash=abort_on_first_run_crash,
                         n_workers=1,)
