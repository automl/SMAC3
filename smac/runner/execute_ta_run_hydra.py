from typing import Any, Dict, Mapping, Optional, Tuple, Type

from smac.runner import StatusType
from smac.runner.algorithm_executer import AlgorithmExecuter, ExecuteTAFuncArray
from smac.runner.execute_ta_run_aclib import ExecuteTARunAClib
from smac.runner.execute_ta_run_old import ExecuteTARunOld
from smac.runner.serial_runner import SerialRunner
from smac.configspace import Configuration

__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"


class ExecuteTARunHydra(SerialRunner):
    """Returns min(cost, cost_portfolio)

    Parameters
    ----------
    cost_oracle: Mapping[str,float]
        cost of oracle per instance
    tae: Type[SerialRunner]
        target algorithm evaluator
    """

    def __init__(
        self,
        cost_oracle: Mapping[str, float],
        tae: Type[SerialRunner] = ExecuteTARunOld,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cost_oracle = cost_oracle
        if tae is ExecuteTARunAClib:
            self.runner = ExecuteTARunAClib(**kwargs)  # type: SerialRunner
        elif tae is ExecuteTARunOld:
            self.runner = ExecuteTARunOld(**kwargs)
        elif tae is AlgorithmExecuter:
            self.runner = AlgorithmExecuter(**kwargs)
        elif tae is ExecuteTAFuncArray:
            self.runner = ExecuteTAFuncArray(**kwargs)
        else:
            raise Exception("TAE not supported")

    def run(
        self,
        config: Configuration,
        instance: str,
        algorithm_walltime_limit: Optional[float] = None,
        seed: int = 12345,
        budget: Optional[float] = None,
        instance_specific: str = "0",
    ) -> Tuple[StatusType, float, float, Dict]:
        """See ~smac.tae.execute_ta_run.ExecuteTARunOld for docstring."""
        if algorithm_walltime_limit is None:
            raise ValueError("algorithm_walltime_limit of type None is not supported")

        status, cost, runtime, additional_info = self.runner.run(
            config=config,
            instance=instance,
            algorithm_walltime_limit=algorithm_walltime_limit,
            seed=seed,
            budget=budget,
            instance_specific=instance_specific,
        )
        if instance in self.cost_oracle:
            oracle_perf = self.cost_oracle[instance]
            if self.run_obj == "runtime":
                self.logger.debug("Portfolio perf: %f vs %f = %f", oracle_perf, runtime, min(oracle_perf, runtime))
                runtime = min(oracle_perf, runtime)
                cost = runtime
            else:
                self.logger.debug("Portfolio perf: %f vs %f = %f", oracle_perf, cost, min(oracle_perf, cost))
                cost = min(oracle_perf, cost)
            if oracle_perf < algorithm_walltime_limit and status is StatusType.TIMEOUT:
                status = StatusType.SUCCESS
        else:
            self.logger.error("Oracle performance missing --- should not happen")

        return status, cost, runtime, additional_info
