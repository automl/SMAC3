import typing

from smac.configspace import Configuration
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.tae.execute_ta_run_aclib import ExecuteTARun
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.tae.execute_func import ExecuteTAFuncArray
from smac.tae.execute_ta_run_old import StatusType

__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"


class ExecuteTARunHydra(ExecuteTARun):

    """Returns min(cost, cost_portfolio)
    """

    def __init__(
        self,
        cost_oracle: typing.Mapping[str, float],
        tae: typing.Type[ExecuteTARun] = ExecuteTARunOld,
        **kwargs: typing.Any
    ) -> None:
        '''
            Constructor

            Arguments
            ---------
            cost_oracle: typing.Mapping[str,float]
                cost of oracle per instance
        '''

        super().__init__(ta=kwargs["ta"],
                         stats=kwargs["stats"], run_obj=kwargs["run_obj"],
                         runhistory=kwargs["runhistory"],
                         par_factor=kwargs["par_factor"],
                         cost_for_crash=kwargs["cost_for_crash"],
                         abort_on_first_run_crash=kwargs["abort_on_first_run_crash"])

        self.cost_oracle = cost_oracle
        self.runner = tae(**kwargs)

    def run(self, config: Configuration,
            instance: str,
            cutoff: typing.Optional[float] = None,
            seed: int = 12345,
            budget: typing.Optional[float] = None,
            instance_specific: str = "0") -> typing.Tuple[StatusType, float, float, typing.Dict]:

        """ see ~smac.tae.execute_ta_run.ExecuteTARunOld for docstring
        """

        if cutoff is None:
            raise ValueError('Cutoff of type None is not supported')

        status, cost, runtime, additional_info = self.runner.run(
            config=config,
            instance=instance,
            cutoff=cutoff,
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
            if oracle_perf < cutoff and status is StatusType.TIMEOUT:
                status = StatusType.SUCCESS
        else:
            self.logger.error("Oracle performance missing --- should not happen")

        return status, cost, runtime, additional_info
