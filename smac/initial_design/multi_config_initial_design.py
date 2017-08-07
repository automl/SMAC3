import typing
import numpy as np

from ConfigSpace.configuration_space import Configuration

from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.single_config_initial_design import SingleConfigInitialDesign
from smac.intensification.intensification import Intensifier

from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class MultiConfigInitialDesign(InitialDesign):
    """ Base class for initial design strategies that evaluates multiple
    configurations

    Attributes
    ----------
    configs : typing.List[Configuration]
        List of configurations to be evaluated
    intensifier
    runhistory
    aggregate_func
    """

    def __init__(self,
                 tae_runner: ExecuteTARun,
                 scenario: Scenario,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 runhistory: RunHistory,
                 rng: np.random.RandomState,
                 configs: typing.List[Configuration],
                 intensifier: Intensifier,
                 aggregate_func: typing.Callable
                 ):
        """Constructor

        Parameters
        ---------
        tae_runner: ExecuteTARun
            Target algorithm execution object.
        scenario: Scenario
            Scenario with all meta information (including configuration space).
        stats: Stats
            Statistics of experiments; needed in case initial design already
            exhausts the budget.
        traj_logger: TrajLogger
            Trajectory logging to add new incumbents found by the initial
            design.
        runhistory: RunHistory
            Runhistory with all target algorithm runs.
        rng: np.random.RandomState
            Random state
        configs: typing.List[Configuration]
            List of initial configurations.
        intensifier: Intensifier
            Intensification object to issue a racing to decide the current
            incumbent.
        aggregate_func: typing:Callable
            Function to aggregate performance of a configuration across
            instances.
        """
        super().__init__(tae_runner=tae_runner,
                         scenario=scenario,
                         stats=stats,
                         traj_logger=traj_logger,
                         rng=rng)

        self.configs = configs
        self.intensifier = intensifier
        self.runhistory = runhistory
        self.aggregate_func = aggregate_func

    def run(self) -> Configuration:
        """Run the initial design.

        Returns
        -------
        incumbent: Configuration
            Initial incumbent configuration
        """
        configs = self.configs

        self.traj_logger.add_entry(train_perf=2**31,
                                   incumbent_id=1,
                                   incumbent=configs[0])

        if len(set(configs)) > 1:
            # intensify will skip all challenger that are identical with the incumbent;
            # if <configs> has only identical configurations,
            # intensifiy will not do any configuration runs
            # (also not on the incumbent)
            # therefore, at least two different configurations have to be in <configs>
            inc, inc_perf = self.intensifier.intensify(challengers=set(configs[1:]),
                                                       incumbent=configs[0],
                                                       run_history=self.runhistory,
                                                       aggregate_func=self.aggregate_func)

        else:
            self.logger.debug("All initial challengers are identical")
            scid = SingleConfigInitialDesign(tae_runner=self.tae_runner,
                                             scenario=self.scenario,
                                             stats=self.stats,
                                             traj_logger=self.traj_logger,
                                             rng=self.rng)

            def get_config():
                return configs[0]
            scid._select_configuration = get_config
            scid.run()
            inc = configs[0]

        return inc
