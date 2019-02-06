import logging
import typing

from ConfigSpace.configuration_space import Configuration
import numpy as np

from smac.intensification.intensification import Intensifier
from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.tae.execute_ta_run import FirstRunCrashedException
from smac.utils import constants

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class InitialDesign:
    """Base class for initial design strategies that evaluates multiple configurations

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
                 intensifier: Intensifier,
                 aggregate_func: typing.Callable,
                 configs: typing.Optional[typing.List[Configuration]]=None,
                 n_configs_x_params: int=10,
                 max_config_fracs: float=0.25,
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
        intensifier: Intensifier
            Intensification object to issue a racing to decide the current
            incumbent.
        aggregate_func: typing:Callable
            Function to aggregate performance of a configuration across
            instances.
        configs: typing.Optional[typing.List[Configuration]]
            List of initial configurations.
        n_configs_x_params: int
            how many configurations will be used at most in the initial design (X*D)
        max_config_fracs: float
            use at most X*budget in the initial design. Not active if a time limit is given.
        """

        self.tae_runner = tae_runner
        self.stats = stats
        self.traj_logger = traj_logger
        self.scenario = scenario
        self.rng = rng
        self.configs = configs
        self.intensifier = intensifier
        self.runhistory = runhistory
        self.aggregate_func = aggregate_func

        self.logger = self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        n_params = len(self.scenario.cs.get_hyperparameters())
        self.init_budget = int(max(1, min(n_configs_x_params * n_params,
                          (max_config_fracs * scenario.ta_run_limit))))
        self.logger.info("Running initial design for %d configurations" %(self.init_budget))

    def select_configurations(self) -> typing.List[Configuration]:

        if self.configs is None:
            return self._select_configurations()
        else:
            return self.configs

    def _select_configurations(self) -> typing.List[Configuration]:
        raise NotImplementedError

    def run(self) -> Configuration:
        """Run the initial design.

        Returns
        -------
        incumbent: Configuration
            Initial incumbent configuration
        """
        configs = self.select_configurations()
        for config in configs:
            if config.origin is None:
                config.origin = 'Initial design'

        # run first design
        # ensures that first design is part of trajectory file
        inc = self._run_first_configuration(configs[0], self.scenario)

        if len(set(configs)) > 1:
            # intensify will skip all challenger that are identical with the incumbent;
            # if <configs> has only identical configurations,
            # intensifiy will not do any configuration runs
            # (also not on the incumbent)
            # therefore, at least two different configurations have to be in <configs>
            inc, _ = self.intensifier.intensify(
                challengers=configs[1:],
                incumbent=configs[0],
                run_history=self.runhistory,
                aggregate_func=self.aggregate_func,
            )

        return inc

    def _run_first_configuration(self, initial_incumbent, scenario):
        """Runs the initial design by calling the target algorithm and adding new entries to the trajectory logger.

        Returns
        -------
        incumbent: Configuration
            Initial incumbent configuration
        """
        if initial_incumbent.origin is None:
            initial_incumbent.origin = 'Initial design'

        # add this incumbent right away to have an entry to time point 0
        self.traj_logger.add_entry(train_perf=2 ** 31,
                                   incumbent_id=1,
                                   incumbent=initial_incumbent)

        rand_inst = self.rng.choice(self.scenario.train_insts)

        if self.scenario.deterministic:
            initial_seed = 0
        else:
            initial_seed = self.rng.randint(0, constants.MAXINT)

        try:
            status, cost, runtime, _ = self.tae_runner.start(
                initial_incumbent,
                instance=rand_inst,
                cutoff=self.scenario.cutoff,
                seed=initial_seed,
                instance_specific=self.scenario.instance_specific.get(rand_inst,
                                                                      "0"))
        except FirstRunCrashedException as err:
            if self.scenario.abort_on_first_run_crash:
                raise err
            else:
                # TODO make it possible to add the failed run to the runhistory
                if self.scenario.run_obj == "quality":
                    cost = self.scenario.cost_for_crash
                else:
                    cost = self.scenario.cutoff * scenario.par_factor

        self.stats.inc_changed += 1  # first incumbent

        self.traj_logger.add_entry(train_perf=cost,
                                   incumbent_id=self.stats.inc_changed,
                                   incumbent=initial_incumbent)

        return initial_incumbent
