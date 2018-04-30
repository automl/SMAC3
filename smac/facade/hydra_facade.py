import logging
import os
import datetime
import time
import typing

import pickle

import numpy as np

from smac.tae.execute_ta_run_hydra import ExecuteTARunHydra
from smac.tae.execute_ta_run_hydra import ExecuteTARunOld
from smac.tae.execute_ta_run_hydra import ExecuteTARun
from smac.tae.execute_ta_run_hydra import ExecuteTARunAClib
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.utils.io.output_directory import create_output_directory
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory import DataOrigin
from smac.optimizer.objective import average_cost

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class Hydra(object):
    """Facade to use Hydra default mode

    Attributes
    ----------
    logger
    stats : Stats
    solver : SMBO
    runhistory : RunHistory
        List with information about previous runs
    trajectory : list
        List of all incumbents
    """

    def __init__(self,
                 scenario: Scenario,
                 n_iterations: int,
                 run_id: int=1,
                 tae: typing.Type[ExecuteTARun]=ExecuteTARunOld,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        scenario : ~smac.scenario.scenario.Scenario
            Scenario object
        n_iterations: int,
            number of Hydra iterations
        """

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.n_iterations = n_iterations

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.scenario = scenario
        self.run_id = run_id
        self.kwargs = kwargs
        self.output_dir = None
        self.top_dir = None
        self.solver = None
        self.rh = RunHistory(average_cost)
        self.tae = tae(ta=self.scenario.ta, run_obj=self.scenario.run_obj)
        self.tae_type = tae

    def optimize(self):
        """Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        -------
        portfolio : typing.List[Configuration]
            Portfolio of found configurations
        """

        portfolio = []
        portfolio_cost = np.inf
        if self.output_dir is None:
            self.top_dir = "hydra-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f'))
            self.scenario.output_dir = os.path.join(self.top_dir, "smac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')))
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)

        self.solver = SMAC(scenario=self.scenario, tae_runner=self.tae, **self.kwargs)
        for i in range(self.n_iterations):
            self.logger.info("Iteration: %d", (i + 1))

            incumbent = self.solver.solver.run()
            self.solver.stats.print_stats()
            self.logger.info("Incumbent of %d-th Iteration", (i + 1))
            self.logger.info(incumbent)
            portfolio.append(incumbent)

            if self.output_dir is not None:
                self.solver.solver.runhistory.save_json(
                    fn=os.path.join(self.solver.output_dir, "runhistory.json")
                )

            # validate incumbent on all trainings instances
            new_rh = self.solver.validate(config_mode='inc',
                                          instance_mode='train',
                                          repetitions=1,
                                          use_epm=False,
                                          n_jobs=1)
            self.rh.update(new_rh, origin=DataOrigin.EXTERNAL_SAME_INSTANCES)
            self.logger.info("Number of validated runs: %d", (len(new_rh.data)))
            # since the TAE uses already the portfolio as an upper limit
            # the following dict already contains oracle performance
            self.logger.info("Start validation of current portfolio")
            cost_per_inst = new_rh.get_instance_costs_for_config(config=incumbent)

            cur_portfolio_cost = np.mean(list(cost_per_inst.values()))
            if portfolio_cost <= cur_portfolio_cost:
                self.logger.info("No further progress (%f) --- terminate hydra", portfolio_cost)
                break
            else:
                portfolio_cost = cur_portfolio_cost
                self.logger.info("Current pertfolio cost: %f", portfolio_cost)

            # modify TAE such that it return oracle performance
            self.tae = ExecuteTARunHydra(ta=self.scenario.ta, run_obj=self.scenario.run_obj,
                                         cost_oracle=cost_per_inst, tae=self.tae_type)

            self.scenario.output_dir = os.path.join(self.top_dir, "smac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')))
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)
            if i != self.n_iterations - 1:
                self.solver = SMAC(scenario=self.scenario, tae_runner=self.tae, **self.kwargs)
        self.rh.save_json(fn=os.path.join(self.top_dir, 'all_runs_runhistory.json'), save_external=True)
        with open(os.path.join(self.top_dir, 'portfolio.pkl'), 'wb') as fh:
            pickle.dump(portfolio, fh)

        return portfolio
