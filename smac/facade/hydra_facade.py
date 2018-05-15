import logging
import os
import datetime
import time
import typing
import copy

import pickle
import multiprocessing

import numpy as np

from ConfigSpace.configuration_space import Configuration

from smac.tae.execute_ta_run_hydra import ExecuteTARunHydra
from smac.tae.execute_ta_run_hydra import ExecuteTARunOld
from smac.tae.execute_ta_run_hydra import ExecuteTARun
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.utils.io.output_directory import create_output_directory
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory import DataOrigin
from smac.optimizer.objective import average_cost
from smac.utils.util_funcs import get_rng

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


def optimize(queue: multiprocessing.Queue,
             scenario: typing.Type[Scenario],
             tae: typing.Type[ExecuteTARun],
             rng: typing.Union[np.random.RandomState, int],
             output_dir: str,
             cost_per_inst: typing.Dict[str, float],
             **kwargs) -> Configuration:
    """
    Unbound method to be called in a subprocess

    Parameters
    ----------
    queue: multiprocessing.Queue
        incumbents (Configurations) of each SMAC call will be pushed to this queue
    scenario: Scenario
        smac.Scenario to initialize SMAC
    tae: ExecuteTARun
        Target Algorithm Runner (supports old and aclib format)
    rng: int/np.random.RandomState
        The randomState/seed to pass to each smac run
    output_dir: str
        The directory in which each smac run should write it's results
    cost_per_inst: dict
        maps instance to cost. Contains portfolio performance on each instance

    Returns
    -------
    incumbent: Configuration
        The incumbent configuration of this run

    """
    if not cost_per_inst:
        tae = tae(ta=scenario.ta, run_obj=scenario.run_obj)
    else:
        tae = ExecuteTARunHydra(ta=scenario.ta, run_obj=scenario.run_obj,
                                cost_oracle=cost_per_inst, tae=tae)
    solver = SMAC(scenario=scenario, tae_runner=tae, rng=rng, **kwargs)
    solver.stats.start_timing()
    solver.stats.print_stats()

    incumbent = solver.solver.run()
    solver.stats.print_stats()

    if output_dir is not None:
        solver.solver.runhistory.save_json(
            fn=os.path.join(solver.output_dir, "runhistory.json")
        )
    queue.put(incumbent)
    queue.close()
    return incumbent


class Hydra(object):
    """
    Facade to use Hydra default mode

    Attributes
    ----------
    logger
    stats : Stats
        loggs information about used resources
    solver : SMBO
        handles the actual algorithm calls
    rh : RunHistory
        List with information about previous runs
    portfolio : list
        List of all incumbents

    """

    def __init__(self,
                 scenario: typing.Type[Scenario],
                 n_iterations: int,
                 val_set: str='train',
                 incs_per_round: int=1,
                 n_optimizers: int=1,
                 rng: typing.Optional[typing.Union[np.random.RandomState, int]]=None,
                 run_id: int=1,
                 tae: typing.Type[ExecuteTARun]=ExecuteTARunOld,
                 **kwargs):
        """
        Constructor

        Parameters
        ----------
        scenario : ~smac.scenario.scenario.Scenario
            Scenario object
        n_iterations: int,
            number of Hydra iterations
        val_set: str
            Set to validate incumbent(s) on. [train, valX].
            train => whole training set,
            valX => train_set * 100/X where X in (0, 100)
        incs_per_round: int
            Number of incumbents to keep per round
        n_optimizers: int
            Number of optimizers to run in parallel per round
        rng: int/np.random.RandomState
            The randomState/seed to pass to each smac run
        run_id: int
            run_id for this hydra run
        tae: ExecuteTARun
            Target Algorithm Runner (supports old and aclib format as well as AbstractTAFunc)

        """
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.n_iterations = n_iterations
        self.scenario = scenario
        self.run_id, self.rng = get_rng(rng, run_id)
        self.kwargs = kwargs
        self.output_dir = None
        self.top_dir = None
        self.solver = None
        self.portfolio = None
        self.rh = RunHistory(average_cost)
        self._tae = tae
        self.tae = tae(ta=self.scenario.ta, run_obj=self.scenario.run_obj)
        if incs_per_round <= 0:
            self.logger.warning('Invalid value in %s: %d. Setting to 1', 'incs_per_round', incs_per_round)
        self.incs_per_round = max(incs_per_round, 1)
        if n_optimizers <= 0:
            self.logger.warning('Invalid value in %s: %d. Setting to 1', 'n_optimizers', n_optimizers)
        self.n_optimizers = max(n_optimizers, 1)
        self.val_set = self._get_validation_set(val_set)
        self.cost_per_inst = {}

    def _get_validation_set(self, val_set: str, delete: bool=True) -> typing.List[str]:
        """
        Create small validation set for hydra to determine incumbent performance

        Parameters
        ----------
        val_set: str
            Set to validate incumbent(s) on. [train, valX].
            train => whole training set,
            valX => train_set * 100/X where X in (0, 100)
        delete: bool
            Flag to delete all validation instances from the training set

        Returns
        -------
        val: typing.List[str]
            List of instance-ids to validate on

        """
        if val_set == 'train':
            return self.scenario.train_insts
        elif val_set[:3] != 'val':
            self.logger.warning('Can not determine validation set size. Using full training-set!')
            return self.scenario.train_insts
        else:
            size = int(val_set[3:])/100
            if size <= 0 or size >= 1:
                raise ValueError('X invalid in valX, should be between 0 and 1')
            insts = np.array(self.scenario.train_insts)
            # just to make sure this also works with the small example we have to round up to 3
            size = max(np.floor(insts.shape[0] * size).astype(int), 3)
            ids = np.random.choice(insts.shape[0], size, replace=False)
            val = insts[ids].tolist()
            if delete:
                self.scenario.train_insts = np.delete(insts, ids).tolist()
            return val

    def optimize(self) -> typing.List[Configuration]:
        """
        Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        -------
        portfolio : typing.List[Configuration]
            Portfolio of found configurations

        """
        # Setup output directory
        self.portfolio = []
        portfolio_cost = np.inf
        if self.output_dir is None:
            self.top_dir = "hydra-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f'))
            self.scenario.output_dir = os.path.join(self.top_dir, "smac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')))
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)

        scen = copy.deepcopy(self.scenario)
        scen.output_dir_for_this_run = None
        scen.output_dir = None
        # parent process SMAC only used for validation purposes
        self.solver = SMAC(scenario=scen, tae_runner=self.tae, rng=self.rng, run_id=self.run_id, **self.kwargs)
        for i in range(self.n_iterations):
            self.logger.info("="*120)
            self.logger.info("Hydra Iteration: %d", (i + 1))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiprocessing part start ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            q = multiprocessing.Queue()
            procs = []
            for p in range(self.n_optimizers):
                proc = multiprocessing.Process(target=optimize,
                                               args=(
                                                   q,                   # Output queue
                                                   self.scenario,       # Scenario object
                                                   self._tae,           # type of tae to run target with
                                                   p,                   # process_id (used in output folder name)
                                                   self.output_dir,     # directory to create outputs in
                                                   self.cost_per_inst   # portfolio cost per instance
                                               ),
                                               kwargs=self.kwargs)
                proc.start()
                procs.append(proc)
            for proc in procs:
                proc.join()
            incs = np.empty((self.n_optimizers, ), dtype=Configuration)
            self.logger.info('*'*120)
            self.logger.info('Incumbents this round:')
            idx = 0
            while not q.empty():
                conf = q.get_nowait()
                self.logger.info(conf)
                incs[idx] = conf
                idx += 1
            q.close()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiprocessing part end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            cur_portfolio_cost = self._update_portfolio(incs)
            if portfolio_cost <= cur_portfolio_cost:
                self.logger.info("No further progress (%f) --- terminate hydra", portfolio_cost)
                break
            else:
                portfolio_cost = cur_portfolio_cost
                self.logger.info("Current pertfolio cost: %f", portfolio_cost)

            # modify TAE such that it return oracle performance
            self.tae = ExecuteTARunHydra(ta=self.scenario.ta, run_obj=self.scenario.run_obj,
                                         cost_oracle=self.cost_per_inst, tae=self._tae)

            self.scenario.output_dir = os.path.join(self.top_dir, "smac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')))
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)
        self.rh.save_json(fn=os.path.join(self.top_dir, 'all_runs_runhistory.json'), save_external=True)
        with open(os.path.join(self.top_dir, 'portfolio.pkl'), 'wb') as fh:
            pickle.dump(self.portfolio, fh)
        self.logger.info("~"*120)
        self.logger.info('Resulting Portfolio:')
        for configuration in self.portfolio:
            self.logger.info(str(configuration))
        self.logger.info("~"*120)

        return self.portfolio

    def _update_portfolio(self, incs: np.ndarray) -> typing.Union[np.float, float]:
        """
        Validates all configurations (in incs) and determines which ones to add to the portfolio

        Parameters
        ----------
        incs: np.ndarray
            List of Configurations

        Returns
        -------
        cur_cost: typing.Union[np.float, float]
            The current cost of the portfolio

        """
        self.logger.info('*'*120)
        self.logger.info('Validating')
        new_rh = self.solver.validate(config_mode=incs,
                                      instance_mode=self.val_set,
                                      repetitions=1,
                                      use_epm=False,
                                      n_jobs=self.n_optimizers)
        self.rh.update(new_rh, origin=DataOrigin.EXTERNAL_SAME_INSTANCES)
        self.logger.info("Number of validated runs: %d", (len(new_rh.data)))
        self.logger.info('*'*120)
        self.logger.info('Determining best incumbents')
        results = []
        config_cost_per_inst = {}
        for incumbent in incs:
            cost_per_inst = new_rh.get_instance_costs_for_config(config=incumbent)
            config_cost_per_inst[incumbent] = cost_per_inst
            results.append(np.mean(list(self.cost_per_inst.values())))
        to_keep_ids = list(map(lambda x: x[0],
                               sorted(enumerate(results), key=lambda y: y[1])))[:self.incs_per_round]
        if len(to_keep_ids) > 1:
            self.logger.info('Keeping incumbents of runs %s', ', '.join(map(str, to_keep_ids)))
        else:
            self.logger.info('Keeping incumbent of run %s', str(to_keep_ids))
        keep_incumbents = incs[to_keep_ids]
        for kept in keep_incumbents:
            self.portfolio.append(kept)
            cost_per_inst = config_cost_per_inst[kept]
            if self.cost_per_inst:
                if len(self.cost_per_inst) != len(cost_per_inst):
                    raise ValueError('Num validated Instances mismatch!')
                for key in cost_per_inst:
                    self.cost_per_inst[key] = min(self.cost_per_inst[key], cost_per_inst[key])
            else:
                self.cost_per_inst = cost_per_inst

        cur_cost = np.mean(list(self.cost_per_inst.values()))  # type: np.float
        return cur_cost
