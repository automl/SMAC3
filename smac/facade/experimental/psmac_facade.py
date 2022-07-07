# type: ignore
# mypy: ignore-errors

from typing import Any, Dict, List, Optional, Type, Union

import copy
import datetime
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
from ConfigSpace.configuration_space import Configuration

from smac.epm.util_funcs import get_rng
from smac.facade.smac_ac_facade import SMAC4AC
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.tae.base import BaseRunner
from smac.tae.execute_ta_run_hydra import ExecuteTARunOld
from smac.utils.io.output_directory import create_output_directory
from smac.utils.io.result_merging import ResultMerger

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


def optimize(
    scenario: Scenario,
    tae_runner: Type[BaseRunner],
    tae_runner_kwargs: Dict,
    rng: Union[np.random.RandomState, int],
    output_dir: str,
    facade_class: Optional[Type[SMAC4AC]] = None,
    **kwargs,
) -> Configuration:
    """
    Unbound method to be called in a subprocess

    Parameters
    ----------
    scenario: Scenario
        smac.Scenario to initialize SMAC
    tae_runner: BaseRunner
        Target Algorithm Runner (supports old and aclib format)
    tae_runner_kwargs: Optional[dict]
        arguments passed to constructor of '~tae'
    rng: int/np.random.RandomState
        The randomState/seed to pass to each smac run
    output_dir: str
        The directory in which each smac run should write its results

    Returns
    -------
    incumbent: Configuration
        The incumbent configuration of this run

    """
    if facade_class is None:
        facade_class = SMAC4AC
    solver = facade_class(
        scenario=scenario, tae_runner=tae_runner, tae_runner_kwargs=tae_runner_kwargs, rng=rng, **kwargs
    )
    solver.stats.start_timing()
    solver.stats.print_stats()

    incumbent = solver.solver.run()
    solver.stats.print_stats()

    if output_dir is not None:
        solver.solver.runhistory.save_json(fn=os.path.join(solver.output_dir, "runhistory.json"))
    return incumbent


class PSMAC(object):
    """
    Facade to use pSMAC [1]_

    With pSMAC you can either run n distinct SMAC optimizations in parallel
    (`shared_model=False`) or you can parallelize the target algorithm evaluations
    (`shared_model=True`).
    In the latter case all SMAC workers share one file directory and communicate via
    the logfiles. You can specify the number of SMAC workers/optimizers with the
    argument `n_workers`.

    You can pass all other kwargs for the SMAC4AC facade.
    In addition, you can access the facade's attributes normally (e.g. smac.stats),
    however each time a new SMAC object is built, reading the information from the
    file system.


    .. [1] Ramage, S. E. A. (2015). Advances in meta-algorithmic software libraries for
        distributed automated algorithm configuration (T). University of British
        Columbia. Retrieved from
        https://open.library.ubc.ca/collections/ubctheses/24/items/1.0167184.

    Parameters
    ----------
    scenario : ~smac.scenario.scenario.Scenario
        Scenario object. Note that the budget/number of evaluations (runcount-limit) is
        used for each worker. So if you specify 40 evaluations and 3 workers, 120
        configurations will be evaluated in total.
    n_workers: int
        Number of optimizers to run in parallel per round
    rng: int/np.random.RandomState
        The randomState/seed to pass to each smac run
    run_id: int
        run_id for this hydra run
    tae_runner: BaseRunner
        Target Algorithm Runner (supports old and aclib format as well as AbstractTAFunc)
    tae_runner_kwargs: Optional[dict]
        arguments passed to constructor of '~tae_runner'
    shared_model: bool
        Flag to indicate whether information is shared between SMAC runs or not
    validate: bool / None
        Flag to indicate whether to validate the found configurations or to use the SMAC estimates
        None => neither and return the full portfolio
    val_set: List[str]
        List of instance-ids to validate on
    **kwargs
        Keyword arguments for the SMAC4AC facade

    Attributes
    ----------
    # TODO update attributes
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

    def __init__(
        self,
        scenario: Scenario,
        rng: Optional[Union[np.random.RandomState, int]] = None,
        run_id: int = 1,
        tae_runner: Type[BaseRunner] = ExecuteTARunOld,
        tae_runner_kwargs: Union[dict, None] = None,
        shared_model: bool = True,
        facade_class: Optional[Type[SMAC4AC]] = None,
        validate: bool = True,
        n_workers: int = 2,
        val_set: Union[List[str], None] = None,
        **kwargs,
    ):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        self.scenario = scenario
        self.run_id, self.rng = get_rng(rng, run_id, logger=self.logger)
        self.kwargs = kwargs
        self.output_dir = None
        if facade_class is None:
            facade_class = SMAC4AC
        self.facade_class = facade_class
        self.rh = RunHistory()
        self._tae_runner = tae_runner
        self._tae_runner_kwargs = tae_runner_kwargs
        if n_workers <= 1:
            self.logger.warning("Invalid value in %s: %d. Setting to 2", "n_workers", n_workers)
        self.n_workers = max(n_workers, 2)
        self.seeds = np.arange(0, self.n_workers, dtype=int)  # seeds for the parallel runs
        self.validate = validate
        self.shared_model = shared_model
        if val_set is None:
            self.val_set = scenario.train_insts
        else:
            self.val_set = val_set

        self.result_merger: Optional[ResultMerger] = None
        self.n_incs: int = 1

    def optimize(self) -> Configuration:
        """
        Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        -------
        incumbent : Configuration
            Best configuration across all workers.

        """
        # Setup output directory
        if self.output_dir is None:
            self.scenario.output_dir = "psmac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H:%M:%S_%f")
            )
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)
            if self.shared_model:
                self.scenario.shared_model = self.shared_model
        if self.scenario.input_psmac_dirs is None:
            self.scenario.input_psmac_dirs = os.path.sep.join((self.scenario.output_dir, "run_*"))

        scen = copy.deepcopy(self.scenario)
        scen.output_dir_for_this_run = None
        scen.output_dir = None
        self.logger.info("+" * 120)
        self.logger.info("PSMAC run")

        incs = joblib.Parallel(n_jobs=self.n_workers)(
            joblib.delayed(optimize)(
                scenario=self.scenario,  # Scenario object
                tae_runner=self._tae_runner,  # type of tae_runner to run target with
                tae_runner_kwargs=self._tae_runner_kwargs,
                rng=int(seed),  # seed for the rng/run_id
                output_dir=self.output_dir,  # directory to create outputs in
                facade_class=self.facade_class,
                **self.kwargs,
            )
            for seed in self.seeds
        )

        inc = self.get_best_incumbent(incs=incs)
        return inc

    def get_best_incumbent(self, incs: List[Configuration]) -> Configuration:
        """
        Determine ID and cost of best configuration (incumbent).

        Parameters
        ----------
        incs : List[Configuration]
            List of incumbents from the workers.

        Returns
        -------
        Configuration
            Best incumbent from all workers.

        """
        if self.validate is True:
            mean_costs_conf_valid, cost_per_config_valid = self.validate_incs(incs)
            val_id = list(map(lambda x: x[0], sorted(enumerate(mean_costs_conf_valid), key=lambda y: y[1])))[0]
            inc = incs[val_id]
        else:
            mean_costs_conf_estimate, cost_per_config_estimate = self._get_mean_costs(incs, self.rh)
            est_id = list(map(lambda x: x[0], sorted(enumerate(mean_costs_conf_estimate), key=lambda y: y[1])))[0]
            inc = incs[est_id]

        return inc

    def _get_mean_costs(self, incs: List[Configuration], new_rh: RunHistory):
        """
        Compute mean cost per instance

        Parameters
        ----------
        incs : List[Configuration]
            incumbents determined by all parallel SMAC runs
        new_rh : RunHistory
            runhistory to determine mean performance

        Returns
        -------
        List[float] means
        Dict(Config -> Dict(inst_id(str) -> float))

        """
        config_cost_per_inst = {}
        results = []
        for incumbent in incs:
            cost_per_inst = new_rh.get_instance_costs_for_config(config=incumbent)
            config_cost_per_inst[incumbent] = cost_per_inst
            values = list(cost_per_inst.values())
            if values:
                results.append(np.mean(values))
            else:
                results.append(np.nan)
        return results, config_cost_per_inst

    def _get_solver(self):
        # TODO specify one output dir or no output dir
        solver = self.facade_class(scenario=self.scenario, rng=self.rng, run_id=None, **self.kwargs)
        return solver

    def validate_incs(self, incs: List[Configuration]):
        """Validation of the incumbents."""
        solver = self._get_solver()
        self.logger.info("*" * 120)
        self.logger.info("Validating")
        new_rh = solver.validate(
            config_mode=incs,
            instance_mode=self.val_set,
            repetitions=1,
            use_epm=False,
            n_jobs=self.n_workers,
        )
        return self._get_mean_costs(incs, new_rh)

    def write_run(self) -> None:
        """
        Write all SMAC files to pSMAC dir

        Returns
        -------
        None

        """
        # write runhistory
        # write configspace file .pcs .json
        # write trajectory traj.json
        # write scenario .txt
        # write stats
        raise NotImplementedError

    def _check_result_merger(self):
        if self.result_merger is None:
            if self.output_dir is None:
                raise ValueError(
                    "Cannot instantiate `ResultMerger` because `output_dir` "
                    "is None. In pSMAC `output_dir` is set after "
                    "`optimize()` has been called. If you already have "
                    "a pSMAC run or rundirs, please directly use "
                    "`smac.utils.io.result_merging.ResultMerger`."
                )
            self.result_merger = ResultMerger(output_dir=Path(self.output_dir).parent)

    def get_runhistory(self) -> Optional[RunHistory]:
        """
        Get merged runhistory from pSMAC workers

        Returns
        -------
        Optional[RunHistory]

        """
        self._check_result_merger()
        return self.result_merger.get_runhistory()

    def get_trajectory(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get trajectory from merged runhistory

        Returns
        -------
        Optional[List[Dict[str, Any]]]

        """
        self._check_result_merger()
        return self.result_merger.get_trajectory()
