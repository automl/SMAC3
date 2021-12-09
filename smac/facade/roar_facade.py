import logging
import typing

import dask.distributed
import numpy as np

from smac.configspace import Configuration
from smac.epm.random_epm import RandomEPM
from smac.facade.smac_ac_facade import SMAC4AC
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.abstract_racer import AbstractRacer
from smac.optimizer.ei_optimization import RandomSearch, AcquisitionFunctionMaximizer
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, RunHistory2EPM4LogCost, AbstractRunHistory2EPM
from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario
from smac.tae.base import BaseRunner

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class ROAR(SMAC4AC):
    """
    Facade to use ROAR mode

    Parameters
    ----------
    scenario: smac.scenario.scenario.Scenario
        Scenario object
    tae_runner: smac.tae.base.BaseRunner or callable
        Callable or implementation of
        :class:`~smac.tae.base.BaseRunner`. In case a
        callable is passed it will be wrapped by
        :class:`~smac.tae.execute_func.ExecuteTAFuncDict`.
        If not set, it will be initialized with the
        :class:`~smac.tae.execute_ta_run_old.ExecuteTARunOld`.
    tae_runner_kwargs: Optional[Dict]
        arguments passed to constructor of '~tae_runner'
    runhistory: RunHistory
        Runhistory to store all algorithm runs
    intensifier: AbstractRacer
        intensification object to issue a racing to decide the current incumbent
    intensifier_kwargs: Optional[Dict]
        arguments passed to the constructor of '~intensifier'
    acquisition_function_optimizer : ~smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer
        Object that implements the :class:`~smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer`.
        Will use :class:`smac.optimizer.ei_optimization.RandomSearch` if not set. Can be used
        to perform random search over a fixed set of configurations.
    acquisition_function_optimizer_kwargs: Optional[dict]
        Arguments passed to constructor of `~acquisition_function_optimizer`
    initial_design : InitialDesign
        initial sampling design
    initial_design_kwargs: Optional[dict]
        arguments passed to constructor of `~initial_design`
    initial_configurations: typing.List[Configuration]
        list of initial configurations for initial design --
        cannot be used together with initial_design
    stats: Stats
        optional stats object
    rng: np.random.RandomState
        Random number generator
    run_id: int, (default: 1)
        Run ID will be used as subfolder for output_dir.
    dask_client : dask.distributed.Client
        User-created dask client, can be used to start a dask cluster and then attach SMAC to it.
    n_jobs : int, optional
        Number of jobs. If > 1 or -1, this creates a dask client if ``dask_client`` is ``None``. Will
        be ignored if ``dask_client`` is not ``None``.
        If ``None``, this value will be set to 1, if ``-1``, this will be set to the number of cpu cores.

    Attributes
    ----------
    logger

    See Also
    --------
    :class:`~smac.facade.smac_ac_facade.SMAC4AC`

    """

    def __init__(self,
                 scenario: Scenario,
                 tae_runner: typing.Optional[
                     typing.Union[typing.Type[BaseRunner], typing.Callable]
                 ] = None,
                 tae_runner_kwargs: typing.Optional[typing.Dict] = None,
                 runhistory: RunHistory = None,
                 intensifier: typing.Optional[typing.Type[AbstractRacer]] = None,
                 intensifier_kwargs: typing.Optional[typing.Dict] = None,
                 acquisition_function_optimizer: typing.Optional[typing.Type[AcquisitionFunctionMaximizer]] = None,
                 acquisition_function_optimizer_kwargs: typing.Optional[dict] = None,
                 initial_design: typing.Optional[typing.Type[InitialDesign]] = None,
                 initial_design_kwargs: typing.Optional[dict] = None,
                 initial_configurations: typing.List[Configuration] = None,
                 stats: Stats = None,
                 rng: np.random.RandomState = None,
                 run_id: int = 1,
                 dask_client: typing.Optional[dask.distributed.Client] = None,
                 n_jobs: typing.Optional[int] = 1,
                 ):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        scenario.acq_opt_challengers = 1   # type: ignore[attr-defined] # noqa F821

        if acquisition_function_optimizer is None:
            acquisition_function_optimizer = RandomSearch

        if scenario.run_obj == "runtime":
            # We need to do this to be on the same scale for imputation (although we only impute with a Random EPM)
            runhistory2epm = RunHistory2EPM4LogCost  # type: typing.Type[AbstractRunHistory2EPM]
        else:
            runhistory2epm = RunHistory2EPM4Cost

        # use SMAC facade
        super().__init__(
            scenario=scenario,
            tae_runner=tae_runner,
            tae_runner_kwargs=tae_runner_kwargs,
            runhistory=runhistory,
            intensifier=intensifier,
            intensifier_kwargs=intensifier_kwargs,
            runhistory2epm=runhistory2epm,
            initial_design=initial_design,
            initial_design_kwargs=initial_design_kwargs,
            initial_configurations=initial_configurations,
            run_id=run_id,
            acquisition_function_optimizer=acquisition_function_optimizer,
            acquisition_function_optimizer_kwargs=acquisition_function_optimizer_kwargs,
            model=RandomEPM,
            rng=rng,
            stats=stats,
            dask_client=dask_client,
            n_jobs=n_jobs,
        )
