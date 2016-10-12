import logging

import numpy as np

from smac.tae.execute_ta_run import ExecuteTARun
from smac.tae.execute_ta_run import StatusType
from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM, RunHistory2EPM4LogCost, RunHistory2EPM4Cost
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.intensification import Intensifier
from smac.smbo.acquisition import AbstractAcquisitionFunction
from smac.epm.random_epm import RandomEPM
from smac.facade.smac_facade import SMAC

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class ROAR(SMAC):

    def __init__(self,
                 scenario: Scenario,
                 tae_runner: ExecuteTARun=None,
                 runhistory: RunHistory=None,
                 intensifier: Intensifier=None,
                 initial_design: InitialDesign=None,
                 stats: Stats=None,
                 rng: np.random.RandomState=None):
        '''
        Facade to use ROAR mode

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        tae_runner: ExecuteTARun or callable
            Callable or implementation of :class:`ExecuteTaRun`. In case a
            callable is passed it will be wrapped by tae.ExecuteTaFunc().
            If not set, it will be initialized with the tae.ExecuteTARunOld()
        runhistory: RunHistory
            runhistory to store all algorithm runs
        intensifier: Intensifier
            intensification object to issue a racing to decide the current incumbent
        initial_design: InitialDesign
            initial sampling design
        stats: Stats
            optional stats object
        rng: np.random.RandomState
            Random number generator
        '''
        self.logger = logging.getLogger("ROAR")

        # initial random number generator
        num_run, rng = self._get_rng(rng=rng)

        # initial EPM
        #use random predictions to simulate random sampling of configurations
        model = RandomEPM(rng=rng)

        # initial conversion of runhistory into EPM data
        # since ROAR does not really use it the converted data
        # we simply use a cheap RunHistory2EPM here
        num_params = len(scenario.cs.get_hyperparameters())
        runhistory2epm = RunHistory2EPM4Cost\
            (scenario=scenario, num_params=num_params,
             success_states=[StatusType.SUCCESS, ],
             impute_censored_data=False, impute_state=None)
                
        # use SMAC facade
        super().__init__(
                         scenario=scenario,
                         tae_runner=tae_runner,
                         runhistory=runhistory,
                         intensifier=intensifier,
                         model=model,
                         runhistory2epm=runhistory2epm,
                         initial_design=initial_design,
                         stats=stats,
                         rng=rng)
