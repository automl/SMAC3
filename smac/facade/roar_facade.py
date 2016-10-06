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
from smac.epm.random_epm import RandomEpm
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
                 runhistory2epm: AbstractRunHistory2EPM=None,
                 initial_design: InitialDesign=None,
                 stats: Stats=None,
                 rng: np.random.RandomState=None):
        '''
        Facade to use ROAR mode

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        tae_runner: ExecuteTARun
            object that implements the following method to call the target
            algorithm (or any other arbitrary function):
            run(self, config)
            If not set, it will be initialized with the tae.ExecuteTARunOld()
        runhistory: RunHistory
            runhistory to store all algorithm runs
        intensifier: Intensifier
            intensification object to issue a racing to decide the current incumbent
        acquisition_function : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction. Will use
            EI if not set.
        model : object
            Model that implements train() and predict(). Will use a
            RandomForest if not set.
        runhistory2epm : RunHistory2EMP
            Object that implements the AbstractRunHistory2EPM. If None,
            will use RunHistory2EPM4Cost if objective is cost or
            RunHistory2EPM4LogCost if objective is runtime.
        initial_design: InitialDesign
            initial sampling design
        stats: Stats
            optional stats object
        rng: np.random.RandomState
            Random number generator
        '''
        self.logger = logging.getLogger("ROAR")

        # initialize random number generator
        if rng is None:
            num_run = np.random.randint(1234567980)
            rng = np.random.RandomState(seed=num_run)
        elif isinstance(rng, int):
            num_run = rng
            rng = np.random.RandomState(seed=rng)
        elif isinstance(rng, np.random.RandomState):
            num_run = rng.randint(1234567980)
            rng = rng
        else:
            raise TypeError('Unknown type %s for argument rng. Only accepts '
                            'None, int or np.random.RandomState' % str(type(rng)))

        # initial EPM
        #use random predictions to simulate random sampling of configurations
        model = RandomEpm(rng=rng)

        # initial conversion of runhistory into EPM data
        if runhistory2epm is None:

            num_params = len(scenario.cs.get_hyperparameters())
            if scenario.run_obj == "runtime":

                # if we log the performance data,
                # the RFRImputator will already get
                # log transform data from the runhistory
                cutoff = np.log10(scenario.cutoff)
                threshold = np.log10(scenario.cutoff *
                                     scenario.par_factor)

                # for ROAR we don't need any imputation of censored data
                runhistory2epm = RunHistory2EPM4LogCost(
                    scenario=scenario, num_params=num_params,
                    success_states=[StatusType.SUCCESS, ],
                    impute_censored_data=False,
                    impute_state=None)

            elif scenario.run_obj == 'quality':
                runhistory2epm = RunHistory2EPM4Cost\
                    (scenario=scenario, num_params=num_params,
                     success_states=[StatusType.SUCCESS, ],
                     impute_censored_data=False, impute_state=None)

            else:
                raise ValueError('Unknown run objective: %s. Should be either '
                                 'quality or runtime.' % self.scenario.run_obj)
                
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
