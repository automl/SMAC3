import logging

import numpy as np

from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory, AbstractRunHistory2EPM
from smac.smbo.acquisition import AbstractAcquisitionFunction
from smac.initial_design.initial_design import InitialDesign

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"

class SMAC(object):
    
    def __init__(self, 
                 scenario:Scenario, 
                 tae_runner:ExecuteTARun=None, 
                 acquisition_function:AbstractAcquisitionFunction=None,
                 model=None, 
                 runhistory2epm:AbstractRunHistory2EPM=None, 
                 initial_design:InitialDesign=None, 
                 stats:Stats=None, 
                 rng:np.random.RandomState=None):
        '''
        Interface that contains the main Bayesian optimization loop

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        tae_runner: ExecuteTARun
            object that implements the following method to call the target
            algorithm (or any other arbitrary function):
            run(self, config)
            If not set, it will be initialized with the tae.ExecuteTARunOld()
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
        self.logger = logging.getLogger("SMAC")
        
        # initialize stats object
        if stats:
            self.stats = stats
        else:
            self.stats = Stats(scenario)

        # initialize empty runhistory
        self.runhistory = RunHistory()

        # initialize random number generator
        if rng is None:
            self.num_run = np.random.randint(1234567980)
            self.rng = np.random.RandomState(seed=self.num_run)
        elif isinstance(rng, int):
            self.num_run = rng
            self.rng = np.random.RandomState(seed=rng)
        elif isinstance(rng, np.random.RandomState):
            self.num_run = rng.randint(1234567980)
            self.rng = rng
        else:
            raise TypeError('Unknown type %s for argument rng. Only accepts '
                            'None, int or np.random.RandomState' % str(type(rng)))

        # initial Trajectory Logger                
        traj_logger = TrajLogger(
            output_dir=self.scenario.output_dir, stats=self.stats)

        # initial EPM
        types = get_types(self.config_space, scenario.feature_array)
        if model is None:
            model = RandomForestWithInstances(types,
                                               scenario.feature_array,
                                               seed=self.rng.randint(
                                                   1234567980))
        else:
            model = model
            
        # initial acquisition function
        if acquisition_function is None:
            acquisition_func = EI(self.model)
        else:
            acquisition_func = acquisition_function

        # initialize optimizer on acquisition function
        local_search = LocalSearch(self.acquisition_func,
                                        self.config_space)
        
        self.incumbent = None

        self.objective = average_cost

        if tae_runner is None:
            self.executor = ExecuteTARunOld(ta=scenario.ta,
                                            stats=self.stats,
                                            run_obj=scenario.run_obj,
                                            runhistory=self.runhistory,
                                            aggregate_func=self.objective,
                                            par_factor=scenario.par_factor)
        else:
            self.executor = tae_runner
            
        if initial_design is None:
            self.initial_design = DefaultDesign(tae_runner=self.executor,
                                                 scenario=self.scenario,
                                                 stats=self.stats,
                                                 traj_logger=self.traj_logger,
                                                 runhistory=self.runhistory,
                                                 rng=self.rng)
        else:
            self.initial_design = initial_design

        self.inten = Intensifier(executor=self.executor,
                                 stats=self.stats,
                                 traj_logger=self.traj_logger,
                                 instances=self.scenario.train_insts,
                                 cutoff=self.scenario.cutoff,
                                 deterministic=self.scenario.deterministic,
                                 run_obj_time=self.scenario.run_obj == "runtime",
                                 instance_specifics=self.scenario.instance_specific)

        num_params = len(self.config_space.get_hyperparameters())

        
        if self.scenario.run_obj == "runtime":

            if runhistory2epm is None:
                # if we log the performance data,
                # the RFRImputator will already get
                # log transform data from the runhistory
                cutoff = np.log10(self.scenario.cutoff)
                threshold = np.log10(self.scenario.cutoff *
                                     self.scenario.par_factor)

                imputor = RFRImputator(cs=self.config_space,
                                       rs=self.rng,
                                       cutoff=cutoff,
                                       threshold=threshold,
                                       model=self.model,
                                       change_threshold=0.01,
                                       max_iter=10)
                self.rh2EPM = RunHistory2EPM4LogCost(
                    scenario=self.scenario, num_params=num_params,
                    success_states=[StatusType.SUCCESS, ],
                    impute_censored_data=True,
                    impute_state=[StatusType.TIMEOUT, ],
                    imputor=imputor)
            else:
                self.rh2EPM = runhistory2epm

        elif self.scenario.run_obj == 'quality':
            if runhistory2epm is None:
                self.rh2EPM = RunHistory2EPM4Cost\
                    (scenario=self.scenario, num_params=num_params,
                     success_states=[StatusType.SUCCESS, ],
                     impute_censored_data=False, impute_state=None)
            else:
                self.rh2EPM = runhistory2epm

        else:
            raise ValueError('Unknown run objective: %s. Should be either '
                             'quality or runtime.' % self.scenario.run_obj)
