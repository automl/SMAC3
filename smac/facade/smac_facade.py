import logging

import numpy as np

from smac.tae.execute_ta_run import ExecuteTARun
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.tae.execute_ta_run import StatusType
from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM, RunHistory2EPM4LogCost, RunHistory2EPM4Cost
from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.intensification.intensification import Intensifier
from smac.smbo.smbo import SMBO
from smac.smbo.objective import average_cost
from smac.smbo.acquisition import EI, AbstractAcquisitionFunction
from smac.smbo.local_search import LocalSearch
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.rfr_imputator import RFRImputator
from smac.epm.base_epm import AbstractEPM
from smac.utils.util_funcs import get_types
from smac.utils.io.traj_logging import TrajLogger


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class SMAC(object):

    def __init__(self,
                 scenario: Scenario,
                 # TODO: once we drop python3.4 add type hint
                 # typing.Union[ExecuteTARun, callable]
                 tae_runner=None,
                 runhistory: RunHistory=None,
                 intensifier: Intensifier=None,
                 acquisition_function: AbstractAcquisitionFunction=None,
                 model:AbstractEPM=None,
                 runhistory2epm: AbstractRunHistory2EPM=None,
                 initial_design: InitialDesign=None,
                 stats: Stats=None,
                 rng: np.random.RandomState=None):
        '''
        Facade to use SMAC default mode

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        tae_runner: ExecuteTARun or callable
            Callable or implementation of :class:`ExecuteTaRun`. In case a
            callable is passed it will be wrapped by tae.ExecuteTaFunc().
            If not set, tae_runner will be initialized with the tae.ExecuteTARunOld()
        runhistory: RunHistory
            runhistory to store all algorithm runs
        intensifier: Intensifier
            intensification object to issue a racing to decide the current incumbent
        acquisition_function : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction. Will use
            EI if not set.
        model : AbstractEPM
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

        aggregate_func = average_cost

        # initialize stats object
        if stats:
            self.stats = stats
        else:
            self.stats = Stats(scenario)

        # initialize empty runhistory
        if runhistory is None:
            runhistory = RunHistory(aggregate_func=aggregate_func)

        # initial random number generator
        num_run, rng = self._get_rng(rng=rng)

        # initial Trajectory Logger
        traj_logger = TrajLogger(
            output_dir=scenario.output_dir, stats=self.stats)

        # initial EPM
        types = get_types(scenario.cs, scenario.feature_array)
        if model is None:
            model = RandomForestWithInstances(types=types,
                                              instance_features=scenario.feature_array,
                                              seed=rng.randint(
                                                  1234567980))
        # initial acquisition function
        if acquisition_function is None:
            acquisition_function = EI(model=model)

        # initialize optimizer on acquisition function
        local_search = LocalSearch(acquisition_function,
                                   scenario.cs)

        # initialize tae_runner
        # First case, if tae_runner is None, the target algorithm is a call
        # string in the scenario file
        if tae_runner is None:
            tae_runner = ExecuteTARunOld(ta=scenario.ta,
                                         stats=self.stats,
                                         run_obj=scenario.run_obj,
                                         runhistory=runhistory,
                                         par_factor=scenario.par_factor)
        # Second case, the tae_runner is a function to be optimized
        elif callable(tae_runner):
            tae_runner = ExecuteTAFuncDict(ta=tae_runner,
                                           stats=self.stats,
                                           run_obj=scenario.run_obj,
                                           runhistory=runhistory,
                                           par_factor=scenario.par_factor)
        # Third case, if it is an ExecuteTaRun we can simply use the
        # instance. Otherwise, the next check raises an exception
        elif not isinstance(tae_runner, ExecuteTARun):
            raise TypeError("Argument 'tae_runner' is %s, but must be "
                            "either a callable or an instance of "
                            "ExecuteTaRun. Passing 'None' will result in the "
                            "creation of target algorithm runner based on the "
                            "call string in the scenario file."
                            % type(tae_runner))

        # inject stats if necessary
        if tae_runner.stats is None:
            tae_runner.stats = self.stats
        # inject runhistory if necessary 
        if tae_runner.runhistory is None:
            tae_runner.runhistory = runhistory

        # initial initial design
        if initial_design is None:
            initial_design = DefaultConfiguration(tae_runner=tae_runner,
                                           scenario=scenario,
                                           stats=self.stats,
                                           traj_logger=traj_logger,
                                           runhistory=runhistory,
                                           rng=rng)

        # initial intensification
        if intensifier is None:
            intensifier = Intensifier(tae_runner=tae_runner,
                                      stats=self.stats,
                                      traj_logger=traj_logger,
                                      rng=rng,
                                      instances=scenario.train_insts,
                                      cutoff=scenario.cutoff,
                                      deterministic=scenario.deterministic,
                                      run_obj_time=scenario.run_obj == "runtime",
                                      instance_specifics=scenario.instance_specific)

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

                imputor = RFRImputator(rs=rng,
                                       cutoff=cutoff,
                                       threshold=threshold,
                                       model=model,
                                       change_threshold=0.01,
                                       max_iter=10)

                runhistory2epm = RunHistory2EPM4LogCost(
                    scenario=scenario, num_params=num_params,
                    success_states=[StatusType.SUCCESS, ],
                    impute_censored_data=True,
                    impute_state=[StatusType.TIMEOUT, ],
                    imputor=imputor)

            elif scenario.run_obj == 'quality':
                runhistory2epm = RunHistory2EPM4Cost\
                    (scenario=scenario, num_params=num_params,
                     success_states=[StatusType.SUCCESS, ],
                     impute_censored_data=False, impute_state=None)

            else:
                raise ValueError('Unknown run objective: %s. Should be either '
                                 'quality or runtime.' % self.scenario.run_obj)

        self.solver = SMBO(scenario=scenario,
                           stats=self.stats,
                           initial_design=initial_design,
                           runhistory=runhistory,
                           runhistory2epm=runhistory2epm,
                           intensifier=intensifier,
                           aggregate_func=aggregate_func,
                           num_run=num_run,
                           model=model,
                           acq_optimizer=local_search,
                           acquisition_func=acquisition_function,
                           rng=rng)

    def _get_rng(self, rng):
        '''
            initial random number generator 
           
            Arguments
            ---------
            rng: np.random.RandomState|int|None
                
            Returns
            -------
            int, np.random.RandomState
        '''
        
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
        return num_run, rng

    def optimize(self):
        '''
            optimize the algorithm provided in scenario (given in constructor)

            Arguments
            ---------
            max_iters: int
                maximal number of iterations
        '''
        incumbent = None
        try:
            incumbent = self.solver.run()
        finally:
            self.solver.stats.print_stats()
            self.logger.info("Final Incumbent: %s" % (self.solver.incumbent))
        return incumbent
