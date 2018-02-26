import logging
import os
import shutil
import typing

import numpy as np

from smac.optimizer.objective import average_cost
from smac.optimizer.ei_optimization import RandomSearch
from smac.tae.execute_ta_run import StatusType, ExecuteTARun
from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.intensification import Intensifier
from smac.facade.smac_facade import SMAC
from smac.configspace import Configuration

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class Hydra(SMAC):
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
                 n_iterations:int,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        scenario : ~smac.scenario.scenario.Scenario
            Scenario object
        n_iterations: int,
            number of Hydra iterations
        """
        
        self.n_iterations = n_iterations

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        
        self.kwargs = kwargs
        
        super().__init__(scenario=scenario, **kwargs)


    def _get_rng(self, rng):
        """Initialize random number generator

        If rng is None, initialize a new generator
        If rng is Int, create RandomState from that
        If rng is RandomState, return it

        Parameters
        ----------
        rng: np.random.RandomState|int|None

        Returns
        -------
        int, np.random.RandomState
        """
        # initialize random number generator
        if rng is None:
            self.logger.debug('no rng given: using default seed of 1')
            num_run = 1
            rng = np.random.RandomState(seed=num_run)
        elif isinstance(rng, int):
            num_run = rng
            rng = np.random.RandomState(seed=rng)
        elif isinstance(rng, np.random.RandomState):
            num_run = rng.randint(MAXINT)
            rng = rng
        else:
            raise TypeError('Unknown type %s for argument rng. Only accepts '
                            'None, int or np.random.RandomState' % str(type(rng)))
        return num_run, rng

    @staticmethod
    def _get_random_configuration_chooser(random_configuration_chooser):
        """Initialize random configuration chooser
        If random_configuration_chooser is falsy, initialize with ChooserNoCoolDown(2.0)

        Parameters
        ----------
        random_configuration_chooser: ChooserNoCoolDown|ChooserLinearCoolDown|None

        Returns
        -------
        ChooserNoCoolDown|ChooserLinearCoolDown
        """
        if not random_configuration_chooser:
            return ChooserNoCoolDown(2.0)
        return random_configuration_chooser

    def optimize(self):
        """Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        ----------
        portfolio : typing.List[Configuration]
            Portfolio of found configurations
        """

        portfolio = []
        for i in range(self.n_iterations):
            incumbent = None
            try:
                    self.logger.info("Iteration: %d" %(i+1))
                    incumbent = self.solver.run()
                    portfolio.append(incumbent)
                    
                    # validate incumbent on all trainings instances
                    
                    # modify TAE
                    
                    super().__init__(scenario=scenario, **kwargs)
                    
                    
            finally:
                self.solver.stats.save()
                self.solver.stats.print_stats()
                self.logger.info("Final Incumbent: %s" % (self.solver.incumbent))
                self.runhistory = self.solver.runhistory
                self.trajectory = self.solver.intensifier.traj_logger.trajectory
        
                if self.output_dir is not None:
                    self.solver.runhistory.save_json(
                        fn=os.path.join(self.output_dir, "runhistory.json")
                    )
        return portfolio

    def validate(self, config_mode='inc', instance_mode='train+test',
                 repetitions=1, use_epm=False, n_jobs=-1, backend='threading'):
        """Create validator-object and run validation, using
        scenario-information, runhistory from smbo and tae_runner from intensify

        Parameters
        ----------
        config_mode: str or list<Configuration>
            string or directly a list of Configuration
            str from [def, inc, def+inc, wallclock_time, cpu_time, all]
            time evaluates at cpu- or wallclock-timesteps of:
            [max_time/2^0, max_time/2^1, max_time/2^3, ..., default]
            with max_time being the highest recorded time
        instance_mode: string
            what instances to use for validation, from [train, test, train+test]
        repetitions: int
            number of repetitions in nondeterministic algorithms (in
            deterministic will be fixed to 1)
        use_epm: bool
            whether to use an EPM instead of evaluating all runs with the TAE
        n_jobs: int
            number of parallel processes used by joblib
        backend: string
            what backend to be used by joblib

        Returns
        -------
        runhistory: RunHistory
            runhistory containing all specified runs
        """
        return self.solver.validate(config_mode, instance_mode, repetitions,
                                    use_epm, n_jobs, backend)

    def get_tae_runner(self):
        """Returns target algorithm evaluator (TAE) object which can run the
        target algorithm given a configuration

        Returns
        -------
        TAE: smac.tae.execute_ta_run.ExecuteTARun
        """
        return self.solver.intensifier.tae_runner

    def get_runhistory(self):
        """Returns the runhistory (i.e., all evaluated configurations and
         the results).

        Returns
        -------
        Runhistory: smac.runhistory.runhistory.RunHistory
        """
        if not hasattr(self, 'runhistory'):
            raise ValueError('SMAC was not fitted yet. Call optimize() prior '
                             'to accessing the runhistory.')
        return self.runhistory

    def get_trajectory(self):
        """Returns the trajectory (i.e., all incumbent configurations over
        time).

        Returns
        -------
        Trajectory : List of :class:`~smac.utils.io.traj_logging.TrajEntry`
        """

        if not hasattr(self, 'trajectory'):
            raise ValueError('SMAC was not fitted yet. Call optimize() prior '
                             'to accessing the runhistory.')
        return self.trajectory

    def get_X_y(self):
        """Simple interface to obtain all data in runhistory in ``X, y`` format.

        Uses
        :meth:`smac.runhistory.runhistory2epm.AbstractRunHistory2EPM.get_X_y()`.

        Returns
        -------
        X: numpy.ndarray
            matrix of all configurations (+ instance features)
        y: numpy.ndarray
            vector of cost values; can include censored runs
        cen: numpy.ndarray
            vector of bools indicating whether the y-value is censored
        """
        return self.solver.rh2EPM.get_X_y(self.runhistory)
