from ConfigSpace import Configuration
import numpy as np

from smac.initial_design.single_config_initial_design import \
    SingleConfigInitialDesign
from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class DefaultConfiguration(SingleConfigInitialDesign):
    """ Initial design that evaluates default configuration

    Attributes
    ----------

    """

    def __init__(self,
                 tae_runner: ExecuteTARun,
                 scenario: Scenario,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState
                 ):
        """Constructor

        Parameters
        ----------
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
        rng: np.random.RandomState
            Random state
        """
        super().__init__(tae_runner=tae_runner,
                         scenario=scenario,
                         stats=stats,
                         traj_logger=traj_logger,
                         rng=rng)

    def _select_configuration(self) -> Configuration:
        """Selects the default configuration.

        Returns
        -------
        config: Configuration
            Initial incumbent configuration.
        """
        config = self.scenario.cs.get_default_configuration()
        config.origin = 'Default'
        return config
