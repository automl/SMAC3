from __future__ import annotations

from typing import List, Tuple

from smac.acquisition_optimizer import AbstractAcquisitionOptimizer
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.utils.stats import Stats


class RandomSearch(AbstractAcquisitionOptimizer):
    """Get candidate solutions via random sampling of configurations.

    Parameters
    ----------
    acquisition_function : ~smac.acquisition.AbstractAcquisitionFunction

    configspace : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def _maximize(
        self,
        runhistory: RunHistory,
        stats: Stats,
        num_points: int,
        _sorted: bool = False,
    ) -> List[Tuple[float, Configuration]]:
        """Randomly sampled configurations.

        Parameters
        ----------
        runhistory: ~smac.runhistory.runhistory.RunHistory
            runhistory object
        stats: ~smac.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`smac.configspace.Configuration`).
        """
        if num_points > 1:
            rand_configs = self.configspace.sample_configuration(size=num_points)
        else:
            rand_configs = [self.configspace.sample_configuration(size=1)]
        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = "Random Search (sorted)"
            return self._sort_configs_by_acq_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = "Random Search"
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]
