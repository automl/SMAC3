from __future__ import annotations

from ConfigSpace import Configuration

from smac.acquisition.maximizer.abstract_acqusition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class RandomSearch(AbstractAcquisitionMaximizer):
    """Get candidate solutions via random sampling of configurations."""

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        _sorted: bool = False,
    ) -> list[tuple[float, Configuration]]:
        """Maximize acquisition function with random search

        Parameters
        ----------
        previous_configs : list[Configuration]
            Not used.
        n_points : int
            Number of configurations to return.
        _sorted : bool, optional
            If True, sort candidates by their acquisition value (descending), by default False

        Returns
        -------
        list[tuple[float, Configuration]]
            Candidates with their acquisition function value. (acq value, candidate)
        """
        if n_points > 1:
            rand_configs = self._configspace.sample_configuration(size=n_points)
        else:
            rand_configs = [self._configspace.sample_configuration(size=1)]

        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = "Acquisition Function Maximizer: Random Search (sorted)"

            return self._sort_by_acquisition_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = "Acquisition Function Maximizer: Random Search"

            return [(0, rand_configs[i]) for i in range(len(rand_configs))]
