from __future__ import annotations

from typing import List, Tuple

import numpy as np

from smac.acquisition import AbstractAcquisitionOptimizer
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.utils.logging import get_logger
from smac.stats import Stats

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class RandomSearch(AbstractAcquisitionOptimizer):
    """Get candidate solutions via random sampling of configurations."""

    def _maximize(
        self,
        previous_configs: List[Configuration],
        num_points: int,
        _sorted: bool = False,
    ) -> List[Tuple[float, Configuration]]:
        """Randomly sampled configurations."""
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
