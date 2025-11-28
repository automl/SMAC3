from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration

from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


def _dominates(a: list[float], b: list[float]) -> bool:
    # Checks if a dominates b
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    return np.count_nonzero(a <= b) >= len(a) and np.count_nonzero(a < b) >= 1


class NewCostDominatesOldCost(AbstractIntensifier):
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """Checks if the configuration should be evaluated against the incumbent while it
        did not run on all the trails the incumbents did. This function checks if the current configuration performance
        improved over its performance in the previous intermediate comparison. As described in MOParamILS.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)

        if not hasattr(self, "_old_config_cost"):
            self._old_config_cost: dict[Configuration, list[float]] = {}

        new_cost: list[float] = self.runhistory.average_cost(config, config_isb_keys)  # type: ignore[assignment]
        if config not in self._old_config_cost:
            self._old_config_cost[config] = new_cost
            return True

        old_cost: list[float] = self._old_config_cost[config]
        if _dominates(new_cost, old_cost):
            self._old_config_cost[config] = new_cost
            return True
        return False


class NewCostDominatesOldCostSkipFirst(AbstractIntensifier):
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """Checks if the configuration should be evaluated against the incumbent while it
        did not run on all the trails the incumbents did. This function checks if the current configuration performance
        improved over its performance in the previous intermediate comparison. As described in MOParamILS.
        However, the first comparison with the incumbent when the configuration dominates the cost after finishing
        its first trial.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)

        if not hasattr(self, "_old_config_cost"):
            self._old_config_cost: dict[Configuration, list[float]] = {}

        new_cost: list[float] = self.runhistory.average_cost(config, config_isb_keys)  # type: ignore[assignment]
        if config not in self._old_config_cost:
            self._old_config_cost[config] = new_cost
            return False

        old_cost: list[float] = self._old_config_cost[config]  # type: ignore[assignment]
        if _dominates(new_cost, old_cost):
            self._old_config_cost[config] = new_cost
        return False


class DoublingNComparison(AbstractIntensifier):
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """Checks if the configuration should be evaluated against the incumbent while it
        did not run on all the trails the incumbents did. This function triggers after every n^2-1 of completed
        trails.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)

        nkeys = len(config_isb_keys)
        return (nkeys + 1) & nkeys == 0  # checks if nkeys+1 is a power of 2 (complies with the sequence (2**n)-1)


class DoublingNComparisonFour(AbstractIntensifier):
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """Checks if the configuration should be evaluated against the incumbent while it
        did not run on all the trails the incumbents did. This function triggers after every n^2-1 of completed
        trails. However, it enforces that at least 4 trails are completed to reduce outlier noise.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)

        max_trigger_number = int(np.ceil(np.log2(self._max_config_calls)))
        trigger_points = [(2**n) - 1 for n in range(2, max_trigger_number + 1)]  # 3, 7, 15, ...
        logger.debug(f"{trigger_points=}")
        logger.debug(f"{len(config_isb_keys)=}")
        return len(config_isb_keys) in trigger_points


class Always(AbstractIntensifier):
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """Checks if the configuration should be evaluated against the incumbent while it
        did not run on all the trails the incumbents did. This function always triggers for a check.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        return True


class Never(AbstractIntensifier):
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """Checks if the configuration should be evaluated against the incumbent while it
        did not run on all the trails the incumbents did. This function never triggers for a check.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        return False