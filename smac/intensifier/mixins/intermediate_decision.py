from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration

from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


def _dominates(a, b) -> bool:
    # Checks if a dominates b
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    return np.count_nonzero(a <= b) >= len(a) and np.count_nonzero(a < b) >= 1


class NewCostDominatesOldCost:
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)

        if not hasattr(self, "_old_config_cost"):
            self._old_config_cost = {}  # TODO remove configuration when done

        new_cost = self.runhistory.average_cost(config, config_isb_keys)
        if config not in self._old_config_cost:
            self._old_config_cost[config] = new_cost
            return True

        old_cost = self._old_config_cost[config]
        if _dominates(new_cost, old_cost):
            self._old_config_cost[config] = new_cost
            return True
        return False


class NewCostDominatesOldCostSkipFirst:
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """Do the first comparison with the incumbent when the configuration dominates the cost after finishing
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
            self._old_config_cost = {}  # TODO remove configuration when done

        new_cost = self.runhistory.average_cost(config, config_isb_keys)
        if config not in self._old_config_cost:
            self._old_config_cost[config] = new_cost
            return False

        old_cost = self._old_config_cost[config]
        if _dominates(new_cost, old_cost):
            self._old_config_cost[config] = new_cost
            return True
        return False


class DoublingNComparison:
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)

        # max_trigger_number = int(np.ceil(np.log2(self._max_config_calls)))
        # trigger_points = [(2**n) - 1 for n in range(1, max_trigger_number + 1)]  # 1, 3, 7, 15, ...
        # logger.debug(f"{trigger_points=}")
        # logger.debug(f"{len(config_isb_keys)=}")
        # return len(config_isb_keys) in trigger_points

        nkeys = len(config_isb_keys)
        return (nkeys + 1) & nkeys == 0  # checks if nkeys+1 is a power of 2 (complies with the sequence (2**n)-1)


class Always:
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        return True


class Never:
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        return False


class DoublingNComparisonFour:
    def _check_for_intermediate_comparison(self, config: Configuration) -> bool:
        """

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which decides if the current configuration should be compared against the incumbent.
        """
        config_isb_keys = self.get_instance_seed_budget_keys(config)

        max_trigger_number = int(np.ceil(np.log2(self._max_config_calls)))
        trigger_points = [(2**n) - 1 for n in range(2, max_trigger_number + 1)]  # 1, 3, 7, 15, ...
        logger.debug(f"{trigger_points=}")
        logger.debug(f"{len(config_isb_keys)=}")
        return len(config_isb_keys) in trigger_points
