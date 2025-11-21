from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import Configuration

from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.utils.configspace import get_config_hash
from smac.utils.logging import get_logger
from smac.utils.pareto_front import _get_costs

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)

class FullIncumbentComparison(AbstractIntensifier):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """ompares the configuration against the incumbent when the configuration did not run on all the trails the
        incumbent did. By default it checks if the performance of configuration is better than the incumbent on the
        trials it completed on so far. In case of multiple incumbents, which occurs with a multi-objetive scenario, one
        all incumbents are considered after which the comparison is made.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A Boolean which indicates if we should continue with this configuration.
        """
        config_hash = get_config_hash(config)
        incumbents = self.get_incumbents()
        config_isb_keys = self.get_instance_seed_budget_keys(config, compare=True)
        incumbent_isb_comparison_keys = self.get_incumbent_instance_seed_budget_keys(compare=True)

        logger.debug(f"Perform intermediate comparions of config {config_hash} with incumbents to see if it is worse")
        # Check if the incumbents ran on all the ones of this config
        if not all([key in incumbent_isb_comparison_keys for key in config_isb_keys]):
            logger.debug("Config ran on other isb_keys than the incumbents. Should not happen.")
            return True

        # Ensure that the config is not part of the incumbent
        if config in incumbents:
            return True

        # Only compare domination between one incumbent (as relaxation measure)
        if config not in incumbents:
            incumbents.append(config)

        # Only the trials of the challenger
        all_incumbent_isb_keys = [config_isb_keys for _ in incumbents]
        new_incumbents = self._calculate_pareto_front(self.runhistory, incumbents, all_incumbent_isb_keys)

        return config in new_incumbents


class SingleIncumbentComparison(AbstractIntensifier):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration against the incumbent when the configuration did not run on all the trails the
        incumbent did. By default it checks if the performance of configuration is better than the incumbent on the
        trials it completed on so far. In case of multiple incumbents, which occurs with a multi-objetive scenario, one
        random incumbent is sampled after which the comparison is made.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A Boolean which indicates if we should continue with this configuration.
        """
        config_hash = get_config_hash(config)
        incumbents = self.get_incumbents()
        config_isb_keys = self.get_instance_seed_budget_keys(config, compare=True)
        incumbent_isb_comparison_keys = self.get_incumbent_instance_seed_budget_keys(compare=True)

        logger.debug(f"Perform intermediate comparions of config {config_hash} with incumbents to see if it is worse")

        # Check if the incumbents ran on all the ones of this config
        if not all([key in incumbent_isb_comparison_keys for key in config_isb_keys]):
            logger.debug("Config ran on other isb_keys than the incumbents. Should not happen.")
            return True # Continue

        # Ensure that the config is not part of the incumbent
        if config in incumbents:
            return True # Continue

        # Only compare domination between one incumbent (as relaxation measure)
        iid = self._rng.choice(len(incumbents))
        incumbents = [incumbents[iid], config]

        # Only the trials of the challenger
        all_incumbent_isb_keys = [config_isb_keys for _ in incumbents]
        new_incumbents = self._calculate_pareto_front(self.runhistory, incumbents, all_incumbent_isb_keys)

        return config in new_incumbents  #if False -> reject the configuration


class ClosestIncumbentComparison(AbstractIntensifier):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration against the incumbent when the configuration did not run on all the trails the
        incumbent did. By default it checks if the performance of configuration is better than the incumbent on the
        trials it completed on so far. In case of multiple incumbents, which occurs with a multi-objetive scenario, one
        incumbent that is closest in the objective space is chosen after which the comparison is made.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A Boolean which indicates if we should continue with this configuration.
        """
        config_hash = get_config_hash(config)
        incumbents = self.get_incumbents()
        config_isb_keys = self.get_instance_seed_budget_keys(config, compare=True)
        incumbent_isb_comparison_keys = self.get_incumbent_instance_seed_budget_keys(compare=True)

        logger.debug(f"Perform intermediate comparisons of config {config_hash} with incumbents to see if it is worse")

        # Ensure that the config is not part of the incumbent
        if config in incumbents:
            return True

        # Check if the incumbents ran on all the ones of this config
        if not all([key in incumbent_isb_comparison_keys for key in config_isb_keys]):
            logger.debug("Config ran on other isb_keys than the incumbents. Should not happen.")
            return True

        # Only compare domination between one incumbent (as relaxation measure)
        inc_costs = _get_costs(self.runhistory, incumbents, [config_isb_keys for _ in incumbents], normalize=True)
        conf_cost = _get_costs(self.runhistory, [config], [config_isb_keys], normalize=True)[0]
        distances = [np.linalg.norm(inc_cost - conf_cost) for inc_cost in inc_costs]
        iid = np.argmin(distances)
        incumbents = [incumbents[iid], config]

        # Only the trials of the challenger
        all_incumbent_isb_keys = [config_isb_keys for _ in incumbents]

        new_incumbents = self._calculate_pareto_front(self.runhistory, incumbents, all_incumbent_isb_keys)

        return config in new_incumbents


class NoComparison(AbstractIntensifier):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Does not perform an intermediate comparison. Is used in later developed facades.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A Boolean which indicates if we should continue with this configuration.
        """
        return True
