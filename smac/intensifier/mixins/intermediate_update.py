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


class DebugComparison(AbstractIntensifier):
    def _register_comparison(self, **kwargs: Any) -> None:
        logger.debug(f"Made intermediate comparison with {kwargs['name']} comparison ")
        if not hasattr(self, "_intermediate_comparisons_log"):
            self._intermediate_comparisons_log = []
        self._intermediate_comparisons_log.append(kwargs)

    def _get_costs_comp(self, config: Configuration) -> dict:
        incumbents = self.get_incumbents()
        if config not in incumbents:
            incumbents.append(config)
        config_isb_keys = self.get_instance_seed_budget_keys(config, compare=True)
        all_incumbent_isb_keys = [config_isb_keys for _ in incumbents]
        costs = _get_costs(self.runhistory, incumbents, all_incumbent_isb_keys)

        return {conf: cost for conf, cost in zip(incumbents, costs)}


class FullIncumbentComparison(DebugComparison):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration against the incumbent

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which indicates if we should continue with this configuration.
        """
        config_hash = get_config_hash(config)
        incumbents = self.get_incumbents()
        config_isb_keys = self.get_instance_seed_budget_keys(config, compare=True)
        incumbent_isb_comparison_keys = self.get_incumbent_instance_seed_budget_keys(compare=True)

        logger.debug(f"Perform intermediate comparions of config {config_hash} with incumbents to see if it is worse")
        # TODO perform comparison with incumbent on current instances.
        # Check if the config with these number of trials is part of the Pareto front

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

        verdict = config in new_incumbents
        self._register_comparison(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=len(config_isb_keys),
            costs=self._get_costs_comp(config),
            prediction=verdict,
            name="FullInc",
        )

        return config in new_incumbents


class SingleIncumbentComparison(DebugComparison):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration against the incumbent

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which indicates if we should continue with this configuration.
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
        iid = self._rng.choice(len(incumbents))
        incumbents = [incumbents[iid], config]

        # Only the trials of the challenger
        all_incumbent_isb_keys = [config_isb_keys for _ in incumbents]

        new_incumbents = self._calculate_pareto_front(self.runhistory, incumbents, all_incumbent_isb_keys)

        verdict = config in new_incumbents
        self._register_comparison(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=len(config_isb_keys),
            costs=self._get_costs_comp(config),
            prediction=verdict,
            name="SingleInc",
        )

        return config in new_incumbents


class ClosestIncumbentComparison(DebugComparison):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration against the incumbent

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which indicates if we should continue with this configuration.
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
        # iid = self._rng.choice(len(incumbents))
        # TODO Normalize to determine closests?
        inc_costs = _get_costs(self.runhistory, incumbents, [config_isb_keys for _ in incumbents], normalize=True)
        conf_cost = _get_costs(self.runhistory, [config], [config_isb_keys], normalize=True)[0]
        distances = [np.linalg.norm(inc_cost - conf_cost) for inc_cost in inc_costs]
        iid = np.argmin(distances)
        incumbents = [incumbents[iid], config]

        # Only the trials of the challenger
        all_incumbent_isb_keys = [config_isb_keys for _ in incumbents]

        new_incumbents = self._calculate_pareto_front(self.runhistory, incumbents, all_incumbent_isb_keys)

        verdict = config in new_incumbents
        self._register_comparison(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=len(config_isb_keys),
            costs=self._get_costs_comp(config),
            prediction=verdict,
            name="ClosestInc",
        )

        return config in new_incumbents


# class RandomComparison(DebugComparison):
#     def _intermediate_comparison(self, config: Configuration) -> bool:
#         """Compares the configuration against the incumbent
#
#         Parameters
#         ----------
#         config: Configuration
#
#         Returns
#         -------
#         A boolean which indicates if we should continue with this configuration.
#         """
#         incumbents = self.get_incumbents()
#         config_isb_keys = self.get_instance_seed_budget_keys(config, compare=True)
#         incumbent_isb_comparison_keys = self.get_incumbent_instance_seed_budget_keys(compare=True)
#
#         # Check if the incumbents ran on all the ones of this config
#         if not all([key in incumbent_isb_comparison_keys for key in config_isb_keys]):
#             logger.debug("Config ran on other isb_keys than the incumbents. Should not happen.")
#             return True
#
#         # Ensure that the config is not part of the incumbent
#         if config in incumbents:
#             return True
#
#         config_isb_keys = self.get_instance_seed_budget_keys(config, compare=True)
#         verdict = self._rng.random() >= 0.5
#         self._register_comparison(
#             config=config,
#             incumbent=self.get_incumbents(),
#             isb_keys=len(config_isb_keys),
#             costs=self._get_costs_comp(config),
#             prediction=verdict,
#             name="Random",
#         )
#         return verdict


class NoComparison(DebugComparison):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Does not perform an intermediate comparison. Is used in later developed facades.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which indicates if we should continue with this configuration.
        """
        incumbents = self.get_incumbents()
        config_isb_keys = self.get_instance_seed_budget_keys(config, compare=True)
        incumbent_isb_comparison_keys = self.get_incumbent_instance_seed_budget_keys(compare=True)

        # Check if the incumbents ran on all the ones of this config
        if not all([key in incumbent_isb_comparison_keys for key in config_isb_keys]):
            logger.debug("Config ran on other isb_keys than the incumbents. Should not happen.")
            return True

        # Ensure that the config is not part of the incumbent
        if config in incumbents:
            return True

        config_isb_keys = self.get_instance_seed_budget_keys(config, compare=True)
        verdict = True
        self._register_comparison(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=len(config_isb_keys),
            costs=self._get_costs_comp(config),
            prediction=verdict,
            name="NoComp",
        )
        return verdict
