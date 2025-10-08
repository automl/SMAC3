from __future__ import annotations

from typing import Any

import itertools

import numpy as np
from ConfigSpace import Configuration
from scipy.stats import binom

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


class RandomComparison(DebugComparison):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration against the incumbent

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
        verdict = self._rng.random() >= 0.5
        self._register_comparison(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=len(config_isb_keys),
            costs=self._get_costs_comp(config),
            prediction=verdict,
            name="Random",
        )
        return verdict


class NoComparison(DebugComparison):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration against the incumbent

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


class BootstrapComparison(DebugComparison):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration by generating bootstraps

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

        if config not in incumbents:
            incumbents.append(config)

        n_samples = 1000
        if len(config_isb_keys) < 7:  # When there are only a limited number of trials available we run all combinations
            samples = list(
                itertools.combinations_with_replacement(list(range(len(config_isb_keys))), r=len(config_isb_keys))
            )
            n_samples = len(samples)
        else:
            samples = np.random.choice(len(config_isb_keys), (n_samples, len(config_isb_keys)), replace=True)

        verdicts = np.zeros(n_samples, dtype=bool)

        for sid, sample in enumerate(samples):
            sample_isb_keys = [config_isb_keys[i] for i in sample]
            all_incumbent_isb_keys = [sample_isb_keys] * len(incumbents)
            new_incumbents = self._calculate_pareto_front(self.runhistory, incumbents, all_incumbent_isb_keys)

            verdicts[sid] = config in new_incumbents

        verdict = (
            np.count_nonzero(verdicts) >= 0.5 * n_samples
        )  # The config is in more than 50% of the times non-dominated
        # P = np.count_nonzero(verdicts)/n_samples
        # print(f"P = {np.count_nonzero(verdicts)}/{n_samples}={P:.2f}")
        self._register_comparison(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=len(config_isb_keys),
            costs=self._get_costs_comp(config),
            prediction=verdict,
            name="Bootstrap",
            probability=np.count_nonzero(verdicts) / n_samples,
            n_samples=n_samples,
        )
        return verdict


class BootstrapSingleComparison(DebugComparison):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration by generating bootstraps

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

        iid = self._rng.choice(len(incumbents))
        incumbents = [incumbents[iid], config]

        n_samples = 1000
        if len(config_isb_keys) < 7:  # When there are only a limited number of trials available we run all combinations
            samples = list(
                itertools.combinations_with_replacement(list(range(len(config_isb_keys))), r=len(config_isb_keys))
            )
            n_samples = len(samples)
        else:
            samples = np.random.choice(len(config_isb_keys), (n_samples, len(config_isb_keys)), replace=True)

        verdicts = np.zeros(n_samples, dtype=bool)

        for sid, sample in enumerate(samples):
            sample_isb_keys = [config_isb_keys[i] for i in sample]
            all_incumbent_isb_keys = [sample_isb_keys] * len(incumbents)
            new_incumbents = self._calculate_pareto_front(self.runhistory, incumbents, all_incumbent_isb_keys)

            verdicts[sid] = config in new_incumbents

        verdict = (
            np.count_nonzero(verdicts) >= 0.5 * n_samples
        )  # The config is in more than 50% of the times non-dominated
        # P = np.count_nonzero(verdicts)/n_samples
        # print(f"P = {np.count_nonzero(verdicts)}/{n_samples}={P:.2f}")
        self._register_comparison(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=len(config_isb_keys),
            costs=self._get_costs_comp(config),
            prediction=verdict,
            name="BootstrapSingle",
            probability=np.count_nonzero(verdicts) / n_samples,
            n_samples=n_samples,
        )
        return bool(verdict)


class BootstrapClosestComparison(DebugComparison):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration by generating bootstraps

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

        inc_costs = _get_costs(self.runhistory, incumbents, [config_isb_keys for _ in incumbents], normalize=True)
        conf_cost = _get_costs(self.runhistory, [config], [config_isb_keys], normalize=True)[0]
        distances = [np.linalg.norm(inc_cost - conf_cost) for inc_cost in inc_costs]
        iid = np.argmin(distances)
        incumbents = [incumbents[iid], config]

        n_samples = 1000
        if len(config_isb_keys) < 7:  # When there are only a limited number of trials available we run all combinations
            samples = list(
                itertools.combinations_with_replacement(list(range(len(config_isb_keys))), r=len(config_isb_keys))
            )
            n_samples = len(samples)
        else:
            samples = np.random.choice(len(config_isb_keys), (n_samples, len(config_isb_keys)), replace=True)

        verdicts = np.zeros(n_samples, dtype=bool)

        for sid, sample in enumerate(samples):
            sample_isb_keys = [config_isb_keys[i] for i in sample]
            all_incumbent_isb_keys = [sample_isb_keys] * len(incumbents)
            new_incumbents = self._calculate_pareto_front(self.runhistory, incumbents, all_incumbent_isb_keys)

            verdicts[sid] = config in new_incumbents

        verdict = (
            np.count_nonzero(verdicts) >= 0.5 * n_samples
        )  # The config is in more than 50% of the times non-dominated
        # P = np.count_nonzero(verdicts)/n_samples
        # print(f"P = {np.count_nonzero(verdicts)}/{n_samples}={P:.2f}")
        self._register_comparison(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=len(config_isb_keys),
            costs=self._get_costs_comp(config),
            prediction=verdict,
            name="BootstrapClosest",
            probability=np.count_nonzero(verdicts) / n_samples,
            n_samples=n_samples,
        )
        return verdict


class SRaceComparison(DebugComparison):
    def _intermediate_comparison(self, config: Configuration) -> bool:
        """Compares the configuration by generating bootstraps

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        A boolean which indicates if we should continue with this configuration.
        """

        def get_alpha(delta: float, n_instances: int) -> float:
            steps = 0
            n = 1
            inst = 0
            while inst < n_instances:
                steps += 1
                inst += n
                n *= 2

            return (1 - delta) / (n_instances) * (steps - 1)

        def dominates(a: list[float], b: list[float]) -> int:
            # Checks if a dominates b
            a = np.array(a)
            b = np.array(b)
            return 1 if np.count_nonzero(a <= b) >= len(a) and np.count_nonzero(a < b) >= 1 else 0

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

        p_values = []
        chall_perf: list[list[float]] = self.runhistory._cost(config, config_isb_keys)  # type: ignore[assignment]
        for incumbent in incumbents:
            inc_perf: list[list[float]] = self.runhistory._cost(incumbent, config_isb_keys)  # type: ignore[assignment]
            n_ij = sum(
                [dominates(_chall_perf, _inc_perf) for _chall_perf, _inc_perf in zip(chall_perf, inc_perf)]
            )  # Number of times the incumbent candidate dominates the challenger
            n_ji = sum(
                [dominates(_chall_perf, _inc_perf) for _chall_perf, _inc_perf in zip(chall_perf, inc_perf)]
            )  # Number of times the challenger dominates the incumbent candidate
            p_value = 1 - binom.cdf(n_ij - 1, n_ij + n_ji, 0.5)
            p_values.append(p_value)

        pvalues_order = np.argsort(p_values)

        # Holm-Bonferroni
        reject = np.zeros(len(p_values), dtype=bool)  # Do not reject any test by default
        alpha = get_alpha(0.05, len(config_isb_keys))
        for i, index in enumerate(pvalues_order):
            corrected_alpha = alpha / (len(p_values) - i)  # Holm-Bonferroni
            if pvalues_order[index] < corrected_alpha:
                # Reject H0 -> winner > candidate
                reject[index] = True
            else:
                break

        verdict = np.count_nonzero(reject) != 0
        # P = np.count_nonzero(verdicts)/n_samples
        # print(f"P = {np.count_nonzero(verdicts)}/{n_samples}={P:.2f}")
        self._register_comparison(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=len(config_isb_keys),
            costs={conf: cost for conf, cost in zip(incumbents, costs)},
            prediction=verdict,
            name="S-Race",
        )
        return verdict
