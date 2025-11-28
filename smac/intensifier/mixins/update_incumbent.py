from __future__ import annotations

from typing import Any

import itertools

import numpy as np
from ConfigSpace import Configuration

from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class NonDominatedUpdate(AbstractIntensifier):
    def _update_incumbent(self, config: Configuration) -> list[Configuration]:
        """Updates the incumbent with the config (which can be the challenger). By default the configuration is added to
        the incumbents after which they are filtered based on Pareto dominance.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        list[Configuration]
            New incumbents after update.
        """
        rh = self.runhistory

        incumbents = self.get_incumbents()

        if config not in incumbents:
            incumbents.append(config)

        isb_keys = self.get_incumbent_instance_seed_budget_keys(compare=True)
        all_incumbent_isb_keys = [isb_keys for _ in range(len(incumbents))]

        # We compare the incumbents now and only return the ones on the Pareto front
        # _calculate_pareto_front returns only non-dominated points
        new_incumbents = self._calculate_pareto_front(rh, incumbents, all_incumbent_isb_keys)

        return new_incumbents


class BootstrapUpdate(AbstractIntensifier):
    def _update_incumbent(self, config: Configuration) -> list[Configuration]:
        """Updates the incumbent with the config (which can be the challenger). This function performance bootstrap
        resampling over the ISB keys and then checks for the dominance relations in each of the bootstrap resamples.
        In case a configuration is non-dominated in 50% of the resamples, it will be considered as an incumbent.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        list[Configuration]
            New incumbents after update.
        """
        incumbents = self.get_incumbents()

        if config not in incumbents:
            incumbents.append(config)

        isb_keys = self.get_incumbent_instance_seed_budget_keys(compare=True)

        n_samples = 1000
        if len(isb_keys) < 7:  # When there are only a limited number of trials available we run all combinations
            samples = list(itertools.combinations_with_replacement(list(range(len(isb_keys))), r=len(isb_keys)))
            n_samples = len(samples)
        else:
            samples = np.random.choice(len(isb_keys), (n_samples, len(isb_keys)), replace=True)

        verdicts = np.zeros((n_samples, len(incumbents)), dtype=bool)

        for sid, sample in enumerate(samples):
            sample_isb_keys = [isb_keys[i] for i in sample]
            all_incumbent_isb_keys = [sample_isb_keys] * len(incumbents)
            new_incumbents = self._calculate_pareto_front(self.runhistory, incumbents, all_incumbent_isb_keys)

            verdicts[sid, :] = [incumbents[i] in new_incumbents for i in range(len(incumbents))]

        probabilities = np.count_nonzero(verdicts, axis=0) / n_samples

        new_incumbent_ids = np.argwhere(
            probabilities >= 0.5
        ).flatten()  # Incumbent needs to be non-dominated at least 50% of the time
        new_incumbents = [incumbents[i] for i in new_incumbent_ids]

        return new_incumbents
