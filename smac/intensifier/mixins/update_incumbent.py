from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Iterator

import copy
import dataclasses
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration
from scipy.stats import binom

import smac
from smac.callback import Callback
from smac.constants import MAXINT
from smac.main.config_selector import ConfigSelector
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import (
    InstanceSeedBudgetKey,
    InstanceSeedKey,
    TrajectoryItem,
    TrialValue,
)
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash, print_config_changes
from smac.utils.logging import get_logger
from smac.utils.pareto_front import (
    _get_costs,
    calculate_pareto_front,
    sort_by_crowding_distance,
)

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class DebugUpdate(object):
    def _register_incumbent_update(self, **kwargs):
        if not hasattr(self, "_update_incumbent_log"):
            self._update_incumbent_log = []
        self._update_incumbent_log.append(kwargs)


class NonDominatedUpdate(DebugUpdate):
    def _update_incumbent(self, config: Configuration) -> list[Configuration]:
        """Updates the incumbent with the config (which can be the challenger)

        Parameters
        ----------
        config: Configuration

        Returns
        -------
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

        self._register_incumbent_update(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=isb_keys,
            new_incumbents=new_incumbents,
            name="NonDominated",
        )

        return new_incumbents


class BootstrapUpdate(DebugUpdate):
    def _update_incumbent(self, config: Configuration) -> list[Configuration]:
        """Updates the incumbent with the config (which can be the challenger)

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        """
        rh = self.runhistory

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

        self._register_incumbent_update(
            config=config,
            incumbent=self.get_incumbents(),
            isb_keys=isb_keys,
            new_incumbents=new_incumbents,
            name="Bootstrap",
            probabilities=probabilities,
            n_samples=n_samples,
        )

        return new_incumbents
