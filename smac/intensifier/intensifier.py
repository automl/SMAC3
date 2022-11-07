from __future__ import annotations

import numpy as np
from typing import Iterator

from ConfigSpace import Configuration

from smac.constants import MAXINT
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import InstanceSeedKey
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class Intensifier(AbstractIntensifier):
    def __init__(
        self,
        scenario: Scenario,
        max_config_calls: int = 2000,
        retries: int = 10,  # Number of iterations to retry if no next trial can be found
        seed: int | None = None,
    ):
        super().__init__(scenario=scenario, seed=seed)

        # Internal variables
        self._max_config_calls = max_config_calls
        self._instance_seed_pairs: list[InstanceSeedKey] | None = None
        self._instance_seed_pairs_validation: list[InstanceSeedKey] | None = None

    @property
    def uses_seeds(self) -> bool:
        return True

    @property
    def uses_budgets(self) -> bool:
        return False

    @property
    def uses_instances(self) -> bool:
        if self._scenario.instances is None:
            return False

        return True

    def __iter__(self) -> Iterator[TrialInfo]:
        rh = self.runhistory

        i = -1
        while True:
            i += 1

            # Incumbents are purely selected by the average cost for each objective (no matter how many trials have
            # been evaluated)
            # Reason: If config with low amount of trials was chosen, it will be evaluated on more trials.
            # If it then performs worse than the previous incumbent, the previous incumbnet becomes incumbent again.
            incumbents = rh.get_incumbents(sort_by="num_trials")

            # Now we get the config with the least number of trials to make it more trustworthy
            for incumbent in incumbents:
                trial = self._get_next_trial(incumbent)
                if trial is not None:
                    yield trial

                    # We break here because we only want to intensify one incumbent at a time
                    # Intensifying all incumbents on all trials would produce a strong overhead
                    break

            # We sample a new configuration
            if i % 2 == 0:
                config = next(self.config_generator)
                trial = self._get_next_trial(config)
                if trial is not None:
                    yield trial
            # A configuration might be not lucky enough to get evaluated on a good performing trial
            # Hence, we select a random configuration and give it another shot to become an incumbent
            else:
                configs = rh.get_configs(sort_by="num_trials")

                # Get rid of incumbents
                for config in incumbents:
                    try:
                        configs.remove(config)
                    except ValueError:
                        pass

                for config in configs:
                    trial = self._get_next_trial(incumbent)
                    if trial is not None:
                        yield trial
                        break

    def get_trials_of_interest(
        self,
        config: Configuration,
        *,
        validate: bool = False,
    ) -> list[TrialInfo]:
        """Returns a list of trials of interest for a given configuration."""
        if (self._instance_seed_pairs is None and not validate) or (
            self._instance_seed_pairs_validation is None and validate
        ):
            instance_seed_pairs: list[InstanceSeedKey] = []
            if validate:
                rng = np.random.RandomState(9999)
            else:
                rng = self._rng

            i = 0
            while len(instance_seed_pairs) < self._max_config_calls:
                if validate:
                    next_seed = int(rng.randint(low=0, high=MAXINT, size=1)[0])
                else:
                    try:
                        next_seed = self._tf_seeds[i]
                    except IndexError:
                        # Use global random generator for a new seed and mark it so it will be reused for another config
                        next_seed = int(rng.randint(low=0, high=MAXINT, size=1)[0])
                        self._tf_seeds.append(next_seed)

                # If no instances are used, tf_instances includes None
                for instance in self._tf_instances:
                    instance_seed_pairs.append(InstanceSeedKey(instance, next_seed))

                # Only use one seed in deterministic case
                if self._scenario.deterministic:
                    break

                # Seed counter
                i += 1

            # Now we cut so that we only have max_config_calls instance_seed_pairs
            # We favor instances over seeds here: That makes sure we always work with the same instance/seed pairs
            if len(instance_seed_pairs) > self._max_config_calls:
                instance_seed_pairs = instance_seed_pairs[: self._max_config_calls]

            # Set it globally
            if not validate:
                self._instance_seed_pairs = instance_seed_pairs
            else:
                self._instance_seed_pairs_validation = instance_seed_pairs

        if not validate:
            assert self._instance_seed_pairs is not None
            instance_seed_pairs = self._instance_seed_pairs
        else:
            assert self._instance_seed_pairs_validation is not None
            instance_seed_pairs = self._instance_seed_pairs_validation

        trials: list[TrialInfo] = []
        for instance_seed in instance_seed_pairs:
            trials.append(TrialInfo(config=config, instance=instance_seed.instance, seed=instance_seed.seed))

        return trials

    def _get_next_trial(self, config: Configuration) -> TrialInfo | None:
        rh = self.runhistory
        trials = self.get_trials_of_interest(config)

        # Now we actually have to check whether the trials have been evaluated already
        evaluated_isbk = rh.get_trials(config, only_max_observed_budget=False)
        for isbk in evaluated_isbk:
            trial = TrialInfo(config, instance=isbk.instance, seed=isbk.seed)
            if trial in trials:
                trials.remove(trial)

        # It's also important to remove running trials from the selection (we don't want to queue them again)
        for trial in rh.get_running_trials():
            if trial in trials:
                trials.remove(trial)

        # Now we shuffle the trials
        # TODO: Shuffle in groups (first all instances, then all seeds)
        self._rng.shuffle(trials)  # type: ignore

        try:
            return trials[0]
        except IndexError:
            return None
