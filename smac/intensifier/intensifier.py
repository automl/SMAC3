from __future__ import annotations

import numpy as np
from typing import Iterator

from ConfigSpace import Configuration

from smac.constants import MAXINT
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.runhistory import TrialInfo
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class Intensifier(AbstractIntensifier):
    def __init__(
        self,
        scenario: Scenario,
        min_config_calls: int = 1,
        max_config_calls: int = 2000,
        min_challenger: int = 2,
        intensify_percentage: float = 0.5,
        race_against: Configuration | None = None,
        seed: int | None = None,
    ):
        if scenario.deterministic:
            if min_challenger != 1:
                logger.info("The number of minimal challengers is set to one for deterministic algorithms.")

            min_challenger = 1

        super().__init__(scenario=scenario)

        # Internal variables
        self._max_config_calls = max_config_calls
        self._intensify_percentage = intensify_percentage

    def __iter__(self) -> Iterator[TrialInfo]:
        rh = self.runhistory
        old_incumbent: Configuration | None = None

        # TODO: How to handle this if we continue optimization run? Just start from scratch?
        rejected: list[Configuration] = []
        pending: list[tuple[Configuration, int, int]] = []  # Configuration, N, seed

        i = -1
        while True:
            i += 1
            # We select a new incumbent, which is based on the average of instance/seed pairs
            # A new incumbent is chosen only if the new configuration is evaluated as on all instance/seed pairs
            # the incumbent has
            incumbent = rh.get_incumbent()

            # Clear the pending queue if incumbent changed
            if old_incumbent != incumbent:
                old_incumbent = incumbent
                pending = []

            # We get ``self._max_config_calls`` == maxR trials to evaluate for the incumbent
            # First the instances are "filled-up" before a new seed is started
            # We don't yield the trials if the trials are marked as running in the runhistory
            initial_incumbent_trials = self._get_missing_trials(incumbent)
            for trial in initial_incumbent_trials:
                # Keep in mind: The generator keeps track of the state so the next time __next__ is called,
                # we start directly after the yield again
                yield trial

            # Get evaluated incumbent trials: Important to check if the challenger has the same number of evaluated ones
            evaluated_incumbent_trials = self._get_evaluated_trials(incumbent)

            # Percentage parameter: We decide whether to intensify or to evaluate a fresh configuration
            if i % 2 == 0:  # random.rand() < self._intensify_percentage:
                config = next(self.config_selector)
                N = 1

                # Seed is responsible for selecting random instance/seed pairs
                seed = int(self._rng.randint(low=0, high=MAXINT, size=1)[0])
            else:
                # Continue pending runs with the latest N
                if len(pending) > 0:
                    config, N, seed = pending.pop()
                else:
                    # If there are no pending configs, we select a [???] config from the runhistory?
                    # Which configuration should be intensified?
                    # - Random
                    # - We choose the configuration with the second lowest cost after incumbent?
                    # - We choose the configuration with the least trials?
                    # Also, how do we select the N? If we select N == 1, then the configuration goes
                    # into the pending queue and might be chosen the next iteration. Should work?
                    # However, if the configuration is already rejected or the incumbent or else,
                    # then the intensification in this iteration is basically skipped.

                    config = self._rng.choice(rh.get_configs())  # type: ignore
                    N = 1
                    seed = int(self._rng.randint(low=0, high=MAXINT, size=1)[0])

            # We don't want to evaluate a rejected configuration or an incumbent
            if config is None or config in rejected or config == incumbent:
                continue

            # Don't return missing trials if marked as ``RUNNING`` in the runhistory
            # Basically, trials which haven't been run yet
            initial_missing_trials = self._get_missing_trials(config, N, seed)
            for trial in initial_missing_trials:
                yield trial

            # Trials which are evaluated already
            evaluated_trials = self._get_evaluated_trials(config, N, seed)
            evaluated_isbk = [trial_info.get_instance_seed_budget_key() for trial_info in evaluated_trials]

            # We only go here if all trials have been evaluated
            if len(initial_missing_trials) == 0:
                # Now we have all trials evaluated and we can do a comparison
                config_cost = rh.average_cost(config, evaluated_isbk, normalize=True)
                incumbent_cost = rh.average_cost(incumbent, evaluated_isbk, normalize=True)
                assert type(config_cost) == float and type(incumbent_cost) == float

                if config_cost > incumbent_cost:
                    # New config is worse than incumbent so we reject the configuration forever
                    rejected.append(config)
                # If we evaluated as much trials as we evaluated the incumbent
                elif len(evaluated_trials) == len(evaluated_incumbent_trials):
                    # New configuration is the new incumbent
                    # However, since the incumbent is evaluated in each iteration, we skip it here
                    pass
                else:
                    # In the original paper, we would double N: In our case, we mark it as pending so it could
                    # be intensified in the next iteration.
                    pending.append((config, N * 2, seed))
            # Trials have not been evaluated yet
            else:
                # We append the current N to the pending, so in the next iteration we check again
                # if the trials have been evaluated
                pending.append((config, N, seed))

    def _get_trials_of_interest(self, config: Configuration, N: int | None = None, seed: int = 0) -> list[TrialInfo]:
        if N is None:
            N = self._max_config_calls

        # tf_seeds might include seeds specified by the user
        trials: list[TrialInfo] = []

        i = 0
        while len(trials) < self._max_config_calls:
            try:
                next_seed = self._tf_seeds[i]
            except IndexError:
                # Use global random generator for a new seed and mark it so it will be reused for another config
                next_seed = int(self._rng.randint(low=0, high=MAXINT, size=1)[0])
                self._tf_seeds.append(next_seed)

            # If no instances are used, tf_instances includes None
            for instance in self._tf_instances:
                trials.append(TrialInfo(config, instance=instance, seed=next_seed))

            # Only use one seed in deterministic case
            if self._scenario.deterministic:
                break

            # Seed counter
            i += 1

        # Now we cut so that we only have max_config_calls trials
        # We favor instances over seeds here: That makes sure we always work with the same instance/seed pairs
        if len(trials) > self._max_config_calls:
            trials = trials[: self._max_config_calls]

        # Now we shuffle the trials based on the seed
        rng = np.random.RandomState(seed)
        rng.shuffle(trials)  # type: ignore

        # Return only N configs
        if len(trials) > N:
            trials = trials[:N]

        return trials

    def _get_missing_trials(self, config: Configuration, N: int | None = None, seed: int = 0) -> list[TrialInfo]:
        """Returns unevaluated/not running trials for the trials of interest. Returns ``max_config_calls`` trials if
        ``N`` is None. Prioritizes instances/seeds found in the runhistory. Seed change happens after all instances are
        added first.
        """
        rh = self.runhistory
        trials_of_interest = self._get_trials_of_interest(config, N=N, seed=seed)

        # Now we actually have to check whether the trials have been evaluated already
        evaluated_isbk = rh.get_trials(config, only_max_observed_budget=False)
        for isbk in evaluated_isbk:
            trial = TrialInfo(config, instance=isbk.instance, seed=isbk.seed)
            if trial in trials_of_interest:
                trials_of_interest.remove(trial)

        # It's also important to remove running trials from the selection (we don't want to queue them again)
        for trial in rh.get_running_trials():
            if trial in trials_of_interest:
                trials_of_interest.remove(trial)

        return trials_of_interest

    def _get_evaluated_trials(self, config: Configuration, N: int | None = None, seed: int = 0) -> list[TrialInfo]:
        """Returns all evaluated trials from the trials of interest."""
        rh = self.runhistory
        trials_of_interest = self._get_trials_of_interest(config, N=N, seed=seed)
        trials: list[TrialInfo] = []

        # We iterate over the trials again
        evaluated_isbk = rh.get_trials(config, only_max_observed_budget=False)
        for isbk in evaluated_isbk:
            trial = TrialInfo(config, instance=isbk.instance, seed=isbk.seed)

            if trial in trials_of_interest:
                trials.append(trial)

        return trials
