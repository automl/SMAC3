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
        intensify_percentage: float = 0.5,
        # if no incumbent is available, the intensifier choses this configuration or the default configspace if none
        # race_against: Configuration | None = None,
        incumbent_configuration: Configuration | None = None,
        retries: int = 10,  # Number of iterations to retry if no next trial can be found
        seed: int | None = None,
    ):
        super().__init__(scenario=scenario, seed=seed)

        if incumbent_configuration is None:
            incumbent_configuration = scenario.configspace.get_default_configuration()

        self._incumbent_configuration = incumbent_configuration

        # Internal variables
        self._min_config_calls = min_config_calls
        self._max_config_calls = max_config_calls
        self._intensify_percentage = intensify_percentage
        self._retries = retries
        self._rejected: list[Configuration] = []
        self._pending: list[tuple[Configuration, int, int]] = []  # Configuration, N, seed

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
        old_incumbent: Configuration | None = None

        # TODO: How to handle this if we continue optimization run? Just start from scratch?
        # We make rejected and pending global so we can better test the behavior
        self._rejected = []
        self._pending = []  # Configuration, N, seed

        fails = 0
        while True:
            # We stop if we have not found new trials for a while
            if fails >= self._retries:
                logger.error(f"Could not find new trials for {fails} iterations.")
                return

            # We select a new incumbent, which is based on the average of instance/seed pairs
            # A new incumbent is chosen only if the new configuration is evaluated as on all instance/seed pairs
            # the incumbent has
            # TODO: How to deal with parego and rejection?
            # Menge von incumbents: was kommt rein und was fliegt raus
            incumbent, incumbent_cost = rh.get_incumbent()

            # Select incumbent configuration if no incumbent is available
            if incumbent is None:
                incumbent = self._incumbent_configuration

            # Clear the pending queue if incumbent changed
            if old_incumbent != incumbent:
                trials = rh.get_trials(incumbent, only_max_observed_budget=False)
                logger.info(f"Incumbent changed with estimated cost {incumbent_cost} on {len(trials)} trials.")
                self.print_config_changes(old_incumbent, incumbent)
                logger.debug("Clearing pending queue.")

                old_incumbent = incumbent
                self._pending = []

            # We get ``self._max_config_calls`` == maxR trials to evaluate for the incumbent
            # First the instances are "filled-up" before a new seed is started
            # We don't yield the trials if the trials are marked as running in the runhistory
            incumbent_missing_trials = self._get_missing_trials(incumbent)
            for trial in incumbent_missing_trials:
                # Keep in mind: The generator keeps track of the state so the next time __next__ is called,
                # we start directly after the yield again
                yield trial

            # We also need to remember how many trials we need
            # TODO: Caching!
            incumbent_trials_of_interest = self.get_trials_of_interest(incumbent)

            # Percentage parameter: We decide whether to intensify or to evaluate a fresh configuration
            if self._rng.rand() > self._intensify_percentage:
                logger.debug("Get next (unseen) configuration.")
                # TODO: Werden running configs im surrogate berücksichtigt? Keine imputation?
                # Mean vom surrogate model nehmen und als halloziniert wert nehmen
                config = next(self.config_generator)
                N = self._min_config_calls  # or 1

                # Seed is responsible for selecting random instance/seed pairs
                seed = int(self._rng.randint(low=0, high=MAXINT, size=1)[0])
            else:
                # Continue pending runs with the latest N
                if len(self._pending) > 0:
                    logger.debug("Selecting config from pending queue.")
                    config, N, seed = self._pending.pop()
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
                    logger.debug("Selecting random config from runhistory.")
                    config = None
                    previous_configs = rh.get_configs()

                    # TODO: Eine finden die nicht rejected ist? -> Overhead
                    # TODO: Nicht für immer rejected: wann ist es nicht mehr rejected?

                    if len(previous_configs) > 0:
                        config_idx = int(self._rng.randint(low=0, high=len(previous_configs), size=1)[0])
                        config = previous_configs[config_idx]
                        N = self._min_config_calls  # or 1
                        seed = int(self._rng.randint(low=0, high=MAXINT, size=1)[0])
                    else:
                        logger.debug("No configurations in runhistory. Do you mark runs as running?")

            # We don't want to evaluate a rejected configuration or an incumbent
            if config is None:
                logger.debug("Skipping intensify config because it is None.")
                fails += 1
                continue

            if config in self._rejected:
                logger.debug("Skipping intensify config because it was rejected before.")
                fails += 1
                continue

            if config == incumbent:
                logger.debug("Skipping intensify config because it is the incumbent.")
                fails += 1
                continue

            # Don't return missing trials if marked as ``RUNNING`` in the runhistory
            # Basically, trials which haven't been run yet
            missing_trials = self._get_missing_trials(config, N, seed)
            for trial in missing_trials:
                fails = 0  # Reset fail counter
                yield trial

            # Trials which are evaluated already
            evaluated_trials = self._get_evaluated_trials(config, N, seed)
            evaluated_isbk = [trial_info.get_instance_seed_budget_key() for trial_info in evaluated_trials]

            # In every iteration we deal with N different trials
            # We need the trials of interest to decide whether we can complete this iteration and increase N
            trials_of_interest = self.get_trials_of_interest(config, N=N, seed=seed)

            # Get evaluated incumbent trials: Important to check if the challenger has the same number of evaluated ones
            incumbent_evaluated_trials = self._get_evaluated_trials(incumbent, N, seed)

            logger.debug(f"-- Config has {len(evaluated_trials)}/{len(trials_of_interest)} evaluated trials.")
            logger.debug(
                f"-- Incumbent has {len(incumbent_evaluated_trials)}/{len(trials_of_interest)} evaluated trials."
            )

            # We only go here if all necessary trials have been evaluated
            if len(evaluated_trials) == len(incumbent_evaluated_trials) == len(trials_of_interest):
                logger.debug("Intensify ...")
                fails = 0  # Reset fail counter

                # Now we have all trials evaluated and we can do a comparison
                config_cost = rh.average_cost(config, evaluated_isbk, normalize=True)
                incumbent_cost = rh.average_cost(incumbent, evaluated_isbk, normalize=True)
                assert type(config_cost) == float and type(incumbent_cost) == float

                if config_cost > incumbent_cost:
                    # New config is worse than incumbent so we reject the configuration forever
                    logger.debug(
                        f"Rejecting config because it is worse than incumbent on {len(evaluated_isbk)} trials."
                    )
                    self._rejected.append(config)
                # If we evaluated as much trials as we evaluated the incumbent
                elif len(trials_of_interest) == len(incumbent_trials_of_interest):
                    # New configuration is the new incumbent
                    # However, since the incumbent is evaluated in each iteration, we skip it here
                    pass
                else:
                    # In the original paper, we would double N: In our case, we mark it as pending so it could
                    # be intensified in the next iteration.
                    logger.debug("Double N and add to pending queue.")
                    self._pending.append((config, N * 2, seed))
            # Trials have not been evaluated yet so we just add the current config with the current N/seed to the queue
            # again
            else:
                # We append the current N to the pending, so in the next iteration we check again
                # if the trials have been evaluated
                logger.debug("Add to pending queue.")
                self._pending.append((config, N, seed))

                # We also have to increase the fails here, otherwise we might get stuck
                fails += 1

    def get_trials_of_interest(
        self,
        config: Configuration,
        *,
        N: int | None = None,
        seed: int = 0,
        validate: bool = False,
    ) -> list[TrialInfo]:
        """Returns a list of trials of interest for a given configuration."""
        if N is None:
            N = self._max_config_calls

        if validate:
            rng = np.random.RandomState(seed)
        else:
            rng = self._rng

        # tf_seeds might include seeds specified by the user
        trials: list[TrialInfo] = []

        i = 0
        while len(trials) < self._max_config_calls:
            if validate:
                next_seed = int(self._rng.randint(low=0, high=MAXINT, size=1)[0])
            else:
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
        trials_of_interest = self.get_trials_of_interest(config, N=N, seed=seed)

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
        trials_of_interest = self.get_trials_of_interest(config, N=N, seed=seed)
        trials: list[TrialInfo] = []

        # We iterate over the trials again
        evaluated_isbk = rh.get_trials(config, only_max_observed_budget=False)
        for isbk in evaluated_isbk:
            trial = TrialInfo(config, instance=isbk.instance, seed=isbk.seed)

            if trial in trials_of_interest:
                trials.append(trial)

        return trials
