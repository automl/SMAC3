from __future__ import annotations

from typing import Any, Iterator

import dataclasses
from collections import defaultdict

import numpy as np
from ConfigSpace import Configuration

from smac.constants import MAXINT
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import InstanceSeedBudgetKey, InstanceSeedKey
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class Intensifier(AbstractIntensifier):
    def __init__(
        self,
        scenario: Scenario,
        max_config_calls: int = 2000,
        max_incumbents: int = 20,
        seed: int | None = None,
    ):
        super().__init__(scenario=scenario, max_incumbents=max_incumbents, seed=seed)

        # The queue for the challengers
        self._queue: list[tuple[Configuration, int]] = []  # (config, N=how many trials should be sampled)
        self._instance_seed_pairs: list[InstanceSeedKey] | None = None
        self._instance_seed_pairs_validation: list[InstanceSeedKey] | None = None

        # Internal variables
        self._max_config_calls = max_config_calls

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"max_config_calls": self._max_config_calls})

        return meta

    @property
    def uses_seeds(self) -> bool:  # noqa: D102
        return True

    @property
    def uses_budgets(self) -> bool:  # noqa: D102
        return False

    @property
    def uses_instances(self) -> bool:  # noqa: D102
        if self._scenario.instances is None:
            return False

        return True

    def get_state(self) -> dict[str, Any]:  # noqa: D102
        instance_seed_pairs: list[dict] | None = None
        if self._instance_seed_pairs is not None:
            instance_seed_pairs = [dataclasses.asdict(item) for item in self._instance_seed_pairs]

        instance_seed_pairs_validation: list[dict] | None = None
        if self._instance_seed_pairs_validation is not None:
            instance_seed_pairs_validation = [dataclasses.asdict(item) for item in self._instance_seed_pairs_validation]

        return {
            "queue": [(self.runhistory.get_config_id(config), n) for config, n in self._queue],
            "instance_seed_pairs": instance_seed_pairs,
            "instance_seed_pairs_validation": instance_seed_pairs_validation,
        }

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: D102
        self._queue = [(self.runhistory.ids_config[id], n) for id, n in state["queue"]]
        self._instance_seed_pairs = None
        self._instance_seed_pairs_validation = None

        if state["instance_seed_pairs"] is not None:
            self._instance_seed_pairs = [InstanceSeedKey(**item) for item in state["instance_seed_pairs"]]

        if state["instance_seed_pairs_validation"] is not None:
            self._instance_seed_pairs_validation = [
                InstanceSeedKey(**item) for item in state["instance_seed_pairs_validation"]
            ]

    def __iter__(self) -> Iterator[TrialInfo]:
        rh = self.runhistory

        # What if there are already trials in the runhistory? Should we queue them up?
        # Because they are part of the runhistory, they might be selected as incumbent. However, they are not
        # intensified because they are not part of the queue. We could add them here to incorporate them in the
        # intensification process.
        # Idea: Add all configs to queue (if it is an incumbent it is removed automatically later on)
        # N=1 is enough here as it will increase automatically in the iterations if the configuration is worthy
        # Note: The incumbents are updated once the runhistory is set (see abstract intensifier)
        # Note 2: If the queue was restored, we don't want to go in here (queue is restored)
        if len(self._queue) == 0:
            for config in rh.get_configs():
                hash = get_config_hash(config)
                self._queue.append((config, 1))
                logger.info(f"Added config {hash} from runhistory to the intensifier queue.")

        fails = -1
        while True:
            fails += 1

            # Some criteria to stop the intensification if nothing can be intensified anymore
            if fails > 8 and fails >= self._scenario.n_workers * 2:
                logger.error("Intensifier could not find any new trials.")
                return

            # Some configs from the runhistory
            running_configs = rh.get_running_configs()
            rejected_configs = self.get_rejected_configs()

            # Now we get the incumbents sorted by number of trials
            # Also, incorporate ``get_incumbent_instances`` here because challenger are only allowed to
            # sample from the incumbent's instances
            incumbents = self.get_incumbents(sort_by="num_trials")
            incumbent_instances = self.get_incumbent_instances()

            # Check if configs in queue are still running
            all_configs_running = True
            for config, _ in self._queue:
                if config not in running_configs:
                    all_configs_running = False
                    break

            if len(self._queue) == 0 or all_configs_running:
                if len(self._queue) == 0:
                    logger.debug("Queue is empty:")
                else:
                    logger.debug("All configs in the queue are running:")

                if len(incumbents) == 0:
                    logger.debug("--- No incumbent to intensify.")

                for incumbent in incumbents:
                    # Instances of this particular incumbent
                    individual_incumbent_instances = rh.get_instances(incumbent)
                    incumbent_hash = get_config_hash(incumbent)

                    # We don't want to intensify an incumbent which is either still running or rejected
                    if incumbent in running_configs:
                        logger.debug(
                            f"--- Skipping intensifying incumbent {incumbent_hash} because it has trials pending."
                        )
                        continue

                    if incumbent in rejected_configs:
                        # This should actually not happen because if a config is rejected the incumbent should
                        # have changed
                        # However, we just keep it here as sanity check
                        logger.debug(f"--- Skipping intensifying incumbent {incumbent_hash} because it was rejected.")
                        continue

                    # If incumbent was evaluated on all incumbent instance intersections but was not evaluated on
                    # the differences, we have to add it here
                    incumbent_instance_differences = self.get_incumbent_instance_differences()

                    # We set shuffle to false because we first want to evaluate the incumbent instances, then the
                    # differences (to make the incumbents equal again)
                    trials = self._get_next_trials(
                        incumbent,
                        from_instances=incumbent_instances + incumbent_instance_differences,
                        shuffle=False,
                    )

                    # If we don't receive any trials, then we try it randomly with any other because we want to
                    # intensify for sure
                    if len(trials) == 0:
                        logger.debug(
                            f"--- Incumbent {incumbent_hash} was already evaluated on all incumbent instances "
                            "and incumbent instance differences so far. Looking for new instances..."
                        )
                        trials = self._get_next_trials(incumbent)
                        logger.debug(f"--- Randomly found {len(trials)} new trials.")

                    if len(trials) > 0:
                        fails = -1
                        logger.debug(
                            f"--- Yielding trial {len(individual_incumbent_instances)+1} of "
                            f"{self._max_config_calls} from incumbent {incumbent_hash}..."
                        )
                        yield trials[0]
                        logger.debug(f"--- Finished yielding for config {incumbent_hash}.")

                        # We break here because we only want to intensify one more trial of one incumbent
                        break
                    else:
                        # assert len(incumbent_instances) == self._max_config_calls
                        logger.debug(
                            f"--- Skipped intensifying incumbent {incumbent_hash} because no new trials have "
                            "been found. Evaluated "
                            f"{len(individual_incumbent_instances)}/{self._max_config_calls} trials."
                        )

                # For each intensification of the incumbent, we also want to intensify the next configuration
                # We simply add it to the queue and intensify it in the next iteration
                # Note: If config generator throws a StopIteration, it will be caught by the SMBO loop
                config = next(self.config_generator)
                config_hash = get_config_hash(config)
                self._queue.append((config, 1))
                logger.debug(f"--- Added a new config {config_hash} to the queue.")
            else:
                logger.debug("Start finding a new challenger in the queue:")
                for i, (config, N) in enumerate(self._queue.copy()):
                    config_hash = get_config_hash(config)

                    # If the config is still running, we ignore it and head to the next config
                    if config in running_configs:
                        logger.debug(f"--- Config {config_hash} is still running. Skipping this config in the queue...")
                        continue

                    # We want to get rid of configs in the queue which are rejected
                    if config in rejected_configs:
                        logger.debug(f"--- Config {config_hash} was removed from the queue because it was rejected.")
                        self._queue.remove((config, N))
                        continue

                    # We don't want to intensify an incumbent here
                    if config in incumbents:
                        logger.debug(f"--- Config {config_hash} was removed from the queue because it is an incumbent.")
                        self._queue.remove((config, N))
                        continue

                    # And then we yield as many trials as we specified N
                    # However, only the same instances as the incumbents are used
                    instances: list[InstanceSeedBudgetKey] | None = None
                    if len(incumbent_instances) > 0:
                        instances = incumbent_instances

                    trials = self._get_next_trials(config, N=N, from_instances=instances)
                    logger.debug(f"--- Yielding {len(trials)} trials to evaluate config {config_hash}...")
                    for trial in trials:
                        fails = -1
                        yield trial

                    logger.debug(f"--- Finished yielding for config {config_hash}.")

                    # Now we have to remove the config
                    self._queue.remove((config, N))

                    # Finally, we add the same config to the queue with a higher N
                    # If the config was rejected by the runhistory, then it's be removed in the next iteration
                    if N < self._max_config_calls:
                        new_pair = (config, N * 2)
                        if new_pair not in self._queue:
                            logger.debug(
                                f"--- Doubled trials of config {config_hash} to N={N*2} and added it to the queue "
                                "again."
                            )
                            self._queue.append((config, N * 2))
                        else:
                            logger.debug(f"--- Config {config_hash} with N={N*2} is already in the queue.")

                    # If we are at this point, it really is important to break because otherwise we would intensify
                    # all configs in the queue in one iteration
                    break

    def get_trials_of_interest(
        self,
        config: Configuration,
        *,
        validate: bool = False,
        seed: int | None = None,
    ) -> list[TrialInfo]:
        """Returns a list of trials of interest for a given configuration. Only ``max_config_calls`` trials are
        returned.

        Warning
        -------
        The passed seed is only used for validation.
        """
        if seed is None:
            seed = 0

        if (self._instance_seed_pairs is None and not validate) or (
            self._instance_seed_pairs_validation is None and validate
        ):
            instance_seed_pairs: list[InstanceSeedKey] = []
            if validate:
                rng = np.random.RandomState(seed)
            else:
                rng = self._rng

            i = 0
            while len(instance_seed_pairs) < self._max_config_calls:
                if validate:
                    next_seed = int(rng.randint(low=0, high=MAXINT, size=1)[0])
                else:
                    try:
                        next_seed = self._tf_seeds[i]
                        logger.info(f"Added existing seed {next_seed} from runhistory to the intensifier.")
                    except IndexError:
                        # Use global random generator for a new seed and mark it so it will be reused for another config
                        next_seed = int(rng.randint(low=0, high=MAXINT, size=1)[0])
                        self._tf_seeds.append(next_seed)
                        logger.info(f"Added new random seed {next_seed} to the intensifier.")

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

    def _get_next_trials(
        self,
        config: Configuration,
        *,
        N: int | None = None,
        from_instances: list[InstanceSeedBudgetKey] | None = None,
        shuffle: bool = True,
    ) -> list[TrialInfo]:
        """Returns the next trials of the configuration based on ``get_trials_of_interest``. If N is specified,
        maximum N trials are returned but not necessarely all of them (depending on evaluated already or still running).

        Parameters
        ----------
        from_instances : list[InstanceSeedBudgetKey], defaults to None
            Only instances from the list are considered for the trials.
        expand_from_instances : list[InstanceSeedBudgetKey], defaults to None
            If no trials are found anymore, ``expand_from_instances`` is used to get more trials. This is especially
            useful in combination with ``from_instances`` as trials can additionally added. Use-case: Next instance
            for incumbent.
        """
        rh = self.runhistory
        trials = self.get_trials_of_interest(config)

        # Keep ``from_instances`` trials only
        if from_instances is not None:
            for trial in trials.copy():
                isbk = trial.get_instance_seed_budget_key()
                if isbk not in from_instances:
                    trials.remove(trial)

        # Counter is important to actually subtract the number of trials that are already evaluated/running
        # Otherwise, evaluated/running trials are not considered
        # Example: max_config_calls=16, N=8, 2 trials are running, 2 trials are evaluated, 4 trials are pending
        # Without counter, we would return 8 trials because there are still so many trials left open
        # With counter, we would return only 4 trials because 4 trials are already evaluated/running
        counter = 0

        # Now we actually have to check whether the trials have been evaluated already
        evaluated_trials = rh.get_trials(config, only_max_observed_budget=False)
        for trial in evaluated_trials:
            if trial in trials:
                counter += 1
                trials.remove(trial)

        # It's also important to remove running trials from the selection (we don't want to queue them again)
        running_trials = rh.get_running_trials()
        for trial in running_trials:
            if trial in trials:
                counter += 1
                trials.remove(trial)

        if shuffle:
            # Now we shuffle the trials in groups (first all instances, then all seeds)
            # - Group by seeds
            # - Shuffle instances in this group of seeds
            # - Attach groups together
            groups = defaultdict(list)
            for trial in trials:
                groups[trial.seed].append(trial)

            # Shuffle groups + attach groups together
            shuffled_trials: list[TrialInfo] = []
            for seed in self._tf_seeds:
                if seed in groups and len(groups[seed]) > 0:
                    # Shuffle trials in the group and add to shuffled trials
                    shuffled = self._rng.choice(groups[seed], size=len(groups[seed]), replace=False)  # type: ignore
                    shuffled_trials += [trial for trial in shuffled]  # type: ignore

            assert len(shuffled_trials) == len(trials)
            trials = shuffled_trials

        # Return only N trials
        if N is not None:
            N = N - counter
            if len(trials) > N:
                trials = trials[:N]

        return trials
