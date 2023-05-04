from __future__ import annotations

from typing import Any, Iterator

from ConfigSpace import Configuration

from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import InstanceSeedBudgetKey
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class Intensifier(AbstractIntensifier):
    """Implementation of an intensifier supporting multi-fidelity, multi-objective, and multi-processing.
    Races challengers against current incumbents.

    The behaviour of this intensifier is as follows:

    - First, adds configs from the runhistory to the queue with N=1 (they will be ignored if they are already
      evaluated).
    - While loop:

      - If queue is empty: Intensifies exactly one more instance of one incumbent and samples a new configuration
        afterwards.
      - If queue is not empty: Configs in the queue are evaluated on N=(N*2) instances if they might be better
        than the incumbents. If not, they are removed from the queue and rejected forever.

    Parameters
    ----------
    max_config_calls : int, defaults to 3
        Maximum number of configuration evaluations. Basically, how many instance-seed keys should be maxed evaluated
        for a configuration.
    max_incumbents : int, defaults to 10
        How many incumbents to keep track of in the case of multi-objective.
    retries : int, defaults to 16
        How many more iterations should be done in case no new trial is found.
    seed : int, defaults to None
        Internal seed used for random events, like shuffle seeds.
    """

    def __init__(
        self,
        scenario: Scenario,
        max_config_calls: int = 3,
        max_incumbents: int = 10,
        retries: int = 16,
        seed: int | None = None,
    ):
        super().__init__(scenario=scenario, max_config_calls=max_config_calls, max_incumbents=max_incumbents, seed=seed)
        self._retries = retries

    def reset(self) -> None:
        """Resets the internal variables of the intensifier including the queue."""
        super().reset()

        # Queue to keep track of the challengers
        # (config, N=how many trials should be sampled)
        self._queue: list[tuple[Configuration, int]] = []

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
        return {
            "queue": [
                (self.runhistory.get_config_id(config), n)
                for config, n in self._queue
                if self.runhistory.has_config(config)
            ],
        }

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: D102
        self._queue = [(self.runhistory.get_config(id), n) for id, n in state["queue"]]

    def __iter__(self) -> Iterator[TrialInfo]:
        """This iter method holds the logic for the intensification loop.
        Some facts about the loop:

        - Adds existing configurations from the runhistory to the queue (that means it supports user-inputs).
        - Everytime an incumbent (with the lowest amount of trials) is intensified, a new challenger is added to the
          queue.
        - If all incumbents are evaluated on the same trials, a new trial is added to one of the incumbents.
        - Only challengers which are not rejected/running/incumbent are intensified by N*2.

        Returns
        -------
        trials : Iterator[TrialInfo]
            Iterator over the trials.
        """
        self.__post_init__()

        rh = self.runhistory
        assert self._max_config_calls is not None

        # What if there are already trials in the runhistory? Should we queue them up?
        # Because they are part of the runhistory, they might be selected as incumbents. However, they are not
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
            if fails > self._retries:
                logger.error("Intensifier could not find any new trials.")
                return

            # Some configs from the runhistory
            running_configs = rh.get_running_configs()
            rejected_configs = self.get_rejected_configs()

            # Now we get the incumbents sorted by number of trials
            # Also, incorporate ``get_incumbent_instance_seed_budget_keys`` here because challengers are only allowed to
            # sample from the incumbent's instances
            incumbents = self.get_incumbents(sort_by="num_trials")
            incumbent_isb_keys = self.get_incumbent_instance_seed_budget_keys()

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
                    individual_incumbent_isb_keys = rh.get_instance_seed_budget_keys(incumbent)
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
                    incumbent_isb_key_differences = self.get_incumbent_instance_seed_budget_key_differences()

                    # We set shuffle to false because we first want to evaluate the incumbent instances, then the
                    # differences (to make the instance-seed keys for the incumbents equal again)
                    trials = self._get_next_trials(
                        incumbent,
                        from_keys=incumbent_isb_keys + incumbent_isb_key_differences,
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
                            f"--- Yielding trial {len(individual_incumbent_isb_keys)+1} of "
                            f"{self._max_config_calls} from incumbent {incumbent_hash}..."
                        )
                        yield trials[0]
                        logger.debug(f"--- Finished yielding for config {incumbent_hash}.")

                        # We break here because we only want to intensify one more trial of one incumbent
                        break
                    else:
                        # assert len(incumbent_isb_keys) == self._max_config_calls
                        logger.debug(
                            f"--- Skipped intensifying incumbent {incumbent_hash} because no new trials have "
                            "been found. Evaluated "
                            f"{len(individual_incumbent_isb_keys)}/{self._max_config_calls} trials."
                        )

                # For each intensification of the incumbent, we also want to intensify the next configuration
                # We simply add it to the queue and intensify it in the next iteration
                try:
                    config = next(self.config_generator)
                    config_hash = get_config_hash(config)
                    self._queue.append((config, 1))
                    logger.debug(f"--- Added a new config {config_hash} to the queue.")

                    # If we added a new config, then we did something in this iteration
                    fails = -1
                except StopIteration:
                    # We stop if we don't find any configuration anymore
                    return
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
                    isk_keys: list[InstanceSeedBudgetKey] | None = None
                    if len(incumbent_isb_keys) > 0:
                        isk_keys = incumbent_isb_keys

                    # TODO: What to do if there are no incumbent instances? (Use-case: call multiple asks)

                    trials = self._get_next_trials(config, N=N, from_keys=isk_keys)
                    logger.debug(f"--- Yielding {len(trials)} trials to evaluate config {config_hash}...")
                    for trial in trials:
                        fails = -1
                        yield trial

                    logger.debug(f"--- Finished yielding for config {config_hash}.")

                    # Now we have to remove the config
                    self._queue.remove((config, N))
                    logger.debug(f"--- Removed config {config_hash} with N={N} from queue.")

                    # Finally, we add the same config to the queue with a higher N
                    # If the config was rejected by the runhistory, then it's been removed in the next iteration
                    if N < self._max_config_calls:
                        new_pair = (config, N * 2)
                        if new_pair not in self._queue:
                            logger.debug(
                                f"--- Doubled trials of config {config_hash} to N={N*2} and added it to the queue "
                                "again."
                            )
                            self._queue.append((config, N * 2))

                            # Also reset fails here
                            fails = -1
                        else:
                            logger.debug(f"--- Config {config_hash} with N={N*2} is already in the queue.")

                    # If we are at this point, it really is important to break because otherwise, we would intensify
                    # all configs in the queue in one iteration
                    break

    def _get_next_trials(
        self,
        config: Configuration,
        *,
        N: int | None = None,
        from_keys: list[InstanceSeedBudgetKey] | None = None,
        shuffle: bool = True,
    ) -> list[TrialInfo]:
        """Returns the next trials of the configuration based on ``get_trials_of_interest``. If N is specified,
        maximum N trials are returned but not necessarily all of them (depending on evaluated already or still running).

        Parameters
        ----------
        N : int | None, defaults to None
            The maximum number of trials to return. If None, all trials (``max_config_calls``) are returned.
            Running and evaluated trials are counted in.
        from_keys : list[InstanceSeedBudgetKey], defaults to None
            Only instances from the list are considered for the trials.
        shuffle : bool, defaults to True
            Shuffles the trials in groups. First, all instances are shuffled, then all seeds.
        """
        rh = self.runhistory
        is_keys = self.get_instance_seed_keys_of_interest()

        # Create trials from the instance seed pairs
        # trials: list[TrialInfo] = []
        # for is_key in is_keys:
        #    trials.append(TrialInfo(config=config, instance=is_key.instance, seed=is_key.seed))

        # Keep ``from_keys`` trials only
        if from_keys is not None:
            valid_is_keys = [key.get_instance_seed_key() for key in from_keys]
            for is_key in is_keys.copy():
                if is_key not in valid_is_keys:
                    is_keys.remove(is_key)

        # Counter is important to actually subtract the number of trials that are already evaluated/running
        # Otherwise, evaluated/running trials are not considered
        # Example: max_config_calls=16, N=8, 2 trials are running, 2 trials are evaluated, 4 trials are pending
        # Without a counter, we would return 8 trials because there are still so many trials left open
        # With counter, we would return only 4 trials because 4 trials are already evaluated/running
        counter = 0

        # Now we actually have to check whether the trials have been evaluated already
        evaluated_isb_keys = rh.get_instance_seed_budget_keys(config, highest_observed_budget_only=False)
        for isb_key in evaluated_isb_keys:
            is_key = isb_key.get_instance_seed_key()
            if is_key in is_keys:
                counter += 1
                is_keys.remove(is_key)

        # It's also important to remove running trials from the selection (we don't want to queue them again)
        running_trials = rh.get_running_trials(config)
        for trial in running_trials:
            is_key = trial.get_instance_seed_key()
            if is_key in is_keys:
                counter += 1
                is_keys.remove(is_key)

        if shuffle:
            is_keys = self._reorder_instance_seed_keys(is_keys)

        # Return only N trials
        if N is not None:
            N = N - counter
            if len(is_keys) > N:
                is_keys = is_keys[:N]

        # Now we convert to trials
        trials: list[TrialInfo] = []
        for is_key in is_keys:
            trials.append(TrialInfo(config=config, instance=is_key.instance, seed=is_key.seed))

        return trials
