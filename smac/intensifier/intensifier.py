from __future__ import annotations

import numpy as np
from typing import Iterator

from ConfigSpace import Configuration

from smac.constants import MAXINT
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import InstanceSeedBudgetKey, InstanceSeedKey
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
        queue: list[tuple[Configuration, int]] = []  # (config, N=how many trials should be sampled)

        i = -1
        while True:
            i += 1

            # Some criteria to stop the intensification if nothing can be intensified anymore
            if i >= self._scenario.n_workers * 2:
                exit()

            # Some configs from the runhistory
            running_configs = rh.get_running_configs()
            rejected_configs = rh.get_rejected_configs()

            # Now we get the incumbents sorted by number of trials
            # Also, incorporate ``get_incumbent_instances`` here because challenger are only allowed to
            # sample from the incumbent's instances
            incumbents = rh.get_incumbents(sort_by="num_trials")
            incumbent_instances = rh.get_incumbent_instances()

            # Check if configs in queue are still running
            all_configs_running = True
            for config, _ in queue:
                if config not in running_configs:
                    all_configs_running = False
                    break

            if len(queue) == 0 or all_configs_running:
                for incumbent in incumbents:
                    # If incumbent was evaluated on all incumbent_instances but was not evaluated on all
                    # max_config_calls, then we have to get a new instance
                    trials = self._get_next_trials(incumbent, from_instances=incumbent_instances, incumbent=True)
                    if len(trials) > 0:
                        i = -1
                        logger.debug("Intensifying trial of one incumbent...")
                        yield trials[0]

                        # We break here because we only want to intensify one more trial of one incumbent
                        break

                # For each intensification of the incumbent, we also want to intensify the next configuration
                # We simply add it to the queue and intensify it in the next iteration
                config = next(self.config_generator)
                queue.append((config, 1))
            else:
                # Now we try to empty the queue
                config, N = queue.pop(0)

                # If the config is still running, we just add it at the end of the queue and continue
                if config in running_configs:
                    queue.append((config, N))
                    continue

                # If the config is rejected, we simply remove it from the queue so that the configuration is never
                # intensified again
                if config not in rejected_configs:
                    trials = self._get_next_trials(config, N=N, from_instances=incumbent_instances)
                    for trial in trials:
                        i = -1
                        yield trial

                    # Finally, we add the same config to the queue with a higher N
                    # If the config was rejected by the runhistory, then it's gonna removed
                    if N < self._max_config_calls:
                        queue.append((config, N * 2))

    def get_trials_of_interest(
        self,
        config: Configuration,
        *,
        validate: bool = False,
    ) -> list[TrialInfo]:
        """Returns a list of trials of interest for a given configuration. Only ``max_config_calls`` trials are
        returned.
        """
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

    def _get_next_trials(
        self,
        config: Configuration,
        *,
        N: int | None = None,
        from_instances: list[InstanceSeedBudgetKey] | None = None,
        incumbent: bool = False,
    ) -> list[TrialInfo]:
        """Returns the next trials of the configuration based on ``get_trials_of_interest``. If N is specified,
        maximum N trials are returned but not necessarely all of them (depending on evaluated already or still running).

        Parameters
        ----------
        incumbent : bool, defaults to False
            If true, ``from_instances`` are expanded.
        """
        rh = self.runhistory
        trials = self.get_trials_of_interest(config)

        # Now we actually have to check whether the trials have been evaluated already
        evaluated_trials = rh.get_trials(config, only_max_observed_budget=False)
        for trial in evaluated_trials:
            if trial in trials:
                trials.remove(trial)

        # It's also important to remove running trials from the selection (we don't want to queue them again)
        for trial in rh.get_running_trials():
            if trial in trials:
                trials.remove(trial)

        # Only leave ``from_instances`` trials
        removed_trials = []
        if from_instances is not None:
            for trial in trials:
                isbk = trial.get_instance_seed_budget_key()
                if isbk not in from_instances:
                    removed_trials.append(trial)
                    trials.remove(trial)

        if incumbent:
            if len(trials) == 0 and len(removed_trials) > 0:
                trials.append(removed_trials[0])

        # Now we shuffle the trials
        # TODO: Shuffle in groups (first all instances, then all seeds)
        self._rng.shuffle(trials)  # type: ignore

        # Return only N trials
        if N is not None:
            if len(trials) > N:
                trials = trials[:N]

        return trials
