from __future__ import annotations
from collections import defaultdict

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

        # What if there are already trials in the runhistory? Should we queue them up?
        # Because they are part of the runhistory, they might be selecdted as incumbent. However, they are not
        # intensified because they are not part of the queue. We could add them here to incorporate them in the
        # intensification process.
        # Idea: Add all configs to queue which are not incumbents
        # N=1 is enough here as it will increase automatically in the iterations if the configuration is worthy
        incumbents = self.get_incumbents()
        for config in rh.get_configs():
            if config not in incumbents:
                logger.info(f"Added config {rh.get_config_id(config)} to the intensifier queue.")
                queue.append((config, 1))
            else:
                logger.info(
                    f"Config {rh.get_config_id(config)} was not added to the intensifier queue "
                    "because it is an incumbent already."
                )

        fails = -1
        while True:
            fails += 1

            # Some criteria to stop the intensification if nothing can be intensified anymore
            if fails > 8 and fails >= self._scenario.n_workers * 2:
                logger.error("Intensifier could not find any new trials.")
                exit()

            # Some configs from the runhistory
            running_configs = rh.get_running_configs()

            # Now we get the incumbents sorted by number of trials
            # Also, incorporate ``get_incumbent_instances`` here because challenger are only allowed to
            # sample from the incumbent's instances
            incumbents = self.get_incumbents(sort_by="num_trials")
            incumbent_instances = self.get_incumbent_instances()

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
                    incumbent_next_instances = self.get_next_incumbent_instances()
                    trials = self._get_next_trials(
                        incumbent,
                        from_instances=incumbent_instances,
                        expand_from_instances=incumbent_next_instances,
                    )

                    if len(trials) > 0:
                        fails = -1
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
                # KEINE UMSORTIERUNG?
                # WIESO CONFIG UND NICHT TRIAL?
                if config in running_configs:
                    queue.append((config, N))
                    continue

                # If the config is rejected, we simply remove it from the queue so that the configuration is never
                # intensified again
                rejected_configs = self.get_rejected_configs()
                if config not in rejected_configs:
                    trials = self._get_next_trials(config, N=N, from_instances=incumbent_instances)
                    for trial in trials:
                        fails = -1
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
        expand_from_instances: list[InstanceSeedBudgetKey] | None = None,
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

        # Now we actually have to check whether the trials have been evaluated already
        evaluated_trials = rh.get_trials(config, only_max_observed_budget=False)
        for trial in evaluated_trials:
            if trial in trials:
                trials.remove(trial)

        # It's also important to remove running trials from the selection (we don't want to queue them again)
        running_trials = rh.get_running_trials()
        for trial in running_trials:
            if trial in trials:
                trials.remove(trial)

        # Keep ``from_instances`` trials only
        if from_instances is not None:
            for trial in trials.copy():
                isbk = trial.get_instance_seed_budget_key()
                if isbk not in from_instances:
                    trials.remove(trial)

        # Special case for intensifying the incumbent: If we have already evaluated all instances on this specific
        # config, we need to expand the incumbent instances
        if expand_from_instances is not None and len(trials) == 0:
            for isbk in expand_from_instances:
                trial = TrialInfo(config=config, instance=isbk.instance, seed=isbk.seed)
                if trial not in evaluated_trials and trial not in running_trials:
                    trials.append(trial)

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

        # Return only N trials
        if N is not None:
            if len(shuffled_trials) > N:
                shuffled_trials = shuffled_trials[:N]

        return shuffled_trials
