from __future__ import annotations

from typing import Any, Iterator


import numpy as np
from ConfigSpace import Configuration
from smac.constants import MAXINT

from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import InstanceSeedBudgetKey
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class NotEvaluatedError(RuntimeError):
    pass


class SuccessiveHalving(AbstractIntensifier):
    def __init__(
        self,
        scenario: Scenario,
        eta: int = 3,
        n_seeds: int = 1,  # How many seeds to use for each instance
        instance_order: str | None = "shuffle",  # shuffle_once, shuffle, None
        incumbent_selection: str = "highest_executed_budget",
        max_incumbents: int = 20,
        seed: int | None = None,
    ):
        super().__init__(scenario=scenario, n_seeds=n_seeds, max_incumbents=max_incumbents, seed=seed)

        # We generate our instance seed pairs once
        instance_seed_pairs = self.get_instance_seed_pairs()

        # Budgets, followed by lots of sanity-checking
        min_budget = scenario.min_budget
        max_budget = scenario.max_budget

        if max_budget is not None and min_budget is not None and max_budget < min_budget:
            raise ValueError("Max budget has to be larger than min budget.")

        if self.uses_instances:
            if isinstance(min_budget, float) or isinstance(max_budget, float):
                raise ValueError("Successive Halving requires integer budgets when using instances.")

            min_budget = min_budget if min_budget is not None else 1
            max_budget = max_budget if max_budget is not None else len(instance_seed_pairs)

            if max_budget > len(instance_seed_pairs):
                raise ValueError(
                    f"Max budget of {max_budget} can not be greater than the number of instance seed "
                    f"pairs ({len(instance_seed_pairs)})."
                )

            if max_budget < len(instance_seed_pairs):
                logger.warning(
                    f"Max budget {max_budget} does not include all instance seed  "
                    f"pairs ({len(instance_seed_pairs)})."
                )
        else:
            if min_budget is None or max_budget is None:
                raise ValueError(
                    "Successive Halving requires the parameters min_budget and max_budget defined in the scenario."
                )

            if len(instance_seed_pairs) != 1:
                raise ValueError("Successive Halving supports only one seed when using budgets.")

        if min_budget is None or min_budget <= 0:
            raise ValueError("Min budget has to be larger than 0.")

        budget_type = "INSTANCES" if self.uses_instances else "BUDGETS"
        logger.info(
            f"Successive Halving uses budget type {budget_type} with eta {eta}, "
            f"min budget {min_budget} and max budget {max_budget}."
        )

        # Pre-computing Successive Halving variables
        max_iterations = int(np.floor(np.log(max_budget / min_budget) / np.log(eta)))
        n_initial_challengers = int(eta**max_iterations)

        # How many configs in each stage
        linspace = -np.linspace(0, max_iterations, max_iterations + 1)
        n_configs = n_initial_challengers * np.power(eta, linspace)
        n_configs = np.array(np.round(n_configs), dtype=int).tolist()

        # How many budgets in each stage
        linspace = -np.linspace(max_iterations, 0, max_iterations + 1)
        budgets = (max_budget * np.power(eta, linspace)).tolist()

        # Global variables
        self._eta = eta
        self._instance_order = instance_order
        self._min_budget = min_budget
        self._max_budget = max_budget
        self._max_iterations = max_iterations
        self._n_configs_in_stage = n_configs
        self._budgets_in_stage = budgets

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "eta": self._eta,
                "instance_order": self._instance_order,
                "min_budget": self._min_budget,
                "max_budget": self._max_budget,
            }
        )

        return meta

    # TODO: State: Also save the rng state

    @property
    def uses_seeds(self) -> bool:  # noqa: D102
        return True

    @property
    def uses_budgets(self) -> bool:  # noqa: D102
        if self._scenario.instances is None:
            return True

        return False

    @property
    def uses_instances(self) -> bool:  # noqa: D102
        if self._scenario.instances is None:
            return False

        return True

    def __iter__(self) -> Iterator[TrialInfo]:
        tracker: dict[int, list[tuple[int | None, list[Configuration]]]] = {}  # dict[stage, tuple[seed, list[config]]]

        # TODO: We have to add existing trials from the runhistory
        # TODO: How to deal with the shuffle index then? Just assign randomly and hope for the best
        # TODO: Configs which are in the tracker have to be ignored

        while True:
            # If we don't yield trials anymore, we have to update
            # Otherwise, we can just keep yielding trials from the tracker
            update = False

            # We iterate over the tracker to do two things:
            # 1) Yield trials of configs that are not yet evaluated/running
            # 2) Update tracker and move better configs to the next stage
            # We start in reverse order to complete higher stages first
            logger.debug("Updating tracker:")
            stages: list[int] = sorted(list(tracker.keys()), reverse=True)
            for stage in stages:
                pairs = tracker[stage].copy()
                for i, (seed, configs) in enumerate(pairs):
                    instances = self._get_instances(stage=stage, seed=seed)

                    # We iterate over the configs and yield trials which are not running/evaluated yet
                    for config in configs:
                        config_hash = get_config_hash(config)
                        trials = self._get_next_trials(config, from_instances=instances)
                        logger.debug(
                            f"--- Yielding {len(trials)}/{len(instances)} for config {config_hash} in "
                            f"stage {stage} with seed {seed}..."
                        )

                        for trial in trials:
                            yield trial
                            update = True

                    # If all configs were evaluated on ``n_configs_required``, we finally can compare
                    try:
                        successful = self._get_best_configs(configs, from_instances=instances)
                    except NotEvaluatedError:
                        # We can't compare anything, so we just continue with the next pairs
                        logger.debug("--- Could not compare configs because not all trials have been evaluated yet.")
                        continue

                    # Update tracker
                    # Remove current shuffle index / config pair
                    del tracker[stage][i]

                    # Add successful to the next stage
                    if stage < self._max_iterations:
                        tracker[stage + 1].append((seed, successful))

                    logger.debug(f"--- Promoted {len(successful)} configs from stage {stage} to stage {stage + 1}.")

            if update:
                continue

            # TODO: Which configs should be compared? It does make a huge difference, right?

            # If we are running out of trials, we want to add configs to the first stage
            # We simply add as many configs to the stage as required (_n_configs_in_stage[0])
            configs = []
            for _ in range(self._n_configs_in_stage[0]):
                try:
                    config = next(self.config_generator)
                    configs.append(config)
                except StopIteration:
                    # We stop if we don't find any configuration anymore
                    return

            # Here we have the option to shuffle the trials when specified by the user
            if self._instance_order == "shuffle":
                seed = self._rng.randint(0, MAXINT)
            elif self._instance_order == "shuffle_once":
                seed = 0
            else:
                seed = None

            # We keep track of the seed so we always evaluate on the same instances
            tracker[0].append((seed, configs))
            logger.debug(f"Added {len(configs)} new configs to stage 0 with seed {seed}.")

    def _get_instances(self, stage: int, seed: int | None = None) -> list[InstanceSeedBudgetKey]:
        """Returns all instances (instance-seed-budget keys in this case) for the given stage. Each stage
        is associated with a budget (N). Two possible options:

        1) Instance based: We return N instances. If a seed is specified, we shuffle the instances, before
        returning the first N instances.
        2) Budget based: We return one instance only but the budget is set to N.
        """
        budget: float | int | None = None
        instance_seed_pairs = self.get_instance_seed_pairs()

        # We have to differentiate between budgets and instance based here
        # If we are budget based, we always have one instance seed pair only
        # If we are in the instance setting, we have to return a specific number of instance seed pairs

        if self.uses_instances:
            # Shuffle instance seed pairs based on the seed
            # TODO: Do it group based
            rng = np.random.RandomState(seed)
            rng.shuffle(instance_seed_pairs)  # type: ignore

            # We only return the first N instances
            N = self._budgets_in_stage[stage]
            instance_seed_pairs = instance_seed_pairs[:N]
        else:
            assert len(instance_seed_pairs) == 1

            # The stage defines which budget should be used (in real-valued setting)
            # No shuffle is needed here because we only have on instance seed pair
            budget = self._budgets_in_stage[stage]

        isbk = []
        for isk in instance_seed_pairs:
            isbk.append(InstanceSeedBudgetKey(instance=isk.instance, seed=isk.seed, budget=budget))

        return isbk

    def _get_next_trials(
        self,
        config: Configuration,
        from_instances: list[InstanceSeedBudgetKey],
    ) -> list[TrialInfo]:
        """Returns trials for a given config from a list of instances (instance-seed-budget keys). The returned trials
        have not run or evaluated yet.
        """
        rh = self.runhistory
        evaluated_trials = rh.get_trials(config, only_max_observed_budget=False)
        running_trials = rh.get_running_trials(config)

        next_trials: list[TrialInfo] = []
        for instance in from_instances:
            trial = TrialInfo(config=config, instance=instance.instance, seed=instance.seed, budget=instance.budget)

            if trial in evaluated_trials or trial in running_trials:
                continue

            next_trials.append(trial)

        return next_trials

    def _get_best_configs(
        self,
        configs: list[Configuration],
        from_instances: list[InstanceSeedBudgetKey],
        *,
        keep: float = 0.5,
    ) -> list[Configuration]:
        rh = self.runhistory

        # TODO: Which ones to keep?
        # We could do pareto front, and if we have less/more than we use average cost
        costs: list[tuple[float, Configuration]] = []
        for config in configs:
            # Small sanity check that config really was evaluated on all instances
            # TODO: Too much overhead?
            available_instances = [
                trial.get_instance_seed_budget_key() for trial in rh.get_trials(config, only_max_observed_budget=False)
            ]

            for instance in from_instances:
                if instance not in available_instances:
                    raise NotEvaluatedError(f"Config {config} was not evaluated on {instance}.")

            cost = rh.average_cost(config, from_instances, normalize=True)
            assert isinstance(cost, float)

            costs.append((cost, config))

        # Sort by cost
        costs.sort(key=lambda x: x[0])

        return costs[: int(len(costs) * keep)]
