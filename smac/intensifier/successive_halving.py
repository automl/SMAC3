from __future__ import annotations
from collections import defaultdict

from typing import Any, Iterator


import numpy as np
from ConfigSpace import Configuration
from smac.constants import MAXINT

from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import InstanceSeedBudgetKey
from smac.runhistory.errors import NotEvaluatedError
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash
from smac.utils.data_structures import batch
from smac.utils.logging import get_logger
from smac.utils.pareto_front import calculate_pareto_front

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class SuccessiveHalving(AbstractIntensifier):
    """
    Parameters
    ----------
    incumbent_selection : str, defaults to "any_budget"
        How to select the incumbent when using budgets. Can be set to:
        * any_budget: Incumbent is the best on any budget i.e., best performance regardless of budget.
        * highest_observed_budget: Incumbent is the best in the highest budget run so far.
        * highest_budget: Incumbent is selected only based on the highest budget.
    """

    def __init__(
        self,
        scenario: Scenario,
        eta: int = 3,
        n_seeds: int = 1,  # How many seeds to use for each instance
        instance_order: str | None = "shuffle",  # shuffle_once, shuffle, None
        max_incumbents: int = 20,
        incumbent_selection: str = "highest_observed_budget",
        seed: int | None = None,
    ):
        super().__init__(
            scenario=scenario,
            n_seeds=n_seeds,
            max_incumbents=max_incumbents,
            seed=seed,
        )

        self._eta = eta
        self._instance_order = instance_order
        self._incumbent_selection = incumbent_selection
        self._highest_observed_budget_only = False if incumbent_selection == "any_budget" else True

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "eta": self._eta,
                "instance_order": self._instance_order,
                "incumbent_selection": self._incumbent_selection,
            }
        )

        return meta

    def reset(self) -> None:
        super().reset()

        # States
        # dict[stage, list[tuple[instance shuffle seed, list[config_id]]]
        self._tracker: dict[int, list[tuple[int | None, list[Configuration]]]] = defaultdict(list)

    def __post_init__(self) -> None:
        """Post initilization steps after the runhistory has been set."""
        # We generate our instance seed pairs once
        is_keys = self.get_instance_seed_keys_of_interest()

        # Budgets, followed by lots of sanity-checking
        eta = self._eta
        min_budget = self._scenario.min_budget
        max_budget = self._scenario.max_budget

        if max_budget is not None and min_budget is not None and max_budget < min_budget:
            raise ValueError("Max budget has to be larger than min budget.")

        if self.uses_instances:
            if isinstance(min_budget, float) or isinstance(max_budget, float):
                raise ValueError("Successive Halving requires integer budgets when using instances.")

            min_budget = min_budget if min_budget is not None else 1
            max_budget = max_budget if max_budget is not None else len(is_keys)

            if max_budget > len(is_keys):
                raise ValueError(
                    f"Max budget of {max_budget} can not be greater than the number of instance-seed "
                    f"keys ({len(is_keys)})."
                )

            if max_budget < len(is_keys):
                logger.warning(
                    f"Max budget {max_budget} does not include all instance seed  " f"pairs ({len(is_keys)})."
                )
        else:
            if min_budget is None or max_budget is None:
                raise ValueError(
                    "Successive Halving requires the parameters min_budget and max_budget defined in the scenario."
                )

            if len(is_keys) != 1:
                raise ValueError("Successive Halving supports only one seed when using budgets.")

        if min_budget is None or min_budget <= 0:
            raise ValueError("Min budget has to be larger than 0.")

        budget_type = "INSTANCES" if self.uses_instances else "BUDGETS"
        logger.info(
            f"Successive Halving uses budget type {budget_type} with eta {eta}, "
            f"min budget {min_budget}, and max budget {max_budget}."
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
        self._min_budget = min_budget
        self._max_budget = max_budget
        self._max_iterations = max_iterations
        self._n_configs_in_stage = n_configs
        self._budgets_in_stage = budgets

    def get_state(self) -> dict[str, Any]:  # noqa: D102
        # Replace config by dict
        tracker: dict[int, list[tuple[int | None, list[dict]]]] = defaultdict(list)
        for stage in self._tracker.keys():
            for seed, configs in self._tracker[stage]:
                tracker[stage].append((seed, [config.get_dictionary() for config in configs]))

        return {"tracker": tracker}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: D102
        self._tracker = defaultdict(list)

        tracker = state["tracker"]
        for stage in tracker.keys():
            for seed, config_dicts in tracker[stage]:
                self._tracker[stage].append(
                    (seed, [Configuration(self._scenario.configspace, config_dict) for config_dict in config_dicts])
                )

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

    def get_trials_of_interest(
        self,
        config: Configuration,
        *,
        validate: bool = False,
        seed: int | None = None,
    ) -> list[TrialInfo]:
        is_keys = self.get_instance_seed_keys_of_interest(validate=validate, seed=seed)
        budget = None

        # When we use budgets, we always evaluated on the highest budget only
        if self.uses_budgets:
            budget = self._max_budget

        trials = []
        for key in is_keys:
            trials.append(TrialInfo(config=config, instance=key.instance, seed=key.seed, budget=budget))

        return trials

    def get_instance_seed_budget_keys(
        self, config: Configuration, compare: bool = False
    ) -> list[InstanceSeedBudgetKey]:
        """Returns the instance-seed-budget keys for a given configuration. This method supports ``highest_budget``,
        which only returns the instance-seed-budget keys for the highest budget (if specified). In this case, the
        incumbents in ``update_incumbents`` are only changed if the costs on the highest budget are lower.

        Parameters
        ----------
        compare : bool, defaults to False
            Get rid of the budget information for comparing if the configuration was evaluated on the same
            instance-seed keys.
        """
        isb_keys = self.runhistory.get_instance_seed_budget_keys(
            config, highest_observed_budget_only=self._highest_observed_budget_only
        )

        # If incumbent should only be changed on the highest budget, we have to kick out all budgets below the highest
        if self.uses_budgets and self._incumbent_selection == "highest_budget":
            isb_keys = [key for key in isb_keys if key.budget == self._max_budget]

        if compare:
            # Get rid of duplicates
            isb_keys = list(
                set([InstanceSeedBudgetKey(instance=key.instance, seed=key.seed, budget=None) for key in isb_keys])
            )

        return isb_keys

    def __iter__(self) -> Iterator[TrialInfo]:
        rh = self.runhistory

        # We have to add already existing trials from the runhistory
        # Idea: We simply add existing configs to the tracker (first stage) but assign a random instance shuffle seed.
        # In the best case, trials (added from the users) are included in the seed and it has not re-computed again.
        # Note: If the intensifier was restored, we don't want to go in here
        if len(self._tracker) == 0:
            # We batch the configs because we need n_configs in each iteration
            # If we don't have n_configs, we sample new ones
            n_configs = self._n_configs_in_stage[0]
            for configs in batch(rh.get_configs(), n_configs):
                n_rh_configs = len(configs)

                if len(configs) < n_configs:
                    try:
                        config = next(self.config_generator)
                        configs.append(config)
                    except StopIteration:
                        # We stop if we don't find any configuration anymore
                        return

                seed = self._get_next_instance_order_seed()
                self._tracker[0].append((seed, configs))
                logger.info(
                    f"Added {n_rh_configs} configs from runhistory and {n_configs - n_rh_configs} new configs to "
                    f"Successive Halving's first stage with seed {seed}."
                )

        while True:
            # If we don't yield trials anymore, we have to update
            # Otherwise, we can just keep yielding trials from the tracker
            update = False

            # We iterate over the tracker to do two things:
            # 1) Yield trials of configs that are not yet evaluated/running
            # 2) Update tracker and move better configs to the next stage
            # We start in reverse order to complete higher stages first
            logger.debug("Updating tracker:")

            # TODO: Sorting? Does it make sense? Or try yielding one trial from each stage?
            # TODO: Logging how many configs are in each stage; add as debugging info
            stages: list[int] = sorted(list(self._tracker.keys()), reverse=True)
            for stage in stages:
                pairs = self._tracker[stage].copy()
                for i, (seed, configs) in enumerate(pairs):
                    isb_keys = self._get_instance_seed_budget_keys_by_stage(stage=stage, seed=seed)

                    # We iterate over the configs and yield trials which are not running/evaluated yet
                    for config in configs:
                        config_hash = get_config_hash(config)
                        trials = self._get_next_trials(config, from_keys=isb_keys)
                        logger.debug(
                            f"--- Yielding {len(trials)}/{len(isb_keys)} for config {config_hash} in "
                            f"stage {stage} with seed {seed}..."
                        )

                        for trial in trials:
                            yield trial
                            update = True

                    # If all configs were evaluated on ``n_configs_required``, we finally can compare
                    try:
                        successful_configs = self._get_best_configs(configs, stage, from_keys=isb_keys)
                    except NotEvaluatedError:
                        # We can't compare anything, so we just continue with the next pairs
                        logger.debug("--- Could not compare configs because not all trials have been evaluated yet.")
                        continue

                    # Update tracker
                    # Remove current shuffle index / config pair
                    del self._tracker[stage][i]

                    # Add successful to the next stage
                    if stage < self._max_iterations:
                        config_ids = [rh.get_config_id(config) for config in successful_configs]
                        self._tracker[stage + 1].append((seed, successful_configs))

                        logger.debug(f"--- Promoted {len(config_ids)} configs from stage {stage} to stage {stage + 1}.")
                    else:
                        logger.debug(f"--- Removed {len(successful_configs)} configs in last stage.")

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

            # We keep track of the seed so we always evaluate on the same instances
            next_seed = self._get_next_instance_order_seed()
            self._tracker[0].append((next_seed, configs))
            logger.debug(f"Added {len(configs)} new configs to stage 0 with seed {next_seed}.")

    def _get_instance_seed_budget_keys_by_stage(
        self, stage: int, seed: int | None = None
    ) -> list[InstanceSeedBudgetKey]:
        """Returns all instances (instance-seed-budget keys in this case) for the given stage. Each stage
        is associated with a budget (N). Two possible options:

        1) Instance based: We return N instances. If a seed is specified, we shuffle the instances, before
        returning the first N instances. The budget is set to None here.
        2) Budget based: We return one instance only but the budget is set to N.
        """
        budget: float | int | None = None
        is_keys = self.get_instance_seed_keys_of_interest()

        # We have to differentiate between budgets and instance based here
        # If we are budget based, we always have one instance seed pair only
        # If we are in the instance setting, we have to return a specific number of instance seed pairs

        if self.uses_instances:
            # Shuffle instance seed pairs group-based
            if seed is not None:
                is_keys = self._reorder_instance_seed_keys(is_keys, seed=seed)

            # We only return the first N instances
            N = self._budgets_in_stage[stage]
            is_keys = is_keys[:N]
        else:
            assert len(is_keys) == 1

            # The stage defines which budget should be used (in real-valued setting)
            # No shuffle is needed here because we only have on instance seed pair
            budget = self._budgets_in_stage[stage]

        isbk = []
        for isk in is_keys:
            isbk.append(InstanceSeedBudgetKey(instance=isk.instance, seed=isk.seed, budget=budget))

        return isbk

    def _get_next_trials(
        self,
        config: Configuration,
        from_keys: list[InstanceSeedBudgetKey],
    ) -> list[TrialInfo]:
        """Returns trials for a given config from a list of instances (instance-seed-budget keys). The returned trials
        have not run or evaluated yet.
        """
        rh = self.runhistory
        evaluated_trials = rh.get_trials(config, highest_observed_budget_only=False)
        running_trials = rh.get_running_trials(config)

        next_trials: list[TrialInfo] = []
        for instance in from_keys:
            trial = TrialInfo(config=config, instance=instance.instance, seed=instance.seed, budget=instance.budget)

            if trial in evaluated_trials or trial in running_trials:
                continue

            next_trials.append(trial)

        return next_trials

    def _get_best_configs(
        self,
        configs: list[Configuration],
        stage: int,
        from_keys: list[InstanceSeedBudgetKey],
    ) -> list[Configuration]:
        try:
            n_configs = self._n_configs_in_stage[stage + 1]
        except IndexError:
            return []

        rh = self.runhistory
        configs = configs.copy()

        # TODO: Make it more efficient?
        for config in configs:
            isb_keys = self.get_instance_seed_budget_keys(config)
            if not all(isb_key in isb_keys for isb_key in from_keys):
                raise NotEvaluatedError

        selected_configs: list[Configuration] = []
        while len(selected_configs) < n_configs:
            # We calculate the pareto front for the given configs
            # We use the same isb keys for all the configs
            all_keys = [from_keys for _ in configs]
            incumbents = calculate_pareto_front(rh, configs, all_keys)

            # Idea: We recursively calculate the pareto front in every iteration
            for incumbent in incumbents:
                configs.remove(incumbent)
                selected_configs.append(incumbent)

        # If we have more selected configs, we remove the last ones
        if len(selected_configs) > n_configs:
            selected_configs = selected_configs[:n_configs]

        return selected_configs

    def _get_next_instance_order_seed(self) -> int | None:
        """Next instances shuffle seed to use."""
        # Here we have the option to shuffle the trials when specified by the user
        if self._instance_order == "shuffle":
            seed = self._rng.randint(0, MAXINT)
        elif self._instance_order == "shuffle_once":
            seed = 0
        else:
            seed = None

        return seed
