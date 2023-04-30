from __future__ import annotations

from typing import Any, Iterator, Optional

import math
from collections import defaultdict

import numpy as np
from ConfigSpace import Configuration

from smac.constants import MAXINT
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.intensifier.stage_information import Stage
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import InstanceSeedBudgetKey
from smac.runhistory.errors import NotEvaluatedError
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash
from smac.utils.logging import get_logger
from smac.utils.pareto_front import calculate_pareto_front, sort_by_crowding_distance

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class SuccessiveHalving(AbstractIntensifier):
    """
    Implementation of Succesive Halving supporting multi-fidelity, multi-objective, and multi-processing.
    Internally, a tracker keeps track of configurations and their bracket and stage.

    The behaviour of this intensifier is as follows:

    - First, adds configurations from the runhistory to the tracker. The first stage is always filled-up. For example,
      the user provided 4 configs with the tell-method but the first stage requires 8 configs: 4 new configs are
      sampled and added together with the provided configs as a group to the tracker.
    - While loop:

      - If a trial in the tracker has not been yielded yet, yield it.
      - If we are running out of trials, we simply add a new batch of configurations to the first stage.

    Note
    ----
    The implementation natively supports brackets from Hyperband. However, in the case of Successive Halving,
    only one bracket is used.

    Parameters
    ----------
    eta : int, defaults to 3
        Input that controls the proportion of configurations discarded in each round of Successive Halving.
    n_seeds : int, defaults to 1
        How many seeds to use for each instance.
    instance_seed_order : str, defaults to "shuffle_once"
        How to order the instance-seed pairs. Can be set to:

        - `None`: No shuffling at all and use the instance-seed order provided by the user.
        - `shuffle_once`: Shuffle the instance-seed keys once and use the same order across all runs.
        - `shuffle`: Shuffles the instance-seed keys for each bracket individually.
    incumbent_selection : str, defaults to "highest_observed_budget"
        How to select the incumbent when using budgets. Can be set to:

        - `any_budget`: Incumbent is the best on any budget i.e., the best performance regardless of budget.
        - `highest_observed_budget`: Incumbent is the best in the highest budget run so far.
        - `highest_budget`: Incumbent is selected only based on the highest budget.
    max_incumbents : int, defaults to 10
        How many incumbents to keep track of in the case of multi-objective.
    seed : int, defaults to None
        Internal seed used for random events like shuffle seeds.
    """

    def __init__(
        self,
        scenario: Scenario,
        eta: int = 3,
        n_seeds: int = 1,
        instance_seed_order: str | None = "shuffle_once",
        max_incumbents: int = 10,
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
        self._instance_seed_order = instance_seed_order
        self._incumbent_selection = incumbent_selection
        self._highest_observed_budget_only = False if incumbent_selection == "any_budget" else True

        # Global variables derived from scenario
        self._min_budget = self._scenario.min_budget
        self._max_budget = self._scenario.max_budget

        # States
        self._tracker: dict[tuple[int, int], list[tuple[int | None, list[Configuration]]]] = defaultdict(list)
        self._configs_from_rh: list[Configuration] = []
        # map repetition, bracket, and stage to stage information
        self._open_stages: dict[tuple[int, int, int], Stage] = dict()
        # map repetition and bracket to seed
        self._seeds_per_bracket: dict[tuple[int, int], Optional[int]] = dict()
        self._next_repetition = 0

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "eta": self._eta,
                "instance_seed_order": self._instance_seed_order,
                "incumbent_selection": self._incumbent_selection,
            }
        )

        return meta

    def reset(self) -> None:
        """Reset the internal variables of the intensifier including the tracker."""
        super().reset()

        # States
        # dict[tuple[bracket, stage], list[tuple[seed to shuffle instance-seed keys, list[config_id]]]
        self._tracker = defaultdict(list)

    def __post_init__(self) -> None:
        """Post initialization steps after the runhistory has been set."""
        super().__post_init__()

        # We generate our instance seed pairs once
        is_keys = self.get_instance_seed_keys_of_interest()

        # Budgets, followed by lots of sanity-checking
        eta = self._eta
        min_budget = self._min_budget
        max_budget = self._max_budget

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
        max_iter = self._get_max_iterations(eta, max_budget, min_budget)
        budgets, n_configs = self._compute_configs_and_budgets_for_stages(eta, max_budget, max_iter)

        # Global variables
        self._min_budget = min_budget
        self._max_budget = max_budget

        # Stage variables, depending on the bracket (0 is the bracket here since SH only has one bracket)
        self._max_iterations: dict[int, int] = {0: max_iter + 1}
        self._n_configs_in_stage: dict[int, list] = {0: n_configs}
        self._budgets_in_stage: dict[int, list] = {0: budgets}

    @staticmethod
    def _get_max_iterations(eta: int, max_budget: float | int, min_budget: float | int) -> int:
        return int(np.floor(np.log(max_budget / min_budget) / np.log(eta)))

    @staticmethod
    def _compute_configs_and_budgets_for_stages(
        eta: int, max_budget: float | int, max_iter: int, s_max: int | None = None
    ) -> tuple[list[int], list[int]]:
        if s_max is None:
            s_max = max_iter

        n_initial_challengers = math.ceil((eta**max_iter) * (s_max + 1) / (max_iter + 1))

        # How many configs in each stage
        lin_space = -np.linspace(0, max_iter, max_iter + 1)
        n_configs_ = np.floor(n_initial_challengers * np.power(eta, lin_space))
        n_configs = np.array(np.round(n_configs_), dtype=int).tolist()

        # How many budgets in each stage
        lin_space = -np.linspace(max_iter, 0, max_iter + 1)
        budgets = (max_budget * np.power(eta, lin_space)).tolist()

        return budgets, n_configs

    def get_state(self) -> dict[str, Any]:  # noqa: D102
        # Replace config by dict
        tracker: dict[str, list[tuple[int | None, list[dict]]]] = defaultdict(list)
        for key in list(self._tracker.keys()):
            for seed, configs in self._tracker[key]:
                # We have to make key serializable
                new_key = f"{key[0]},{key[1]}"
                tracker[new_key].append((seed, [config.get_dictionary() for config in configs]))

        return {"tracker": tracker}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: D102
        self._tracker = defaultdict(list)

        tracker = state["tracker"]
        for old_key in list(tracker.keys()):
            keys = [k for k in old_key.split(",")]
            new_key = (int(keys[0]), int(keys[1]))
            for seed, config_dicts in tracker[old_key]:
                seed = None if seed is None else int(seed)
                self._tracker[new_key].append(
                    (
                        seed,
                        [Configuration(self._scenario.configspace, config_dict) for config_dict in config_dicts],
                    )
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

    def print_tracker(self) -> None:
        """Prints the number of configurations in each bracket/stage."""
        messages = []
        for (bracket, stage), others in self._tracker.items():
            counter = 0
            for _, config_ids in others:
                counter += len(config_ids)

            if counter > 0:
                messages.append(f"--- Bracket {bracket} / Stage {stage}: {counter} configs")

        if len(messages) > 0:
            logger.debug(f"{self.__class__.__name__} statistics:")

        for message in messages:
            logger.debug(message)

    def get_trials_of_interest(
        self,
        config: Configuration,
        *,
        validate: bool = False,
        seed: int | None = None,
    ) -> list[TrialInfo]:  # noqa: D102
        is_keys = self.get_instance_seed_keys_of_interest(validate=validate, seed=seed)
        budget = None

        # When we use budgets, we always evaluated on the highest budget only
        if self.uses_budgets:
            budget = self._max_budget

        trials = []
        for key in is_keys:
            trials.append(TrialInfo(config=config, instance=key.instance, seed=key.seed, budget=budget))

        return trials

    def _sub_select_incumbent_isb(
        self, incumbents: list[Configuration], isb_keys: list[list[InstanceSeedBudgetKey]]
    ) -> list[list[InstanceSeedBudgetKey]]:

        if not self._incumbent_selection == "any_budget":
            return isb_keys

        for i in range(len(incumbents)):
            isb_keys[i] = self._get_best_cost_for_config(incumbents[i], isb_keys[i])

        return isb_keys

    def _get_best_cost_for_config(
        self, config: Configuration, isb_keys: list[InstanceSeedBudgetKey]
    ) -> list[InstanceSeedBudgetKey]:
        # step one: merge all the usb keys for one budget
        budgets = set([isb.budget for isb in isb_keys])
        configs_by_budgets: dict[Optional[float], list[InstanceSeedBudgetKey]] = {budget: [] for budget in budgets}
        for isb in isb_keys:
            configs_by_budgets[isb.budget].append(isb)

        # step two: get the average performance for each budget
        runhistory_average_cost_by_budget = {}
        for budget, isbs in configs_by_budgets.items():
            average_cost = self.runhistory.average_cost(config=config, instance_seed_budget_keys=isbs)

            if type(average_cost) == list:
                raise NotImplementedError(
                    "SH with any-budget incumbent selection only supports single-objective " "scenarios."
                )

            runhistory_average_cost_by_budget[budget] = average_cost

        # step three: get the best performance
        best_budget = min(runhistory_average_cost_by_budget, key=runhistory_average_cost_by_budget.get)  # type: ignore

        return configs_by_budgets[best_budget]

    def get_instance_seed_budget_keys(
        self, config: Configuration, compare: bool = False
    ) -> list[InstanceSeedBudgetKey]:
        """Returns the instance-seed-budget keys for a given configuration. This method supports ``highest_budget``,
        which only returns the instance-seed-budget keys for the highest budget (if specified). In this case, the
        incumbents in ``update_incumbents`` are only changed if the costs on the highest budget are lower.

        Parameters
        ----------
        config: Configuration
            The Configuration to be queried
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

    def __iter__(self) -> Iterator[TrialInfo]:  # noqa: D102
        self._log_state()

        self.__post_init__()
        self._check_runhistory_for_invalid_existing_trials()
        self._configs_from_rh = self.runhistory.get_configs()

        # Initialize the first stage
        self._seeds_per_bracket[(0, 0)] = self._get_next_order_seed()
        isb_keys = self._get_instance_seed_budget_keys_by_stage(0, 0, self._seeds_per_bracket[(0, 0)])
        self._open_stages[(0, 0, 0)] = Stage(amount_configs_to_yield=self._n_configs_in_stage[0][0], isb_keys=isb_keys)
        self._next_repetition = 1

        while True:

            stages_to_delete = []
            stages_to_add = {}

            for repetition, bracket, stage in sorted(self._open_stages.keys()):
                stage_info = self._open_stages[(repetition, bracket, stage)]

                if not stage_info.is_done:
                    if stage == 0:
                        try:
                            if len(self._configs_from_rh) > 0:
                                config = self._configs_from_rh.pop(0)
                            else:
                                config = next(self._config_generator)  # type: ignore

                            stage_info.add_config(config)
                            seed = self._seeds_per_bracket[(repetition, bracket)]

                            # We yield trials which are not running/evaluated yet
                            config_hash = get_config_hash(config)
                            trials_for_config = self._get_next_trials(config, from_keys=stage_info.isb_keys)
                            stage_info.add_trials_for_config(config, trials_for_config)
                            logger.debug(
                                f"--- Yielding {len(trials_for_config)}/{len(stage_info.isb_keys)} for config "
                                f"{config_hash} in "
                                f"stage {stage} with seed {seed}..."
                            )

                            for trial in trials_for_config:
                                yield trial

                            stage_info.amount_configs_yielded += 1

                        except StopIteration:
                            # We ran out of configs
                            # TODO - only in this stage - might still be able to progress in other stages if we wait.
                            # TODO currently this will lead to a forever loop if we run out of configs
                            # logger.debug(f"--- No more configs available in stage {stage}.")
                            # stage_info.amount_configs_to_yield = stage_info.amount_configs_yielded
                            return

                # When we have yielded all the configs, we check if we want to promote configs.
                if stage_info.is_done:
                    try:
                        successful_configs = self._get_best_configs(
                            stage_info.configs, bracket, stage, stage_info.isb_keys
                        )

                        # Mark current stage for removal from tracker
                        stages_to_delete.append((repetition, bracket, stage))

                        # Add successful to the next stage
                        if stage < self._max_iterations[bracket] - 1:
                            config_ids = [self.runhistory.get_config_id(config) for config in successful_configs]
                            stages_to_add[(repetition, bracket, stage + 1)] = Stage(
                                amount_configs_to_yield=self._n_configs_in_stage[bracket][stage + 1],
                                isb_keys=self._get_instance_seed_budget_keys_by_stage(
                                    bracket, stage + 1, self._seeds_per_bracket[(repetition, bracket)]
                                ),
                            )

                            logger.debug(
                                f"--- Promoted {len(config_ids)} configs from stage {stage} to stage {stage + 1} in "
                                f"bracket {bracket}."
                            )
                        else:
                            logger.debug(
                                f"--- Removed {len(successful_configs)} configs to last stage in bracket {bracket}."
                            )
                            self._open_new_repetition_if_necessary(bracket, stages_to_add)
                    except NotEvaluatedError:
                        # We can't compare anything, so we just continue with the next pairs
                        logger.debug("--- Could not compare configs because not all trials have been evaluated yet.")
                        self._open_new_repetition_if_necessary(bracket, stages_to_add)

            # We have to update the tracker and remove stages that are done and add new stages
            for repetition, bracket, stage in stages_to_delete:
                del self._open_stages[repetition, bracket, stage]

            for repetition, bracket, stage in stages_to_add.keys():
                self._open_stages[(repetition, bracket, stage)] = stages_to_add[(repetition, bracket, stage)]

    def _open_new_repetition_if_necessary(self, bracket: int, stages_to_add: dict[tuple[int, int, int], Stage]) -> None:
        if np.all([open_stage.is_done for open_stage in self._open_stages.values()]):
            self._seeds_per_bracket[(self._next_repetition, bracket)] = self._get_next_order_seed()
            stages_to_add[(self._next_repetition, 0, 0)] = Stage(
                amount_configs_to_yield=self._n_configs_in_stage[0][0],
                isb_keys=self._get_instance_seed_budget_keys_by_stage(
                    0, 0, self._seeds_per_bracket[(self._next_repetition, 0)]
                ),
            )
            self._next_repetition += 1

    def _log_state(self) -> None:
        # Log brackets/stages
        logger.info("Number of configs in stage:")
        for bracket, n in self._n_configs_in_stage.items():
            logger.info(f"--- Bracket {bracket}: {n}")
        logger.info("Budgets in stage:")
        for bracket, budgets in self._budgets_in_stage.items():
            logger.info(f"--- Bracket {bracket}: {budgets}")

    def _check_runhistory_for_invalid_existing_trials(self) -> None:
        # Print ignored budgets
        ignored_budgets = []
        for k in self.runhistory.keys():
            if k.budget not in self._budgets_in_stage[0] and k.budget not in ignored_budgets:
                ignored_budgets.append(k.budget)
        if len(ignored_budgets) > 0:
            logger.warning(
                f"Trials with budgets {ignored_budgets} will been ignored. Consider adding trials with budgets "
                f"{self._budgets_in_stage[0]}."
            )

    def _get_instance_seed_budget_keys_by_stage(
        self,
        bracket: int,
        stage: int,
        seed: int | None = None,
    ) -> list[InstanceSeedBudgetKey]:
        """Returns all instance-seed-budget keys (isb keys) for the given stage. Each stage
        is associated with a budget (budget_n). Two possible options:

        1) Instance based: We return budget_n isb keys. If a seed is specified, we shuffle the keys before
        returning the first budget_n instances. The budget is set to None here.
        2) Budget based: We return one isb only but the budget is set to budget_n.
        """
        budget: float | int | None = None
        is_keys = self.get_instance_seed_keys_of_interest()

        # We have to differentiate between budgets and instances based here
        # If we are budget based, we always have one instance seed pair only
        # If we are in the instance setting, we have to return a specific number of instance seed pairs

        if self.uses_instances:
            # Shuffle instance seed pairs group-based
            if seed is not None:
                is_keys = self._reorder_instance_seed_keys(is_keys, seed=seed)

            # We only return the first budget_n instances
            budget_n = int(self._budgets_in_stage[bracket][stage])
            is_keys = is_keys[:budget_n]
        else:
            assert len(is_keys) == 1

            # The stage defines which budget should be used (in real-valued setting)
            # No shuffle is needed here because we only have on instance seed pair
            budget = self._budgets_in_stage[bracket][stage]

        isb_keys = []
        for isk in is_keys:
            isb_keys.append(InstanceSeedBudgetKey(instance=isk.instance, seed=isk.seed, budget=budget))

        return isb_keys

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
        bracket: int,
        stage: int,
        from_keys: list[InstanceSeedBudgetKey],
    ) -> list[Configuration]:
        """Returns the best configurations. The number of configurations is depending on the stage. Raises
        ``NotEvaluatedError`` if not all trials have been evaluated.
        """
        try:
            n_configs = self._n_configs_in_stage[bracket][stage + 1]
        except IndexError:
            return []

        rh = self.runhistory
        configs = configs.copy()

        for config in configs:
            isb_keys = rh.get_instance_seed_budget_keys(config)
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

        # If we have more selected configs, we remove the ones with the smallest crowding distance
        if len(selected_configs) > n_configs:
            selected_configs = sort_by_crowding_distance(rh, configs, all_keys)[:n_configs]
            logger.debug("Found more configs than required. Removed configs with smallest crowding distance.")

        return selected_configs

    def _get_next_order_seed(self) -> int | None:
        """Next instances shuffle seed to use."""
        # Here we have the option to shuffle the trials when specified by the user
        if self._instance_seed_order == "shuffle":
            seed = self._rng.randint(0, MAXINT)
        elif self._instance_seed_order == "shuffle_once":
            seed = 0
        else:
            seed = None

        return seed

    def _get_next_bracket(self) -> int:
        """Successive Halving only uses one bracket. Therefore, we always return 0 here."""
        return 0
