from __future__ import annotations

from typing import Any, Iterator, Optional

import math
from collections import defaultdict

import numpy as np
from ConfigSpace import Configuration

from smac.callback.multifidelity_stopping_callback import MultiFidelityStoppingCallback
from smac.constants import MAXINT
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.intensifier.search_space_modifier import AbstractSearchSpaceModifier
from smac.intensifier.stage_information import Stage
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import InstanceSeedBudgetKey
from smac.runhistory.errors import NotEvaluatedError
from smac.scenario import Scenario
from smac.utils.logging import get_logger
from smac.utils.pareto_front import calculate_pareto_front, sort_by_crowding_distance

__copyright__ = "Copyright 2023, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class SuccessiveHalving(AbstractIntensifier):
    """
    Implementation of Successive Halving supporting multi-fidelity, multi-objective, and multi-processing.
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
    sample_brackets_at_once : bool, defaults to False
        Whether to sample all configurations in a bracket at once or one at a time, after a potential surrogate model
        retrain.
    seed : int, defaults to None
        Internal seed used for random events like shuffle seeds.
    early_stopping : MultiFidelityStoppingCallback, defaults to None
        Callback used to stop the single fidelities.
    remove_stopped_fidelities_mode : str, defaults to None
        Whether to remove stopped fidelities from the tracker or not. Can be set to:

        - `incrementally`: Remove fidelities only if the previous fidelity has also been stopped.
        - `single`: Any fidelity can be removed independently of the other fidelities.
        - `cascade`: If a fidelity is stopped, all lower fidelities are also removed.
    only_go_to_next_fidelity_after_early_stopping : bool, defaults to False
        Whether to only go to the next fidelity after early stopping or not (essentially removes bracket size
        limitations).
    modify_search_space_on_stop : AbstractSearchSpaceModifier, defaults to None
        Search space modifier to use when stopping a fidelity.
    """

    def __init__(
        self,
        scenario: Scenario,
        eta: int = 3,
        n_seeds: int = 1,
        instance_seed_order: str | None = "shuffle_once",
        max_incumbents: int = 10,
        incumbent_selection: str = "highest_observed_budget",
        sample_brackets_at_once: bool = False,
        seed: int | None = None,
        early_stopping: MultiFidelityStoppingCallback | None = None,
        remove_stopped_fidelities_mode: str = None,
        only_go_to_next_fidelity_after_early_stopping: bool = False,
        modify_search_space_on_stop: AbstractSearchSpaceModifier = None,
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
        self._sample_brackets_at_once = sample_brackets_at_once
        self._early_stopping = early_stopping
        self._remove_stopped_fidelities_mode = remove_stopped_fidelities_mode
        self._only_go_to_next_fidelity_after_early_stopping = only_go_to_next_fidelity_after_early_stopping
        self._shrink_search_space_on_stop = modify_search_space_on_stop

        # Global variables derived from scenario
        self._min_budget = self._scenario.min_budget
        self._max_budget = self._scenario.max_budget

        # States
        self._tracker: dict[tuple[int, int], list[tuple[int | None, list[Configuration]]]] = defaultdict(list)
        self._configs_from_rh: list[Configuration] = []
        # map repetition, bracket, and stage to stage information
        self._open_stages: dict[tuple[int, int, int], Stage] = dict()
        self._stopped_fidelities: set[float] = set()
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

        if (
            self._only_go_to_next_fidelity_after_early_stopping or self._remove_stopped_fidelities_mode is not None
        ) and self._early_stopping is None:
            raise ValueError(
                f"Early stopping has to be enabled for the options "
                f"only_go_to_next_fidelity_after_early_stopping and remove_stopped_fidelities_"
                f"incrementally. Current options are: "
                f"only_go_to_next_fidelity_after_early_stopping={self._only_go_to_next_fidelity_after_early_stopping}, "
                f"remove_stopped_fidelities_mode={self._remove_stopped_fidelities_mode}, "
                f"early_stopping={self._early_stopping}."
            )

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
    ) -> tuple[list[float], list[int]]:
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
        for budget, isb_s in configs_by_budgets.items():
            average_cost = self.runhistory.average_cost(config=config, instance_seed_budget_keys=isb_s)

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
        self.__post_init__()
        self._log_initial_state()
        self._check_runhistory_for_invalid_existing_trials()
        self._configs_from_rh = self.runhistory.get_configs()

        # Initialize the first stage
        self._get_next_bracket()  # move bracket counter
        self._seeds_per_bracket[(0, 0)] = self._get_next_order_seed()
        isb_keys = self._get_instance_seed_budget_keys_by_stage(0, 0, self._seeds_per_bracket[(0, 0)])
        first_stage = Stage(
            repetition=0,
            bracket=0,
            stage=0,
            budget=self._budgets_in_stage[0][0],
            amount_configs_to_yield=self._n_configs_in_stage[0][0],
            isb_keys=isb_keys,
        )
        self._open_stages[(0, 0, 0)] = first_stage
        self._next_repetition = 1

        while True:
            logger.debug(
                f"Starting Successive Halving iteration with open stages:"
                f" {[open_stage_ for open_stage_ in self._open_stages]}"
            )
            if len(self._open_stages) == 0:
                logger.info("No more stages to process. Exiting.")
                return

            stages_to_delete = []
            stages_to_add = {}

            for repetition, bracket, stage in sorted(self._open_stages.keys()):
                stage_info = self._open_stages[(repetition, bracket, stage)]
                logger.debug(f"Processing repetition {repetition}, bracket {bracket}, stage {stage}")
                logger.debug(f"{stage_info}")

                # skip stage if possible
                if self._skip_stage(stage_info):
                    logger.info(f"Skipping the rest of stage {stage}")
                    stage_info.terminate()
                    if self._shrink_search_space_on_stop is not None:
                        self._shrink_search_space_on_stop.modify_search_space(
                            self._scenario.configspace, self.runhistory
                        )
                elif self._only_go_to_next_fidelity_after_early_stopping and stage_info.all_configs_yielded:
                    stage_info.amount_configs_to_yield += 1

                # --- yield configs if necessary
                if not stage_info.all_configs_yielded and not stage_info.terminated:
                    # sample configs if necessary
                    if (
                        stage_info.amount_configs_yielded >= len(stage_info.configs)
                        and not stage_info.all_configs_generated
                    ):
                        configs_to_sample = (
                            1 if not self._sample_brackets_at_once else stage_info.amount_configs_to_yield
                        )

                        for i in range(configs_to_sample):
                            try:
                                if len(self._configs_from_rh) > 0:
                                    config = self._configs_from_rh.pop(0)
                                else:
                                    config = next(self._config_generator)  # type: ignore

                                stage_info.add_config(config)
                            except StopIteration:
                                logger.info(f"--- No more configs available in stage {stage}.")
                                stage_info.terminate()
                                if np.all(
                                    [
                                        open_stage.all_configs_yielded or open_stage.terminated
                                        for open_stage in self._open_stages.values()
                                    ]
                                ):
                                    return
                                continue
                            except IndexError:
                                logger.info(
                                    f"--- IndexError: No more configs available in stage {stage} due to search"
                                    f"space shrinking."
                                )
                                stage_info.terminate()
                                if np.all(
                                    [
                                        open_stage.all_configs_yielded or open_stage.terminated
                                        for open_stage in self._open_stages.values()
                                    ]
                                ):
                                    return
                                continue

                    # yield the trials for the next config
                    config = stage_info.get_next_config()
                    trials_for_config = self._get_next_trials(config, from_keys=stage_info.isb_keys)
                    stage_info.add_trials_for_config(config, trials_for_config)
                    logger.debug(
                        f"--- Yielding {len(trials_for_config)}/{len(stage_info.isb_keys)} in "
                        f"stage {stage} with seed {self._seeds_per_bracket[(repetition, bracket)]}..."
                    )

                    for trial in trials_for_config:
                        yield trial

                # --- promote configs if possible
                if (
                    stage_info.all_configs_yielded and not self._only_go_to_next_fidelity_after_early_stopping
                ) or stage_info.terminated:
                    try:
                        successful_configs = self._get_best_configs(stage_info)

                        # Remove current stage from tracker
                        stages_to_delete.append((repetition, bracket, stage))

                        # Add successful to the next stage
                        amount_configs_to_yield = len(successful_configs)
                        if stage < self._max_iterations[bracket] - 1:
                            stages_to_add[(repetition, bracket, stage + 1)] = Stage(
                                repetition=repetition,
                                bracket=bracket,
                                stage=stage + 1,
                                budget=self._budgets_in_stage[bracket][stage + 1],
                                amount_configs_to_yield=amount_configs_to_yield,
                                isb_keys=self._get_instance_seed_budget_keys_by_stage(
                                    bracket,
                                    stage + 1,
                                    self._seeds_per_bracket[(repetition, bracket)],
                                ),
                                configs=successful_configs,
                            )

                            logger.debug(
                                f"--- Promoted {len(successful_configs)} configs from stage {stage} to stage "
                                f"{stage + 1} in bracket {bracket}."
                            )
                        else:
                            logger.debug(
                                f"--- Removed {len(successful_configs)} configs to last stage in bracket {bracket}."
                            )
                    except NotEvaluatedError:
                        logger.debug("--- Could not compare configs because not all trials have been evaluated yet.")

                    # Check if we need to open a new repetition or bracket
                    self._open_new_repetition_or_bracket_if_necessary(repetition, stages_to_add)

            # Update the tracker and remove stages that are done and add new stages
            for repetition, bracket, stage in stages_to_delete:
                logger.debug(f"--- Removing repetition {repetition}, bracket {bracket}, stage {stage} from tracker.")
                del self._open_stages[repetition, bracket, stage]

            for repetition, bracket, stage in stages_to_add.keys():
                self._open_stages[(repetition, bracket, stage)] = stages_to_add[(repetition, bracket, stage)]

    def _open_new_repetition_or_bracket_if_necessary(
        self, repetition: int, stages_to_add: dict[tuple[int, int, int], Stage]
    ) -> None:
        if len(stages_to_add) > 0:
            return
        if np.all(
            [open_stage.all_configs_yielded or open_stage.terminated for open_stage in self._open_stages.values()]
        ):
            # Increase counters
            next_bracket = self._get_next_bracket()

            if next_bracket == 0:
                next_repetition = self._next_repetition
                self._next_repetition += 1
            else:
                next_repetition = repetition

            self._seeds_per_bracket[(next_repetition, next_bracket)] = self._get_next_order_seed()
            next_stage = 0

            # Incrementally check if the stage can still be opened, or the fidelity has already been closed
            if self._remove_stopped_fidelities_mode is not None:
                while self._budgets_in_stage[next_bracket][next_stage] in self._stopped_fidelities:
                    next_stage += 1

                    if next_stage > len(self._budgets_in_stage[next_bracket]) - 1:
                        logger.info("--- All fidelities have been closed.")
                        return

            new_stage = Stage(
                repetition=next_repetition,
                bracket=next_bracket,
                stage=next_stage,
                budget=self._budgets_in_stage[next_bracket][next_stage],
                amount_configs_to_yield=self._n_configs_in_stage[next_bracket][next_stage],
                isb_keys=self._get_instance_seed_budget_keys_by_stage(
                    next_bracket, next_stage, self._seeds_per_bracket[(next_repetition, next_bracket)]
                ),
            )
            stages_to_add[(next_repetition, next_bracket, next_stage)] = new_stage
            logger.debug(f"--- Opened new repetition {next_repetition}, bracket {next_bracket}, stage {next_stage}.")
        else:
            logger.debug("--- Not all stages are done yet, so no new repetition or bracket is opened.")

    def _log_initial_state(self) -> None:
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

    def _get_best_configs(self, stage_info: Stage) -> list[Configuration]:
        """Returns the best configurations. The number of configurations is depending on the stage. Raises
        ``NotEvaluatedError`` if not all trials have been evaluated.
        """
        try:
            n_configs_next_stage = self._n_configs_in_stage[stage_info.bracket][stage_info.stage + 1]
        except IndexError:
            return []

        if self._only_go_to_next_fidelity_after_early_stopping:
            n_configs_next_stage = max(n_configs_next_stage, int(stage_info.amount_configs_yielded / self._eta))

        configs = stage_info.yielded_configs.copy()
        from_keys = stage_info.isb_keys

        # check that all configs have been evaluated - however, not necessary in the case of early stopping, in which
        # case we simply collect all evaluated configs
        evaluated = []
        for config in configs:
            isb_keys = self.runhistory.get_instance_seed_budget_keys(config)

            if all(isb_key in isb_keys for isb_key in from_keys):
                evaluated.append(config)
            elif not stage_info.terminated:
                raise NotEvaluatedError

        # We only select the best configs out of the evaluated ones in the case of early stopping
        configs = evaluated
        n_configs_next_stage = min(n_configs_next_stage, len(configs))

        selected_configs: list[Configuration] = []
        while len(selected_configs) < n_configs_next_stage:
            # We calculate the pareto front for the given configs
            # We use the same isb keys for all the configs
            all_keys = [from_keys for _ in configs]
            incumbents = calculate_pareto_front(self.runhistory, configs, all_keys)

            # Idea: We recursively calculate the pareto front in every iteration
            for incumbent in incumbents:
                configs.remove(incumbent)
                selected_configs.append(incumbent)

        # If we have more selected configs, we remove the ones with the smallest crowding distance
        if len(selected_configs) > n_configs_next_stage:
            selected_configs = sort_by_crowding_distance(self.runhistory, configs, all_keys)[:n_configs_next_stage]
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

    def _skip_stage(self, stage_info: Stage) -> bool:
        if self._early_stopping is None:
            return False

        # no early stopping if no config evaluated or all already yielded (except in cases where skipping is beneficial
        # even when all configs already have been yielded)
        skip_last_config = (
            self._only_go_to_next_fidelity_after_early_stopping or self._remove_stopped_fidelities_mode is not None
        )
        if stage_info.amount_configs_yielded == 0 or (stage_info.all_configs_yielded and (not skip_last_config)):
            return False

        stop = self._early_stopping.should_stage_stop(self.runhistory, self._scenario, stage_info)

        # If stopped and either the stages' budget is the minimum budget or the previous fidelity has also been stopped,
        # we add the budget to the stopped fidelities
        if stop and self._remove_stopped_fidelities_mode is not None:
            fidelities_to_stop = []

            if self._remove_stopped_fidelities_mode == "single":
                fidelities_to_stop = [stage_info.budget]
            elif self._remove_stopped_fidelities_mode == "incrementally":
                is_min_budget = stage_info.budget == self._budgets_in_stage[0][0]
                previous_fidelity_stopped = self._budgets_in_stage[0][stage_info.stage - 1] in self._stopped_fidelities
                if is_min_budget or previous_fidelity_stopped:
                    fidelities_to_stop = [stage_info.budget]
            elif self._remove_stopped_fidelities_mode == "cascade":
                for budget in self._budgets_in_stage[0]:
                    if budget > stage_info.budget:
                        break
                    fidelities_to_stop.append(budget)
            else:
                raise ValueError(f"Unknown remove_stopped_fidelities_mode: {self._remove_stopped_fidelities_mode}")

            for fidelity in fidelities_to_stop:
                self._stopped_fidelities.add(fidelity)

                logger.info(f"Stopping fidelity {fidelity} due to early stopping.")
                for stage in self._open_stages.values():
                    if stage.budget == fidelity:
                        stage.terminate()

        return stop
