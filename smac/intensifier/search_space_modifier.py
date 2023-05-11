from typing import Callable

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)

from smac import RunHistory
from smac.constants import VERY_SMALL_NUMBER
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class AbstractSearchSpaceModifier:
    def modify_search_space(self, search_space: ConfigurationSpace, runhistory: RunHistory) -> None:
        """Modifies the search space."""
        raise NotImplementedError()


class MultiFidelitySearchSpaceShrinker(AbstractSearchSpaceModifier):
    """Shrinks the search space by setting the boundaries around a percentage of best performing configurations."""

    def __init__(
        self,
        get_hyperparameter_for_bounds: dict[str, Callable],
        percentage_configurations: float,
        max_shrinkage: float = 0.5,
        seed: int = 0,
    ):
        self.percentage_configurations = percentage_configurations
        self.max_shrinkage = max_shrinkage
        self.get_hyperparameter_for_bounds = get_hyperparameter_for_bounds
        self.random_state = np.random.RandomState(seed)

    def modify_search_space(self, search_space: ConfigurationSpace, runhistory: RunHistory) -> None:  # noqa: D102
        # find best configurations
        configs = runhistory.get_configs(sort_by="cost")
        selected_configs = configs[: int(len(configs) * self.percentage_configurations)]

        hparams = search_space.get_hyperparameter_names()
        hparam_values: dict[str, list] = {hparam: [config[hparam] for config in selected_configs] for hparam in hparams}

        # compute the new boundaries
        for hyperparameter in hparams:
            if isinstance(search_space[hyperparameter], NumericalHyperparameter):
                old_min_boundary = search_space[hyperparameter].lower
                old_max_boundary = search_space[hyperparameter].upper

                new_min_boundary = min(hparam_values[hyperparameter])
                new_max_boundary = max(hparam_values[hyperparameter])

                # don't expand
                new_min_boundary = max(new_min_boundary, old_min_boundary)
                new_max_boundary = min(new_max_boundary, old_max_boundary)

                # don't move beyond legal borders
                new_min_boundary = min(new_min_boundary, old_max_boundary)
                new_max_boundary = max(new_max_boundary, old_min_boundary)

                # compute shrinkage for both directions
                shrinkage_lower = new_min_boundary - old_min_boundary
                shrinkage_upper = old_max_boundary - new_max_boundary
                old_size = old_max_boundary - old_min_boundary
                new_size = new_max_boundary - new_min_boundary

                if new_size == 0:
                    shrinkage_lower -= VERY_SMALL_NUMBER if shrinkage_lower != 0 else 0
                    shrinkage_upper -= VERY_SMALL_NUMBER if shrinkage_upper != 0 else 0
                    new_size = old_size - (shrinkage_upper + shrinkage_lower)

                if new_size / old_size < self.max_shrinkage:
                    new_new_size = old_size * self.max_shrinkage
                    shrinkage_lower_percentage = shrinkage_lower / (shrinkage_upper + shrinkage_lower)
                    shrinkage_upper_percentage = shrinkage_upper / (shrinkage_upper + shrinkage_lower)

                    shrinkage_lower = shrinkage_lower_percentage * (old_size - new_new_size)
                    shrinkage_upper = shrinkage_upper_percentage * (old_size - new_new_size)

                    if isinstance(search_space[hyperparameter], IntegerHyperparameter):
                        shrinkage_lower = round(shrinkage_lower)
                        shrinkage_upper = round(shrinkage_upper)

                        if (old_size - (shrinkage_upper + shrinkage_lower) / old_size) < self.max_shrinkage:
                            if shrinkage_lower_percentage > shrinkage_upper_percentage:
                                shrinkage_upper = shrinkage_upper - 1
                            elif shrinkage_upper_percentage > shrinkage_lower_percentage:
                                shrinkage_lower = shrinkage_lower - 1
                            else:
                                shrinkage_upper = shrinkage_upper - 1
                                shrinkage_lower = shrinkage_lower - 1

                    new_min_boundary = old_min_boundary + shrinkage_lower
                    new_max_boundary = old_max_boundary - shrinkage_upper

                    new_size = new_max_boundary - new_min_boundary

                if new_size != old_size:
                    # min and max cannot be the same
                    if isinstance(search_space[hyperparameter], IntegerHyperparameter):
                        new_min_boundary = round(new_min_boundary)
                        new_max_boundary = round(new_max_boundary)

                        # don't completely remove the hyperparameter
                        if new_min_boundary == new_max_boundary:
                            # if lower shrinkage greater than upper shrinkage, increase upper bound
                            if shrinkage_lower > shrinkage_upper and new_max_boundary + 1 <= old_max_boundary:
                                new_max_boundary = new_max_boundary + 1
                            elif shrinkage_upper < shrinkage_lower:
                                new_min_boundary = new_max_boundary - 1
                            else:
                                # if both shrinkages are equal, move both bounds
                                new_max_boundary = new_max_boundary + 1
                                new_min_boundary = new_min_boundary - 1

                    logger.debug(
                        f"Shrinking {hyperparameter}, old min {old_min_boundary}, old max {old_max_boundary}, "
                        f"shrinkage lower {shrinkage_lower}, shrinkage upper {shrinkage_upper}, "
                        f"new lower {new_min_boundary}, new upper {new_max_boundary}"
                    )

                    new_hyperparameter = self.get_hyperparameter_for_bounds[hyperparameter](
                        new_min_boundary, new_max_boundary
                    )
                    search_space._hyperparameters[hyperparameter] = new_hyperparameter

            elif isinstance(search_space[hyperparameter], CategoricalHyperparameter):
                choices = search_space[hyperparameter].sequence
                amount_choice_chosen = {choice: 0 for choice in choices}

                for config in selected_configs:
                    amount_choice_chosen[config[hyperparameter]] += 1

                new_choices = []
                for choice in choices:
                    if amount_choice_chosen[choice] > 0:
                        new_choices.append(choice)

                if len(choices) == len(new_choices):
                    continue

                # don't shrink too much
                if len(new_choices) == 0 or len(amount_choice_chosen) / len(new_choices) < self.max_shrinkage:
                    num_additional_choices = int(len(amount_choice_chosen) * self.max_shrinkage) + 1 - len(new_choices)

                    if isinstance(search_space[hyperparameter], OrdinalHyperparameter):

                        # add choices that were not chosen
                        maximal_choice = choices.index(new_choices[-1])
                        minimal_choice = choices.index(new_choices[0])

                        upper_left_out = len(choices) - maximal_choice - 1
                        lower_left_out = minimal_choice

                        percentage_upper_left_out = upper_left_out / (upper_left_out + lower_left_out)
                        percentage_lower_left_out = lower_left_out / (upper_left_out + lower_left_out)

                        # add more choices to side where fewer choices were left out
                        add_choices_bottom = round(num_additional_choices * (1 - percentage_lower_left_out))
                        add_choices_top = round(num_additional_choices * (1 - percentage_upper_left_out))

                        # add choices
                        for i in range(add_choices_bottom):
                            new_choices.insert(0, choices[minimal_choice - i - 1])

                        for i in range(add_choices_top):
                            new_choices.append(choices[maximal_choice + i + 1])
                    # only categorical, no order
                    else:
                        # randomly select hyperparameter to add
                        left_out_choices = [choice for choice in choices if choice not in new_choices]
                        new_choices.extend(
                            self.random_state.choice(left_out_choices, replace=False, size=num_additional_choices)
                        )

                new_hyperparameter = OrdinalHyperparameter(hyperparameter, new_choices)
                search_space._hyperparameters[hyperparameter] = new_hyperparameter
