from typing import Callable

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
    NumericalHyperparameter,
    OrdinalHyperparameter,
    NormalIntegerHyperparameter,
    NormalFloatHyperparameter,
)

from smac.constants import VERY_SMALL_NUMBER
from smac.runhistory.runhistory import RunHistory
from smac.utils.logging import get_logger
from copy import deepcopy
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
        range_multiplier: float = 2.0,
        seed: int = 0,
    ):
        # TODO find better name than range_multiplier
        self.get_hyperparameter_for_bounds = get_hyperparameter_for_bounds
        self.percentage_configurations = percentage_configurations
        self.max_shrinkage = max_shrinkage
        self.random_state = np.random.RandomState(seed)
        self.range_multiplier = range_multiplier
        logger.warning("Max Shrinkage should get removed")

    def modify_search_space(self, search_space: ConfigurationSpace, runhistory: RunHistory) -> None:  # noqa: D102
        # find best configurations
        configs = runhistory.get_configs(sort_by="cost")
        # select all configs, the cost of each config being the highest observed budget-cost for the config
        selected_configs = configs[: int(len(configs) * self.percentage_configurations)]

#        hyperparameter_names = search_space.get_hyperparameter_names()
 #       hparam_values: dict[str, list] = {hparam: [config[hparam] for config in selected_configs] for hparam in hyperparameter_names}

        # compute the new boundaries
        for hyperparameter_name in search_space.get_hyperparameter_names():
            if isinstance(search_space[hyperparameter_name], NumericalHyperparameter):
                # TODO check hyperparamter distribution
                # Extract Relevant values
                hyperparameter = search_space[hyperparameter_name]
                hyperparameter_values = [config[hyperparameter_name] for config in selected_configs]
                avg_hyperparamter_value = np.mean(hyperparameter_values)
                std_hyperparamter_value = np.std(hyperparameter_values)

                # Create new Hyperparameter in normal distribution
                new_hyperparameter_class = NormalIntegerHyperparameter if isinstance(hyperparameter, IntegerHyperparameter) else NormalFloatHyperparameter
                # TODO we need to check whether the distribution is more spiked than it was before
                new_hyperparameter = new_hyperparameter_class(
                    name = hyperparameter_name,
                    mu = avg_hyperparamter_value,
                    sigma = self.range_multiplier * std_hyperparamter_value,
                    lower = hyperparameter.lower,
                    upper = hyperparameter.upper,
                    log = hyperparameter.log,
                    meta = hyperparameter.meta,
                    q = hyperparameter.q,
                )

                search_space._hyperparameters[hyperparameter_name] = new_hyperparameter

            elif isinstance(search_space[hyperparameter_name], CategoricalHyperparameter) or isinstance(
                search_space[hyperparameter_name], OrdinalHyperparameter
            ):
                raise NotImplementedError("Categorical and Ordinal hyperparameters are not yet supported.")
                if not isinstance(search_space[hyperparameter_name], OrdinalHyperparameter):
                    choices = search_space[hyperparameter_name].choices
                else:
                    choices = search_space[hyperparameter_name].sequence
                amount_choice_chosen = {choice: 0 for choice in choices}

                for config in selected_configs:
                    if config[hyperparameter_name] in amount_choice_chosen.keys():
                        amount_choice_chosen[config[hyperparameter_name]] += 1

                new_choices = []
                for choice in choices:
                    if amount_choice_chosen[choice] > 0:
                        new_choices.append(choice)

                if len(choices) == len(new_choices):
                    continue

                # don't shrink too much
                if len(new_choices) == 0 or len(new_choices) / len(choices) < self.max_shrinkage:
                    num_additional_choices = int(len(choices) * self.max_shrinkage) + 1 - len(new_choices)

                    if isinstance(search_space[hyperparameter_name], OrdinalHyperparameter) and len(new_choices) != 0:
                        # add choices that were not chosen
                        maximal_choice = choices.index(new_choices[-1])
                        minimal_choice = choices.index(new_choices[0])

                        upper_left_out = len(choices) - maximal_choice - 1
                        lower_left_out = minimal_choice

                        percentage_upper_left_out = upper_left_out / (upper_left_out + lower_left_out)
                        percentage_lower_left_out = lower_left_out / (upper_left_out + lower_left_out)

                        add_choices_bottom = round(num_additional_choices * percentage_lower_left_out)
                        add_choices_top = round(num_additional_choices * percentage_upper_left_out)

                        # add choices
                        for i in range(add_choices_bottom):
                            new_choices.insert(0, choices[minimal_choice - i - 1])

                        for i in range(add_choices_top):
                            new_choices.append(choices[maximal_choice + i + 1])
                    # only categorical, no order (or need to select randomly because none chosen for ordinal)
                    else:
                        # randomly select hyperparameter to add
                        left_out_choices = [choice for choice in choices if choice not in new_choices]
                        # it has to be done this way otherwise numpy converts the types

                        if isinstance(search_space[hyperparameter_name], CategoricalHyperparameter):
                            new_choice_indices = self.random_state.choice(
                                len(left_out_choices), replace=False, size=num_additional_choices
                            )
                        else:
                            # generate a random starting point, and select as many as needed around it
                            max_starting_point_index = len(left_out_choices) - num_additional_choices
                            starting_index = self.random_state.randint(max_starting_point_index + 1)
                            new_choice_indices = list(range(starting_index, starting_index + num_additional_choices))
                        new_choices.extend(left_out_choices[i] for i in new_choice_indices)

                new_hyperparameter = OrdinalHyperparameter(hyperparameter_name, new_choices)
                search_space._hyperparameters[hyperparameter_name] = new_hyperparameter
