from typing import Callable

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    IntegerHyperparameter,
    NumericalHyperparameter,
    NormalIntegerHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    NormalFloatHyperparameter,
)

from smac.runhistory.runhistory import RunHistory
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class AbstractSearchSpaceModifier:
    def modify_search_space(
        self, search_space: ConfigurationSpace, runhistory: RunHistory
    ) -> None:
        """Modifies the search space."""
        raise NotImplementedError()


class MultiFidelitySearchSpaceShrinker(AbstractSearchSpaceModifier):
    """Shrinks the search space by setting the boundaries around a percentage of best performing configurations."""

    def __init__(
        self,
        get_hyperparameter_for_bounds: dict[str, Callable],
        percentage_configurations: float,
        range_multiplier: float = 2.0,
        seed: int = 0,
    ):
        # TODO find better name than range_multiplier
        self.get_hyperparameter_for_bounds = get_hyperparameter_for_bounds
        self.percentage_configurations = percentage_configurations
        self.random_state = np.random.RandomState(seed)
        self.range_multiplier = range_multiplier
        logger.warning("Max Shrinkage should get removed")

    def modify_search_space(
        self, search_space: ConfigurationSpace, runhistory: RunHistory
    ) -> None:  # noqa: D102
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
                # Allowed are uniform and normally distributed hyperparameters
                # if the hyperparameter is uniform, we can just shrink the range
                if not isinstance(
                    search_space[hyperparameter_name],
                    (
                        UniformFloatHyperparameter,
                        UniformIntegerHyperparameter,
                        NormalIntegerHyperparameter,
                        NormalFloatHyperparameter,
                    ),
                ):
                    raise NotImplementedError(
                        "Only uniform and normal hyperparameters are supported."
                    )

                # Extract Relevant values
                hyperparameter = search_space[hyperparameter_name]
                hyperparameter_values = [
                    config[hyperparameter_name] for config in selected_configs
                ]
                avg_hyperparamter_value = np.mean(hyperparameter_values)
                std_hyperparamter_value = np.std(hyperparameter_values)

                # Check that std does not increase
                if isinstance(hyperparameter, (NormalFloatHyperparameter, NormalIntegerHyperparameter)):
                    if std_hyperparamter_value > hyperparameter.sigma:
                        raise ValueError(
                            f"Standard deviation of hyperparameter {hyperparameter_name} is larger than the "
                            f"original sigma. This is not supported."
                        )

                # Create new Hyperparameter in normal distribution
                new_hyperparameter_class = (
                    NormalIntegerHyperparameter
                    if isinstance(hyperparameter, IntegerHyperparameter)
                    else NormalFloatHyperparameter
                )
                # TODO we need to check whether the distribution is more spiked than it was before
                new_hyperparameter = new_hyperparameter_class(
                    name=hyperparameter_name,
                    mu=avg_hyperparamter_value,
                    sigma=self.range_multiplier * std_hyperparamter_value,
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    log=hyperparameter.log,
                    meta=hyperparameter.meta,
                    q=hyperparameter.q,
                )

                search_space._hyperparameters[hyperparameter_name] = new_hyperparameter

            else:
                raise NotImplementedError(
                    "Only numerical hyperparameters are supported."
                )
