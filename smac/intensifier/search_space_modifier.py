from typing import Callable

from functools import reduce

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import IntegerHyperparameter

from smac import RunHistory
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
    ):
        self.percentage_configurations = percentage_configurations
        self.max_shrinkage = max_shrinkage
        self.get_hyperparameter_for_bounds = get_hyperparameter_for_bounds

    def modify_search_space(self, search_space: ConfigurationSpace, runhistory: RunHistory) -> None:  # noqa: D102
        # find best configurations
        configs = runhistory.get_configs(sort_by="cost")

        # select the best configurations
        selected_configs = configs[: int(len(configs) * self.percentage_configurations)]

        hparams = search_space.get_hyperparameter_names()
        hparam_values: dict[str, list] = {hparam: [] for hparam in hparams}

        # collect the hyperparameter values
        for config in selected_configs:
            for hyperparameter in hparams:
                hparam_values[hyperparameter].append(config[hyperparameter])

        # compute the new boundaries
        hparam_shrinkage = {}
        hparam_boundaries = {}
        new_configspace = ConfigurationSpace()
        for hyperparameter in hparams:
            old_min_boundary = search_space[hyperparameter].lower
            old_max_boundary = search_space[hyperparameter].upper

            # if the boundaries are the same, don't shrink
            if old_min_boundary == old_max_boundary:
                new_configspace.add_hyperparameter(search_space.get_hyperparameter(hyperparameter))
                continue

            new_min_boundary = min(hparam_values[hyperparameter])
            new_max_boundary = max(hparam_values[hyperparameter])

            # don't expand (fallback, should not happen)
            if new_min_boundary < old_min_boundary:
                new_min_boundary = old_min_boundary

            if new_max_boundary > old_max_boundary:
                new_max_boundary = old_max_boundary

            # compute shrinkage for both directions
            shrinkage_lower = new_min_boundary - old_min_boundary
            shrinkage_upper = old_max_boundary - new_max_boundary
            relative_shrinkage = (shrinkage_lower + shrinkage_upper) / (old_max_boundary - old_min_boundary)
            hparam_shrinkage[hyperparameter] = relative_shrinkage
            hparam_boundaries[hyperparameter] = (shrinkage_lower, shrinkage_upper, old_min_boundary, old_max_boundary)

        # compute the total shrinkage
        total_shrinkage = reduce(lambda x, y: x * y, hparam_shrinkage.values())
        clipped_total_shrinkage = min(total_shrinkage, self.max_shrinkage)

        if clipped_total_shrinkage != total_shrinkage:
            logger.debug(f"Clipping total shrinkage from {total_shrinkage} to {clipped_total_shrinkage}")
            for hparam in hparam_shrinkage:
                boundaries = hparam_boundaries[hparam]
                new_min_boundary = boundaries[2] + (boundaries[0] * clipped_total_shrinkage / total_shrinkage)
                new_max_boundary = boundaries[3] - (boundaries[1] * clipped_total_shrinkage / total_shrinkage)

                # set the new boundaries
                logger.debug(f"Setting new boundaries for {hparam} to ({new_min_boundary}, {new_max_boundary})")

                if isinstance(search_space[hparam], IntegerHyperparameter):
                    new_min_boundary = round(new_min_boundary)
                    new_max_boundary = round(new_max_boundary)

                new_hyperparameter = self.get_hyperparameter_for_bounds[hparam](new_min_boundary, new_max_boundary)
                new_configspace.add_hyperparameter(new_hyperparameter)

                search_space._hyperparameters[hparam] = new_hyperparameter
