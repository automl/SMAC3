from typing import Callable

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import IntegerHyperparameter

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
    ):
        self.percentage_configurations = percentage_configurations
        self.max_shrinkage = max_shrinkage
        self.get_hyperparameter_for_bounds = get_hyperparameter_for_bounds

    def modify_search_space(self, search_space: ConfigurationSpace, runhistory: RunHistory) -> None:  # noqa: D102
        # find best configurations
        configs = runhistory.get_configs(sort_by="cost")
        selected_configs = configs[: int(len(configs) * self.percentage_configurations)]

        hparams = search_space.get_hyperparameter_names()
        hparam_values: dict[str, list] = {hparam: [config[hparam] for config in selected_configs] for hparam in hparams}

        # compute the new boundaries
        for hyperparameter in hparams:
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

                new_min_boundary = old_min_boundary + shrinkage_lower
                new_max_boundary = old_max_boundary - shrinkage_upper

                new_size = new_new_size

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
