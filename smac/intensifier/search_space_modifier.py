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
        selected_configs = configs[: int(len(configs) * self.percentage_configurations)]

        hparams = search_space.get_hyperparameter_names()
        hparam_values: dict[str, list] = {hparam: [config[hparam] for config in selected_configs] for hparam in hparams}

        # compute the new boundaries
        hparam_shrinkage = []
        hparam_boundaries = {}
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
            relative_shrinkage = (shrinkage_lower + shrinkage_upper) / (old_max_boundary - old_min_boundary)

            if relative_shrinkage != 0:
                hparam_shrinkage.append(relative_shrinkage)
            hparam_boundaries[hyperparameter] = (shrinkage_lower, shrinkage_upper, old_min_boundary, old_max_boundary)
            logger.debug(
                f"Shrink {hyperparameter}: lower {shrinkage_lower}, upper {shrinkage_upper}, "
                f"old bounds {old_min_boundary} - {old_max_boundary}, "
                f"new bounds {new_min_boundary} - {new_max_boundary}"
            )

        if len(hparam_shrinkage) == 0:
            logger.debug("Did not shrink search space.")
            return

        # compute the total shrinkage
        total_shrinkage = reduce(lambda x, y: x * y, hparam_shrinkage)

        clipped_total_shrinkage = min(total_shrinkage, self.max_shrinkage)
        # equally divide the reduction of shrinkage among all hyperparameters with this factor
        factor = (clipped_total_shrinkage / total_shrinkage) ** (1 / len(hparams))

        if clipped_total_shrinkage != total_shrinkage:
            logger.debug(
                f"Clipping total shrinkage from {total_shrinkage} to {clipped_total_shrinkage}" f" (factor: {factor})"
            )

        for hparam in hparams:
            boundaries = hparam_boundaries[hparam]
            new_min_boundary = boundaries[2] + (boundaries[0] * factor)
            new_max_boundary = boundaries[3] - (boundaries[1] * factor)

            # set the new boundaries
            logger.debug(f"Setting new boundaries for {hparam} to ({new_min_boundary}, {new_max_boundary})")

            # min and max cannot be the same
            if isinstance(search_space[hparam], IntegerHyperparameter):
                new_min_boundary = round(new_min_boundary)
                new_max_boundary = round(new_max_boundary)

                # don't completely remove the hyperparameter
                if new_min_boundary == new_max_boundary:
                    # if lower shrinkage greater than upper shrinkage, increase upper bound
                    if boundaries[0] > boundaries[1] and new_max_boundary + 1 <= boundaries[3]:
                        new_max_boundary = new_max_boundary + 1
                    elif boundaries[0] < boundaries[1]:
                        new_min_boundary = new_max_boundary - 1
                    else:
                        # if both shrinkages are equal, move both bounds
                        new_max_boundary = new_max_boundary + 1
                        new_min_boundary = new_min_boundary - 1

            logger.debug(f"Parameter {hparam}")
            logger.debug(
                f"Old min: {boundaries[2]}, new min: {new_min_boundary}, "
                f"adjustment: {boundaries[0] * factor}, adjustment without factor: {boundaries[0]}"
            )
            logger.debug(
                f"Old max: {boundaries[3]}, new max: {new_max_boundary}, "
                f"adjustment: {boundaries[1] * factor}, adjustment without factor: {boundaries[1]}"
            )

            new_hyperparameter = self.get_hyperparameter_for_bounds[hparam](new_min_boundary, new_max_boundary)
            search_space._hyperparameters[hparam] = new_hyperparameter
