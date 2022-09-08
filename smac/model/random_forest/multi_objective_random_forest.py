from __future__ import annotations

from typing import Any

from ConfigSpace import ConfigurationSpace
from smac.model.abstract_model import AbstractModel
from smac.model.multi_objective_model import MultiObjectiveModel
from smac.model.random_forest.random_forest import (
    RandomForest,
)

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class MultiObjectiveRandomForest(MultiObjectiveModel):
    """Wrapper for the random forest to predict multiple targets.

    Only a list with the target names and the types array for the
    underlying forest model are mandatory. All other hyperparameters to
    the random forest can be passed via kwargs. Consult the documentation of
    the random forest for the hyperparameters and their meanings.
    """

    def construct_estimators(
        self,
        configspace: ConfigurationSpace,
        model_kwargs: dict[str, Any],
    ) -> list[AbstractModel]:
        """
        Construct a list of estimators. The number of the estimators equals 'self.num_targets'
        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space to tune for.
        model_kwargs : Dict[str, Any]
            model kwargs for initializing models
        Returns
        -------
        estimators: List[BaseEPM]
            A list of Random Forests
        """
        return [RandomForest(configspace, **model_kwargs) for _ in range(self.num_targets)]
