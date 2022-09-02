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
        types : List[int]
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass [3, 0]. Note that we count starting from 0.
        bounds : List[Tuple[float, float]]
            bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan) for categorical dims
        model_kwargs : Dict[str, Any]
            model kwargs for initializing models
        Returns
        -------
        estimators: List[BaseEPM]
            A list of Random Forests
        """
        return [RandomForest(configspace, **model_kwargs) for _ in range(self.num_targets)]
