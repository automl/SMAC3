from typing import Any, Dict, List, Tuple

from smac.configspace import ConfigurationSpace
from smac.epm.base_epm import AbstractEPM
from smac.epm.base_uncorrelated_mo_model import UncorrelatedMultiObjectiveModel
from smac.epm.rf_with_instances import RandomForestWithInstances

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class UncorrelatedMultiObjectiveRandomForestWithInstances(UncorrelatedMultiObjectiveModel):
    """Wrapper for the random forest to predict multiple targets.

    Only a list with the target names and the types array for the
    underlying forest model are mandatory. All other hyperparameters to
    the random forest can be passed via kwargs. Consult the documentation of
    the random forest for the hyperparameters and their meanings.
    """

    def construct_estimators(
        self,
        configspace: ConfigurationSpace,
        types: List[int],
        bounds: List[Tuple[float, float]],
        model_kwargs: Dict[str, Any],
    ) -> List[AbstractEPM]:
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
        estimators: List[AbstractEPM]
            A list of Random Forests
        """
        return [RandomForestWithInstances(configspace, types, bounds, **model_kwargs) for _ in range(self.num_targets)]
