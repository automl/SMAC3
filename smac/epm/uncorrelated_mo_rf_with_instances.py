from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from smac.configspace import ConfigurationSpace
from smac.epm.base_uncorrelated_mo_model import UncorrelatedMultiObjectiveModel
from smac.epm.rf_with_instances import RandomForestWithInstances

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class UncorrelatedMultiObjectiveRandomForestWithInstances(UncorrelatedMultiObjectiveModel):
    """Wrapper for the random forest to predict multiple targets.

    Only the a list with the target names and the types array for the
    underlying forest model are mandatory. All other hyperparameters to
    the random forest can be passed via kwargs. Consult the documentation of
    the random forest for the hyperparameters and their meanings.


    Parameters
    ----------
    target_names : list
        List of str, each entry is the name of one target dimension. Length
        of the list will be ``n_objectives``.
    types : List[int]
        Specifies the number of categorical values of an input dimension where
        the i-th entry corresponds to the i-th input dimension. Let's say we
        have 2 dimension where the first dimension consists of 3 different
        categorical choices and the second dimension is continuous than we
        have to pass [3, 0]. Note that we count starting from 0.
    bounds : List[Tuple[float, float]]
        bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan) for categorical dims
    instance_features : np.ndarray (I, K)
        Contains the K dimensional instance features of the I different instances
    pca_components : float
        Number of components to keep when using PCA to reduce dimensionality of instance features. Requires to
        set n_feats (> pca_dims).


    Attributes
    ----------
    target_names
    num_targets
    estimators
    """

    def __init__(
        self,
        target_names: List[str],
        configspace: ConfigurationSpace,
        types: List[int],
        bounds: List[Tuple[float, float]],
        seed: int,
        rf_kwargs: Optional[Dict[str, Any]] = None,
        instance_features: Optional[np.ndarray] = None,
        pca_components: Optional[int] = None,
    ) -> None:
        super().__init__(
            target_names=target_names,
            configspace=configspace,
            bounds=bounds,
            types=types,
            seed=seed,
            instance_features=instance_features,
            pca_components=pca_components,
        )
        if rf_kwargs is None:
            rf_kwargs = {}
        rf_kwargs['seed'] = seed
        rf_kwargs = rf_kwargs
        self.estimators = [
            RandomForestWithInstances(configspace, types, bounds, **rf_kwargs) for _ in range(self.num_targets)
        ]
