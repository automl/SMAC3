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
    """

    def construct_estimators(self,
                             configspace: ConfigurationSpace,
                             types: List[int],
                             bounds: List[Tuple[float, float]],
                             model_kwargs: Dict[str, Any]) -> List[RandomForestWithInstances]:
        return [
            RandomForestWithInstances(configspace, types, bounds, **model_kwargs) for _ in range(self.num_targets)
        ]
