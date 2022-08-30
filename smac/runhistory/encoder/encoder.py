from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

from smac.configspace import convert_configurations_to_array
from smac.multi_objective.utils import normalize_costs
from smac.runhistory.encoder import AbstractRunHistoryEncoder
from smac.runhistory.runhistory import RunHistory, TrialKey, TrialValue
from smac.utils.logging import get_logger
from smac.scenario import Scenario
from smac.runhistory.enumerations import StatusType

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryEncoder(AbstractRunHistoryEncoder):
    """TODO."""

    def _build_matrix(
        self,
        run_dict: Mapping[TrialKey, TrialValue],
        runhistory: RunHistory,
        store_statistics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        # First build nan-matrix of size #configs x #params+1
        n_rows = len(run_dict)
        n_cols = self.n_params
        X = np.ones([n_rows, n_cols + self.n_features]) * np.nan

        # For now we keep it as 1
        # TODO: Extend for native multi-objective
        y = np.ones([n_rows, 1])

        if self.multi_objective_algorithm is not None:
            self.multi_objective_algorithm.update_on_iteration_start()

        # Then populate matrix
        for row, (key, run) in enumerate(run_dict.items()):
            # Scaling is automatically done in configSpace
            conf = runhistory.ids_config[key.config_id]
            conf_vector = convert_configurations_to_array([conf])[0]
            if self.n_features > 0 and self.instance_features is not None:
                assert isinstance(key.instance, str)
                feats = self.instance_features[key.instance]
                X[row, :] = np.hstack((conf_vector, feats))
            else:
                X[row, :] = conf_vector
            # run_array[row, -1] = instances[row]

            if self.n_objectives > 1:
                assert self.multi_objective_algorithm is not None
                assert isinstance(run.cost, list)

                # Let's normalize y here
                # We use the objective_bounds calculated by the runhistory
                y_ = normalize_costs(run.cost, runhistory.objective_bounds)
                y_agg = self.multi_objective_algorithm(y_)
                y[row] = y_agg
            else:
                # if return_time_as_y:
                #    y[row, 0] = run.time
                # else:
                y[row] = run.cost

        if y.size > 0:
            if store_statistics:
                self.perc = np.percentile(y, self.scale_percentage, axis=0)
                self.min_y = np.min(y, axis=0)
                self.max_y = np.max(y, axis=0)

        y = self.transform_response_values(values=y)
        return X, y

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values. Returns the input values.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        return values
