from __future__ import annotations

from typing import Mapping

import numpy as np

from smac.runhistory.encoder import AbstractRunHistoryEncoder
from smac.runhistory.runhistory import TrialKey, TrialValue
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger
from smac.utils.multi_objective import normalize_costs

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryEIPSEncoder(AbstractRunHistoryEncoder):
    """Encoder specifically for the EIPS (expected improvement per second) acquisition function."""

    def _build_matrix(
        self,
        trials: Mapping[TrialKey, TrialValue],
        store_statistics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if store_statistics:
            # store_statistics is currently not necessary
            pass

        # First build nan-matrix of size #configs x #params+1
        n_rows = len(trials)
        n_cols = self._n_params
        X = np.ones([n_rows, n_cols + self._n_features]) * np.nan
        y = np.ones([n_rows, 2])

        # Then populate matrix
        for row, (key, run) in enumerate(trials.items()):
            # Scaling is automatically done in configSpace
            conf = self.runhistory.ids_config[key.config_id]
            conf_vector = convert_configurations_to_array([conf])[0]
            if self._n_features > 0 and self._instance_features is not None:
                assert isinstance(key.instance, str)
                feats = self._instance_features[key.instance]
                X[row, :] = np.hstack((conf_vector, feats))
            else:
                X[row, :] = conf_vector

            if self._n_objectives > 1:
                assert self._multi_objective_algorithm is not None
                assert isinstance(run.cost, list)

                # Let's normalize y here
                # We use the objective_bounds calculated by the runhistory
                y_ = normalize_costs(run.cost, self.runhistory.objective_bounds)
                y_agg = self._multi_objective_algorithm(y_)
                y[row, 0] = y_agg
            else:
                y[row, 0] = run.cost

            y[row, 1] = run.time

        y_transformed = self.transform_response_values(values=y)

        return X, y_transformed

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values. Transform the runtimes by a log transformation
        log(1. + runtime).

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        # We need to ensure that time remains positive after the log transform.
        values[:, 1] = np.log(1 + values[:, 1])
        return values
