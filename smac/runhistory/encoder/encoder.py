from __future__ import annotations

from typing import Mapping

import numpy as np

from smac.runhistory.encoder import AbstractRunHistoryEncoder
from smac.runhistory.runhistory import TrialKey, TrialValue
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger
from smac.utils.multi_objective import normalize_costs

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryEncoder(AbstractRunHistoryEncoder):
    def _build_matrix(
        self,
        trials: Mapping[TrialKey, TrialValue],
        store_statistics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_obj = self._n_objectives
        n_rows = len(trials)
        y_dim = self._n_objectives if self._native_multi_objective else 1

        if n_rows == 0:
            X = np.empty((0, self._n_params + self._n_features))
            y = np.empty((0, y_dim))
            return X, y

        # First build nan-matrix of size #configs x #params+1
        n_cols = self._n_params
        X = np.ones([n_rows, n_cols + self._n_features]) * np.nan

        n_obj = self._n_objectives
        y_raw = np.zeros((n_rows, n_obj), dtype=float)

        # Then populate matrix
        for row, (key, run) in enumerate(trials.items()):
            # Scaling is automatically done in configSpace
            conf = self.runhistory._ids_config[key.config_id]
            conf_vector = convert_configurations_to_array([conf])[0]

            if self._n_features > 0 and self._instance_features is not None:
                assert isinstance(key.instance, str)
                feats = self._instance_features[key.instance]
                X[row, :] = np.hstack((conf_vector, feats))
            else:
                X[row, :] = conf_vector

            if n_obj == 1:
                y_raw[row, 0] = run.cost
            else:
                cost = np.array(run.cost, dtype=float)
                y_raw[row, :] = cost

        finite_mask = np.isfinite(y_raw)

        # Worst observed finite value per objective.
        max_finite = np.array([self.runhistory.objective_bounds[o][1] for o in range(n_obj)], dtype=float)

        # Replace inf/nan objective values (e.g. from crashed runs)
        if not np.any(finite_mask):
            y_raw[:] = max_finite
        else:
            y_raw = np.where(finite_mask, y_raw, max_finite)

        y = np.zeros([n_rows, y_dim])

        if n_obj == 1:
            y = y_raw
        else:
            assert self._multi_objective_algorithm is not None

            bounds = self.runhistory.objective_bounds

            for row in range(n_rows):
                y_ = normalize_costs(y_raw[row], bounds) if self._normalize else y_raw[row]
                y[row] = self._multi_objective_algorithm(y_)

        if y.size > 0:
            if store_statistics:
                self._percentile = np.percentile(y, self._scale_percentage, axis=0)
                self._min_y = np.min(y, axis=0)
                self._max_y = np.max(y, axis=0)

        return X, y

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Returns the input values."""
        return values
