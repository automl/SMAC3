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


class RunHistoryEIPSEncoder(AbstractRunHistoryEncoder):
    """Encoder specifically for the EIPS (expected improvement per second) acquisition function."""

    def _build_matrix(
        self,
        trials: Mapping[TrialKey, TrialValue],
        store_statistics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_obj = self._n_objectives
        n_rows = len(trials)
        y_dim = self._n_objectives if self._native_multi_objective else 1

        if len(trials) == 0:
            X = np.empty((0, self._n_params + self._n_features))
            y = np.empty((0, y_dim + 1))
            return X, y

        if store_statistics:
            # store_statistics is currently not necessary
            pass

        # First build nan-matrix of size #configs x #params+1
        n_cols = self._n_params
        X = np.ones([n_rows, n_cols + self._n_features]) * np.nan
        y_raw = np.zeros([n_rows, n_obj + 1], dtype=float)

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

            if n_obj == 1:
                y_raw[row, 0] = run.cost
            else:
                y_raw[row, :n_obj] = np.asarray(run.cost, dtype=float)

            y_raw[row, -1] = float(run.time)

        obj_matrix = y_raw[:, :n_obj]

        finite_mask = np.isfinite(obj_matrix)

        # Worst observed finite value per objective.
        max_finite = np.array([self.runhistory.objective_bounds[o][1] for o in range(n_obj)], dtype=float)

        # Replace inf/nan objective values (e.g. from crashed runs)
        if not np.any(finite_mask):
            obj_matrix[:] = max_finite
        else:
            obj_matrix = np.where(finite_mask, obj_matrix, max_finite)

        # write back
        y_raw[:, :n_obj] = obj_matrix

        y = np.zeros([n_rows, y_dim + 1])
        if n_obj == 1:
            y[:, 0] = y_raw[:, 0]
            y[:, 1] = y_raw[:, -1]

        else:
            assert self._multi_objective_algorithm is not None

            bounds = self.runhistory.objective_bounds

            for row in range(n_rows):
                time = y_raw[row, -1]
                y_ = normalize_costs(y_raw[row, :n_obj], bounds) if self._normalize else y_raw[row, :n_obj]
                y[row, :y_dim] = self._multi_objective_algorithm(y_)
                y[row, -1] = time

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
        values[:, -1] = np.log(1 + values[:, -1])
        return values
