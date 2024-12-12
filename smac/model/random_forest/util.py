#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import numpy as np
from pyrfr import regression

if TYPE_CHECKING:
    from pyrfr.regression import default_data_container as DataContainer
    from numpy import typing as npt


def init_data_container(
        X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], bounds: Iterable[tuple[float, float]]
) -> DataContainer:
    """Fills a pyrfr default data container s.t. the forest knows categoricals and bounds for continous data.

    Parameters
    ----------
    X : np.ndarray [#samples, #hyperparameter + #features]
        Input data points.
    Y : np.ndarray [#samples, #objectives]
        The corresponding target values.

    Returns
    -------
    data : DataContainer
        The filled data container that pyrfr can interpret.
    """
    # Retrieve the types and the bounds from the ConfigSpace
    data = regression.default_data_container(X.shape[1])

    for i, (mn, mx) in enumerate(bounds):
        if np.isnan(mx):
            data.set_type_of_feature(i, mn)
        else:
            data.set_bounds_of_feature(i, mn, mx)

    for row_X, row_y in zip(X, y):
        data.add_data_point(row_X, row_y)

    return data
