from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


def normalize_costs(values: list[float], bounds: list[tuple[float, float]] | None = None) -> list[float]:
    """
    Normalizes a list of floats with corresponding bounds.

    Parameters
    ----------
    values : list[float]
        List of costs to be normalized.
    bounds : list[tuple[float, float]] | None, optional
        List of tuple of bounds. By default None. If no bounds are passed, only ones are returned.

    Returns
    -------
    normalized_costs : list[float]
        If no bounds were passed or
    """
    if bounds is None:
        return np.ones(len(values)).tolist()

    costs = []
    for v, b in zip(values, bounds):
        p = v - b[0]
        q = b[1] - b[0]

        if q < 1e-10:
            cost = 1.0
        else:
            cost = p / q
        costs += [cost]

    return costs
