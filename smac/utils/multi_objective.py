from typing import List, Optional, Tuple, Union

import numpy as np


def normalize_costs(
    values: Union[np.ndarray, List, List[List], List[np.ndarray]],
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Normalizes the costs to be between 0 and 1 if no bounds are given. Otherwise, the costs are
    normalized according to the bounds.

    Example
    -------

    [0, 10, 5] -> [[0], [1], [0.5]]
    [[0], [10], [5]] -> [[0], [1], [0.5]]
    [[0, 0], [10, 50], [5, 200]] -> [[0, 0], [1, 0.25], [0.5, 1]]

    Parameters
    ----------
    values : Union[np.ndarray, List, List[List]]
        Cost values which should be normalized.
        If array/list is one-dimensional, it is expanded by one dimension.
    bounds : Optional[List[Tuple[float, float]]], optional
        Min and max bounds which should be applied to the values, by default None.
        If bounds are None the min and max values from the data are used.

    Returns
    -------
    np.ndarray
        Normalized costs.
    """
    _values: np.ndarray
    if isinstance(values, list):
        _values = np.array(values)
    else:
        _values = values

    if len(_values.shape) == 1:
        _values = np.expand_dims(_values, axis=-1)

    normalized_values = []
    for col in range(_values.shape[1]):
        data = _values[:, col].astype(float)

        if bounds is not None:
            assert len(bounds) == _values.shape[1]

            min_value = bounds[col][0]
            max_value = bounds[col][1]
        else:
            min_value = np.min(data)
            max_value = np.max(data)

        denominator = max_value - min_value

        # Prevent divide by zero
        if denominator < 1e-10:
            # Return ones
            normalized_values.append(np.ones_like(data))
        else:
            numerator = data - min_value
            normalized_values.append(numerator / denominator)

    return np.swapaxes(np.array(normalized_values), 0, 1)
