from typing import Union, List, Tuple
import numpy as np


def normalize_costs(values: Union[np.ndarray, list],
                    bounds: List[Tuple[float, float]] = None) -> np.ndarray:

    if isinstance(values, list):
        values = np.array(values)

    normalized_values = np.empty_like(values)
    for axis in range(values.shape[1]):
        data = values[:, axis]
        
        if bounds is not None and len(bounds) == values.shape[1]:
            min_value = bounds[axis][0]
            max_value = bounds[axis][1]
        else:
            min_value = np.min(data)
            max_value = np.max(data)
        
        denominator = max_value - min_value
        
        # Prevent divide by zero
        if denominator == 0:
            normalized_values[:, axis] = data
        else:
            normalized_values[:, axis] = (data - min_value) / denominator

    return normalized_values
