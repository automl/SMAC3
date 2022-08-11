from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union


import numpy as np
import scipy.spatial.distance
import scipy.special

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


def get_conditional_hyperparameters(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Returns conditional hyperparameters."""
    # Taking care of conditional hyperparameters according to Levesque et al.
    X_cond = X <= -1
    if Y is not None:
        Y_cond = Y <= -1
    else:
        Y_cond = X <= -1
    active = ~((np.expand_dims(X_cond, axis=1) != Y_cond).any(axis=2))
    return active
