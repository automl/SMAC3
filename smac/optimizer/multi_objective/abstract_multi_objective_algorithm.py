import typing
from abc import ABC

import numpy as np


class AbstractMultiObjectiveAlgorithm(ABC):
    """
    A general interface for multi-objective optimizer, depending on different strategies.
    It can be applied to rh2epm or epmchooser.
    """

    def __init__(self, num_obj: int, rng: typing.Optional[np.random.RandomState] = None):

        if rng is None:
            rng = np.random.RandomState(0)

        self.num_obj = num_obj
        self.rng = rng
