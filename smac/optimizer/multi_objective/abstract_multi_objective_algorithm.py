import typing
import numpy as np
from abc import ABC


class AbstractMultiObjectiveAlgorithm(ABC):
    """
    A general interface for mutli-objective optimizer, dependeing on different streagies, it can be applied to rh2epm
    or epmchooser.
    """

    def __init__(self,
                 num_obj: int,
                 rng: typing.Optional[np.random.RandomState] = None):

        if rng is not None:
            rng = np.random.RandomState(0)

        self.num_obj = num_obj
        self.rng = rng
