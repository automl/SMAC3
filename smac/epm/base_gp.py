import numpy as np

from smac.epm.base_epm import AbstractEPM


class BaseModel(AbstractEPM):

    def __init__(self, types, bounds, seed):
        """
        Abstract base class for all Gaussian process models.
        """
        super().__init__(types=types, bounds=bounds, seed=seed, instance_features=None, pca_components=None)

        self.rng = np.random.RandomState(seed)

        lower = []
        upper = []
        for bound in bounds:
            lower.append(bound[0])
            upper.append(bound[1])

        self.lower = np.array(lower)
        self.upper = np.array(upper)

        self.X = None
        self.y = None
