import numpy as np

from smac.epm.base_epm import AbstractEPM


class BaseModel(AbstractEPM):

    def __init__(self):
        """
        Abstract base class for all models
        """
        bounds = [(l, u) for l, u in zip(self.lower, self.upper)]
        types = np.zeros(len(bounds))
        super().__init__(types=types, bounds=bounds, instance_features=None, pca_components=None)

        self.X = None
        self.y = None
