import numpy as np

from smac.epm.base_epm import AbstractEPM


class BaseModel(AbstractEPM):

    def __init__(self, types, bounds, seed, **kwargs):
        """
        Abstract base class for all Gaussian process models.
        """
        super().__init__(types=types, bounds=bounds, seed=seed, **kwargs)

        self.rng = np.random.RandomState(seed)

    def _normalize_y(self, y):
        self.mean_y_ = np.mean(y)
        self.std_y_ = np.std(y)
        if self.std_y_ == 0:
            self.std_y_ = 1
        return (y - self.mean_y_) / self.std_y_

    def _untransform_y(self, y, var=None):
        y = y * self.std_y_ + self.mean_y_
        if var is not None:
            var = var * self.std_y_ ** 2
            return y, var
        return y
