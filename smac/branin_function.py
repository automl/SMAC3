from ConfigSpace import ConfigurationSpace, Float, Configuration
import numpy as np

class Branin(object):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x0 = Float("x0", (-5, 10), default=-5, log=False)
        x1 = Float("x1", (0, 15), default=2, log=False)
        cs.add([x0, x1])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Branin function

        Parameters
        ----------
        config : Configuration
            Contains two continuous hyperparameters, x0 and x1
        seed : int, optional
            Not used, by default 0

        Returns
        -------
        float
            Branin function value
        """
        x0 = config["x0"]
        x1 = config["x1"]
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)
        ret = a * (x1 - b * x0**2 + c * x0 - r) ** 2 + s * (1 - t) * np.cos(x0) + s

        return ret

