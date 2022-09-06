import pytest
from ConfigSpace import ConfigurationSpace, Float, Configuration


class Rosenbrock2D:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x0 = Float("x0", (-5, 10), default=-3)
        x1 = Float("x1", (-5, 10), default=-4)
        cs.add_hyperparameters([x0, x1])

        return cs

    def train(self, config: Configuration, seed: int = 0, budget: float = 0, instance: str = "") -> float:
        """The 2-dimensional Rosenbrock function as a toy model.
        The Rosenbrock function is well-known in the optimization community and
        often serves as a toy problem. It can be defined for arbitrary
        dimensions. The minimum is always at x_i = 1 with a function value of
        zero. All input parameters are continuous. The search domain for
        all x's is the interval [-5, 10].
        """
        x1 = config["x0"]
        x2 = config["x1"]

        cost = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
        return cost


@pytest.fixture
def rosenbrock():
    return Rosenbrock2D()
