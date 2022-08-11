"""
Synthetic Function
^^^^^^^^^^^^^^^^^^

An example of applying SMAC to optimize a synthetic function (2D Rosenbrock function).

We use the black-box facade because it is designed for black-box function optimization.
The black-box facade uses a :term:`Gaussian Process<GP>` as its surrogate model.
The facade works best on numerical hyperparameter configuration space and should not
be applied to problems with large evaluation budgets (up to 1000 evaluations).
"""

from ConfigSpace import Configuration, ConfigurationSpace, Float

from smac import BlackBoxFacade, Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def rosenbrock_2d(x: Configuration) -> float:
    """The 2-dimensional Rosenbrock function as a toy model.
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 10].
    """
    x1 = x["x0"]
    x2 = x["x1"]

    cost = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
    return cost


if __name__ == "__main__":
    # Build configuration space which defines all parameters and their ranges
    configspace = ConfigurationSpace()
    x0 = Float("x0", (-5, 10), default=-3)
    x1 = Float("x1", (-5, 10), default=-4)
    configspace.add_hyperparameters([x0, x1])

    # Scenario object
    scenario = Scenario(configspace, n_runs=100)

    # Example call of the target algorithm
    default_value = rosenbrock_2d(configspace.get_default_configuration())
    print(f"Default value: {round(default_value, 2)}")

    # Now we use SMAC to find the best hyperparameters
    smac = BlackBoxFacade(scenario, rosenbrock_2d)
    incumbent = smac.optimize()

    incumbent_value = rosenbrock_2d(incumbent)
    print(f"Incumbent value: {round(incumbent_value, 2)}")
