"""Output Constraints
# Flags: doc-Runnable

An example of optimizing an objective subject to a bound on a second, measured quantity.

An output constraint restricts something the target function reports rather than something the configuration
space can express. "Keep the validation error low, but the model has to stay under 50 kilobytes" is an output
constraint: whether a configuration satisfies it is only known after evaluating it, exactly like the objective.
Forbidden clauses cannot express this, because they can only rule out configurations by their parameters.

SMAC handles this by modelling each constrained output with its own surrogate and weighting the acquisition
function by the probability that every bound holds:

    a_c(x) = a(x) * prod_i P(y_i(x) satisfies constraint i)

Weighting rather than penalizing matters. Reporting a bad cost for a violating configuration would teach the
objective model a cliff that is not a feature of the objective, which degrades its predictions inside the
feasible region too, and would throw away the measured value that locates the boundary.

Two things follow from the constraint being modelled rather than known:

* The incumbent is the best *feasible* configuration, not the best one overall.
* Until something feasible turns up there is no improvement to expect over, so the search maximizes the
  probability of feasibility alone.

To use it, declare the bounds on the scenario and report a value for each constrained output from the target
function, next to the cost.
"""

from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float

from smac import HyperparameterOptimizationFacade, Scenario

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class ConstrainedBranin:
    """The Branin function with a linear region of the domain declared infeasible.

    The unconstrained optimum lies inside the infeasible region, so an optimizer that ignores the constraint
    reports an answer that cannot be used.
    """

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        cs.add(Float("x1", (-5.0, 10.0), default=0.0), Float("x2", (0.0, 15.0), default=0.0))

        return cs

    def train(self, config: Configuration, seed: int = 0) -> tuple[float, dict]:
        """Returns the cost and, alongside it, the value of the constrained output."""
        x1 = config["x1"]
        x2 = config["x2"]

        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        cost = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

        # The constrained output. Anything the target function measures can be bounded this way.
        distance = x1 + x2

        return cost, {"distance": distance}


if __name__ == "__main__":
    model = ConstrainedBranin()

    # The bound is declared next to the objective, not as one of them: the run stays single objective and
    # `distance` is never optimized, only kept inside its bound.
    scenario = Scenario(
        model.configspace,
        objectives="cost",
        constraints=["distance <= 5.0"],
        n_trials=100,
        deterministic=True,
    )

    smac = HyperparameterOptimizationFacade(scenario, model.train, overwrite=True)
    incumbent = smac.optimize()

    x1 = incumbent["x1"]
    x2 = incumbent["x2"]

    print(f"Incumbent: x1={x1:.4f}, x2={x2:.4f}")
    print(f"Cost: {smac.validate(incumbent):.4f}")
    print(f"Constrained output: distance={x1 + x2:.4f} (bound: <= 5.0)")
