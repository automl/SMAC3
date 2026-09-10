from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


# "name <= bound" or "name >= bound", tolerating arbitrary surrounding whitespace.
_EXPRESSION = re.compile(r"^\s*(?P<name>\S+)\s*(?P<op><=|>=)\s*(?P<bound>\S+)\s*$")

LEQ = "<="
GEQ = ">="


@dataclass(frozen=True)
class OutcomeConstraint:
    """A bound on a measured output of the target function.

    An outcome constraint restricts a quantity that is observed alongside the objective, for example
    ``latency <= 100``. In contrast to a forbidden clause, which restricts the configuration space itself, the
    constrained quantity is a black-box output and has to be modelled.

    Parameters
    ----------
    name : str
        Name of the constrained output. The target function has to report a value under this name for every trial.
    op : str
        Either ``"<="`` or ``">="``.
    bound : float
        The threshold the output is compared against.
    """

    name: str
    op: str
    bound: float

    def __post_init__(self) -> None:
        if self.op not in (LEQ, GEQ):
            raise ValueError(f"Unknown comparison operator {self.op!r}. Use {LEQ!r} or {GEQ!r}.")

    def __str__(self) -> str:
        return f"{self.name} {self.op} {self.bound}"

    def is_satisfied(self, value: float) -> bool:
        """Whether an observed value satisfies this constraint."""
        if self.op == LEQ:
            return value <= self.bound

        return value >= self.bound

    def violation(self, value: float) -> float:
        """How far an observed value falls outside the bound.

        Returns 0.0 for a satisfied constraint and a positive amount otherwise. Used to rank configurations while
        no feasible one has been observed.
        """
        if self.op == LEQ:
            return max(0.0, value - self.bound)

        return max(0.0, self.bound - value)


def parse_constraint(expression: str) -> OutcomeConstraint:
    """Parses a single constraint expression such as ``"latency <= 100"``.

    Parameters
    ----------
    expression : str
        An expression of the form ``"<name> <= <bound>"`` or ``"<name> >= <bound>"``.

    Returns
    -------
    constraint : OutcomeConstraint

    Raises
    ------
    ValueError
        If the expression is malformed or the bound is not a number.
    """
    if not isinstance(expression, str):
        raise ValueError(f"Expected a constraint expression as a string but got {expression!r}.")

    match = _EXPRESSION.match(expression)
    if match is None:
        raise ValueError(
            f"Could not parse the constraint {expression!r}. "
            f"Expected something like 'latency <= 100' or 'accuracy >= 0.9'."
        )

    bound = match.group("bound")
    try:
        bound_value = float(bound)
    except ValueError:
        raise ValueError(f"The bound {bound!r} in the constraint {expression!r} is not a number.")

    return OutcomeConstraint(name=match.group("name"), op=match.group("op"), bound=bound_value)


def parse_constraints(expressions: list[str] | None) -> list[OutcomeConstraint]:
    """Parses a list of constraint expressions and rejects duplicated names.

    Parameters
    ----------
    expressions : list[str] | None
        Constraint expressions. ``None`` yields an empty list.

    Returns
    -------
    constraints : list[OutcomeConstraint]

    Raises
    ------
    ValueError
        If an expression is malformed or two constraints bound the same output.
    """
    if expressions is None:
        return []

    if isinstance(expressions, str):
        raise ValueError(
            f"Expected a list of constraint expressions but got the single string {expressions!r}. "
            f"Pass [{expressions!r}] instead."
        )

    constraints = [parse_constraint(expression) for expression in expressions]

    seen: set[str] = set()
    for constraint in constraints:
        if constraint.name in seen:
            raise ValueError(f"The output {constraint.name!r} is constrained more than once.")

        seen.add(constraint.name)

    return constraints


def is_feasible(constraints: list[OutcomeConstraint], values: dict[str, float] | None) -> bool:
    """Whether all constraints are satisfied by the observed values.

    A missing or non-finite value counts as infeasible: the trial did not demonstrate that it satisfies the bound.

    Parameters
    ----------
    constraints : list[OutcomeConstraint]
    values : dict[str, float] | None
        The constraint values observed for a trial.

    Returns
    -------
    feasible : bool
    """
    if len(constraints) == 0:
        return True

    if values is None:
        return False

    for constraint in constraints:
        value = values.get(constraint.name)
        if value is None:
            return False

        if not _is_finite(value):
            return False

        if not constraint.is_satisfied(value):
            return False

    return True


def total_violation(constraints: list[OutcomeConstraint], values: dict[str, float] | None) -> float:
    """Sums how far the observed values fall outside their bounds.

    Returns ``inf`` if a value is missing or non-finite, so that a trial which reported nothing never outranks one
    which merely overshot its bound.

    Parameters
    ----------
    constraints : list[OutcomeConstraint]
    values : dict[str, float] | None

    Returns
    -------
    violation : float
    """
    if len(constraints) == 0:
        return 0.0

    if values is None:
        return float("inf")

    violation = 0.0
    for constraint in constraints:
        value = values.get(constraint.name)
        if value is None or not _is_finite(value):
            return float("inf")

        violation += constraint.violation(value)

    return violation


def probability_of_feasibility(
    constraints: list[OutcomeConstraint],
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Probability that all constraints hold, under a Gaussian belief about each constrained output.

    For an upper bound this is $\\Phi((bound - \\mu) / \\sigma)$ and for a lower bound
    $\\Phi((\\mu - bound) / \\sigma)$. The constrained outputs are assumed to be independent of each other and of
    the objective, so the individual probabilities multiply. That independence assumption is what makes the
    feasibility-weighted acquisition function equal to the constrained expected improvement of Gardner et al.

    Note
    ----
    A Gaussian posterior is exact for a Gaussian process, but a random forest reports a variance across trees
    rather than a calibrated posterior. The resulting probability is then an approximation, which is the usual
    trade-off for using a forest as a surrogate.

    Parameters
    ----------
    constraints : list[OutcomeConstraint]
        The constraints, in the same column order as ``means`` and ``variances``.
    means : np.ndarray [N, K]
        Predicted mean of each constrained output.
    variances : np.ndarray [N, K]
        Predicted variance of each constrained output.

    Returns
    -------
    probabilities : np.ndarray [N, 1]
        Probability that every constraint is satisfied.
    """
    means = np.atleast_2d(means)
    variances = np.atleast_2d(variances)

    if means.shape != variances.shape:
        raise ValueError(f"Means of shape {means.shape} do not match variances of shape {variances.shape}.")

    if means.shape[1] != len(constraints):
        raise ValueError(f"Got {means.shape[1]} predicted outputs for {len(constraints)} constraints.")

    # A non-positive variance means the surrogate is certain; the comparison then degenerates to a step function,
    # which np.where below applies directly instead of dividing by zero.
    stds = np.sqrt(np.clip(variances, 0.0, None))

    probabilities = np.ones((means.shape[0], 1))
    for index, constraint in enumerate(constraints):
        mean = means[:, index]
        std = stds[:, index]

        if constraint.op == LEQ:
            slack = constraint.bound - mean
        else:
            slack = mean - constraint.bound

        certain = std <= 0.0
        safe_std = np.where(certain, 1.0, std)
        probability = np.where(certain, (slack >= 0.0).astype(float), norm.cdf(slack / safe_std))

        probabilities *= probability.reshape((-1, 1))

    return probabilities


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
