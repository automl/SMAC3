from __future__ import annotations

from typing import Any, Mapping

import re
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr

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

    def residual(self, value: float) -> float:
        """Signed distance of a value from the bound, negative when the constraint holds.

        Expressing both directions as "residual <= 0" lets the surrogate, the feasibility probability and the
        violation all share one convention, with the bound and the direction folded into the residual.
        """
        if self.op == LEQ:
            return value - self.bound

        return self.bound - value

    def is_satisfied(self, value: float) -> bool:
        """Whether an observed value satisfies this constraint."""
        return self.residual(value) <= 0.0

    def violation(self, value: float) -> float:
        """How far an observed value falls outside the bound.

        Returns 0.0 for a satisfied constraint and a positive amount otherwise. Used to rank configurations while
        no feasible one has been observed.
        """
        return max(0.0, self.residual(value))


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


def bilog(residuals: np.ndarray) -> np.ndarray:
    r"""Compresses constraint residuals around the feasibility boundary.

    $$
    \text{bilog}(r) = \text{sign}(r) \log(1 + |r|)
    $$

    See "Scalable Constrained Bayesian Optimization" by David Eriksson and Matthias Poloczek
    [[EP21][EP21]]. The transform magnifies values near zero and flattens extreme ones, which is what a
    constraint model wants: its accuracy only matters near the boundary, and a single wildly violating
    observation should not dominate the fit. Constraint values are raw measurements and routinely span orders
    of magnitude, so this matters in practice.

    The transform is strictly increasing and maps 0 to 0, so it leaves feasibility unchanged: a residual is
    non-positive exactly when its transform is.

    Parameters
    ----------
    residuals : np.ndarray
        Signed distances from the bound, as returned by ``OutcomeConstraint.residual``.

    Returns
    -------
    transformed : np.ndarray
    """
    residuals = np.asarray(residuals, dtype=float)

    return np.sign(residuals) * np.log1p(np.abs(residuals))


def inverse_bilog(transformed: np.ndarray) -> np.ndarray:
    """Maps transformed residuals back to their original units."""
    transformed = np.asarray(transformed, dtype=float)

    return np.sign(transformed) * np.expm1(np.abs(transformed))


def probability_of_feasibility(means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    r"""Probability that every constraint holds, given a Gaussian belief about each residual.

    The predictions are residuals, so a constraint is satisfied exactly when its residual is non-positive and
    the probability is $\Phi(-\mu / \sigma)$ regardless of the direction of the original bound. Residuals are
    assumed independent of each other and of the objective, which is the assumption that makes the
    feasibility-weighted acquisition function equal the constrained expected improvement.

    Note
    ----
    A Gaussian posterior is exact for a Gaussian process, but a random forest reports a variance across trees
    rather than a calibrated posterior. The resulting probability is then an approximation, which is the usual
    trade-off for using a forest as a surrogate.

    Parameters
    ----------
    means : np.ndarray [N, K]
        Predicted mean residual of each constraint.
    variances : np.ndarray [N, K]
        Predicted variance of each residual.

    Returns
    -------
    probabilities : np.ndarray [N, 1]
        Probability that every constraint is satisfied.
    """
    return np.exp(log_probability_of_feasibility(means, variances))


def log_probability_of_feasibility(means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    r"""Log of :func:`probability_of_feasibility`, summed rather than multiplied.

    The product underflows to exactly zero once enough constraints are unlikely, which destroys the ranking the
    acquisition maximizer depends on. Accumulating $\log \Phi$ with a stable implementation avoids that, and
    is what "Unexpected Improvements to Expected Improvement for Bayesian Optimization" by Sebastian Ament et
    al. [[ADE+23][ADE+23]] recommends.

    Parameters
    ----------
    means : np.ndarray [N, K]
    variances : np.ndarray [N, K]

    Returns
    -------
    log_probabilities : np.ndarray [N, 1]
    """
    means = np.atleast_2d(means)
    variances = np.atleast_2d(variances)

    if means.shape != variances.shape:
        raise ValueError(f"Means of shape {means.shape} do not match variances of shape {variances.shape}.")

    # A non-positive variance means the surrogate is certain; the comparison then degenerates to a step
    # function, which is applied directly instead of dividing by zero.
    stds = np.sqrt(np.clip(variances, 0.0, None))
    certain = stds <= 0.0
    safe_stds = np.where(certain, 1.0, stds)

    feasible_when_certain = np.where(means <= 0.0, 0.0, -np.inf)
    log_probabilities = np.where(certain, feasible_when_certain, log_ndtr(-means / safe_stds))

    return log_probabilities.sum(axis=1).reshape((-1, 1))


def extract_constraint_values(
    constraints: list[OutcomeConstraint], values: Mapping[str, Any]
) -> dict[str, float] | None:
    """Reads the constrained outputs out of the information returned alongside the cost.

    A target function reports constraint values by returning ``(cost, {"latency": 93.2})``, and that dictionary
    reaches SMAC as ``additional_info`` whether the target function was run by SMAC itself or by the caller of
    ``tell``. This picks the declared names out of it.

    A missing value is not an error. A crashed trial never gets the chance to report one, and it is treated as
    infeasible downstream.

    Parameters
    ----------
    constraints : list[OutcomeConstraint]
        The declared constraints, as parsed from ``Scenario.constraints``.
    values : Mapping[str, Any]
        What the target function returned alongside the cost.

    Returns
    -------
    dict[str, float] | None
        The observed values of the constrained outputs, or ``None`` if no constraints are declared.
    """
    if len(constraints) == 0:
        return None

    constraint_values = {}
    for constraint in constraints:
        if constraint.name in values:
            constraint_values[constraint.name] = float(values[constraint.name])

    return constraint_values


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
