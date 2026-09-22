from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from smac.utils.constraints import (
    OutcomeConstraint,
    bilog,
    inverse_bilog,
    is_feasible,
    log_probability_of_feasibility,
    parse_constraint,
    parse_constraints,
    probability_of_feasibility,
    total_violation,
)

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


def test_parse_upper_and_lower_bounds():
    """Both comparison operators parse into the matching constraint.

    Parses one expression of each direction and checks name, operator and bound.
    """
    assert parse_constraint("latency <= 100") == OutcomeConstraint("latency", "<=", 100.0)
    assert parse_constraint("accuracy >= 0.9") == OutcomeConstraint("accuracy", ">=", 0.9)


def test_parse_tolerates_surrounding_whitespace():
    """Whitespace around the operands does not change the parse.

    Parses a heavily padded expression and compares against the compact form.
    """
    assert parse_constraint("   latency   <=   100.5  ") == OutcomeConstraint("latency", "<=", 100.5)


@pytest.mark.parametrize(
    "expression",
    ["latency < 100", "latency > 100", "latency = 100", "latency", "", "latency <= abc", "a <= 1 <= 2"],
)
def test_parse_rejects_malformed_expressions(expression):
    """Expressions that are not a single name/operator/number triple are rejected.

    Feeds each malformed expression to the parser and expects a ValueError rather than a silent misparse.
    """
    with pytest.raises(ValueError):
        parse_constraint(expression)


def test_parse_constraints_rejects_duplicate_names():
    """One output cannot carry two bounds.

    Parses a list that bounds the same name twice and expects a ValueError naming the output.
    """
    with pytest.raises(ValueError, match="constrained more than once"):
        parse_constraints(["latency <= 100", "latency >= 10"])


def test_parse_constraints_rejects_a_bare_string():
    """Passing a single string instead of a list is caught early.

    A bare string would otherwise iterate character by character, so the error message suggests wrapping it.
    """
    with pytest.raises(ValueError, match="single string"):
        parse_constraints("latency <= 100")


def test_parse_constraints_of_none_is_empty():
    """An unconstrained scenario yields no constraints.

    Parses None and checks the result is an empty list rather than None.
    """
    assert parse_constraints(None) == []


def test_unknown_operator_is_rejected_on_construction():
    """The dataclass validates its operator independently of the parser.

    Constructs a constraint directly with an unsupported operator and expects a ValueError.
    """
    with pytest.raises(ValueError, match="Unknown comparison operator"):
        OutcomeConstraint("latency", "<", 100.0)


@pytest.mark.parametrize(
    "op,bound,value,expected",
    [
        ("<=", 100.0, 99.0, True),
        ("<=", 100.0, 100.0, True),
        ("<=", 100.0, 101.0, False),
        (">=", 0.9, 0.95, True),
        (">=", 0.9, 0.9, True),
        (">=", 0.9, 0.85, False),
    ],
)
def test_is_satisfied_is_inclusive_at_the_bound(op, bound, value, expected):
    """A value exactly on the bound counts as satisfying it.

    Checks values just inside, exactly on, and just outside the bound for both directions.
    """
    assert OutcomeConstraint("m", op, bound).is_satisfied(value) is expected


def test_feasibility_requires_every_constraint():
    """A configuration is feasible only when all bounds hold.

    Evaluates a two-constraint set against value sets that satisfy both, then each one alone.
    """
    constraints = parse_constraints(["latency <= 100", "accuracy >= 0.9"])

    assert is_feasible(constraints, {"latency": 90.0, "accuracy": 0.95})
    assert not is_feasible(constraints, {"latency": 110.0, "accuracy": 0.95})
    assert not is_feasible(constraints, {"latency": 90.0, "accuracy": 0.5})


@pytest.mark.parametrize("values", [None, {}, {"latency": float("nan")}, {"latency": float("inf")}])
def test_missing_or_non_finite_values_are_infeasible(values):
    """A trial that did not report a usable value has not shown itself feasible.

    Checks a missing dict, an empty dict, and non-finite readings against a single constraint.
    """
    constraints = parse_constraints(["latency <= 100"])
    assert not is_feasible(constraints, values)


def test_no_constraints_is_always_feasible():
    """An unconstrained run treats every trial as feasible.

    Checks the empty constraint list against a missing value dict.
    """
    assert is_feasible([], None)
    assert total_violation([], None) == 0.0


def test_total_violation_sums_the_overshoot():
    """Violation measures how far outside the bounds the values fall.

    Sums the overshoot of an upper and a lower bound and compares against the hand-computed total.
    """
    constraints = parse_constraints(["latency <= 100", "accuracy >= 0.9"])

    assert total_violation(constraints, {"latency": 90.0, "accuracy": 0.95}) == 0.0
    assert total_violation(constraints, {"latency": 120.0, "accuracy": 0.8}) == pytest.approx(20.1)


def test_total_violation_of_a_missing_value_is_infinite():
    """A trial that reported nothing never outranks one that merely overshot.

    Compares the violation of a missing value against that of a large but finite overshoot.
    """
    constraints = parse_constraints(["latency <= 100"])

    assert total_violation(constraints, None) == float("inf")
    assert total_violation(constraints, {"latency": 1e9}) < float("inf")


def test_bilog_is_monotone_and_fixes_the_boundary():
    """The transform preserves ordering and leaves the feasibility boundary where it was.

    Transforms residuals spanning three orders of magnitude on both sides of zero and checks the result is
    strictly increasing and maps 0 to 0, which is what keeps feasibility verdicts unchanged.
    """
    residuals = np.array([-1000.0, -5.0, -1e-6, 0.0, 1e-6, 5.0, 1000.0])

    transformed = bilog(residuals)

    assert transformed[3] == 0.0
    assert np.all(np.diff(transformed) > 0)
    assert np.all(np.sign(transformed) == np.sign(residuals))


def test_bilog_compresses_extremes_far_more_than_the_boundary():
    """Extreme residuals are flattened while near-boundary detail is kept.

    Compares how much the transform shrinks a residual of 1000 against one of 1, which is the property that
    stops one wild violation from dominating the constraint model.
    """
    near, far = bilog(np.array([1.0, 1000.0]))

    assert near == pytest.approx(np.log(2.0))
    assert far / near < 12.0  # a thousandfold gap in raw units becomes roughly tenfold


def test_bilog_round_trips():
    """The inverse recovers the original residuals.

    Transforms and inverts a spread of residuals and compares against the input.
    """
    residuals = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])

    assert inverse_bilog(bilog(residuals)) == pytest.approx(residuals)


def test_probability_of_feasibility_matches_the_normal_cdf():
    """The probability is the normal CDF of the standardised residual.

    Predicts three mean residuals around the boundary with a known variance and compares against norm.cdf.
    """
    means = np.array([[0.0], [-10.0], [20.0]])
    variances = np.array([[25.0], [25.0], [25.0]])

    probabilities = probability_of_feasibility(means, variances).ravel()

    assert probabilities == pytest.approx([norm.cdf(0.0), norm.cdf(2.0), norm.cdf(-4.0)])


def test_probability_of_feasibility_is_direction_agnostic():
    """Both bound directions reduce to the same residual-space computation.

    Builds the residuals of an upper and a lower bound for values equally far inside their bounds and checks
    the resulting probabilities match.
    """
    upper, lower = parse_constraints(["latency <= 100", "accuracy >= 0.9"])

    assert upper.residual(95.0) == pytest.approx(-5.0)
    assert lower.residual(0.95) == pytest.approx(-0.05)

    both = np.array([[upper.residual(95.0), lower.residual(0.95) * 100.0]])
    probabilities = probability_of_feasibility(both, np.ones((1, 2)))

    assert probabilities.ravel() == pytest.approx([norm.cdf(5.0) * norm.cdf(5.0)])


def test_probability_of_feasibility_multiplies_across_constraints():
    """Independent constraints combine as a product.

    Predicts both residuals exactly on the boundary, so each probability is 0.5 and the product is 0.25.
    """
    assert probability_of_feasibility(np.zeros((1, 2)), np.ones((1, 2))).ravel() == pytest.approx([0.25])


def test_probability_of_feasibility_is_a_step_function_without_variance():
    """A certain prediction gives a hard zero or one instead of dividing by zero.

    Predicts a feasible and an infeasible residual with zero variance and checks the degenerate probabilities.
    """
    means = np.array([[-10.0], [10.0]])

    assert probability_of_feasibility(means, np.zeros((2, 1))).ravel() == pytest.approx([1.0, 0.0])


def test_probability_of_feasibility_returns_a_column():
    """The result is shaped like an acquisition value so it can weight one directly.

    Predicts four points against two constraints and checks the output shape is [N, 1].
    """
    assert probability_of_feasibility(np.zeros((4, 2)), np.ones((4, 2))).shape == (4, 1)


def test_probability_of_feasibility_rejects_mismatched_shapes():
    """Means and variances have to describe the same predictions.

    Passes differently shaped arrays and expects a ValueError rather than a broadcast.
    """
    with pytest.raises(ValueError, match="do not match"):
        probability_of_feasibility(np.zeros((3, 2)), np.ones((4, 2)))


def test_log_probability_of_feasibility_agrees_with_the_product():
    """In the regime where the product is representable, the log form matches it.

    Compares exp(log probability) against the plain product for moderate residuals.
    """
    means = np.array([[0.5, -1.0], [2.0, 0.25]])
    variances = np.ones((2, 2))

    log_probabilities = log_probability_of_feasibility(means, variances)

    assert np.exp(log_probabilities) == pytest.approx(probability_of_feasibility(means, variances))


def test_log_probability_survives_an_underflowing_product():
    """Many unlikely constraints destroy the product but not the log sum.

    Predicts sixty residuals at six sigma outside the bound, where the product underflows to exactly zero and
    every candidate would rank equally, and checks the log form stays finite and still discriminates.
    """
    worse = np.full((1, 60), 6.0)
    better = np.full((1, 60), 5.0)
    variances = np.ones((1, 60))

    assert probability_of_feasibility(worse, variances).ravel()[0] == 0.0
    assert probability_of_feasibility(better, variances).ravel()[0] == 0.0

    log_worse = log_probability_of_feasibility(worse, variances).ravel()[0]
    log_better = log_probability_of_feasibility(better, variances).ravel()[0]

    assert np.isfinite(log_worse) and np.isfinite(log_better)
    assert log_better > log_worse
