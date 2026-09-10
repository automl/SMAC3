from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from smac.utils.constraints import (
    OutcomeConstraint,
    is_feasible,
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


def test_probability_of_feasibility_matches_the_normal_cdf():
    """The upper-bound probability is the normal CDF of the standardised slack.

    Predicts three means around a bound with a known variance and compares against norm.cdf by hand.
    """
    constraints = parse_constraints(["latency <= 100"])
    means = np.array([[100.0], [90.0], [120.0]])
    variances = np.array([[25.0], [25.0], [25.0]])

    probabilities = probability_of_feasibility(constraints, means, variances).ravel()

    assert probabilities == pytest.approx([norm.cdf(0.0), norm.cdf(2.0), norm.cdf(-4.0)])


def test_probability_of_feasibility_flips_for_a_lower_bound():
    """A lower bound standardises the slack in the opposite direction.

    Predicts one mean below and one above a lower bound and checks the probabilities are mirrored.
    """
    constraints = parse_constraints(["accuracy >= 0.9"])
    means = np.array([[0.95], [0.85]])
    variances = np.array([[0.0025], [0.0025]])

    probabilities = probability_of_feasibility(constraints, means, variances).ravel()

    assert probabilities == pytest.approx([norm.cdf(1.0), norm.cdf(-1.0)])


def test_probability_of_feasibility_multiplies_across_constraints():
    """Independent constraints combine as a product.

    Predicts both outputs exactly on their bounds, so each probability is 0.5 and the product is 0.25.
    """
    constraints = parse_constraints(["a <= 10", "b >= 5"])
    means = np.array([[10.0, 5.0]])
    variances = np.array([[1.0, 1.0]])

    assert probability_of_feasibility(constraints, means, variances).ravel() == pytest.approx([0.25])


def test_probability_of_feasibility_is_a_step_function_without_variance():
    """A certain prediction gives a hard zero or one instead of dividing by zero.

    Predicts a feasible and an infeasible mean with zero variance and checks the degenerate probabilities.
    """
    constraints = parse_constraints(["latency <= 100"])
    means = np.array([[90.0], [110.0]])
    variances = np.zeros((2, 1))

    assert probability_of_feasibility(constraints, means, variances).ravel() == pytest.approx([1.0, 0.0])


def test_probability_of_feasibility_returns_a_column():
    """The result is shaped like an acquisition value so it can weight one directly.

    Predicts four points against two constraints and checks the output shape is [N, 1].
    """
    constraints = parse_constraints(["a <= 1", "b <= 1"])

    probabilities = probability_of_feasibility(constraints, np.zeros((4, 2)), np.ones((4, 2)))

    assert probabilities.shape == (4, 1)


def test_probability_of_feasibility_rejects_mismatched_shapes():
    """Predicting a different number of outputs than there are constraints is an error.

    Passes a single predicted column for a two-constraint set and expects a ValueError.
    """
    constraints = parse_constraints(["a <= 1", "b <= 1"])

    with pytest.raises(ValueError, match="for 2 constraints"):
        probability_of_feasibility(constraints, np.zeros((3, 1)), np.ones((3, 1)))

    with pytest.raises(ValueError, match="do not match"):
        probability_of_feasibility(constraints, np.zeros((3, 2)), np.ones((4, 2)))
