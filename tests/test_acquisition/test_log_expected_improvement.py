from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from smac.acquisition.function import EI, LogEI
from smac.acquisition.function.log_expected_improvement import log_h

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class GridModel:
    """Surrogate whose prediction is read straight off the input, for exact expected values."""

    def __init__(self, means, variances):
        self._means = np.asarray(means, dtype=float).reshape((-1, 1))
        self._variances = np.asarray(variances, dtype=float).reshape((-1, 1))

    def predict_marginalized(self, X):
        return self._means, self._variances


def test_log_h_matches_the_direct_form_where_it_is_representable():
    """The stable implementation agrees with the textbook one wherever floats can hold the latter.

    Evaluates both over a wide range and compares only where phi(z) + z*Phi(z) has not underflowed.
    """
    z = np.linspace(-25.0, 5.0, 2000)
    direct = norm.pdf(z) + z * norm.cdf(z)
    representable = direct > 1e-300

    assert log_h(z)[representable] == pytest.approx(np.log(direct[representable]), abs=1e-9)


def test_log_h_stays_finite_where_the_direct_form_is_zero():
    """Far below the incumbent the plain form collapses and the log form does not.

    Evaluates at standardised improvements from -50 to -1e9, checks the direct form is exactly zero there, and
    checks the log form is finite and still ordered.
    """
    z = np.array([-50.0, -100.0, -1e3, -1e5, -1e9])

    assert np.all(norm.pdf(z) + z * norm.cdf(z) == 0.0)

    values = log_h(z)

    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) < 0)


def test_log_h_is_strictly_increasing():
    """A larger standardised improvement is always worth more.

    Checks monotonicity across the boundary between the implementation's branches.
    """
    assert np.all(np.diff(log_h(np.linspace(-60.0, 5.0, 3000))) > 0)


def test_log_ei_equals_the_log_of_ei():
    """Where EI is representable, LogEI is exactly its logarithm.

    Predicts a spread of means with a fixed variance and compares the two acquisition functions.
    """
    means = np.linspace(-1.0, 3.0, 50)
    model = GridModel(means, np.full_like(means, 0.25))

    log_ei = LogEI()
    log_ei.update(model=model, eta=1.0)
    plain = EI()
    plain.update(model=model, eta=1.0)

    X = np.zeros((len(means), 1))
    ei_values = plain._compute(X).reshape(-1)
    representable = ei_values > 1e-300

    assert log_ei._compute(X).reshape(-1)[representable] == pytest.approx(
        np.log(ei_values[representable]), abs=1e-9
    )


def test_log_ei_ranks_identically_to_ei():
    """The logarithm does not reorder candidates.

    Compares the full ranking of fifty predictions under both acquisition functions.
    """
    means = np.linspace(0.0, 2.0, 50)
    model = GridModel(means, np.linspace(0.05, 0.5, 50))
    X = np.zeros((len(means), 1))

    log_ei = LogEI()
    log_ei.update(model=model, eta=1.0)
    plain = EI()
    plain.update(model=model, eta=1.0)

    assert np.array_equal(
        np.argsort(log_ei._compute(X).reshape(-1)), np.argsort(plain._compute(X).reshape(-1))
    )


def test_log_ei_discriminates_where_ei_is_flat_zero():
    """The reason the class exists: EI gives the maximizer nothing to climb, LogEI does.

    Predicts three hopeless but differently hopeless points, where EI is identically zero, and checks LogEI
    still orders them.
    """
    model = GridModel([50.0, 60.0, 70.0], [1.0, 1.0, 1.0])
    X = np.zeros((3, 1))

    plain = EI()
    plain.update(model=model, eta=0.0)
    log_ei = LogEI()
    log_ei.update(model=model, eta=0.0)

    assert np.all(plain._compute(X) == 0.0)

    values = log_ei._compute(X).reshape(-1)

    assert np.all(np.isfinite(values))
    assert values[0] > values[1] > values[2]


def test_log_ei_declares_itself_logarithmic():
    """Wrappers need to know to add rather than multiply.

    Reads the log flag off LogEI and off plain EI.
    """
    assert LogEI().log is True
    assert EI().log is False


def test_log_ei_requires_an_incumbent():
    """Computing before update is an error, as for EI.

    Calls _compute without having supplied eta.
    """
    log_ei = LogEI()
    log_ei.model = GridModel([1.0], [1.0])

    with pytest.raises(ValueError, match="No current best specified"):
        log_ei._compute(np.zeros((1, 1)))
