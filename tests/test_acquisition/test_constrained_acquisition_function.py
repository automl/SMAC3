from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac.acquisition.function import EI, LCB, ConstrainedAcquisitionFunction
from smac.runhistory.runhistory import RunHistory
from smac.utils.constraints import parse_constraints, probability_of_feasibility

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class FixedModel:
    """Surrogate returning a constant mean and variance per output, independent of the input."""

    def __init__(self, means: list[float], variances: list[float]):
        self.means = np.array(means, dtype=float)
        self.variances = np.array(variances, dtype=float)
        self.trained_on: tuple[np.ndarray, np.ndarray] | None = None

    def train(self, X, Y):
        self.trained_on = (X, Y)
        return self

    def predict_marginalized(self, X):
        n = X.shape[0]

        return np.tile(self.means, (n, 1)), np.tile(self.variances, (n, 1))

    @property
    def meta(self):
        return {"name": "FixedModel"}


@pytest.fixture
def configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)
    cs.add(Float("x", (0.0, 1.0)))

    return cs


def _runhistory(configspace: ConfigurationSpace, latencies: list[float | None]) -> RunHistory:
    runhistory = RunHistory()
    for i, latency in enumerate(latencies):
        configspace.seed(i)
        config = configspace.sample_configuration()
        constraint_values = None if latency is None else {"latency": latency}
        runhistory.add(config=config, cost=0.5, seed=0, constraint_values=constraint_values)

    return runhistory


def _make(objective_model, constraint_model, inner=None):
    acquisition = ConstrainedAcquisitionFunction(
        acquisition_function=inner if inner is not None else EI(),
        constraints=parse_constraints(["latency <= 100"]),
        constraint_model=constraint_model,
    )
    acquisition.model = objective_model

    return acquisition


def test_at_least_one_constraint_is_required():
    """Wrapping an acquisition function without constraints is a mistake, not a no-op.

    Constructs the wrapper with an empty constraint list and expects a ValueError.
    """
    with pytest.raises(ValueError, match="at least one constraint"):
        ConstrainedAcquisitionFunction(EI(), [], FixedModel([0.0], [1.0]))


def test_value_is_the_acquisition_times_the_probability_of_feasibility(configspace):
    """The weighted value is exactly the inner acquisition value times the feasibility probability.

    Runs the wrapper and a bare EI over the same fixed models and compares against the hand-computed product.
    """
    objective_model = FixedModel([0.5], [1.0])
    constraint_model = FixedModel([90.0], [25.0])

    acquisition = _make(objective_model, constraint_model)
    acquisition.update(model=objective_model, eta=1.0, runhistory=_runhistory(configspace, [50.0]))

    X = np.array([[0.2], [0.7]])
    values = acquisition._compute(X)

    bare = EI()
    bare.update(model=objective_model, eta=acquisition._eta)
    expected_feasibility = probability_of_feasibility(
        parse_constraints(["latency <= 100"]), *constraint_model.predict_marginalized(X)
    )

    assert values == pytest.approx(bare._compute(X).reshape((-1, 1)) * (expected_feasibility + 1e-12))
    assert values.shape == (2, 1)


def test_an_infeasible_region_is_suppressed(configspace):
    """A configuration predicted to violate the bound scores far below a feasible one.

    Compares the weighted value under a constraint model predicting well inside the bound against one
    predicting well outside it, with the objective model held fixed.
    """
    objective_model = FixedModel([0.5], [1.0])
    runhistory = _runhistory(configspace, [50.0])
    X = np.array([[0.5]])

    feasible = _make(objective_model, FixedModel([50.0], [1.0]))
    feasible.update(model=objective_model, eta=1.0, runhistory=runhistory)

    infeasible = _make(objective_model, FixedModel([150.0], [1.0]))
    infeasible.update(model=objective_model, eta=1.0, runhistory=runhistory)

    assert infeasible._compute(X) < feasible._compute(X)


def test_without_a_feasible_observation_only_feasibility_matters(configspace):
    """Before anything feasible turns up the search maximizes the probability of feasibility.

    Builds a runhistory whose single trial violates the bound and checks the value equals the feasibility
    probability alone, ignoring the objective.
    """
    objective_model = FixedModel([0.5], [1.0])
    constraint_model = FixedModel([90.0], [25.0])

    acquisition = _make(objective_model, constraint_model)
    acquisition.update(model=objective_model, eta=1.0, runhistory=_runhistory(configspace, [150.0]))

    X = np.array([[0.2], [0.7]])
    expected = probability_of_feasibility(
        parse_constraints(["latency <= 100"]), *constraint_model.predict_marginalized(X)
    )

    assert acquisition._has_feasible is False
    assert acquisition._compute(X) == pytest.approx(expected)


def test_the_incumbent_is_the_best_feasible_one(configspace):
    """The wrapped acquisition function is given the best feasible value, not the best overall.

    Uses an objective model whose prediction depends on the input so that the feasible and infeasible
    configurations differ, then checks the eta handed on matches the feasible one.
    """

    class SlopedModel(FixedModel):
        def predict_marginalized(self, X):
            return X[:, :1].copy(), np.ones((X.shape[0], 1))

    runhistory = RunHistory()
    configspace.seed(5)
    good_but_infeasible = configspace.sample_configuration()
    configspace.seed(0)
    worse_but_feasible = configspace.sample_configuration()

    # The infeasible configuration has to be the better one for this test to mean anything
    assert good_but_infeasible.get_array()[0] < worse_but_feasible.get_array()[0]

    runhistory.add(config=good_but_infeasible, cost=0.1, seed=0, constraint_values={"latency": 150.0})
    runhistory.add(config=worse_but_feasible, cost=0.9, seed=0, constraint_values={"latency": 50.0})

    objective_model = SlopedModel([0.0], [1.0])
    acquisition = _make(objective_model, FixedModel([50.0], [1.0]))
    acquisition.update(model=objective_model, eta=0.0, runhistory=runhistory)

    assert acquisition._has_feasible is True
    assert acquisition._eta == pytest.approx(worse_but_feasible.get_array()[0])


def test_constraint_model_trains_on_raw_values(configspace):
    """The constraint surrogate sees the observed values in their own units.

    Runs an update over a runhistory with known latencies and inspects what the constraint model was trained on.
    """
    constraint_model = FixedModel([50.0], [1.0])
    acquisition = _make(FixedModel([0.5], [1.0]), constraint_model)
    acquisition.update(model=acquisition.model, eta=1.0, runhistory=_runhistory(configspace, [50.0, 150.0]))

    _, Y = constraint_model.trained_on

    assert sorted(Y.ravel().tolist()) == [50.0, 150.0]


def test_trials_without_constraint_values_are_not_trained_on(configspace):
    """A trial that reported no value cannot be a training point.

    Mixes reporting and non-reporting trials in a runhistory and checks only the reporting ones were used.
    """
    constraint_model = FixedModel([50.0], [1.0])
    acquisition = _make(FixedModel([0.5], [1.0]), constraint_model)
    acquisition.update(model=acquisition.model, eta=1.0, runhistory=_runhistory(configspace, [50.0, None, 150.0]))

    _, Y = constraint_model.trained_on

    assert sorted(Y.ravel().tolist()) == [50.0, 150.0]


def test_falls_back_to_the_bare_acquisition_without_any_observation(configspace):
    """With nothing observed at all the wrapper does not invent a feasibility weight.

    Updates over an empty runhistory and checks the value matches the unwrapped acquisition function.
    """
    objective_model = FixedModel([0.5], [1.0])
    acquisition = _make(objective_model, FixedModel([50.0], [1.0]))
    acquisition.update(model=objective_model, eta=1.0, runhistory=RunHistory())

    X = np.array([[0.2]])
    bare = EI()
    bare.update(model=objective_model, eta=1.0)

    assert acquisition._trained is False
    assert acquisition._compute(X) == pytest.approx(bare._compute(X))


def test_a_negative_inner_acquisition_is_shifted_before_weighting(configspace):
    """LCB returns negative values, which would invert the ranking if multiplied directly.

    Wraps LCB, computes the weighted values, and checks they are non-negative.
    """
    objective_model = FixedModel([0.5], [1.0])
    acquisition = _make(objective_model, FixedModel([50.0], [1.0]), inner=LCB())
    acquisition.update(
        model=objective_model,
        eta=1.0,
        num_data=10,
        runhistory=_runhistory(configspace, [50.0]),
    )

    assert acquisition._rescale is True
    assert np.all(acquisition._compute(np.array([[0.2], [0.7]])) >= 0.0)


def test_meta_describes_the_wrapping(configspace):
    """The metadata records the constraints and the wrapped acquisition function.

    Reads meta off a constructed wrapper and checks the constraint expressions round trip as strings.
    """
    acquisition = _make(FixedModel([0.5], [1.0]), FixedModel([50.0], [1.0]))
    meta = acquisition.meta

    assert meta["name"] == "ConstrainedAcquisitionFunction"
    assert meta["constraints"] == ["latency <= 100.0"]
    assert meta["acquisition_function"]["name"] == "EI"
