from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac.acquisition.function import EI, LCB, ConstrainedAcquisitionFunction, LogEI
from smac.runhistory.runhistory import RunHistory
from smac.acquisition.function.abstract_acquisition_function import AcquisitionScale
from smac.utils.constraints import (
    bilog,
    log_probability_of_feasibility,
    parse_constraints,
    probability_of_feasibility,
)

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class FixedModel:
    """Surrogate returning a constant mean and variance per output, independent of the input.

    Used both as the objective model and as the constraint model; in the latter case the outputs are residuals,
    negative when the constraint holds.
    """

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
    constraint_model = FixedModel([-10.0], [25.0])   # residual: ten below the bound

    acquisition = _make(objective_model, constraint_model)
    acquisition.update(model=objective_model, eta=1.0, runhistory=_runhistory(configspace, [50.0]))

    X = np.array([[0.2], [0.7]])
    values = acquisition._compute(X)

    bare = EI()
    bare.update(model=objective_model, eta=acquisition._eta)
    expected_feasibility = probability_of_feasibility(*constraint_model.predict_marginalized(X))

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

    feasible = _make(objective_model, FixedModel([-50.0], [1.0]))
    feasible.update(model=objective_model, eta=1.0, runhistory=runhistory)

    infeasible = _make(objective_model, FixedModel([50.0], [1.0]))
    infeasible.update(model=objective_model, eta=1.0, runhistory=runhistory)

    assert infeasible._compute(X) < feasible._compute(X)


def test_without_a_feasible_observation_only_feasibility_matters(configspace):
    """Before anything feasible turns up the search maximizes the probability of feasibility.

    Builds a runhistory whose single trial violates the bound and checks the value equals the feasibility
    probability alone, ignoring the objective.
    """
    objective_model = FixedModel([0.5], [1.0])
    constraint_model = FixedModel([-10.0], [25.0])

    acquisition = _make(objective_model, constraint_model)
    acquisition.update(model=objective_model, eta=1.0, runhistory=_runhistory(configspace, [150.0]))

    X = np.array([[0.2], [0.7]])
    expected = probability_of_feasibility(*constraint_model.predict_marginalized(X))

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
    acquisition = _make(objective_model, FixedModel([-50.0], [1.0]))
    acquisition.update(model=objective_model, eta=0.0, runhistory=runhistory)

    assert acquisition._has_feasible is True
    assert acquisition._eta == pytest.approx(worse_but_feasible.get_array()[0])


def test_constraint_model_trains_on_transformed_residuals(configspace):
    """The constraint surrogate is fitted on the compressed distance to the bound, not the raw value.

    Runs an update over a runhistory with one satisfying and one violating latency, and checks the training
    targets are the bilog of their signed residuals, so that the boundary sits at zero.
    """
    constraint_model = FixedModel([0.0], [1.0])
    acquisition = _make(FixedModel([0.5], [1.0]), constraint_model)
    acquisition.update(model=acquisition.model, eta=1.0, runhistory=_runhistory(configspace, [50.0, 150.0]))

    _, Y = constraint_model.trained_on

    # residuals against "latency <= 100" are -50 (satisfied) and +50 (violated)
    assert sorted(Y.ravel().tolist()) == pytest.approx(sorted(bilog(np.array([-50.0, 50.0])).tolist()))


def test_the_transform_can_be_turned_off(configspace):
    """Disabling the transform fits the untouched residuals.

    Runs the same update with transform=False and checks the training targets are the plain signed residuals.
    """
    constraint_model = FixedModel([0.0], [1.0])
    acquisition = ConstrainedAcquisitionFunction(
        acquisition_function=EI(),
        constraints=parse_constraints(["latency <= 100"]),
        constraint_model=constraint_model,
        transform=False,
    )
    acquisition.model = FixedModel([0.5], [1.0])
    acquisition.update(model=acquisition.model, eta=1.0, runhistory=_runhistory(configspace, [50.0, 150.0]))

    _, Y = constraint_model.trained_on

    assert sorted(Y.ravel().tolist()) == [-50.0, 50.0]


def test_a_lower_bound_becomes_a_negative_residual_when_satisfied(configspace):
    """A ">=" constraint is folded into the same "residual <= 0" convention.

    Fits a lower-bounded constraint on one satisfying and one violating observation and checks the signs.
    """
    constraint_model = FixedModel([0.0], [1.0])
    acquisition = ConstrainedAcquisitionFunction(
        acquisition_function=EI(),
        constraints=parse_constraints(["accuracy >= 0.9"]),
        constraint_model=constraint_model,
        transform=False,
    )
    acquisition.model = FixedModel([0.5], [1.0])

    runhistory = RunHistory()
    for i, accuracy in enumerate([0.95, 0.80]):
        configspace.seed(i)
        runhistory.add(config=configspace.sample_configuration(), cost=0.5, seed=0,
                       constraint_values={"accuracy": accuracy})

    acquisition.update(model=acquisition.model, eta=1.0, runhistory=runhistory)

    _, Y = constraint_model.trained_on

    assert sorted(Y.ravel().tolist()) == pytest.approx([-0.05, 0.10])


def test_trials_without_constraint_values_are_not_trained_on(configspace):
    """A trial that reported no value cannot be a training point.

    Mixes reporting and non-reporting trials in a runhistory and checks only the reporting ones were used.
    """
    constraint_model = FixedModel([0.0], [1.0])
    acquisition = _make(FixedModel([0.5], [1.0]), constraint_model)
    acquisition.update(model=acquisition.model, eta=1.0, runhistory=_runhistory(configspace, [50.0, None, 150.0]))

    _, Y = constraint_model.trained_on

    assert Y.shape[0] == 2


def test_falls_back_to_the_bare_acquisition_without_any_observation(configspace):
    """With nothing observed at all the wrapper does not invent a feasibility weight.

    Updates over an empty runhistory and checks the value matches the unwrapped acquisition function.
    """
    objective_model = FixedModel([0.5], [1.0])
    acquisition = _make(objective_model, FixedModel([-50.0], [1.0]))
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

    assert acquisition._scale is AcquisitionScale.SIGNED
    assert np.all(acquisition._compute(np.array([[0.2], [0.7]])) >= 0.0)


def test_meta_describes_the_wrapping(configspace):
    """The metadata records the constraints and the wrapped acquisition function.

    Reads meta off a constructed wrapper and checks the constraint expressions round trip as strings.
    """
    acquisition = _make(FixedModel([0.5], [1.0]), FixedModel([-50.0], [1.0]))
    meta = acquisition.meta

    assert meta["name"] == "ConstrainedAcquisitionFunction"
    assert meta["constraints"] == ["latency <= 100.0"]
    assert meta["acquisition_function"]["name"] == "EI"


def _log_make(objective_model, constraint_model, constraints=None):
    acquisition = ConstrainedAcquisitionFunction(
        acquisition_function=LogEI(),
        constraints=parse_constraints(constraints or ["latency <= 100"]),
        constraint_model=constraint_model,
    )
    acquisition.model = objective_model

    return acquisition


def test_a_log_inner_function_makes_the_wrapper_logarithmic(configspace):
    """The wrapper reports the space it is working in so that callers can compose correctly.

    Reads the log flag off wrappers around LogEI and around EI.
    """
    assert _log_make(FixedModel([0.5], [1.0]), FixedModel([-50.0], [1.0])).value_scale is AcquisitionScale.LOG
    assert _make(FixedModel([0.5], [1.0]), FixedModel([-50.0], [1.0])).value_scale is AcquisitionScale.LINEAR


def test_log_weighting_adds_instead_of_multiplying(configspace):
    """In log space the feasibility weight is a sum of log probabilities.

    Compares the wrapper against LogEI plus the log feasibility computed independently.
    """
    objective_model = FixedModel([0.5], [1.0])
    constraint_model = FixedModel([-10.0], [25.0])

    acquisition = _log_make(objective_model, constraint_model)
    acquisition.update(model=objective_model, eta=1.0, runhistory=_runhistory(configspace, [50.0]))

    X = np.array([[0.2], [0.7]])

    bare = LogEI()
    bare.update(model=objective_model, eta=acquisition._eta)
    expected = bare._compute(X).reshape((-1, 1)) + log_probability_of_feasibility(
        *constraint_model.predict_marginalized(X)
    )

    assert acquisition._compute(X) == pytest.approx(expected)


def test_log_and_plain_weighting_rank_the_same_when_nothing_underflows(configspace):
    """The log form is the same criterion, not a different one.

    Ranks candidates predicted at a spread of residuals under both forms and compares the orderings.
    """

    class SlopedConstraint(FixedModel):
        def predict_marginalized(self, X):
            return (X[:, :1] * 40.0) - 20.0, np.full((X.shape[0], 1), 4.0)

    objective_model = FixedModel([0.5], [1.0])
    X = np.linspace(0.0, 1.0, 25).reshape((-1, 1))
    runhistory = _runhistory(configspace, [50.0])

    plain = _make(objective_model, SlopedConstraint([0.0], [1.0]))
    plain.update(model=objective_model, eta=1.0, runhistory=runhistory)

    logged = _log_make(objective_model, SlopedConstraint([0.0], [1.0]))
    logged.update(model=objective_model, eta=1.0, runhistory=runhistory)

    assert np.array_equal(
        np.argsort(plain._compute(X).reshape(-1)), np.argsort(logged._compute(X).reshape(-1))
    )


def test_log_weighting_survives_what_the_product_cannot(configspace):
    """Many unlikely constraints flatten the plain product to zero; the log form keeps ranking.

    Predicts sixty residuals six sigma outside their bounds, with one candidate strictly closer to feasible
    than the other, and checks the plain form ties them while the log form separates them.
    """
    constraints = [f"c{i} <= 0" for i in range(60)]

    class Hopeless(FixedModel):
        def predict_marginalized(self, X):
            # the first candidate is nearer the boundary than the second, in every constraint
            residuals = np.where(X[:, :1] < 0.5, 6.0, 7.0)

            return np.tile(residuals, (1, 60)), np.ones((X.shape[0], 60))

    objective_model = FixedModel([0.5], [1.0])
    X = np.array([[0.2], [0.8]])

    runhistory = RunHistory()
    configspace.seed(0)
    runhistory.add(
        config=configspace.sample_configuration(),
        cost=0.5,
        seed=0,
        constraint_values={f"c{i}": -1.0 for i in range(60)},
    )

    plain = ConstrainedAcquisitionFunction(EI(), parse_constraints(constraints), Hopeless([0.0], [1.0]))
    plain.model = objective_model
    plain.update(model=objective_model, eta=1.0, runhistory=runhistory)

    logged = ConstrainedAcquisitionFunction(LogEI(), parse_constraints(constraints), Hopeless([0.0], [1.0]))
    logged.model = objective_model
    logged.update(model=objective_model, eta=1.0, runhistory=runhistory)

    plain_values = plain._compute(X).reshape(-1)
    log_values = logged._compute(X).reshape(-1)

    # The product has underflowed, so the two candidates are indistinguishable
    assert plain_values[0] == plain_values[1]

    assert np.all(np.isfinite(log_values))
    assert log_values[0] > log_values[1]


def test_the_log_fallback_is_the_log_probability_of_feasibility(configspace):
    """With nothing feasible the log wrapper returns log feasibility, not feasibility.

    Builds a runhistory whose only trial violates the bound and compares against the log probability directly.
    """
    objective_model = FixedModel([0.5], [1.0])
    constraint_model = FixedModel([-10.0], [25.0])

    acquisition = _log_make(objective_model, constraint_model)
    acquisition.update(model=objective_model, eta=1.0, runhistory=_runhistory(configspace, [150.0]))

    X = np.array([[0.2], [0.7]])

    assert acquisition._has_feasible is False
    assert acquisition._compute(X) == pytest.approx(
        log_probability_of_feasibility(*constraint_model.predict_marginalized(X))
    )
