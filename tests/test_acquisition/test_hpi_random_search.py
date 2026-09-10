from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    InCondition,
    Integer,
)
from ConfigSpace.hyperparameters import Constant

from smac.acquisition.function import EI
from smac.acquisition.maximizer import HPIRandomSearch
from smac.model.random_forest.random_forest import RandomForest
from smac.utils.configspace import convert_configurations_to_array

pytest.importorskip("hypershap")

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


@pytest.fixture
def configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)
    cs.add([Float("a", (0, 1)), Float("b", (0, 1)), Float("c", (0, 1))])

    return cs


@pytest.fixture
def configspace_conditional() -> ConfigurationSpace:
    """An MLP-like space: some hyperparameters are only active for specific values of ``solver``."""
    cs = ConfigurationSpace(seed=0)

    n_layer = Integer("n_layer", (1, 5), default=1)
    n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
    solver = Categorical("solver", ["lbfgs", "sgd", "adam"], default="adam")
    learning_rate_init = Float("learning_rate_init", (0.0001, 1.0), default=0.001, log=True)

    cs.add([n_layer, n_neurons, solver, learning_rate_init])
    cs.add(InCondition(child=learning_rate_init, parent=solver, values=["sgd", "adam"]))

    return cs


@pytest.fixture
def configspace_forbidden() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)
    a = Float("a", (0, 1), default=0.5)
    b = Categorical("b", ["x", "y"], default="x")
    cs.add([a, b])
    cs.add(ForbiddenAndConjunction(ForbiddenEqualsClause(b, "y")))

    return cs


@pytest.fixture
def acquisition_function(configspace: ConfigurationSpace) -> EI:
    model = RandomForest(configspace, n_trees=5, seed=0)
    rng = np.random.RandomState(0)
    X = rng.rand(30, len(list(configspace.values())))
    y = 1 - (np.sum(X, axis=1) / len(list(configspace.values())))
    model.train(X, y)

    ei = EI()
    ei.update(model=model, eta=0.5)

    return ei


# --------------------------------------------------------------
# Construction & validation
# --------------------------------------------------------------


def test_default_construction():
    rs = HPIRandomSearch()
    assert rs._threshold == [0.0, 0.8, 0.0]
    assert rs._fixing_strategy == "incumbent"


def test_invalid_fixing_strategy_raises(configspace):
    with pytest.raises(ValueError):
        HPIRandomSearch(configspace, fixing_strategy="unknown")


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_scalar_threshold_raises(configspace, threshold):
    with pytest.raises(ValueError):
        HPIRandomSearch(configspace, threshold=threshold)


def test_invalid_list_threshold_raises(configspace):
    with pytest.raises(ValueError):
        HPIRandomSearch(configspace, threshold=[0.0, 1.5])


@pytest.mark.parametrize("random_prob", [-0.1, 1.1])
def test_invalid_random_prob_raises(configspace, random_prob):
    with pytest.raises(ValueError):
        HPIRandomSearch(configspace, random_prob=random_prob)


def test_meta_includes_hpi_specific_fields(configspace):
    rs = HPIRandomSearch(configspace, n_trials=100, threshold=0.5, fixing_strategy="default", random_prob=0.2)

    meta = rs.meta

    assert meta["n_trials"] == 100
    assert meta["threshold"] == 0.5
    assert meta["fixing_strategy"] == "default"
    assert meta["random_prob"] == 0.2


# --------------------------------------------------------------
# _get_current_threshold
# --------------------------------------------------------------


def test_scalar_threshold_is_constant(configspace):
    rs = HPIRandomSearch(configspace, threshold=0.5)
    rs._n_evaluated_trials = 0
    assert rs._get_current_threshold() == 0.5

    rs._n_evaluated_trials = 1000
    assert rs._get_current_threshold() == 0.5


def test_list_threshold_picks_phase_and_clamps(configspace):
    rs = HPIRandomSearch(configspace, threshold=[0.0, 0.8, 0.0], n_trials=90)

    for n_evaluated, expected in [(0, 0.0), (29, 0.0), (30, 0.8), (59, 0.8), (60, 0.0), (89, 0.0), (1000, 0.0)]:
        rs._n_evaluated_trials = n_evaluated
        assert rs._get_current_threshold() == expected


# --------------------------------------------------------------
# _get_reference_configuration
# --------------------------------------------------------------


def test_get_reference_configuration_default_strategy_ignores_previous_configs(configspace):
    rs = HPIRandomSearch(configspace, fixing_strategy="default")
    assert rs._get_reference_configuration([]) == configspace.get_default_configuration()


def test_get_reference_configuration_random_strategy_samples_a_configuration(configspace):
    rs = HPIRandomSearch(configspace, fixing_strategy="random")
    reference = rs._get_reference_configuration([])

    assert isinstance(reference, Configuration)
    assert set(reference.keys()) == set(configspace.keys())


def test_get_reference_configuration_incumbent_strategy_picks_best_predicted_and_caches_context(
    configspace, acquisition_function
):
    rs = HPIRandomSearch(configspace, acquisition_function=acquisition_function)
    previous_configs = configspace.sample_configuration(10)

    reference = rs._get_reference_configuration(previous_configs)

    assert rs._context_configs == previous_configs
    assert len(rs._context_Y) == len(previous_configs)

    predicted = acquisition_function.model.predict_marginalized(convert_configurations_to_array(previous_configs))[0]
    assert reference == previous_configs[int(np.argmin(predicted))]


# --------------------------------------------------------------
# _compute_important_hps / _predict_config
# --------------------------------------------------------------


def test_predict_config_returns_negated_model_prediction(configspace, acquisition_function):
    rs = HPIRandomSearch(configspace, acquisition_function=acquisition_function)
    config = configspace.get_default_configuration()

    expected = -acquisition_function.model.predict(np.array([config.get_array()]))[0][0]
    assert rs._predict_config(config) == pytest.approx(expected)


def test_compute_important_hps_returns_hyperparameter_names(configspace, acquisition_function):
    """Regression test: HyperSHAP keys its results by name-tuples (e.g. `("a",)`), which must be unwrapped to
    plain hyperparameter names -- otherwise nothing is ever recognized as important, see `_reduce_configspace`."""
    rs = HPIRandomSearch(configspace, acquisition_function=acquisition_function)
    reference = configspace.get_default_configuration()

    important_hps = rs._compute_important_hps(reference, threshold=0.8)

    assert len(important_hps) > 0
    assert all(isinstance(hp, str) and hp in configspace.keys() for hp in important_hps)


# --------------------------------------------------------------
# _select_important_hps
# --------------------------------------------------------------


def test_select_important_hps_reaches_threshold(configspace):
    rs = HPIRandomSearch(configspace)
    shapley_values = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}

    assert rs._select_important_hps(shapley_values, threshold=0.8) == ["a", "b"]
    assert rs._select_important_hps(shapley_values, threshold=1.0) == ["a", "b", "c", "d"]


def test_select_important_hps_ignores_non_positive_values(configspace):
    rs = HPIRandomSearch(configspace)
    shapley_values = {"a": 0.5, "b": -0.3, "c": 0.0}

    assert rs._select_important_hps(shapley_values, threshold=1.0) == ["a"]


# --------------------------------------------------------------
# _reduce_configspace
# --------------------------------------------------------------


def test_reduce_configspace_fixes_unimportant_and_mutates_configspace(configspace_conditional):
    rs = HPIRandomSearch(configspace_conditional)
    reference = configspace_conditional.get_default_configuration()

    rs._reduce_configspace(["n_layer"], reference)

    assert not isinstance(rs._configspace["n_layer"], Constant)
    assert isinstance(rs._configspace["n_neurons"], Constant)
    assert rs._configspace["n_neurons"].default_value == reference["n_neurons"]


def test_reduce_configspace_promotes_parent_of_important_child(configspace_conditional):
    rs = HPIRandomSearch(configspace_conditional)
    reference = configspace_conditional.get_default_configuration()

    rs._reduce_configspace(["learning_rate_init"], reference)

    assert not isinstance(rs._configspace["solver"], Constant)
    condition_children = {condition.child.name for condition in rs._configspace.conditions}
    assert "learning_rate_init" in condition_children

    for config in rs._configspace.sample_configuration(20):
        assert config["n_layer"] == reference["n_layer"]


def test_reduce_configspace_keeps_sibling_condition_of_promoted_parent(configspace):
    """Regression test: a sibling condition on a to-be-promoted parent must not be dropped just because it was
    visited before the condition that triggers the promotion (order-dependence of a single pass)."""
    cs = ConfigurationSpace(seed=0)
    p = Categorical("p", ["x", "y"], default="x")
    unimportant_child = Float("unimportant_child", (0, 1), default=0.5)
    important_child = Float("important_child", (0, 1), default=0.5)
    cs.add([p, unimportant_child, important_child])
    cs.add(InCondition(child=unimportant_child, parent=p, values=["x"]))
    cs.add(InCondition(child=important_child, parent=p, values=["y"]))

    rs = HPIRandomSearch(cs)
    reference = cs.get_default_configuration()

    # Only "important_child" is important; "p" gets promoted since it's its parent.
    rs._reduce_configspace(["important_child"], reference)

    condition_children = {condition.child.name for condition in rs._configspace.conditions}
    assert "unimportant_child" in condition_children
    assert "important_child" in condition_children


def test_reduce_configspace_default_strategy_uses_reference_value_or_default(configspace_conditional):
    rs = HPIRandomSearch(configspace_conditional, fixing_strategy="default")

    # solver="adam" (default): learning_rate_init is active in this reference, keep its value.
    active_reference = configspace_conditional.get_default_configuration()
    rs._reduce_configspace(["n_layer"], active_reference)
    assert rs._configspace["learning_rate_init"].default_value == active_reference["learning_rate_init"]

    # solver="lbfgs": learning_rate_init is inactive in this reference, fall back to its own default.
    inactive_reference = Configuration(
        configspace_conditional, values={"n_layer": 1, "n_neurons": 10, "solver": "lbfgs"}
    )
    rs._reduce_configspace(["n_layer"], inactive_reference)
    assert (
        rs._configspace["learning_rate_init"].default_value
        == configspace_conditional["learning_rate_init"].default_value
    )


# --------------------------------------------------------------
# _incumbent_value_for
# --------------------------------------------------------------


def test_incumbent_value_for_uses_best_config_where_hp_is_active():
    cs = ConfigurationSpace(seed=0)
    solver = Categorical("solver", ["sgd", "adam"], default="sgd")
    lr = Float("learning_rate_init", (0.0001, 1.0), default=0.001, log=True)
    cs.add([solver, lr])
    cs.add(InCondition(child=lr, parent=solver, values=["adam"]))

    rs = HPIRandomSearch(cs)
    rs._context_configs = [
        Configuration(cs, values={"solver": "adam", "learning_rate_init": 0.5}),
        Configuration(cs, values={"solver": "adam", "learning_rate_init": 0.01}),
        Configuration(cs, values={"solver": "sgd"}),
    ]
    rs._context_Y = np.array([[10.0], [1.0], [5.0]])

    # The global incumbent (config with lowest y overall) has solver="adam", lr=0.01, so this matches the
    # simple case; the interesting part is that this is derived from "adam" trials specifically, not sgd ones.
    assert rs._incumbent_value_for(cs["learning_rate_init"]) == 0.01


def test_incumbent_value_for_falls_back_to_default_if_never_active():
    cs = ConfigurationSpace(seed=0)
    solver = Categorical("solver", ["sgd", "adam", "lbfgs"], default="lbfgs")
    momentum = Float("momentum", (0, 1), default=0.9)
    cs.add([solver, momentum])
    cs.add(InCondition(child=momentum, parent=solver, values=["sgd"]))

    rs = HPIRandomSearch(cs)
    rs._context_configs = [Configuration(cs, values={"solver": "adam"})]
    rs._context_Y = np.array([[1.0]])

    assert rs._incumbent_value_for(cs["momentum"]) == momentum.default_value


# --------------------------------------------------------------
# _drop_forbidden
# --------------------------------------------------------------


def test_drop_forbidden_is_noop_without_forbidden_clauses(configspace):
    rs = HPIRandomSearch(configspace)
    candidates = configspace.sample_configuration(10)

    assert rs._drop_forbidden(candidates) == candidates


def test_drop_forbidden_filters_violations(configspace_forbidden):
    rs = HPIRandomSearch(configspace_forbidden)

    unconstrained = ConfigurationSpace(seed=0)
    unconstrained.add(list(configspace_forbidden.values()))
    candidates = unconstrained.sample_configuration(200)
    assert any(config["b"] == "y" for config in candidates)  # sanity check: violations exist

    filtered = rs._drop_forbidden(candidates)
    assert len(filtered) < len(candidates)
    assert all(config["b"] != "y" for config in filtered)


# --------------------------------------------------------------
# _maximize
# --------------------------------------------------------------


def test_maximize_zero_threshold_uses_plain_random_search(configspace, acquisition_function):
    rs = HPIRandomSearch(configspace, acquisition_function=acquisition_function, threshold=0.0, random_prob=0.0)

    values = rs._maximize(configspace.sample_configuration(10), 10)

    assert len(values) == 10
    assert all(config.origin == "Acquisition Function Maximizer: Random Search (sorted)" for _, config in values)


def test_maximize_random_prob_one_always_samples_from_original_space(configspace, acquisition_function):
    rs = HPIRandomSearch(configspace, acquisition_function=acquisition_function, threshold=0.8, random_prob=1.0)
    # Sampled from `rs._original_cs`'s own (continuously advancing) stream, not a separately-reset copy of
    # `configspace`: `__init__` reseeds both `self._configspace` (== `configspace`, same object) and
    # `self._original_cs` to the *same* seed, so independently-drawn samples would coincide exactly.
    previous_configs = rs._original_cs.sample_configuration(20)

    values = rs._maximize(previous_configs, 10)

    assert len(values) > 0
    assert all(v == 0 for v, _ in values)
    assert all(config.origin == "Acquisition Function Maximizer: HPI Random Search (random)" for _, config in values)
    assert all(config not in previous_configs for _, config in values)


def test_maximize_reduction_path_produces_sorted_configs(configspace, acquisition_function):
    """End-to-end: the reduction path (random_prob=0, threshold>0) estimates hyperparameter importance via
    HyperSHAP and returns candidates sorted by acquisition value."""
    rs = HPIRandomSearch(configspace, acquisition_function=acquisition_function, threshold=0.8, random_prob=0.0)
    previous_configs = configspace.sample_configuration(20)

    values = rs._maximize(previous_configs, 5)

    assert len(values) > 0
    assert all(config.origin == "Acquisition Function Maximizer: HPI Random Search (sorted)" for _, config in values)

    acq_values = [v for v, _ in values]
    assert acq_values == sorted(acq_values, reverse=True)
