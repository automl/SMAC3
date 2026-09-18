import numpy as np
import pytest
from ConfigSpace import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    Categorical,
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    Integer,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac.acquisition.weight import ConfigSpacePrior
from smac.utils.configspace import (
    create_prior_configspace_copy,
    create_uniform_configspace_copy,
)


@pytest.fixture
def non_uniform_configspace():
    configspace = ConfigurationSpace(name="test_configspace", seed=42, meta={"info": "test configspace"})
    configspace.add(
        BetaFloatHyperparameter(
            "beta_float",
            alpha=2.0,
            beta=5.0,
            lower=0.0,
            upper=1.0,
            default_value=0.5,
            log=False,
        )
    )
    configspace.add(
        BetaIntegerHyperparameter(
            "beta_int",
            alpha=2.0,
            beta=5.0,
            lower=1,
            upper=10,
            default_value=5,
            log=False,
        )
    )
    configspace.add(
        NormalFloatHyperparameter(
            "normal_float",
            mu=0.0,
            sigma=1.0,
            lower=-3.0,
            upper=3.0,
            default_value=0.0,
            log=False,
        )
    )
    configspace.add(
        NormalIntegerHyperparameter("normal_int", mu=5, sigma=2, lower=1, upper=10, default_value=5, log=False)
    )
    configspace.add(UniformFloatHyperparameter("uniform_float", lower=0.0, upper=10.0, default_value=5.0, log=False))
    configspace.add(UniformIntegerHyperparameter("uniform_int", lower=1, upper=100, default_value=50, log=False))
    configspace.add(
        CategoricalHyperparameter(
            "categorical",
            choices=["red", "green", "blue"],
            default_value="green",
            weights=[0.2, 0.5, 0.3],
        )
    )
    configspace.add(Constant("constant", value=3.14))
    return configspace


@pytest.fixture
def uniform_configspace():
    configspace = ConfigurationSpace(name="test_configspace", seed=42, meta={"info": "test configspace"})
    configspace.add(UniformFloatHyperparameter("beta_float", lower=0.0, upper=1.0, default_value=0.5, log=False))
    configspace.add(UniformIntegerHyperparameter("beta_int", lower=1, upper=10, default_value=5, log=False))
    configspace.add(UniformFloatHyperparameter("normal_float", lower=-3.0, upper=3.0, default_value=0.0, log=False))
    configspace.add(UniformIntegerHyperparameter("normal_int", lower=1, upper=10, default_value=5, log=False))
    configspace.add(UniformFloatHyperparameter("uniform_float", lower=0.0, upper=10.0, default_value=5.0, log=False))
    configspace.add(UniformIntegerHyperparameter("uniform_int", lower=1, upper=100, default_value=50, log=False))
    configspace.add(CategoricalHyperparameter("categorical", choices=["red", "green", "blue"], default_value="green"))
    configspace.add(Constant("constant", value=3.14))
    return configspace


def test_create_uniform_configspace_copy(
    non_uniform_configspace: ConfigurationSpace, uniform_configspace: ConfigurationSpace
):
    adapted_configspace = create_uniform_configspace_copy(non_uniform_configspace)
    assert adapted_configspace == uniform_configspace


def test_create_uniform_configspace_copy_preserves_forbiddens():
    cs = ConfigurationSpace(seed=42)
    x = NormalFloatHyperparameter("x", mu=0.0, sigma=1.0, lower=-3.0, upper=3.0)
    # default is "b", so the default config doesn't violate the forbidden clause
    y = CategoricalHyperparameter("y", choices=["a", "b", "c"], default_value="b")
    cs.add([x, y])
    cs.add(ForbiddenAndConjunction(ForbiddenEqualsClause(x, -3.0), ForbiddenEqualsClause(y, "a")))

    adapted = create_uniform_configspace_copy(cs)

    assert len(adapted.forbidden_clauses) == 1
    for _ in range(100):
        config = adapted.sample_configuration()
        config.check_valid_configuration()
        assert not (config["x"] == -3.0 and config["y"] == "a")


def test_create_prior_configspace_copy_places_a_belief():
    configspace = ConfigurationSpace(seed=0)
    configspace.add(
        Float("lr", (1e-4, 1.0), log=True),
        Integer("n", (1, 10)),
        Categorical("k", ["a", "b", "c"]),
    )

    prior = create_prior_configspace_copy(configspace, {"lr": 0.01, "n": 3, "k": "b"})
    hyperparameters = dict(prior)

    assert isinstance(hyperparameters["lr"], NormalFloatHyperparameter)
    assert isinstance(hyperparameters["n"], NormalIntegerHyperparameter)
    assert hyperparameters["lr"].mu == 0.01
    assert hyperparameters["n"].mu == 3

    # The density has to peak at the believed value, log scale included.
    values = np.linspace(0, 1, 2001)
    densities = hyperparameters["lr"].pdf_vector(values)
    assert hyperparameters["lr"].to_value(np.array([values[int(np.argmax(densities))]]))[0] == pytest.approx(0.01)

    assert list(hyperparameters["k"].probabilities) == pytest.approx([0.1, 0.8, 0.1])


def test_create_prior_configspace_copy_leaves_the_rest_uniform():
    """That is how a belief about part of the search space is stated."""
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("a", (0.0, 1.0)), Float("b", (0.0, 1.0)))

    prior = create_prior_configspace_copy(configspace, {"a": 0.25})
    hyperparameters = dict(prior)

    assert isinstance(hyperparameters["a"], NormalFloatHyperparameter)
    assert isinstance(hyperparameters["b"], UniformFloatHyperparameter)


def test_create_prior_configspace_copy_leaves_a_constant_alone():
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Constant("c", 5), Float("a", (0.0, 1.0)))

    prior = create_prior_configspace_copy(configspace, {"a": 0.25})

    assert isinstance(dict(prior)["c"], Constant)


def test_create_prior_configspace_copy_preserves_conditions_and_forbiddens():
    configspace = ConfigurationSpace(seed=0)
    a = Categorical("a", ["x", "y"])
    b = Float("b", (0.0, 1.0))
    configspace.add(a, b)
    configspace.add(EqualsCondition(b, a, "x"))
    configspace.add(ForbiddenEqualsClause(a, "y"))

    prior = create_prior_configspace_copy(configspace, {"b": 0.25})

    assert len(prior.conditions) == 1
    assert len(prior.forbidden_clauses) == 1

    # The forbidden clause has to point at the new hyperparameter, not the old one.
    assert prior.forbidden_clauses[0].hyperparameter is dict(prior)["a"]


def test_a_sharper_belief_has_a_smaller_standard_deviation():
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("a", (0.0, 1.0)))

    wide = dict(create_prior_configspace_copy(configspace, {"a": 0.5}, std_denominator=2.0))["a"]
    narrow = dict(create_prior_configspace_copy(configspace, {"a": 0.5}, std_denominator=20.0))["a"]

    assert narrow.sigma < wide.sigma


def test_an_impossible_belief_is_rejected():
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("a", (0.0, 1.0)), Categorical("k", ["x", "y"]))

    with pytest.raises(ValueError, match="outside"):
        create_prior_configspace_copy(configspace, {"a": 2.0})

    with pytest.raises(ValueError, match="not one of"):
        create_prior_configspace_copy(configspace, {"k": "z"})

    with pytest.raises(ValueError, match="not in the search space"):
        create_prior_configspace_copy(configspace, {"nope": 1.0})

    with pytest.raises(ValueError, match="denominator must be positive"):
        create_prior_configspace_copy(configspace, {"a": 0.5}, std_denominator=0.0)

    with pytest.raises(ValueError, match="categorical weight must be in"):
        create_prior_configspace_copy(configspace, {"k": "x"}, categorical_weight=1.5)


def test_the_belief_can_be_read_as_a_prior():
    """The whole point: the result is something the acquisition function can be weighted by."""
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("a", (0.0, 1.0)), Float("b", (0.0, 1.0)))

    prior = ConfigSpacePrior(create_prior_configspace_copy(configspace, {"a": 0.25}))
    prior.validate_against(configspace)

    at_the_belief = prior.pdf(np.array([[0.25, 0.5]]))
    elsewhere = prior.pdf(np.array([[0.95, 0.5]]))

    assert at_the_belief[0, 0] > elsewhere[0, 0]
