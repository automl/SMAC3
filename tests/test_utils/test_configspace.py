import pytest
from ConfigSpace import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.utils.configspace import create_uniform_configspace_copy


@pytest.fixture
def non_uniform_configspace():
    configspace = ConfigurationSpace(
        name="test_configspace", seed=42, meta={"info": "test configspace"}
    )
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
        NormalIntegerHyperparameter(
            "normal_int", mu=5, sigma=2, lower=1, upper=10, default_value=5, log=False
        )
    )
    configspace.add(
        UniformFloatHyperparameter(
            "uniform_float", lower=0.0, upper=10.0, default_value=5.0, log=False
        )
    )
    configspace.add(
        UniformIntegerHyperparameter(
            "uniform_int", lower=1, upper=100, default_value=50, log=False
        )
    )
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
    configspace = ConfigurationSpace(
        name="test_configspace", seed=42, meta={"info": "test configspace"}
    )
    configspace.add(
        UniformFloatHyperparameter(
            "beta_float", lower=0.0, upper=1.0, default_value=0.5, log=False
        )
    )
    configspace.add(
        UniformIntegerHyperparameter(
            "beta_int", lower=1, upper=10, default_value=5, log=False
        )
    )
    configspace.add(
        UniformFloatHyperparameter(
            "normal_float", lower=-3.0, upper=3.0, default_value=0.0, log=False
        )
    )
    configspace.add(
        UniformIntegerHyperparameter(
            "normal_int", lower=1, upper=10, default_value=5, log=False
        )
    )
    configspace.add(
        UniformFloatHyperparameter(
            "uniform_float", lower=0.0, upper=10.0, default_value=5.0, log=False
        )
    )
    configspace.add(
        UniformIntegerHyperparameter(
            "uniform_int", lower=1, upper=100, default_value=50, log=False
        )
    )
    configspace.add(
        CategoricalHyperparameter(
            "categorical", choices=["red", "green", "blue"], default_value="green"
        )
    )
    configspace.add(Constant("constant", value=3.14))
    return configspace


def test_create_uniform_configspace_copy(
    non_uniform_configspace: ConfigurationSpace, uniform_configspace: ConfigurationSpace
):
    adapted_configspace = create_uniform_configspace_copy(non_uniform_configspace)
    assert adapted_configspace == uniform_configspace
