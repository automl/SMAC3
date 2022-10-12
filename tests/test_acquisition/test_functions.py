from typing import Any

import numpy as np
import pytest

from smac.acquisition.function import (
    EI,
    EIPS,
    LCB,
    PI,
    TS,
    IntegratedAcquisitionFunction,
    PriorAcquisitionFunction,
)

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class ConfigurationMock:
    def __init__(self, values=None):
        self.values = values

    def get_array(self):
        return self.values


class MockModel:
    def __init__(self, num_targets=1, seed=0):
        self.num_targets = num_targets
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)

    def predict_marginalized(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 1)), np.array(
            [np.mean(X, axis=1).reshape((1, -1))] * self.num_targets
        ).reshape((-1, 1))


class MockModelDual:
    def __init__(self, num_targets=1):
        self.num_targets = num_targets

    def predict_marginalized(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 2)), np.array(
            [np.mean(X, axis=1).reshape((1, -1))] * self.num_targets
        ).reshape((-1, 2))


class MockPrior:
    def __init__(self, pdf, max_density):
        self.pdf = pdf
        self.max_density = max_density

    def _pdf(self, X):
        return self.pdf(X)

    def get_max_density(self):
        return self.max_density


class GetHPDict:
    def __init__(self, hyperparameter_dict) -> None:
        self._hyperparameter_dict = None
        self._return_value = None
        self.return_value = hyperparameter_dict

    @property
    def hyperparameter_dict(self):
        return self._hyperparameter_dict

    @hyperparameter_dict.setter
    def hyperparameter_dict(self, value):
        self._hyperparameter_dict = value
        self._return_value = value

    @property
    def return_value(self):
        return self._return_value

    @return_value.setter
    def return_value(self, value):
        self._return_value = value
        self._hyperparameter_dict = value

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.hyperparameter_dict


class MockConfigurationSpace:
    def __init__(self, hyperparameter_dict):
        self.hyperparameter_dict = hyperparameter_dict
        self.get_hyperparameters_dict = GetHPDict(hyperparameter_dict=hyperparameter_dict)


class PriorMockModel:
    def __init__(self, hyperparameter_dict=None, num_targets=1, seed=0):
        self.num_targets = num_targets
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)
        self._configspace = MockConfigurationSpace(hyperparameter_dict)
        self.hyperparameter_dict = hyperparameter_dict
        # since the PriorAcquisitionFunction needs to return the hyperparameters in dict
        # form through two function calls (self.model.get_configspace().get_hyperparameters_dict()),
        # we need a slightly intricate solution
        self._configspace.get_hyperparameters_dict.return_value = self.hyperparameter_dict

    def get_configspace(self):
        return self._configspace

    def predict_marginalized(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 1)), np.array(
            [np.mean(X, axis=1).reshape((1, -1))] * self.num_targets
        ).reshape((-1, 1))

    def update_prior(self, hyperparameter_dict):
        self._configspace.get_hyperparameters_dict.return_value = hyperparameter_dict


class MockModelRNG(MockModel):
    def __init__(self, num_targets=1, seed=0):
        self.num_targets = num_targets
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)


class MockModelSampler(MockModelRNG):
    def __init__(self, num_targets=1, seed=0):
        self.num_targets = num_targets
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    def sample_functions(self, X, n_funcs=1):
        m = np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1,))
        var = np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1,))
        var = np.diag(var)
        return self._rng.multivariate_normal(m, var, n_funcs).T


@pytest.fixture
def model():
    return MockModel()


@pytest.fixture
def acquisition_function(model):
    ei = EI()
    ei.model = model

    return ei


# --------------------------------------------------------------
# Test AbstractAcquisitionFunction
# --------------------------------------------------------------


def test_update_model_and_eta(model, acquisition_function):
    model = "abc"
    assert acquisition_function._eta is None
    acquisition_function.update(model=model, eta=0.1)
    assert acquisition_function.model == model
    assert acquisition_function._eta == 0.1


def test_update_with_kwargs(acquisition_function):
    acquisition_function.update(model="abc", eta=0.0, other="hi there:)")
    assert acquisition_function.model == "abc"


def test_update_without_required(acquisition_function):
    with pytest.raises(
        TypeError,
    ):
        acquisition_function.update(other=None)


# --------------------------------------------------------------
# Test IntegratedAcquisitionFunction
# --------------------------------------------------------------


@pytest.fixture
def multimodel():
    model = MockModel()
    model.models = [MockModel()] * 3
    return model


@pytest.fixture
def acq_multi(multimodel, acquisition_function):
    acquisition_function.model = multimodel
    return acquisition_function


@pytest.fixture
def iaf(multimodel, acq_multi):
    iaf = IntegratedAcquisitionFunction(acquisition_function=acq_multi)
    iaf.model = multimodel
    return iaf


def test_integrated_acquisition_function_update(iaf, model, multimodel):
    iaf.update(model=multimodel, eta=2)
    for func in iaf._functions:
        assert func._eta == 2

    with pytest.raises(ValueError) as exc_info:
        iaf.update(model=MockModel())
    assert exc_info.value.args[0] == "IntegratedAcquisitionFunction requires at least one model to integrate!"

    with pytest.raises(ValueError) as exc_info:
        iaf.model.models = []
        iaf.update(model=model)
    assert exc_info.value.args[0] == "IntegratedAcquisitionFunction requires at least one model to integrate!"


def test_integrated_acquisition_function_compute(iaf, multimodel):
    class CountingMock:
        counter = 0
        long_name = "CountingMock"

        def _compute(self, *args, **kwargs):
            self.counter += 1
            return self.counter

        def update(self, **kwargs):
            pass

    iaf = IntegratedAcquisitionFunction(acquisition_function=CountingMock())
    iaf.update(model=multimodel)
    configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
    rval = iaf(configurations)
    assert rval == 1

    # Test that every counting mock is called only once!
    for counting_mock in iaf._functions:
        assert counting_mock.counter == 1


def test_integrated_acquisition_function_compute_with_different_numbers_of_models(iaf, multimodel):
    for i in range(1, 3):
        multimodel.models = [MockModel()] * i
        iaf.update(model=multimodel, eta=1)
        configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
        rval = iaf(configurations)
        assert rval.shape == (1, 1)

        configurations = [
            ConfigurationMock([1.0, 1.0, 1.0]),
            ConfigurationMock([1.0, 2.0, 3.0]),
        ]
        rval = iaf(configurations)
        assert rval.shape == (2, 1)


# --------------------------------------------------------------
# Test PriorAcquisitionFunction
# --------------------------------------------------------------


@pytest.fixture
def x0_prior():
    return MockPrior(pdf=lambda x: 2 * x, max_density=2)


@pytest.fixture
def hyperparameter_dict(x0_prior):
    return {"x0": x0_prior}


@pytest.fixture
def prior_model(hyperparameter_dict):
    return PriorMockModel(hyperparameter_dict=hyperparameter_dict)


@pytest.fixture
def beta():
    return 2


@pytest.fixture
def prior_floor():
    return 1e-1


def test_prior_init_ei(prior_model, acquisition_function, beta):
    acq_ei = acquisition_function
    paf = PriorAcquisitionFunction(prior_model, acq_ei, beta)
    assert paf._rescale is False


def test_prior_init_ts(prior_model, acq_ts, beta):
    paf = PriorAcquisitionFunction(acquisition_function=acq_ts, decay_beta=beta)
    paf.update(model=prior_model, eta=1, num_data=1)
    assert paf._rescale is True


def test_prior_update(prior_model, acquisition_function, beta):
    paf = PriorAcquisitionFunction(acquisition_function=acquisition_function, decay_beta=beta)
    paf.update(model=prior_model, eta=2)
    assert paf._eta == 2
    assert paf._acquisition_function._eta == 2
    assert paf._iteration_number == 1


def test_prior_compute_prior_Nx1(prior_model, hyperparameter_dict, acquisition_function, beta):
    prior_model.update_prior(hyperparameter_dict)
    paf = PriorAcquisitionFunction(acquisition_function=acquisition_function, decay_beta=beta)
    paf.update(model=prior_model, eta=1)

    X = np.array([0, 0.5, 1]).reshape(3, 1)
    prior_values = paf._compute_prior(X)

    assert prior_values.shape == (3, 1)
    assert prior_values[0][0] == 0
    assert prior_values[1][0] == 1
    assert prior_values[2][0] == 2


def test_prior_compute_prior_NxD(prior_model, hyperparameter_dict, acquisition_function, beta):
    prior_model.update_prior(hyperparameter_dict)
    paf = PriorAcquisitionFunction(acquisition_function=acquisition_function, decay_beta=beta)
    paf.update(model=prior_model, eta=1)

    X = np.array([[0, 0], [0, 1], [1, 1]])
    prior_values = paf._compute_prior(X)

    assert prior_values.shape == (3, 1)
    assert prior_values[0][0] == 0
    assert prior_values[1][0] == 0
    assert prior_values[2][0] == 2


def test_prior_compute_prior_1xD(prior_model, acquisition_function, beta):
    x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
    x1_prior = MockPrior(pdf=lambda x: np.ones_like(x), max_density=1)
    hyperparameter_dict = {"x0": x0_prior, "x1": x1_prior}

    prior_model.update_prior(hyperparameter_dict)
    paf = PriorAcquisitionFunction(acquisition_function=acquisition_function, decay_beta=beta)
    paf.update(model=prior_model, eta=1)

    X = np.array([[0.5, 0.5]])
    prior_values = paf._compute_prior(X)

    assert prior_values.shape == (1, 1)
    assert prior_values[0][0] == 1


def test_prior_compute_prior_1x1(prior_model, hyperparameter_dict, acquisition_function, beta):
    prior_model.update_prior(hyperparameter_dict)
    paf = PriorAcquisitionFunction(acquisition_function=acquisition_function, decay_beta=beta)
    paf.update(model=prior_model, eta=1)

    X = np.array([0.5]).reshape(1, 1)
    prior_values = paf._compute_prior(X)

    assert prior_values.shape == (1, 1)
    assert prior_values[0][0] == 1


@pytest.fixture
def x1_prior():
    return MockPrior(pdf=lambda x: np.ones_like(x), max_density=1)


@pytest.fixture
def x2_prior():
    return MockPrior(pdf=lambda x: 2 - 2 * x, max_density=2)


@pytest.fixture
def hp_dict3(x0_prior, x1_prior, x2_prior):
    return {"x0": x0_prior, "x1": x1_prior, "x2": x2_prior}


def test_prior_1xD(hp_dict3, prior_model, acquisition_function, beta, prior_floor):
    prior_model.update_prior(hp_dict3)
    paf = PriorAcquisitionFunction(acquisition_function=acquisition_function, decay_beta=beta, prior_floor=prior_floor)
    paf.update(model=prior_model, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
    acq = paf(configurations)
    assert acq.shape == (1, 1)

    prior_0_factor = np.power(2.0 * 1.0 * 0.0 + paf._prior_floor, beta / 1.0)

    assert np.isclose(acq[0][0], 0.3989422804014327 * prior_0_factor)


def test_prior_NxD(hp_dict3, prior_model, acquisition_function, beta, prior_floor):
    prior_model.update_prior(hp_dict3)
    paf = PriorAcquisitionFunction(acquisition_function=acquisition_function, decay_beta=beta, prior_floor=prior_floor)
    paf.update(model=prior_model, eta=1.0)

    # These are the exact same numbers as in the EI tests below
    configurations = [
        ConfigurationMock([0.0, 0.0, 0.0]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = paf(configurations)
    assert acq.shape == (3, 1)

    prior_0_factor = np.power(0.0 * 1.0 * 2.0 + paf._prior_floor, beta / 1.0)
    prior_1_factor = np.power(0.2 * 1.0 * 1.8 + paf._prior_floor, beta / 1.0)
    prior_2_factor = np.power(2.0 * 1.0 * 0.0 + paf._prior_floor, beta / 1.0)

    # We do only one update, so we are at iteration 1 (beta/iteration_nbr=2)
    assert np.isclose(acq[0][0], 0.0 * prior_0_factor)
    assert np.isclose(acq[1][0], 0.90020601136712231 * prior_1_factor)
    assert np.isclose(acq[2][0], 0.3989422804014327 * prior_2_factor)


def test_prior_NxD_TS(prior_model, hp_dict3, acq_ts, beta, prior_floor):
    prior_model.update_prior(hp_dict3)
    paf = PriorAcquisitionFunction(acquisition_function=acq_ts, decay_beta=beta, prior_floor=prior_floor)

    eta = 1.0
    paf.update(model=prior_model, eta=eta, num_data=1)

    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = paf(configurations)
    assert acq.shape == (3, 1)

    # retrieved from TS example
    ts_value_0 = -0.00988738
    ts_value_1 = -0.22654082
    ts_value_2 = -2.76405235

    prior_0_factor = np.power(0.0002 * 1 * 1.9998 + paf._prior_floor, beta / 1.0)
    prior_1_factor = np.power(0.2 * 1.0 * 1.8 + paf._prior_floor, beta / 1.0)
    prior_2_factor = np.power(2.0 * 1.0 * 0.0 + paf._prior_floor, beta / 1.0)

    # rescaling to avoid negative values, and keep the TS ranking intact
    combined_value_0 = np.clip(ts_value_0 + eta, 0, np.inf) * prior_0_factor
    combined_value_1 = np.clip(ts_value_1 + eta, 0, np.inf) * prior_1_factor
    combined_value_2 = np.clip(ts_value_2 + eta, 0, np.inf) * prior_2_factor

    assert np.isclose(acq[0][0], combined_value_0)
    assert np.isclose(acq[1][0], combined_value_1)
    assert np.isclose(acq[2][0], combined_value_2)


def test_prior_decay(hp_dict3, prior_model, acquisition_function, beta, prior_floor):
    prior_model.update_prior(hp_dict3)
    paf = PriorAcquisitionFunction(acquisition_function=acquisition_function, decay_beta=beta, prior_floor=prior_floor)
    paf.update(model=prior_model, eta=1.0)
    configurations = [ConfigurationMock([0.1, 0.1, 0.1])]

    for i in range(1, 6):
        prior_factor = np.power(0.2 * 1.0 * 1.8 + paf._prior_floor, beta / i)
        acq = paf(configurations)
        print(acq, 0.90020601136712231 * prior_factor)
        assert np.isclose(acq[0][0], 0.90020601136712231 * prior_factor)
        paf.update(model=prior_model, eta=1.0)  # increase iteration number


def test_prior_discretize_pdf(prior_model, acquisition_function, beta, prior_floor):
    x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
    hyperparameter_dict = {"x0": x0_prior}
    prior_model.update_prior(hyperparameter_dict)
    paf = PriorAcquisitionFunction(
        acquisition_function=acquisition_function, decay_beta=beta, prior_floor=prior_floor, discretize=True
    )
    paf.update(model=prior_model, eta=1)

    number_of_bins_1 = 13
    number_of_bins_2 = 27521
    number_of_points = 1001

    discrete_values_1 = paf._compute_discretized_pdf(
        x0_prior, np.linspace(0, 1, number_of_points), number_of_bins=number_of_bins_1
    )
    discrete_values_2 = paf._compute_discretized_pdf(
        x0_prior, np.linspace(0, 1, number_of_points), number_of_bins=number_of_bins_2
    )
    number_unique_values_1 = len(np.unique(discrete_values_1))
    number_unique_values_2 = len(np.unique(discrete_values_2))

    assert number_unique_values_1 == number_of_bins_1
    assert number_unique_values_2 == number_of_points
    with pytest.raises(ValueError):
        paf._compute_discretized_pdf(x0_prior, np.linspace(0, 1, number_of_points), number_of_bins=-1)


# --------------------------------------------------------------
# Test EI
# --------------------------------------------------------------


def test_ei_1xD(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.3989422804014327)


def test_ei_NxD(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.0, 0.0, 0.0]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = ei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.0)
    assert np.isclose(acq[1][0], 0.90020601136712231)
    assert np.isclose(acq[2][0], 0.3989422804014327)


def test_ei_1x1(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([1.0])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.3989422804014327)


def test_ei_Nx1(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.0001]),
        ConfigurationMock([1.0]),
        ConfigurationMock([2.0]),
    ]
    acq = ei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.9999)
    assert np.isclose(acq[1][0], 0.3989422804014327)
    assert np.isclose(acq[2][0], 0.19964122837424575)


def test_ei_zero_variance(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    X = np.array([ConfigurationMock([0.0])])
    acq = np.array(ei(X))
    assert np.isclose(acq[0][0], 0.0)


# --------------------------------------------------------------
# Test EIPS
# --------------------------------------------------------------


@pytest.fixture
def model_eips():
    return MockModelDual()


@pytest.fixture
def acq_eips(model_eips):
    ei = EIPS()
    ei.model = model_eips
    return ei


def test_eips_1xD(model_eips, acq_eips):
    ei = acq_eips
    ei.update(model=model_eips, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0]), ConfigurationMock([1.0, 1.0])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.3989422804014327)


def test_eips_fail(model_eips, acq_eips):
    ei = acq_eips
    with pytest.raises(ValueError):
        configurations = [ConfigurationMock([1.0, 1.0])]
        ei(configurations)


@pytest.fixture
def acq_logei(model):
    ei = EI(log=True)
    ei.model = model
    return ei


def test_logei_1xD(model, acq_logei):
    ei = acq_logei
    ei.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.6480973967332011)


def test_logei_NxD(model, acq_logei):
    ei = acq_logei
    ei.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.1, 0.0, 0.0]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = ei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 1.6670107375002425)
    assert np.isclose(acq[1][0], 1.5570607606556273)
    assert np.isclose(acq[2][0], 0.6480973967332011)


# --------------------------------------------------------------
# Test PI
# --------------------------------------------------------------


@pytest.fixture
def acq_pi(model):
    pi = PI()
    pi.model = model
    return pi


def test_pi_1xD(model, acq_pi):
    ei = acq_pi
    ei.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.7602499389065233)


def test_pi_1xD_zero(model, acq_pi):
    ei = acq_pi
    ei.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([100, 100, 100])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0)


def test_pi_NxD(model, acq_pi):
    ei = acq_pi
    ei.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = ei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 1.0)
    assert np.isclose(acq[1][0], 0.99778673707104)
    assert np.isclose(acq[2][0], 0.5)


# --------------------------------------------------------------
# Test LCB
# --------------------------------------------------------------


@pytest.fixture
def acq_lcb(model):
    lcb = LCB()
    lcb.model = model
    return lcb


def test_lcb_1xD(model, acq_lcb):
    ei = acq_lcb
    ei.update(model=model, eta=1.0, par=1, num_data=3)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 1.315443985917585)
    ei.update(model=model, eta=1.0, par=1, num_data=100)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 2.7107557771721433)


def test_lcb_1xD_no_improvement_vs_improvement(model, acq_lcb):
    ei = acq_lcb
    ei.update(model=model, par=1, num_data=1)
    configurations = [ConfigurationMock([100, 100])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], -88.22589977)
    configurations = [ConfigurationMock([0.001, 0.001])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.03623297)


def test_lcb_NxD(model, acq_lcb):
    ei = acq_lcb
    ei.update(model=model, eta=1.0, num_data=100)
    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = ei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.045306943655446116)
    assert np.isclose(acq[1][0], 1.3358936353814157)
    assert np.isclose(acq[2][0], 3.5406943655446117)


# --------------------------------------------------------------
# Test TS
# --------------------------------------------------------------


@pytest.fixture
def acq_ts(model):
    ts = TS()
    ts.model = model
    return ts


def test_ts_1xD(model, acq_ts):
    ei = acq_ts
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], -1.74737338)


def test_ts_NxD(model, acq_ts):
    ei = acq_ts
    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = ei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], -0.00988738)
    assert np.isclose(acq[1][0], -0.22654082)
    assert np.isclose(acq[2][0], -2.76405235)


def test_ts_rng():
    """Test TS acqusition function with model that only has attribute 'rng'"""
    model = MockModelRNG()
    ts = TS()
    ts.model = model


def test_ts_sampler():
    "Test TS acqusition function with model that only has attribute 'sample_functions'"
    model = MockModelSampler()
    ts = TS()
    ts.model = model
