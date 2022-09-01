import unittest
import unittest.mock

import pytest

import numpy as np

from smac.acquisition.functions import (
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


class ConfigurationMock(object):
    def __init__(self, values=None):
        self.values = values
        self.configuration_space = unittest.mock.MagicMock()
        self.configuration_space.get_hyperparameters.return_value = []

    def get_array(self):
        return self.values


class MockModel(object):
    def __init__(self, num_targets=1, seed=0):
        self.num_targets = num_targets
        self.seed = seed

    def predict_marginalized_over_instances(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 1)), np.array(
            [np.mean(X, axis=1).reshape((1, -1))] * self.num_targets
        ).reshape((-1, 1))


class MockModelDual(object):
    def __init__(self, num_targets=1):
        self.num_targets = num_targets

    def predict_marginalized_over_instances(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 2)), np.array(
            [np.mean(X, axis=1).reshape((1, -1))] * self.num_targets
        ).reshape((-1, 2))


class MockPrior(object):
    def __init__(self, pdf, max_density):
        self.pdf = pdf
        self.max_density = max_density

    def _pdf(self, X):
        return self.pdf(X)

    def get_max_density(self):
        return self.max_density


class PriorMockModel(object):
    def __init__(self, hyperparameter_dict=None, num_targets=1, seed=0):
        self.num_targets = num_targets
        self.seed = seed
        self.configuration_space = unittest.mock.MagicMock()
        self.hyperparameter_dict = hyperparameter_dict
        # since the PriorAcquisitionFunction needs to return the hyperparameters in dict
        # form through two function calls (self.model.get_configspace().get_hyperparameters_dict()),
        # we need a slightly intricate solution
        self.configuration_space.get_hyperparameters_dict.return_value = self.hyperparameter_dict

    def get_configspace(self):
        return self.configuration_space

    def predict_marginalized_over_instances(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 1)), np.array(
            [np.mean(X, axis=1).reshape((1, -1))] * self.num_targets
        ).reshape((-1, 1))

    def update_prior(self, hyperparameter_dict):
        self.configuration_space.get_hyperparameters_dict.return_value = hyperparameter_dict


class MockModelRNG(MockModel):
    def __init__(self, num_targets=1, seed=0):
        self.num_targets = num_targets
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)


class MockModelSampler(MockModelRNG):
    def __init__(self, num_targets=1, seed=0):
        self.num_targets = num_targets
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def sample_functions(self, X, n_funcs=1):
        m = np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1,))
        var = np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1,))
        var = np.diag(var)
        return self.rng.multivariate_normal(m, var, n_funcs).T


@pytest.fixture
def model():
    return MockModel()


@pytest.fixture
def acquisition_function(model):
    ei = EI()
    ei._set_model(model=model)
    return ei


def test_update_model_and_eta(model, acquisition_function):
    model = "abc"
    assert acquisition_function.eta is None
    acquisition_function.update(model=model, eta=0.1)
    assert acquisition_function.model == model
    assert acquisition_function.eta == 0.1


def test_update_with_kwargs(acquisition_function):
    acquisition_function.update(model="abc", eta=0., other="hi there:)")
    assert acquisition_function.model == "abc"


def test_update_without_required(acquisition_function):
    with pytest.raises(
            TypeError,
        ):
            acquisition_function.update(other=None)

# class TestAcquisitionFunction(unittest.TestCase):
#     def setUp(self):
#         self.model = unittest.mock.Mock()
#         self.acq = EI(model=self.model)

    # def test_update_model_and_eta(self):
    #     model = "abc"
    #     self.assertIsNone(self.acq.eta)
    #     self.acq.update(model=model, eta=0.1)
    #     self.assertEqual(self.acq.model, model)
    #     self.assertEqual(self.acq.eta, 0.1)

    # def test_update_other(self):
    #     self.acq.other = "other"

    #     with self.assertRaisesRegex(
    #         ValueError,
    #         r"Acquisition function EI needs to be updated with key model, but only got keys " r"\['other'\].",
    #     ):
    #         self.acq.update(other=None)

    #     model = "abc"
    #     self.acq.update(model=model, eta=0.1, other=None)
    #     self.assertEqual(self.acq.other, "other")


@pytest.fixture
def multimodel():
    model = MockModel()
    model.models = [MockModel()] * 3
    return model


@pytest.fixture
def acq_multi(multimodel, acquisition_function):
    acquisition_function._set_model(multimodel)
    return acquisition_function


@pytest.fixture
def iaf(multimodel, acq_multi):
    iaf = IntegratedAcquisitionFunction(acquisition_function=acq_multi)
    iaf._set_model(model=multimodel)
    return iaf


def test_integrated_acquisition_function_update(iaf, model, multimodel):
    iaf.update(model=multimodel, eta=2)
    for func in iaf._functions:
        assert func.eta == 2

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


# class TestIntegratedAcquisitionFunction(unittest.TestCase):
#     def setUp(self):
#         self.model = unittest.mock.Mock()
#         self.model.models = [MockModel(), MockModel(), MockModel()]
#         self.ei = EI(self.model)

    # def test_update(self):
    #     iaf = IntegratedAcquisitionFunction(model=self.model, acquisition_function=self.ei)
    #     iaf.update(model=self.model, eta=2)
    #     for func in iaf._functions:
    #         self.assertEqual(func.eta, 2)
    #
    #     with self.assertRaisesRegex(
    #         ValueError,
    #         "IntegratedAcquisitionFunction requires at least one model to integrate!",
    #     ):
    #         iaf.update(model=MockModel())
    #
    #     with self.assertRaisesRegex(
    #         ValueError,
    #         "IntegratedAcquisitionFunction requires at least one model to integrate!",
    #     ):
    #         self.model.models = []
    #         iaf.update(model=self.model)

    # def test_compute(self):
    #     class CountingMock:
    #         counter = 0
    #         long_name = "CountingMock"
    #
    #         def _compute(self, *args, **kwargs):
    #             self.counter += 1
    #             return self.counter
    #
    #         def update(self, **kwargs):
    #             pass
    #
    #     iaf = IntegratedAcquisitionFunction(model=self.model, acquisition_function=CountingMock())
    #     iaf.update(model=self.model)
    #     configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
    #     rval = iaf(configurations)
    #     self.assertEqual(rval, 1)
    #
    #     # Test that every counting mock is called only once!
    #     for counting_mock in iaf._functions:
    #         self.assertEqual(counting_mock.counter, 1)

    # def test_compute_with_different_numbers_of_models(self):
    #
    #     for i in range(1, 3):
    #         self.model.models = [MockModel()] * i
    #         iaf = IntegratedAcquisitionFunction(model=self.model, acquisition_function=self.ei)
    #         iaf.update(model=self.model, eta=1)
    #         configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
    #         rval = iaf(configurations)
    #         self.assertEqual(rval.shape, (1, 1))
    #
    #         configurations = [
    #             ConfigurationMock([1.0, 1.0, 1.0]),
    #             ConfigurationMock([1.0, 2.0, 3.0]),
    #         ]
    #         rval = iaf(configurations)
    #         self.assertEqual(rval.shape, (2, 1))


# class TestPriorAcquisitionFunction(unittest.TestCase):
#     def setUp(self):
#         x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
#         hyperparameter_dict = {"x0": x0_prior}
#         self.model = PriorMockModel(hyperparameter_dict=hyperparameter_dict)
#         self.ei = EI(self.model)
#         self.ts = TS(self.model)
#         self.beta = 2
#         self.prior_floor = 1e-1
#
#     def test_init_ei(self):
#         paf = PriorAcquisitionFunction(self.model, self.ei, self.beta)
#         self.assertFalse(paf.rescale_acq)
#
#     def test_init_ts(self):
#         paf = PriorAcquisitionFunction(self.model, self.ts, self.beta)
#         self.assertTrue(paf.rescale_acq)
#
#     def test_update(self):
#         paf = PriorAcquisitionFunction(self.model, self.ei, self.beta)
#         paf.update(model=self.model, eta=2)
#         self.assertEqual(paf.eta, 2)
#         self.assertEqual(paf.acq.eta, 2)
#         self.assertEqual(paf.iteration_number, 1)
#
#     def test_compute_prior_Nx1(self):
#         x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
#         hyperparameter_dict = {"x0": x0_prior}
#         self.model.update_prior(hyperparameter_dict)
#         paf = PriorAcquisitionFunction(self.model, self.ei, self.beta)
#
#         X = np.array([0, 0.5, 1]).reshape(3, 1)
#         prior_values = paf._compute_prior(X)
#
#         self.assertEqual(prior_values.shape, (3, 1))
#         self.assertEqual(prior_values[0][0], 0)
#         self.assertEqual(prior_values[1][0], 1)
#         self.assertEqual(prior_values[2][0], 2)
#
#     def test_compute_prior_NxD(self):
#         x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
#         x1_prior = MockPrior(pdf=lambda x: np.ones_like(x), max_density=1)
#         hyperparameter_dict = {"x0": x0_prior, "x1": x1_prior}
#         self.model.update_prior(hyperparameter_dict)
#         paf = PriorAcquisitionFunction(self.model, self.ei, self.beta)
#
#         X = np.array([[0, 0], [0, 1], [1, 1]])
#         prior_values = paf._compute_prior(X)
#
#         self.assertEqual(prior_values.shape, (3, 1))
#         self.assertEqual(prior_values[0][0], 0)
#         self.assertEqual(prior_values[1][0], 0)
#         self.assertEqual(prior_values[2][0], 2)
#
#     def test_compute_prior_1xD(self):
#         x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
#         x1_prior = MockPrior(pdf=lambda x: np.ones_like(x), max_density=1)
#         hyperparameter_dict = {"x0": x0_prior, "x1": x1_prior}
#         self.model.update_prior(hyperparameter_dict)
#         paf = PriorAcquisitionFunction(self.model, self.ei, self.beta)
#
#         X = np.array([[0.5, 0.5]])
#         prior_values = paf._compute_prior(X)
#
#         self.assertEqual(prior_values.shape, (1, 1))
#         self.assertEqual(prior_values[0][0], 1)
#
#     def test_compute_prior_1x1(self):
#         x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
#         hyperparameter_dict = {"x0": x0_prior}
#         self.model.update_prior(hyperparameter_dict)
#         paf = PriorAcquisitionFunction(self.model, self.ei, self.beta)
#
#         X = np.array([0.5]).reshape(1, 1)
#         prior_values = paf._compute_prior(X)
#
#         self.assertEqual(prior_values.shape, (1, 1))
#         self.assertEqual(prior_values[0][0], 1)
#
#     def test_1xD(self):
#         x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
#         x1_prior = MockPrior(pdf=lambda x: np.ones_like(x), max_density=1)
#         x2_prior = MockPrior(pdf=lambda x: 2 - 2 * x, max_density=2)
#         hyperparameter_dict = {"x0": x0_prior, "x1": x1_prior, "x2": x2_prior}
#         self.model.update_prior(hyperparameter_dict)
#         paf = PriorAcquisitionFunction(self.model, self.ei, self.beta, self.prior_floor)
#
#         paf.update(model=self.model, eta=1.0)
#         configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
#         acq = paf(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#
#         prior_0_factor = np.power(2.0 * 1.0 * 0.0 + paf.prior_floor, self.beta / 1.0)
#
#         self.assertAlmostEqual(acq[0][0], 0.3989422804014327 * prior_0_factor)
#
#     def test_NxD(self):
#         x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
#         x1_prior = MockPrior(pdf=lambda x: np.ones_like(x), max_density=1)
#         x2_prior = MockPrior(pdf=lambda x: 2 - 2 * x, max_density=2)
#         hyperparameter_dict = {"x0": x0_prior, "x1": x1_prior, "x2": x2_prior}
#         self.model.update_prior(hyperparameter_dict)
#         paf = PriorAcquisitionFunction(self.model, self.ei, self.beta, self.prior_floor)
#
#         paf.update(model=self.model, eta=1.0)
#         # These are the exact same numbers as in the EI tests below
#         configurations = [
#             ConfigurationMock([0.0, 0.0, 0.0]),
#             ConfigurationMock([0.1, 0.1, 0.1]),
#             ConfigurationMock([1.0, 1.0, 1.0]),
#         ]
#         acq = paf(configurations)
#         self.assertEqual(acq.shape, (3, 1))
#
#         prior_0_factor = np.power(0.0 * 1.0 * 2.0 + paf.prior_floor, self.beta / 1.0)
#         prior_1_factor = np.power(0.2 * 1.0 * 1.8 + paf.prior_floor, self.beta / 1.0)
#         prior_2_factor = np.power(2.0 * 1.0 * 0.0 + paf.prior_floor, self.beta / 1.0)
#
#         # We do only one update, so we are at iteration 1 (beta/iteration_nbr=2)
#         self.assertAlmostEqual(acq[0][0], 0.0 * prior_0_factor)
#         self.assertAlmostEqual(acq[1][0], 0.90020601136712231 * prior_1_factor)
#         self.assertAlmostEqual(acq[2][0], 0.3989422804014327 * prior_2_factor)
#
#     def test_NxD_TS(self):
#         # since there is some rescaling that needs to be done for TS and UCB (the same
#         # scaling for the two of them), it justifies testing as well
#         x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
#         x1_prior = MockPrior(pdf=lambda x: np.ones_like(x), max_density=1)
#         x2_prior = MockPrior(pdf=lambda x: 2 - 2 * x, max_density=2)
#         hyperparameter_dict = {"x0": x0_prior, "x1": x1_prior, "x2": x2_prior}
#         self.model.update_prior(hyperparameter_dict)
#         paf = PriorAcquisitionFunction(self.model, self.ts, self.beta, self.prior_floor)
#         eta = 1.0
#
#         paf.update(model=self.model, eta=eta)
#
#         configurations = [
#             ConfigurationMock([0.0001, 0.0001, 0.0001]),
#             ConfigurationMock([0.1, 0.1, 0.1]),
#             ConfigurationMock([1.0, 1.0, 1.0]),
#         ]
#         acq = paf(configurations)
#         self.assertEqual(acq.shape, (3, 1))
#
#         # retrieved from TS example
#         ts_value_0 = -0.00988738
#         ts_value_1 = -0.22654082
#         ts_value_2 = -2.76405235
#
#         prior_0_factor = np.power(0.0002 * 1 * 1.9998 + paf.prior_floor, self.beta / 1.0)
#         prior_1_factor = np.power(0.2 * 1.0 * 1.8 + paf.prior_floor, self.beta / 1.0)
#         prior_2_factor = np.power(2.0 * 1.0 * 0.0 + paf.prior_floor, self.beta / 1.0)
#
#         # rescaling to avoid negative values, and keep the TS ranking intact
#         combined_value_0 = np.clip(ts_value_0 + eta, 0, np.inf) * prior_0_factor
#         combined_value_1 = np.clip(ts_value_1 + eta, 0, np.inf) * prior_1_factor
#         combined_value_2 = np.clip(ts_value_2 + eta, 0, np.inf) * prior_2_factor
#
#         self.assertAlmostEqual(acq[0][0], combined_value_0)
#         self.assertAlmostEqual(acq[1][0], combined_value_1)
#         self.assertAlmostEqual(acq[2][0], combined_value_2)
#
#     def test_decay(self):
#         x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
#         x1_prior = MockPrior(pdf=lambda x: np.ones_like(x), max_density=1)
#         x2_prior = MockPrior(pdf=lambda x: 2 - 2 * x, max_density=2)
#         hyperparameter_dict = {"x0": x0_prior, "x1": x1_prior, "x2": x2_prior}
#         self.model.update_prior(hyperparameter_dict)
#         paf = PriorAcquisitionFunction(self.model, self.ei, self.beta, self.prior_floor)
#         configurations = [ConfigurationMock([0.1, 0.1, 0.1])]
#
#         for i in range(1, 6):
#             paf.update(model=self.model, eta=1.0)
#             prior_factor = np.power(0.2 * 1.0 * 1.8 + paf.prior_floor, self.beta / i)
#             acq = paf(configurations)
#             self.assertAlmostEqual(acq[0][0], 0.90020601136712231 * prior_factor)
#
#     def test_discretize_pdf(self):
#         x0_prior = MockPrior(pdf=lambda x: 2 * x, max_density=2)
#         hyperparameter_dict = {"x0": x0_prior}
#         self.model.update_prior(hyperparameter_dict)
#         paf = PriorAcquisitionFunction(self.model, self.ei, self.beta, self.prior_floor, discretize=True)
#
#         number_of_bins_1 = 13
#         number_of_bins_2 = 27521
#         number_of_points = 1001
#
#         discrete_values_1 = paf._compute_discretized_pdf(
#             x0_prior, np.linspace(0, 1, number_of_points), number_of_bins=number_of_bins_1
#         )
#         discrete_values_2 = paf._compute_discretized_pdf(
#             x0_prior, np.linspace(0, 1, number_of_points), number_of_bins=number_of_bins_2
#         )
#         number_unique_values_1 = len(np.unique(discrete_values_1))
#         number_unique_values_2 = len(np.unique(discrete_values_2))
#
#         self.assertEqual(number_unique_values_1, number_of_bins_1)
#         self.assertEqual(number_unique_values_2, number_of_points)
#         with self.assertRaises(ValueError):
#             paf._compute_discretized_pdf(x0_prior, np.linspace(0, 1, number_of_points), number_of_bins=-1)
#
#

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


# class TestEI(unittest.TestCase):
#     def setUp(self):
#         self.model = MockModel()
#         self.ei = EI(self.model)
#
#     def test_1xD(self):
#         self.ei.update(model=self.model, eta=1.0)
#         configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], 0.3989422804014327)
#
#     def test_NxD(self):
#         self.ei.update(model=self.model, eta=1.0)
#         configurations = [
#             ConfigurationMock([0.0, 0.0, 0.0]),
#             ConfigurationMock([0.1, 0.1, 0.1]),
#             ConfigurationMock([1.0, 1.0, 1.0]),
#         ]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (3, 1))
#         self.assertAlmostEqual(acq[0][0], 0.0)
#         self.assertAlmostEqual(acq[1][0], 0.90020601136712231)
#         self.assertAlmostEqual(acq[2][0], 0.3989422804014327)
#
#     def test_1x1(self):
#         self.ei.update(model=self.model, eta=1.0)
#         configurations = [ConfigurationMock([1.0])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], 0.3989422804014327)
#
#     def test_Nx1(self):
#         self.ei.update(model=self.model, eta=1.0)
#         configurations = [
#             ConfigurationMock([0.0001]),
#             ConfigurationMock([1.0]),
#             ConfigurationMock([2.0]),
#         ]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (3, 1))
#         self.assertAlmostEqual(acq[0][0], 0.9999)
#         self.assertAlmostEqual(acq[1][0], 0.3989422804014327)
#         self.assertAlmostEqual(acq[2][0], 0.19964122837424575)
#
#     def test_zero_variance(self):
#         self.ei.update(model=self.model, eta=1.0)
#         X = np.array([ConfigurationMock([0.0])])
#         acq = np.array(self.ei(X))
#         self.assertAlmostEqual(acq[0][0], 0.0)
#
#
# class TestEIPS(unittest.TestCase):
#     def setUp(self):
#         self.model = MockModelDual()
#         self.ei = EIPS(self.model)
#
#     def test_1xD(self):
#         self.ei.update(model=self.model, eta=1.0)
#         configurations = [ConfigurationMock([1.0, 1.0]), ConfigurationMock([1.0, 1.0])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], 0.3989422804014327)
#
#     def test_fail(self):
#         with self.assertRaises(ValueError):
#             configurations = [ConfigurationMock([1.0, 1.0])]
#             self.ei(configurations)
#
#
# class TestLogEI(unittest.TestCase):
#     def setUp(self):
#         self.model = MockModel()
#         self.ei = LogEI(self.model)
#
#     def test_1xD(self):
#         self.ei.update(model=self.model, eta=1.0)
#         configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], 0.6480973967332011)
#
#     def test_NxD(self):
#         self.ei.update(model=self.model, eta=1.0)
#         configurations = [
#             ConfigurationMock([0.1, 0.0, 0.0]),
#             ConfigurationMock([0.1, 0.1, 0.1]),
#             ConfigurationMock([1.0, 1.0, 1.0]),
#         ]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (3, 1))
#         self.assertAlmostEqual(acq[0][0], 1.6670107375002425)
#         self.assertAlmostEqual(acq[1][0], 1.5570607606556273)
#         self.assertAlmostEqual(acq[2][0], 0.6480973967332011)
#
#
# class TestPI(unittest.TestCase):
#     def setUp(self):
#         self.model = MockModel()
#         self.ei = PI(self.model)
#
#     def test_1xD(self):
#         self.ei.update(model=self.model, eta=1.0)
#         configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], 0.7602499389065233)
#
#     def test_1xD_zero(self):
#         self.ei.update(model=self.model, eta=1.0)
#         configurations = [ConfigurationMock([100, 100, 100])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], 0)
#
#     def test_NxD(self):
#         self.ei.update(model=self.model, eta=1.0)
#         configurations = [
#             ConfigurationMock([0.0001, 0.0001, 0.0001]),
#             ConfigurationMock([0.1, 0.1, 0.1]),
#             ConfigurationMock([1.0, 1.0, 1.0]),
#         ]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (3, 1))
#         self.assertAlmostEqual(acq[0][0], 1.0)
#         self.assertAlmostEqual(acq[1][0], 0.99778673707104)
#         self.assertAlmostEqual(acq[2][0], 0.5)
#
#
# class TestLCB(unittest.TestCase):
#     def setUp(self):
#         self.model = MockModel()
#         self.ei = LCB(self.model)
#
#     def test_1xD(self):
#         self.ei.update(model=self.model, eta=1.0, par=1, num_data=3)
#         configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], 1.315443985917585)
#         self.ei.update(model=self.model, eta=1.0, par=1, num_data=100)
#         configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], 2.7107557771721433)
#
#     def test_1xD_no_improvement_vs_improvement(self):
#         self.ei.update(model=self.model, par=1, num_data=1)
#         configurations = [ConfigurationMock([100, 100])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], -88.22589977)
#         configurations = [ConfigurationMock([0.001, 0.001])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], 0.03623297)
#
#     def test_NxD(self):
#         self.ei.update(model=self.model, eta=1.0, num_data=100)
#         configurations = [
#             ConfigurationMock([0.0001, 0.0001, 0.0001]),
#             ConfigurationMock([0.1, 0.1, 0.1]),
#             ConfigurationMock([1.0, 1.0, 1.0]),
#         ]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (3, 1))
#         self.assertAlmostEqual(acq[0][0], 0.045306943655446116)
#         self.assertAlmostEqual(acq[1][0], 1.3358936353814157)
#         self.assertAlmostEqual(acq[2][0], 3.5406943655446117)
#
#
# class TestTS(unittest.TestCase):
#     def setUp(self):
#         # Test TS acqusition function with model that only has attribute 'seed'
#         self.model = MockModel()
#         self.ei = TS(self.model)
#
#     def test_1xD(self):
#         self.ei.update(model=self.model)
#         configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (1, 1))
#         self.assertAlmostEqual(acq[0][0], -1.74737338)
#
#     def test_NxD(self):
#         self.ei.update(model=self.model)
#         configurations = [
#             ConfigurationMock([0.0001, 0.0001, 0.0001]),
#             ConfigurationMock([0.1, 0.1, 0.1]),
#             ConfigurationMock([1.0, 1.0, 1.0]),
#         ]
#         acq = self.ei(configurations)
#         self.assertEqual(acq.shape, (3, 1))
#         self.assertAlmostEqual(acq[0][0], -0.00988738)
#         self.assertAlmostEqual(acq[1][0], -0.22654082)
#         self.assertAlmostEqual(acq[2][0], -2.76405235)
#
#
# class TestTSRNG(TestTS):
#     def setUp(self):
#         # Test TS acqusition function with model that only has attribute 'rng'
#         self.model = MockModelRNG()
#         self.ei = TS(self.model)
#
#
# class TestTSSampler(TestTS):
#     def setUp(self):
#         # Test TS acqusition function with model that only has attribute 'sample_functions'
#         self.model = MockModelSampler()
#         self.ei = TS(self.model)
