import unittest
import unittest.mock

import numpy as np

from smac.optimizer.acquisition import (
    EI,
    LogEI,
    EIPS,
    PI,
    LCB,
    TS,
    IntegratedAcquisitionFunction,
)

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
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
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 1)),\
            np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 1))


class MockModelDual(object):
    def __init__(self, num_targets=1):
        self.num_targets = num_targets

    def predict_marginalized_over_instances(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 2)), \
            np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 2))


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
        m = np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, ))
        var = np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, ))
        var = np.diag(var)
        return self.rng.multivariate_normal(m, var, n_funcs).T


class TestAcquisitionFunction(unittest.TestCase):
    def setUp(self):
        self.model = unittest.mock.Mock()
        self.acq = EI(model=self.model)

    def test_update_model_and_eta(self):
        model = 'abc'
        self.assertIsNone(self.acq.eta)
        self.acq.update(model=model, eta=0.1)
        self.assertEqual(self.acq.model, model)
        self.assertEqual(self.acq.eta, 0.1)

    def test_update_other(self):
        self.acq.other = 'other'

        with self.assertRaisesRegex(
            ValueError,
            r"Acquisition function EI needs to be updated with key model, but only got keys "
            r"\['other'\].",
        ):
            self.acq.update(other=None)

        model = 'abc'
        self.acq.update(model=model, eta=0.1, other=None)
        self.assertEqual(self.acq.other, 'other')


class TestIntegratedAcquisitionFunction(unittest.TestCase):
    def setUp(self):
        self.model = unittest.mock.Mock()
        self.model.models = [MockModel(), MockModel(), MockModel()]
        self.ei = EI(self.model)

    def test_update(self):
        iaf = IntegratedAcquisitionFunction(model=self.model, acquisition_function=self.ei)
        iaf.update(model=self.model, eta=2)
        for func in iaf._functions:
            self.assertEqual(func.eta, 2)

        with self.assertRaisesRegex(
            ValueError,
            'IntegratedAcquisitionFunction requires at least one model to integrate!',
        ):
            iaf.update(model=MockModel())

        with self.assertRaisesRegex(
            ValueError,
            'IntegratedAcquisitionFunction requires at least one model to integrate!',
        ):
            self.model.models = []
            iaf.update(model=self.model)

    def test_compute(self):
        class CountingMock:
            counter = 0
            long_name = 'CountingMock'

            def _compute(self, *args, **kwargs):
                self.counter += 1
                return self.counter

            def update(self, **kwargs):
                pass

        iaf = IntegratedAcquisitionFunction(model=self.model, acquisition_function=CountingMock())
        iaf.update(model=self.model)
        configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
        rval = iaf(configurations)
        self.assertEqual(rval, 1)

        # Test that every counting mock is called only once!
        for counting_mock in iaf._functions:
            self.assertEqual(counting_mock.counter, 1)

    def test_compute_with_different_numbers_of_models(self):

        for i in range(1, 3):
            self.model.models = [MockModel()] * i
            iaf = IntegratedAcquisitionFunction(model=self.model, acquisition_function=self.ei)
            iaf.update(model=self.model, eta=1)
            configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
            rval = iaf(configurations)
            self.assertEqual(rval.shape, (1, 1))

            configurations = [ConfigurationMock([1.0, 1.0, 1.0]), ConfigurationMock([1.0, 2.0, 3.0])]
            rval = iaf(configurations)
            self.assertEqual(rval.shape, (2, 1))


class TestEI(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.ei = EI(self.model)

    def test_1xD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.3989422804014327)

    def test_NxD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = ([ConfigurationMock([0.0, 0.0, 0.0]),
                           ConfigurationMock([0.1, 0.1, 0.1]),
                           ConfigurationMock([1.0, 1.0, 1.0])])
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], 0.0)
        self.assertAlmostEqual(acq[1][0], 0.90020601136712231)
        self.assertAlmostEqual(acq[2][0], 0.3989422804014327)

    def test_1x1(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([1.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.3989422804014327)

    def test_Nx1(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([0.0001]),
                          ConfigurationMock([1.0]),
                          ConfigurationMock([2.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], 0.9999)
        self.assertAlmostEqual(acq[1][0], 0.3989422804014327)
        self.assertAlmostEqual(acq[2][0], 0.19964122837424575)

    def test_zero_variance(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([ConfigurationMock([0.0])])
        acq = np.array(self.ei(X))
        self.assertAlmostEqual(acq[0][0], 0.0)


class TestEIPS(unittest.TestCase):
    def setUp(self):
        self.model = MockModelDual()
        self.ei = EIPS(self.model)

    def test_1xD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([1.0, 1.0]), ConfigurationMock([1.0, 1.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.3989422804014327)

    def test_fail(self):
        with self.assertRaises(ValueError):
            configurations = [ConfigurationMock([1.0, 1.0])]
            self.ei(configurations)


class TestLogEI(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.ei = LogEI(self.model)

    def test_1xD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.6480973967332011)

    def test_NxD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([0.1, 0.0, 0.0]),
                          ConfigurationMock([0.1, 0.1, 0.1]),
                          ConfigurationMock([1.0, 1.0, 1.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], 1.6670107375002425)
        self.assertAlmostEqual(acq[1][0], 1.5570607606556273)
        self.assertAlmostEqual(acq[2][0], 0.6480973967332011)


class TestPI(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.ei = PI(self.model)

    def test_1xD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([.5, .5, .5])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.7602499389065233)

    def test_1xD_zero(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([100, 100, 100])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0)

    def test_NxD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = ([ConfigurationMock([0.0001, 0.0001, 0.0001]),
                           ConfigurationMock([0.1, 0.1, 0.1]),
                           ConfigurationMock([1.0, 1.0, 1.0])])
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], 1.0)
        self.assertAlmostEqual(acq[1][0], 0.99778673707104)
        self.assertAlmostEqual(acq[2][0], 0.5)


class TestLCB(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.ei = LCB(self.model)

    def test_1xD(self):
        self.ei.update(model=self.model, eta=1.0, par=1, num_data=3)
        configurations = [ConfigurationMock([.5, .5, .5])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 1.315443985917585)
        self.ei.update(model=self.model, eta=1.0, par=1, num_data=100)
        configurations = [ConfigurationMock([.5, .5, .5])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 2.7107557771721433)

    def test_1xD_no_improvement_vs_improvement(self):
        self.ei.update(model=self.model, par=1, num_data=1)
        configurations = [ConfigurationMock([100, 100])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], -88.22589977)
        configurations = [ConfigurationMock([0.001, 0.001])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.03623297)

    def test_NxD(self):
        self.ei.update(model=self.model, eta=1.0, num_data=100)
        configurations = ([ConfigurationMock([0.0001, 0.0001, 0.0001]),
                           ConfigurationMock([0.1, 0.1, 0.1]),
                           ConfigurationMock([1.0, 1.0, 1.0])])
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], 0.045306943655446116)
        self.assertAlmostEqual(acq[1][0], 1.3358936353814157)
        self.assertAlmostEqual(acq[2][0], 3.5406943655446117)


class TestTS(unittest.TestCase):
    def setUp(self):
        # Test TS acqusition function with model that only has attribute 'seed'
        self.model = MockModel()
        self.ei = TS(self.model)

    def test_1xD(self):
        self.ei.update(model=self.model)
        configurations = [ConfigurationMock([.5, .5, .5])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], -1.74737338)

    def test_NxD(self):
        self.ei.update(model=self.model)
        configurations = ([ConfigurationMock([0.0001, 0.0001, 0.0001]),
                           ConfigurationMock([0.1, 0.1, 0.1]),
                           ConfigurationMock([1.0, 1.0, 1.0])])
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], -0.00988738)
        self.assertAlmostEqual(acq[1][0], -0.22654082)
        self.assertAlmostEqual(acq[2][0], -2.76405235)


class TestTSRNG(TestTS):
    def setUp(self):
        # Test TS acqusition function with model that only has attribute 'rng'
        self.model = MockModelRNG()
        self.ei = TS(self.model)


class TestTSSampler(TestTS):
    def setUp(self):
        # Test TS acqusition function with model that only has attribute 'sample_functions'
        self.model = MockModelSampler()
        self.ei = TS(self.model)
