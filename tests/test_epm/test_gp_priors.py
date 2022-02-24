import unittest

import numpy as np
import scipy.optimize

from smac.epm.gp_base_prior import TophatPrior, HorseshoePrior, LognormalPrior, GammaPrior, SoftTopHatPrior
from smac.utils.constants import VERY_SMALL_NUMBER

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def wrap_ln(theta, prior):
    return np.exp(prior.lnprob(np.log(theta)))


class TestTophatPrior(unittest.TestCase):

    def test_lnprob_and_grad_scalar(self):
        prior = TophatPrior(
            lower_bound=np.exp(-10),
            upper_bound=np.exp(2),
            rng=np.random.RandomState(1),
        )

        # Legal scalar
        for val in (-1, 0, 1):
            self.assertEqual(prior.lnprob(val), 0, msg=str(val))
            self.assertEqual(prior.gradient(val), 0, msg=str(val))

        # Boundary
        for val in (-10, 2):
            self.assertEqual(prior.lnprob(val), 0)
            self.assertEqual(prior.gradient(val), 0)

        # Values outside the boundary
        for val in (-10 - VERY_SMALL_NUMBER, 2 + VERY_SMALL_NUMBER, -50, 50):
            self.assertTrue(np.isinf(prior.lnprob(val)))
            self.assertEqual(prior.gradient(val), 0)

    def test_sample_from_prior(self):
        prior = TophatPrior(lower_bound=np.exp(-10), upper_bound=np.exp(2), rng=np.random.RandomState(1))
        samples = prior.sample_from_prior(10)
        np.testing.assert_array_equal(samples >= -10, True)
        np.testing.assert_array_equal(samples <= 2, True)
        # Test that the rng is set
        self.assertAlmostEqual(samples[0], -4.995735943569112)

    def test_sample_from_prior_shapes(self):
        rng = np.random.RandomState(1)
        lower_bound = 2 + rng.random_sample() * 50
        upper_bound = lower_bound + rng.random_sample() * 50
        prior = TophatPrior(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            rng=np.random.RandomState(1),
        )
        sample = prior.sample_from_prior(1)
        self.assertEqual(sample.shape, (1,))
        sample = prior.sample_from_prior(2)
        self.assertEqual(sample.shape, (2,))
        sample = prior.sample_from_prior(10)
        self.assertEqual(sample.shape, (10,))
        with self.assertRaises(ValueError):
            prior.sample_from_prior(0)
        with self.assertRaises(ValueError):
            prior.sample_from_prior((2,))


class TestHorseshoePrior(unittest.TestCase):

    def test_lnprob_and_grad_scalar(self):
        prior = HorseshoePrior(scale=1, rng=np.random.RandomState(1))

        # Legal scalar
        self.assertEqual(prior.lnprob(-1), 1.1450937952919953)
        self.assertEqual(prior.gradient(-1), -0.6089187456211098)

        # Boundary
        self.assertTrue(np.isinf(prior._lnprob(0)))
        self.assertTrue(np.isinf(prior._gradient(0)))

    def test_sample_from_prior(self):
        prior = HorseshoePrior(scale=1, rng=np.random.RandomState(1))
        samples = prior.sample_from_prior(10)
        # Test that the rng is set
        self.assertAlmostEqual(samples[0], 1.0723988839129437)

    def test_sample_from_prior_shapes(self):
        rng = np.random.RandomState(1)
        lower_bound = 2 + rng.random_sample() * 50
        upper_bound = lower_bound + rng.random_sample() * 50
        prior = TophatPrior(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            rng=np.random.RandomState(1),
        )
        sample = prior.sample_from_prior(1)
        self.assertEqual(sample.shape, (1,))
        sample = prior.sample_from_prior(2)
        self.assertEqual(sample.shape, (2,))
        sample = prior.sample_from_prior(10)
        self.assertEqual(sample.shape, (10,))
        with self.assertRaises(ValueError):
            prior.sample_from_prior(0)
        with self.assertRaises(ValueError):
            prior.sample_from_prior((2,))

    def test_gradient(self):
        for scale in (0.1, 0.5, 1., 2.):
            prior = HorseshoePrior(scale=scale, rng=np.random.RandomState(1))
            # The function appears to be unstable above 15
            for theta in range(-20, 15):
                if theta == 0:
                    continue
                error = scipy.optimize.check_grad(
                    lambda x: prior.lnprob(x[0]),
                    lambda x: prior.gradient(x[0]),
                    np.array([theta]),
                    epsilon=1e-5,
                )
                self.assertAlmostEqual(error, 0, delta=5)


class TestGammaPrior(unittest.TestCase):

    def test_lnprob_and_grad_scalar(self):
        prior = GammaPrior(a=0.5, scale=1 / 2, loc=0, rng=np.random.RandomState(1))

        # Legal scalar
        x = -1
        self.assertEqual(prior.lnprob(x), -0.46155023498761205)
        self.assertEqual(prior.gradient(x), -1.2357588823428847)

    def test_lnprob_and_grad_array(self):
        prior = GammaPrior(a=0.5, scale=1 / 2, loc=0, rng=np.random.RandomState(1))
        val = np.array([-1, -1])
        with self.assertRaises(NotImplementedError):
            prior.lnprob(val)
        with self.assertRaises(NotImplementedError):
            prior.gradient(val)

    def test_gradient(self):
        for scale in (0.5, 1., 2.):
            prior = GammaPrior(a=2, scale=scale, loc=0, rng=np.random.RandomState(1))
            # The function appears to be unstable above 10
            for theta in np.arange(1e-15, 10, 0.01):
                if theta == 0:
                    continue
                error = scipy.optimize.check_grad(
                    lambda x: prior.lnprob(x[0]),
                    lambda x: prior.gradient(x[0]),
                    np.array([theta]),
                    epsilon=1e-5,
                )
                self.assertAlmostEqual(error, 0, delta=5, msg=str(theta))


class TestLogNormalPrior(unittest.TestCase):

    def test_gradient(self):
        for sigma in (0.5, 1., 2.):
            prior = LognormalPrior(mean=0, sigma=sigma, rng=np.random.RandomState(1))
            # The function appears to be unstable above 15
            for theta in range(0, 15):
                # Gradient approximation becomes unstable when going closer to zero
                theta += 1e-2
                error = scipy.optimize.check_grad(
                    lambda x: prior.lnprob(x[0]),
                    lambda x: prior.gradient(x[0]),
                    np.array([theta]),
                    epsilon=1e-5,
                )
                self.assertAlmostEqual(error, 0, delta=5, msg=theta)


class TestSoftTopHatPrior(unittest.TestCase):

    def test_lnprob(self):
        prior = SoftTopHatPrior(
            lower_bound=np.exp(-5),
            upper_bound=np.exp(5),
            exponent=2,
            rng=np.random.RandomState(1),
        )

        # Legal values
        self.assertEqual(prior.lnprob(-5), 0)
        self.assertEqual(prior.lnprob(0), 0)
        self.assertEqual(prior.lnprob(5), 0)

        # Illegal values
        self.assertAlmostEqual(prior.lnprob(-5.1), -0.01)
        self.assertAlmostEqual(prior.lnprob(-6), -1)
        self.assertAlmostEqual(prior.lnprob(-7), -4)
        self.assertAlmostEqual(prior.lnprob(5.1), -0.01)
        self.assertAlmostEqual(prior.lnprob(6), -1)
        self.assertAlmostEqual(prior.lnprob(7), -4)

    def test_grad(self):
        prior = SoftTopHatPrior(
            lower_bound=np.exp(-5),
            upper_bound=np.exp(5),
            exponent=2,
            rng=np.random.RandomState(1),
        )

        # Legal values
        self.assertEqual(prior.gradient(-5), 0)
        self.assertEqual(prior.gradient(0), 0)
        self.assertEqual(prior.gradient(5), 0)

        for theta in [-10, -7, -6, -5.1, 5.1, 6, 7, 10]:
            # Gradient approximation becomes unstable when going closer to zero
            theta += 1e-2
            grad = prior.gradient(theta)
            grad_vector = prior.gradient(theta)
            self.assertEqual(grad, grad_vector)

            def prob(x):
                return prior.lnprob(x[0])

            def grad(x):
                return prior.gradient(x[0])

            error = scipy.optimize.check_grad(prob, grad, np.array([theta]), epsilon=1e-5)
            self.assertAlmostEqual(error, 0, delta=5, msg=theta)
