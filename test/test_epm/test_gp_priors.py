import unittest

import numpy as np
import scipy.optimize

from smac.epm.gp_base_prior import TophatPrior, HorseshoePrior, LognormalPrior
from smac.utils.constants import VERY_SMALL_NUMBER


class TestTophatPrior(unittest.TestCase):

    def test_lnprob_and_grad_scalar(self):
        prior = TophatPrior(l_bound=-10, u_bound=2)

        # Legal scalar
        for val in (-1, 0, 1):
            self.assertEqual(prior.lnprob(val), 0)
            self.assertEqual(prior.gradient(val), 0)

        # Boundary
        for val in (-10, 2):
            self.assertEqual(prior.lnprob(val), 0)
            self.assertEqual(prior.gradient(val), 0)

        # Values outside the boundary
        for val in (-10 - VERY_SMALL_NUMBER, 2 + VERY_SMALL_NUMBER, -50, 50):
            self.assertTrue(np.isinf(prior.lnprob(val)))
            self.assertEqual(prior.gradient(val), 0)

    def test_lnprob_and_grad_array(self):
        prior = TophatPrior(l_bound=-10, u_bound=2)

        # Legal arrays
        for val in ([-1, -1], [0, 0], [1, 1]):
            val = np.array(val)
            self.assertEqual(prior.lnprob(val), 0)
            np.testing.assert_array_almost_equal(prior.gradient(val), [0, 0])

        # Boundary
        self.assertEqual(prior.lnprob(np.array([-10, 2])), 0)
        np.testing.assert_array_almost_equal(prior.gradient(np.array([-10, 2])), [0, 0])

        # Values outside the boundary
        for val in ([-20, 1], [-5, 20], [-11, -11], [3, 3]):
            val = np.array(val)
            self.assertTrue(np.isinf(prior.lnprob(val)))
            np.testing.assert_array_almost_equal(prior.gradient(val), [0, 0])

    def test_sample_from_prior(self):
        prior = TophatPrior(l_bound=-10, u_bound=2, rng=np.random.RandomState(1))
        samples = prior.sample_from_prior(10)
        np.testing.assert_array_equal(samples >= -10, True)
        np.testing.assert_array_equal(samples <= 2, True)
        # Test that the rng is set
        self.assertAlmostEqual(samples[0][0], -4.995735943569112)


class TestHoreshoePrior(unittest.TestCase):

    def test_lnprob_and_grad_scalar(self):
        prior = HorseshoePrior(scale=1)

        # Legal scalar
        self.assertEqual(prior.lnprob(-1), 1.1450937952919953)
        self.assertEqual(prior.gradient(-1), -0.6089187456211098)

        # Boundary
        self.assertTrue(np.isinf(prior.lnprob(0)))
        self.assertTrue(np.isinf(prior.gradient(0)))

    def test_lnprob_and_grad_array(self):
        prior = HorseshoePrior(scale=1)

        # Legal array
        val = np.array([-1, -1])
        self.assertEqual(prior.lnprob(val), 1.838240975836031)
        np.testing.assert_array_almost_equal(prior.gradient(val), [-0.608919, -0.608919])

        # Boundary
        self.assertTrue(np.isinf(prior.lnprob(np.array([0, 2]))))
        self.assertTrue(np.isinf(prior.gradient(np.array([0, 2]))).all())

    def test_sample_from_prior(self):
        prior = HorseshoePrior(scale=1, rng=np.random.RandomState(1))
        samples = prior.sample_from_prior(10)
        # Test that the rng is set
        self.assertAlmostEqual(samples[0][0], 1.0723988839129437)

    def test_gradient(self):
        for scale in (0.5, 1., 2.):
            prior = HorseshoePrior(scale=scale)
            # The function appears to be unstable above 15
            for theta in range(-20, 15):
                if theta == 0:
                    continue
                grad = prior.gradient(theta)
                grad_vector = prior.gradient(np.array([theta]))
                self.assertEqual(grad, grad_vector)
                error = scipy.optimize.check_grad(prior.lnprob, prior.gradient, np.array([theta]), epsilon=1e-5)
                self.assertAlmostEqual(error, 0, delta=5)


class TestLogNormalPrior(unittest.TestCase):

    def test_gradient(self):
        for sigma in (0.5, 1., 2.):
            prior = LognormalPrior(mean=0, sigma=sigma)
            # The function appears to be unstable above 15
            for theta in range(0, 15):
                # Gradient approximation becomes unstable when going closer to zero
                theta += 1e-2
                grad = prior.gradient(theta)
                grad_vector = prior.gradient(np.array([theta]))
                self.assertEqual(grad, grad_vector)
                error = scipy.optimize.check_grad(prior.lnprob, prior.gradient, np.array([theta]), epsilon=1e-5)
                self.assertAlmostEqual(error, 0, delta=5, msg=theta)
