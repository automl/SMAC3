import unittest

import numpy as np
import scipy.optimize
import scipy.stats as scst
import scipy.integrate as scin

from smac.epm.gp_base_prior import TophatPrior, HorseshoePrior, LognormalPrior, GammaPrior, SoftTopHatPrior
from smac.utils.constants import VERY_SMALL_NUMBER


def compare_to_scalar(self, prior, bounds):
    """
    1) Samples 1000 configs within bounds
    2) Computes lnprob using prior
    3) ASSERTS lnprog for array is the same as lnprob for scalars
    """
    samples = np.linspace(bounds[0], bounds[1], 5)
    lns = np.array([prior.lnprob(np.log(s)) for s in samples])
    self.assertEqual(np.nansum(lns), prior.lnprob(np.log(samples)))
    return lns


def wrap_ln(theta, prior):
    return np.exp(prior.lnprob(np.log(theta)))


class TestTophatPrior(unittest.TestCase):

    def test_lnprob_sums_to_1(self):
        rng = np.random.RandomState(1)
        for i in range(100):
            lower_bound = 2+rng.random_sample()*50
            upper_bound = lower_bound + rng.random_sample()*50
            prior = TophatPrior(lower_bound=lower_bound, upper_bound=upper_bound)
            compare_to_scalar(self, prior, [lower_bound, upper_bound])

        # Doesn't work yet
        #integral = scin.quad(wrap_ln, args=(TophatPrior(lower_bound=-1, upper_bound=1), ), a=-5, b=5)
        #self.assertLess(integral[1], 0.1)
        #self.assertEqual(integral[0], 1)

    def test_lnprob_and_grad_scalar(self):
        prior = TophatPrior(lower_bound=-10, upper_bound=2)

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
        prior = TophatPrior(lower_bound=-10, upper_bound=2)

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
        prior = TophatPrior(lower_bound=-10, upper_bound=2, rng=np.random.RandomState(1))
        samples = prior.sample_from_prior(10)
        np.testing.assert_array_equal(samples >= -10, True)
        np.testing.assert_array_equal(samples <= 2, True)
        # Test that the rng is set
        self.assertAlmostEqual(samples[0][0], -4.995735943569112)


class TestHorseshoePrior(unittest.TestCase):

    def test_lnprob_sums_to_1(self):
        rng = np.random.RandomState(1)
        for i in range(100):
            lower_bound = 1+rng.random_sample()
            upper_bound = lower_bound + rng.random_sample()*5
            scale = 0.5+rng.random_sample()
            prior = HorseshoePrior(scale=scale)
            compare_to_scalar(self, prior, [lower_bound, upper_bound])

        # Does not work
        #integral = scin.quad(wrap_ln, args=(HorseshoePrior(scale=1), ), a=1e-10, b=5)#, b=100)
        #self.assertLess(integral[1], 0.1)
        #self.assertEqual(integral[0], 1)

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


class TestGammaPrior(unittest.TestCase):

    def test_lnprob_sums_to_1(self):
        rng = np.random.RandomState(1)
        for i in range(100):
            lower_bound = rng.randint(low=1, high=5)
            upper_bound = lower_bound + 5
            loc = lower_bound-0.5
            a = rng.random_sample()
            scale = 0.1 + rng.random_sample()
            prior = GammaPrior(a=a, loc=loc, scale=scale)
            compare_to_scalar(self, prior, [lower_bound, upper_bound])

        integral = scin.quad(wrap_ln, args=(GammaPrior(a=1, loc=0, scale=1),), a=0, b=100)
        self.assertLess(integral[1], 0.1)
        self.assertAlmostEqual(integral[0], 1, 2)

    def test_lnprob_and_grad_scalar(self):
        prior = GammaPrior(a=0.5, scale=1/2, loc=0)

        # Legal scalar
        x = -1
        self.assertEqual(prior.lnprob(x), -0.46155023498761205)
        self.assertEqual(prior.gradient(x), 0.7458042693520156)

    def test_lnprob_and_grad_array(self):
        prior = GammaPrior(a=0.5, scale=1/2, loc=0)

        # Legal array
        val = np.array([-1, -1])
        print(prior.lnprob(val))
        self.assertEqual(prior.lnprob(val), -0.9231004699752241)
        np.testing.assert_array_almost_equal(prior.gradient(val), [0.7458042693520156, 0.7458042693520156])

    def test_gradient(self):
        for scale in (0.5, 1., 2.):
            prior = GammaPrior(a=0.5, scale=scale, loc=0)
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

    def test_lnprob_sums_to_1(self):
        rng = np.random.RandomState(1)
        for i in range(100):
            lower_bound = 0.001
            upper_bound = 5
            scale = 0.1+rng.random_sample()
            prior = LognormalPrior(mean=0, sigma=scale)
            compare_to_scalar(self, prior, [lower_bound, upper_bound])

        integral = scin.quad(wrap_ln, args=(LognormalPrior(mean=0, sigma=1), ), a=1e-10, b=100)
        self.assertLess(integral[1], 0.1)
        self.assertAlmostEqual(integral[0], 1, 2)

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


class TestSoftTopHatPrior(unittest.TestCase):

    def test_lnprob_sums_to_1(self):
        rng = np.random.RandomState(1)
        for i in range(100):
            lower_bound = -rng.randint(low=-50, high=-6)
            upper_bound = -lower_bound
            exponent = 2
            prior = SoftTopHatPrior(lower_bound=lower_bound+5, upper_bound=upper_bound-5, exponent=exponent)
            compare_to_scalar(self, prior, [lower_bound, upper_bound])

        # Does not work
        #integral = scin.quad(wrap_ln, args=(SoftTopHatPrior(lower_bound=1, upper_bound=5, exponent=2),), a=-10, b=100)
        #self.assertLess(integral[1], 0.1)
        #self.assertAlmostEqual(integral[0], 1, 2)

    def test_lnprob(self):
        prior = SoftTopHatPrior(lower_bound=-5, upper_bound=5)

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
        prior = SoftTopHatPrior(lower_bound=-5, upper_bound=5)

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
            error = scipy.optimize.check_grad(prior.lnprob, prior.gradient, np.array([theta]), epsilon=1e-5)
            self.assertAlmostEqual(error, 0, delta=5, msg=theta)
