import unittest

from nose.plugins.attrib import attr
from scipy.optimize import fmin_l_bfgs_b
import shutil
from smac.facade.func_facade import fmin_smac


def rosenbrock_2d(x, seed=1):

    return 100. * (x[1] - x[0] ** 2.) ** 2. + (1 - x[0]) ** 2.


class TestSMACFacade(unittest.TestCase):

    def setUp(self):
        self.output_dirs = []

    def tearDown(self):
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)

    @attr('slow')
    def test_func_smac(self):
        func = rosenbrock_2d
        x0 = [-3, -4]
        bounds = [(-5, 5), (-5, 5)]

        x, f, smac = fmin_smac(func, x0, bounds, maxfun=10)
        x_s, f_s, _ = fmin_l_bfgs_b(func, x0, bounds, maxfun=10,
                                    approx_grad=True)

        self.assertEqual(type(x), type(x_s))
        self.assertEqual(type(f), type(f_s))

        self.output_dirs.append(smac.scenario.output_dir)

    def test_parameter_order(self):
        def func(x):
            for i in range(len(x)):
                self.assertLess(i - 1, x[i])
                self.assertGreater(i, x[i])
            return 1

        default = [i - 0.5 for i in range(10)]
        bounds = [(i - 1, i) for i in range(10)]
        print(default, bounds)
        _, _, smac = fmin_smac(func=func, x0=default,
                               bounds=bounds,
                               maxfun=1)

        self.output_dirs.append(smac.scenario.output_dir)

