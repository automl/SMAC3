import unittest

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
