import unittest

import numpy as np

from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, ChooserProb, ChooserLinearCoolDown

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestRandomConfigurationChooser(unittest.TestCase):

    def test_no_cool_down(self):
        c = ChooserNoCoolDown(rng=np.random.RandomState(), modulus=3.0)
        self.assertFalse(c.check(1))
        self.assertFalse(c.check(2))
        self.assertTrue(c.check(3))
        self.assertFalse(c.check(4))
        self.assertFalse(c.check(5))
        self.assertTrue(c.check(6))
        self.assertTrue(c.check(30))
        c.next_smbo_iteration()
        self.assertFalse(c.check(1))
        self.assertFalse(c.check(2))
        self.assertTrue(c.check(3))
        self.assertFalse(c.check(4))
        self.assertFalse(c.check(5))
        self.assertTrue(c.check(6))
        self.assertTrue(c.check(30))
        c = ChooserNoCoolDown(rng=np.random.RandomState(), modulus=1.0)
        self.assertTrue(c.check(1))
        self.assertTrue(c.check(2))
        c.next_smbo_iteration()
        self.assertTrue(c.check(1))
        self.assertTrue(c.check(2))

    def test_linear_cool_down(self):
        c = ChooserLinearCoolDown(None, 2.0, 1.0, 4.0)
        self.assertFalse(c.check(1))
        self.assertTrue(c.check(2))
        self.assertFalse(c.check(3))
        self.assertTrue(c.check(4))
        self.assertFalse(c.check(5))
        self.assertTrue(c.check(6))
        self.assertFalse(c.check(7))
        self.assertTrue(c.check(8))
        c.next_smbo_iteration()
        self.assertFalse(c.check(1))
        self.assertFalse(c.check(2))
        self.assertTrue(c.check(3))
        self.assertFalse(c.check(4))
        self.assertFalse(c.check(5))
        self.assertTrue(c.check(6))
        self.assertFalse(c.check(7))
        self.assertFalse(c.check(8))
        c.next_smbo_iteration()
        self.assertFalse(c.check(1))
        self.assertFalse(c.check(2))
        self.assertFalse(c.check(3))
        self.assertTrue(c.check(4))
        self.assertFalse(c.check(5))
        self.assertFalse(c.check(6))
        self.assertFalse(c.check(7))
        self.assertTrue(c.check(8))
        c.next_smbo_iteration()
        self.assertFalse(c.check(1))
        self.assertFalse(c.check(2))
        self.assertFalse(c.check(3))
        self.assertTrue(c.check(4))
        self.assertFalse(c.check(5))
        self.assertFalse(c.check(6))
        self.assertFalse(c.check(7))
        self.assertTrue(c.check(8))

    def test_chooser_prob(self):
        for i in range(10):
            c = ChooserProb(rng=np.random.RandomState(1), prob=0.1 * i)
            stats = []
            for j in range(100000):
                stats.append(c.check(j))
            print(np.sum(stats) / 100000, 0.1 * i)
            self.assertAlmostEqual(np.sum(stats) / 100000, 0.1 * i, places=2)
