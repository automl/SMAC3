import numpy as np

from smac.random_design.annealing_design import CosineAnnealingRandomDesign

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def test_cosineannealing_down():
    c = CosineAnnealingRandomDesign(seed=1, min_probability=0.0, max_probability=1.0, restart_iteration=5)
    expected = [1.0, 0.8535, 0.5, 0.1464, 0.0]
    for i in range(20):
        stats = []
        for j in range(10000):
            stats.append(c.check(j))
        assert np.isclose(np.mean(stats), expected[i % 5], atol=10**-2)
        c.next_iteration()
