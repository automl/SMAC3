import numpy as np

from smac.random_design.probability_design import ProbabilityRandomDesign

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def test_chooser_prob():
    for i in range(10):
        c = ProbabilityRandomDesign(seed=1, probability=0.1 * i)
        stats = []
        for j in range(100000):
            stats.append(c.check(j))
        assert np.isclose(np.mean(stats), 0.1 * i, atol=10**-2)

