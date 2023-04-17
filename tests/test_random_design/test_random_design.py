from smac.random_design.annealing_design import CosineAnnealingRandomDesign
from smac.random_design.modulus_design import (
    DynamicModulusRandomDesign,
    ModulusRandomDesign,
)
from smac.random_design.probability_design import ProbabilityRandomDesign

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def test_chooser_rng():
    dc = {
        "NoCoolDownRandomDesign": [ModulusRandomDesign, {"modulus": 2}],
        "LinearCoolDownRandomDesign": [
            DynamicModulusRandomDesign,
            {
                "start_modulus": 2,
                "modulus_increment": 1,
                "end_modulus": 10,
            },
        ],
        "ProbabilityRandomDesign": [ProbabilityRandomDesign, {"probability": 0.5}],
        "CosineAnnealingRandomDesign": [
            CosineAnnealingRandomDesign,
            {"min_probability": 0.0, "max_probability": 0.9, "restart_iteration": 7},
        ],
    }

    for method in dc:
        for i in range(100):
            c1 = dc[method][0](**dc[method][1], seed=i)
            c2 = dc[method][0](**dc[method][1], seed=i)
            for i in range(100):
                assert c1.check(0) == c2.check(0)
                assert c1.check(10) == c2.check(10)
                c1.next_iteration()
                c2.next_iteration()
