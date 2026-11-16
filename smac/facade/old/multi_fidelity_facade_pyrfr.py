from __future__ import annotations

from smac.facade.multi_fidelity_facade import MultiFidelityFacade
from smac.facade.old.hyperparameter_optimization_facade_pyrfr import (
    HyperparameterOptimizationRFRFacade,
)

__copyright__ = "Copyright 2026, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class MultiFidelityRFRFacade(MultiFidelityFacade, HyperparameterOptimizationRFRFacade):
    pass
