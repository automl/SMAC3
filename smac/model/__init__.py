from smac.model.abstract_model import AbstractModel
from smac.model.multi_objective_model import MultiObjectiveModel
from smac.model.random_model import RandomModel

__all__ = ["AbstractModel", "MultiObjectiveModel", "RandomModel"]

try:
    from smac.model.tabPFNv2 import TabPFNModel

    __all__ = ["AbstractModel", "MultiObjectiveModel", "RandomModel", "TabPFNModel"]
except ImportError as e:
    raise ImportError(
        "TabPFNModel requires tabpfn to be installed and Python >=3.9. " "Install with pip install tabpfn"
    ) from e
