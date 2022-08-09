from __future__ import annotations

from typing import Any, Dict

import json
from dataclasses import dataclass
from enum import Enum

from smac.configspace import Configuration
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class StatusType(Enum):
    """Class to define numbers for status types."""

    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORT = 4
    MEMOUT = 5
    CAPPED = 6

    # Only relevant for SH/HB. Run might have a results, but should not be considered further.
    # By default, these runs will always be considered for building the model. Potential use cases:
    # 1) The run has converged and does not benefit from a higher budget
    # 2) The run has exhausted given resources and will not benefit from higher budgets
    DONOTADVANCE = 7

    # Gracefully exit SMAC - wait for currently executed runs to finish
    STOP = 8

    # In case a job was submited, but it has not finished
    RUNNING = 9

    @staticmethod
    def enum_hook(obj: Dict) -> Any:
        """Hook function passed to json-deserializer as "object_hook".

        EnumEncoder in runhistory/runhistory.
        """
        if "__enum__" in obj:
            # object is marked as enum
            name, member = obj["__enum__"].split(".")
            if name == "StatusType":
                return getattr(globals()[name], member)
        return obj


@dataclass(frozen=True)
class RunKey:
    config_id: int
    instance: str | None = None
    seed: int | None = None
    budget: float = 0.0


@dataclass(frozen=True)
class InstanceSeedKey:
    instance: str
    seed: int


@dataclass(frozen=True)
class InstanceSeedBudgetKey:
    instance: str
    seed: int
    budget: float


@dataclass(frozen=True)
class RunValue:
    cost: float | list[float]
    time: float
    status: StatusType
    starttime: float
    endtime: float
    additional_info: dict[str, Any]


@dataclass(frozen=True)
class RunInfo:
    config: Configuration
    instance: str | None
    # instance_specific: str | None
    seed: int
    budget: float = 0.0
    source_id: int = 0


# TODO: Do we really need this?
class EnumEncoder(json.JSONEncoder):
    """Custom encoder for enum-serialization (implemented for StatusType from tae).

    Using encoder implied using object_hook as defined in StatusType to deserialize from json.
    """

    def default(self, obj: object) -> Any:
        """Returns the default encoding of the passed object."""
        if isinstance(obj, StatusType):
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


class DataOrigin(Enum):
    """Definition of how data in the runhistory is used.

    * ``INTERNAL``: internal data which was gathered during the current
      optimization run. It will be saved to disk, used for building EPMs and
      during intensify.
    * ``EXTERNAL_SAME_INSTANCES``: external data, which was gathered by running
       another program on the same instances as the current optimization run
       runs on (for example pSMAC). It will not be saved to disk, but used both
       for EPM building and during intensify.
    * ``EXTERNAL_DIFFERENT_INSTANCES``: external data, which was gathered on a
       different instance set as the one currently used, but due to having the
       same instance features can still provide useful information. Will not be
       saved to disk and only used for EPM building.
    """

    INTERNAL = 1
    EXTERNAL_SAME_INSTANCES = 2
    EXTERNAL_DIFFERENT_INSTANCES = 3
