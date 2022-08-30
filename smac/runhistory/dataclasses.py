from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field

from ConfigSpace import Configuration

from smac.runhistory.enumerations import StatusType

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


@dataclass(frozen=True)
class InstanceSeedKey:
    instance: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class InstanceSeedBudgetKey:
    instance: str | None = None
    seed: int | None = None
    budget: float = 0.0

    def __lt__(self, other: InstanceSeedBudgetKey) -> bool:
        return self.budget < other.budget


@dataclass(frozen=True)
class TrialKey:
    config_id: int
    instance: str | None = None
    seed: int | None = None
    budget: float = 0.0


@dataclass  # (frozen=True)
class TrialValue:
    cost: float | list[float]
    time: float = 0.0
    status: StatusType = StatusType.SUCCESS
    starttime: float = 0.0
    endtime: float = 0.0
    additional_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialInfo:
    config: Configuration
    instance: str | None
    seed: int | None
    budget: float = 0.0
    source: int = 0


@dataclass
class TrajectoryItem:
    """Replaces `TrajEntry` from the original code."""

    cost: float
    incumbent: Configuration | dict[str, Any]
    walltime_used: float
    target_algorithm_walltime_used: float
    target_algorithm_runs: int
    budget: float

    def __post_init__(self) -> None:
        # Transform configuration to dict
        if isinstance(self.incumbent, Configuration):
            self.incumbent = self.incumbent.get_dictionary()
