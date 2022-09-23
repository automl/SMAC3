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
    budget: float | None = None

    def __lt__(self, other: InstanceSeedBudgetKey) -> bool:
        if self.budget is not None and other.budget is not None:
            return self.budget < other.budget

        if self.instance is not None and other.instance is not None:
            return self.instance < other.instance

        if self.seed is not None and other.seed is not None:
            return self.seed < other.seed

        raise RuntimeError("Could not compare InstanceSeedBudgetKey.")


@dataclass(frozen=True)
class TrialKey:
    config_id: int
    instance: str | None = None
    seed: int | None = None
    budget: float | None = None


@dataclass(frozen=True)
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
    instance: str | None = None
    seed: int | None = None
    budget: float | None = None
    source: int = 0


@dataclass
class TrajectoryItem:
    """Replaces `TrajEntry` from the original code."""

    incumbent: Configuration | dict[str, Any]
    cost: float | list[float]
    budget: float | None
    walltime_used: float
    num_trial: int

    def __post_init__(self) -> None:
        # Transform configuration to dict
        if isinstance(self.incumbent, Configuration):
            self.incumbent = self.incumbent.get_dictionary()
