from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field

from ConfigSpace import Configuration

from smac.runhistory.enumerations import StatusType

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


@dataclass(frozen=True)
class InstanceSeedKey:
    """Key for instance and seed.

    Parameters
    ----------
    instance : str | None, defaults to None
    seed : int | None, defaults to None
    """

    instance: str | None = None
    seed: int | None = None

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, InstanceSeedKey):
            if self.instance == other.instance and self.seed == other.seed:
                return True

        return False


@dataclass(frozen=True)
class InstanceSeedBudgetKey:
    """Key for instance, seed and budget.

    Parameters
    ----------
    instance : str | None, defaults to None
    seed : int | None, defaults to None
    budget : float | None, defaults to None
    """

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

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, InstanceSeedBudgetKey):
            if self.instance == other.instance and self.seed == other.seed and self.budget == other.budget:
                return True

        return False

    def get_instance_seed_key(self) -> InstanceSeedKey:
        """Returns the instance-seed key. The budget is omitted."""
        return InstanceSeedKey(instance=self.instance, seed=self.seed)


@dataclass(frozen=True)
class TrialKey:
    """Key of a trial.

    Parameters
    ----------
    config_id : int
    instance : str | None, defaults to None
    seed : int | None, defaults to None
    budget : float | None, defaults to None
    """

    config_id: int
    instance: str | None = None
    seed: int | None = None
    budget: float | None = None


@dataclass(frozen=True)
class TrialValue:
    """Values of a trial.

    Parameters
    ----------
    cost : float | list[float]
    time : float, defaults to 0.0
    status : StatusType, defaults to StatusType.SUCCESS
    starttime : float, defaults to 0.0
    endtime : float, defaults to 0.0
    additional_info : dict[str, Any], defaults to {}
    """

    cost: float | list[float]
    time: float = 0.0
    status: StatusType = StatusType.SUCCESS
    starttime: float = 0.0
    endtime: float = 0.0
    additional_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialInfo:
    """Information about a trial.

    Parameters
    ----------
    config : Configuration
    instance : str | None, defaults to None
    seed : int | None, defaults to None
    budget : float | None, defaults to None
    """

    config: Configuration
    instance: str | None = None
    seed: int | None = None
    budget: float | None = None

    def get_instance_seed_key(self) -> InstanceSeedKey:
        """Instantiates and returns an InstanceSeedKey object"""
        return InstanceSeedKey(instance=self.instance, seed=self.seed)

    def get_instance_seed_budget_key(self) -> InstanceSeedBudgetKey:
        """Instantiates and returns an InstanceSeedBudgetKey object."""
        return InstanceSeedBudgetKey(instance=self.instance, seed=self.seed, budget=self.budget)


@dataclass
class TrajectoryItem:
    """Item of a trajectory.

    Parameters
    ----------
    config_ids : list[int]
        Configuration ids of the current incumbents.
    costs : list[float | list[float]]
        Costs of the current incumbents. In case of multi-objective, this is a list of lists.
    trial : int
        How many trials have been evaluated so far.
    walltime : float
        How much walltime has been used so far.
    """

    config_ids: list[int]
    costs: list[float | list[float]]
    trial: int
    walltime: float
