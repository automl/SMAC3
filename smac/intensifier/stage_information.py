from ConfigSpace import Configuration

from smac.runhistory import InstanceSeedBudgetKey, TrialInfo

__copyright__ = "Copyright 2023, automl.org"
__license__ = "3-clause BSD"


class Stage:
    """Class to store information about a stage in the multi-fidelity optimization process."""

    def __init__(
        self,
        repetition: int,
        bracket: int,
        stage: int,
        budget: float,
        amount_configs_to_yield: int,
        isb_keys: list[InstanceSeedBudgetKey],
        configs: list[Configuration] = None,
    ):
        self.repetition: int = repetition
        self.bracket: int = bracket
        self.stage: int = stage
        self.budget: float = budget
        if configs is None:
            configs = []
        self.configs: list[Configuration] = configs
        self.trials: dict[Configuration, list[TrialInfo]] = {config: [] for config in configs}
        self.isb_keys: list[InstanceSeedBudgetKey] = isb_keys
        self.amount_configs_yielded: int = 0
        self.amount_configs_to_yield: int = amount_configs_to_yield
        self.terminated: bool = False

    @property
    def all_configs_yielded(self) -> bool:
        """Returns whether the stage is done."""
        return self.amount_configs_yielded >= self.amount_configs_to_yield

    @property
    def yielded_configs(self) -> list[Configuration]:
        """Returns the configs that have been yielded."""
        return self.configs[: self.amount_configs_yielded]

    @property
    def all_configs_generated(self) -> bool:
        """Returns whether all configs have been generated."""
        return len(self.configs) >= self.amount_configs_to_yield

    def add_config(self, config: Configuration) -> None:
        """Adds a config to the stage."""
        if len(self.configs) >= self.amount_configs_to_yield:
            raise ValueError("Cannot add more configs to stage than specified in amount_configs_to_yield.")
        self.configs.append(config)
        self.trials[config] = []

    def add_trials_for_config(self, config: Configuration, trials_for_config: list[TrialInfo]) -> None:
        """Adds trials for a config to the stage."""
        self.trials[config].extend(trials_for_config)

    def terminate(self) -> None:
        """Terminates the stage."""
        self.terminated = True

    def get_next_config(self) -> Configuration:
        """Returns the next config to evaluate."""
        if self.all_configs_yielded:
            raise ValueError("All configs have already been yielded.")
        config = self.configs[self.amount_configs_yielded]
        self.amount_configs_yielded += 1
        return config

    def __str__(self) -> str:
        return (
            f"Stage progress: {self.amount_configs_yielded} / {self.amount_configs_to_yield}, "
            f"isb_keys={len(self.isb_keys)}, configs={len(self.configs)}, trials={len(self.trials)})"
        )
