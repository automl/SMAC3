from ConfigSpace import Configuration

from smac.runhistory import InstanceSeedBudgetKey, TrialInfo


class Stage:
    """Class to store information about a stage in the multi-fidelity optimization process."""

    def __init__(self, amount_configs_to_yield: int, isb_keys: list[InstanceSeedBudgetKey]):
        self.configs: list[Configuration] = []
        self.trials: dict[Configuration, list[TrialInfo]] = {}
        self.isb_keys: list[InstanceSeedBudgetKey] = isb_keys
        self.amount_configs_yielded: int = 0
        self.amount_configs_to_yield: int = amount_configs_to_yield

    @property
    def is_done(self) -> bool:
        """Returns whether the stage is done."""
        return self.amount_configs_yielded >= self.amount_configs_to_yield

    def add_config(self, config: Configuration) -> None:
        """Adds a config to the stage."""
        self.configs.append(config)
        self.trials[config] = []

    def add_trials_for_config(self, config: Configuration, trials_for_config: list[TrialInfo]) -> None:
        """Adds trials for a config to the stage."""
        self.trials[config].extend(trials_for_config)
