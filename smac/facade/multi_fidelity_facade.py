from __future__ import annotations

from ConfigSpace import Configuration

from smac.callback.multifidelity_stopping_callback import MultiFidelityStoppingCallback
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade,
)
from smac.initial_design.random_design import RandomInitialDesign
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario

__copyright__ = "Copyright 2023, automl.org"
__license__ = "3-clause BSD"


class MultiFidelityFacade(HyperparameterOptimizationFacade):
    """This facade configures SMAC in a multi-fidelity setting."""

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        eta: int = 3,
        n_seeds: int = 1,
        instance_seed_order: str | None = "shuffle_once",
        max_incumbents: int = 10,
        incumbent_selection: str = "highest_observed_budget",
        sample_brackets_at_once: bool = False,
        early_stopping: MultiFidelityStoppingCallback = None,
        remove_stopped_fidelities_incrementally: bool = False,
        only_go_to_next_fidelity_after_early_stopping: bool = False,
    ) -> Hyperband:
        """Returns a Hyperband intensifier instance. Budgets are supported.

        eta : int, defaults to 3
            Input that controls the proportion of configurations discarded in each round of Successive Halving.
        n_seeds : int, defaults to 1
            How many seeds to use for each instance.
        instance_seed_order : str, defaults to "shuffle_once"
            How to order the instance-seed pairs. Can be set to:
            * None: No shuffling at all and use the instance-seed order provided by the user.
            * "shuffle_once": Shuffle the instance-seed keys once and use the same order across all runs.
            * "shuffle": Shuffles the instance-seed keys for each bracket individually.
        incumbent_selection : str, defaults to "any_budget"
            How to select the incumbent when using budgets. Can be set to:
            * "any_budget": Incumbent is the best on any budget, i.e., the best performance regardless of budget.
            * "highest_observed_budget": Incumbent is the best in the highest budget run so far.
            refer to `runhistory.get_trials` for more details. Crucially, if true, then a
            for a given config-instance-seed, only the highest (so far executed) budget is used for
            comparison against the incumbent. Notice, that if the highest observed budget is smaller
            than the highest budget of the incumbent, the configuration will be queued again to
            be intensified again.
            * "highest_budget": Incumbent is selected only based on the absolute highest budget
            available only.
        max_incumbents : int, defaults to 10
            How many incumbents to keep track of in the case of multi-objective.
        sample_brackets_at_once : bool, defaults to False
            Whether to sample all brackets at once or one config at a time.
        early_stopping : MultiFidelityStoppingCallback, defaults to False
            Whether to stop the optimization early.
        remove_stopped_fidelities_incrementally : bool, defaults to False
            Whether to remove stopped fidelities incrementally or not remove them completely at all.
        only_go_to_next_fidelity_after_early_stopping : bool, defaults to False
            Whether to only go to the next fidelity after early stopping or not.
        """
        return Hyperband(
            scenario=scenario,
            eta=eta,
            n_seeds=n_seeds,
            instance_seed_order=instance_seed_order,
            max_incumbents=max_incumbents,
            incumbent_selection=incumbent_selection,
            sample_brackets_at_once=sample_brackets_at_once,
            early_stopping=early_stopping,
            remove_stopped_fidelities_incrementally=remove_stopped_fidelities_incrementally,
            only_go_to_next_fidelity_after_early_stopping=only_go_to_next_fidelity_after_early_stopping,
        )

    @staticmethod
    def get_initial_design(  # type: ignore
        scenario: Scenario,
        *,
        n_configs: int | None = None,
        n_configs_per_hyperparameter: int = 10,
        max_ratio: float = 0.25,
        additional_configs: list[Configuration] = None,
    ) -> RandomInitialDesign:
        """Returns a random initial design.

        Parameters
        ----------
        scenario : Scenario
        n_configs : int | None, defaults to None
            Number of initial configurations (disables the arguments ``n_configs_per_hyperparameter``).
        n_configs_per_hyperparameter: int, defaults to 10
            Number of initial configurations per hyperparameter. For example, if my configuration space covers five
            hyperparameters and ``n_configs_per_hyperparameter`` is set to 10, then 50 initial configurations will be
            samples.
        max_ratio: float, defaults to 0.1
            Use at most ``scenario.n_trials`` * ``max_ratio`` number of configurations in the initial design.
            Additional configurations are not affected by this parameter.
        additional_configs: list[Configuration], defaults to []
            Adds additional configurations to the initial design.
        """
        additional_configs = [] if additional_configs is None else additional_configs
        return RandomInitialDesign(
            scenario=scenario,
            n_configs=n_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparameter,
            max_ratio=max_ratio,
            additional_configs=additional_configs,
        )
