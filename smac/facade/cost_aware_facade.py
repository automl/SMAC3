from __future__ import annotations

from typing import Any, Callable

from functools import wraps

from ConfigSpace import Configuration

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.cost_aware_acquisition_function import (
    CostAwareAcquisitionFunction,
)
from smac.acquisition.function.expected_improvement import EI
from smac.callback.budget_exhausted_callback import BudgetExhaustedCallback
from smac.callback.callback import Callback
from smac.callback.cost_surrogate_callback import CostSurrogateCallback
from smac.callback.update_cost_callback import UpdateCostCallback
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.initial_design.cost_aware_initial_design import CostAwareInitialDesign
from smac.model.abstract_model import AbstractModel
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.runner.abstract_runner import AbstractRunner
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class CostAwareFacade(BlackBoxFacade):
    """Facade to configure SMAC for cost-aware black-box optimization.

    This facade extends ``BlackBoxFacade`` with cost-awareness: configurations are evaluated
    under a finite resource budget, and the acquisition function is weighted by predicted cost
    (EI-Cool strategy). The optimization terminates when the cumulative cost exceeds
    ``total_resource_budget``.

    The target function must return a dictionary with two keys: ``"performance"`` for the
    value to be minimized, and ``"cost"`` for the resource cost of the evaluation. The facade
    wraps this internally into SMAC's expected ``(cost, additional_info)`` format.

    Parameters
    ----------
    scenario : Scenario
        The scenario object, holding all environmental information.
    target_function : Callable | str | AbstractRunner | None, defaults to None
        The target function to optimize. If a callable, it must return a dictionary with
        ``"performance"`` and ``"cost"`` keys.
    total_resource_budget : float
        The total budget for the optimization in terms of resource/cost.
    cost_model : AbstractModel | None, defaults to None
        The cost model to predict the cost of configurations.
        Mutually exclusive with ``cost_formula``.
    cost_formula : Callable | None, defaults to None
        A callable that calculates the cost of a configuration.
        Used if ``cost_model`` is not provided.
    initial_design : AbstractInitialDesign | None, defaults to None
        The initial design strategy. If None, ``CostAwareInitialDesign`` is used.
    initial_design_budget_ratio : float, defaults to 0.125
        The fraction of ``total_resource_budget`` to be used for the initial design.
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
        The acquisition function to use. If None, ``CostAwareAcquisitionFunction(EI())`` is used.
    overwrite : bool, defaults to False
        When True, overwrites the run results if a previous run is found that is
        consistent in the meta data with the current setup.
    **kwargs: Any
        Additional keyword arguments passed to ``BlackBoxFacade``.
    """

    def __init__(
        self,
        scenario: Scenario,
        target_function: Callable | str | AbstractRunner | None = None,
        *,
        total_resource_budget: float,
        cost_model: AbstractModel | None = None,
        cost_formula: Callable | None = None,
        initial_design: AbstractInitialDesign | None = None,
        initial_design_budget_ratio: float = 0.125,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ):
        self._total_resource_budget = total_resource_budget
        self._initial_design_budget = total_resource_budget * initial_design_budget_ratio

        # Wrap the target function to extract performance/cost from dict return
        if callable(target_function):
            target_function = self._wrap_target_function(target_function)

        # Resolve cost model (shared between initial design and acquisition function)
        cost_model = self.get_cost_model(scenario, cost_model=cost_model, cost_formula=cost_formula)

        # Create initial design if not provided
        if initial_design is None:
            initial_design = self.get_initial_design(
                scenario,
                cost_model=cost_model,
                initial_budget=self._initial_design_budget,
            )

        # Assemble cost-aware callbacks and acquisition function
        callbacks = list(kwargs.pop("callbacks", []))
        callbacks, acquisition_function = self._prepare_callbacks(
            scenario=scenario,
            cost_model=cost_model,
            acquisition_function=acquisition_function,
            callbacks=callbacks,
        )

        super().__init__(
            scenario=scenario,
            target_function=target_function,
            initial_design=initial_design,
            acquisition_function=acquisition_function,
            overwrite=overwrite,
            callbacks=callbacks,
            **kwargs,
        )

    def _prepare_callbacks(
        self,
        scenario: Scenario,
        cost_model: AbstractModel,
        acquisition_function: AbstractAcquisitionFunction | None,
        callbacks: list[Callback],
    ) -> tuple[list[Callback], AbstractAcquisitionFunction]:
        """Assembles cost-surrogate, budget-exhaustion, and cost-update callbacks.

        Parameters
        ----------
        scenario : Scenario
        cost_model : AbstractModel
            The resolved cost model.
        acquisition_function : AbstractAcquisitionFunction | None
            The user-provided acquisition function, or None for the default.
        callbacks : list[Callback]
            The existing callbacks list to extend.

        Returns
        -------
        callbacks : list[Callback]
            The extended callbacks list.
        acquisition_function : AbstractAcquisitionFunction
            The resolved acquisition function.
        """
        callbacks = list(callbacks)

        # Ensure a CostSurrogateCallback exists
        cost_surrogate_cb = next((cb for cb in callbacks if isinstance(cb, CostSurrogateCallback)), None)
        if cost_surrogate_cb is None:
            logger.debug("No CostSurrogateCallback found, creating default.")
            cost_surrogate_cb = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)
            callbacks.append(cost_surrogate_cb)

        # Resolve cost-aware acquisition function
        if acquisition_function is None:
            acquisition_function = self.get_acquisition_function(scenario, cost_surrogate_callback=cost_surrogate_cb)

        # Shared mutable tracker between budget and cost-update callbacks
        cumulative_cost_tracker = [0.0]

        callbacks.append(
            BudgetExhaustedCallback(
                total_resource_budget=self._total_resource_budget,
                cumulative_cost_tracker=cumulative_cost_tracker,
            )
        )

        if isinstance(acquisition_function, CostAwareAcquisitionFunction):
            callbacks.append(
                UpdateCostCallback(
                    acquisition_function=acquisition_function,
                    total_budget=self._total_resource_budget,
                    initial_design_budget=self._initial_design_budget,
                    cumulative_cost_tracker=cumulative_cost_tracker,
                )
            )

        return callbacks, acquisition_function

    def _update_dependencies(self) -> None:
        """Ensures runhistory is propagated to cost-aware initial design."""
        super()._update_dependencies()

        if isinstance(self._initial_design, CostAwareInitialDesign):
            self._initial_design._runhistory = self._runhistory

    def _validate(self) -> None:
        """Validates the SMBO configuration for cost-aware optimization."""
        super()._validate()

        if self._total_resource_budget <= 0:
            raise ValueError("The `total_resource_budget` must be greater than zero.")

    @staticmethod
    def get_cost_model(
        scenario: Scenario,
        *,
        cost_model: AbstractModel | None = None,
        cost_formula: Callable | None = None,
    ) -> AbstractModel:
        """Returns a surrogate cost model.

        If neither ``cost_model`` nor ``cost_formula`` is provided, a default random forest
        model is created via ``HyperparameterOptimizationFacade.get_model``.

        Parameters
        ----------
        scenario : Scenario
        cost_model : AbstractModel | None, defaults to None
            An explicit cost model instance. Mutually exclusive with ``cost_formula``.
        cost_formula : Callable | None, defaults to None
            A callable that calculates the cost of a configuration.
            Mutually exclusive with ``cost_model``.

        Returns
        -------
        AbstractModel
            The resolved cost model instance.

        Raises
        ------
        ValueError
            If both ``cost_model`` and ``cost_formula`` are provided.
        """
        if cost_model is not None and cost_formula is not None:
            raise ValueError("Cannot provide both `cost_model` and `cost_formula`.")

        if cost_model is not None:
            return cost_model

        if cost_formula is not None:
            return HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)

        # Default: use a random forest surrogate trained on observed costs
        from smac.facade.hyperparameter_optimization_facade import (
            HyperparameterOptimizationFacade,
        )

        return HyperparameterOptimizationFacade.get_model(scenario)

    @staticmethod
    def get_acquisition_function(  # type: ignore
        scenario: Scenario,
        *,
        cost_model: AbstractModel | None = None,
        cost_formula: Callable | None = None,
        cost_surrogate_callback: CostSurrogateCallback | None = None,
        xi: float = 0.0,
    ) -> CostAwareAcquisitionFunction:
        """Returns a cost-aware Expected Improvement acquisition function.

        The acquisition function wraps EI and uses a cost surrogate to weight
        configurations by their predicted cost, favoring cheap-to-evaluate regions
        (EI-Cool strategy).

        Parameters
        ----------
        scenario : Scenario
        cost_model : AbstractModel | None, defaults to None
            A model for predicting configuration costs. Mutually exclusive with ``cost_formula``.
            Ignored if ``cost_surrogate_callback`` is provided.
        cost_formula : Callable | None, defaults to None
            A callable that calculates the cost of a configuration. Mutually exclusive with
            ``cost_model``. Ignored if ``cost_surrogate_callback`` is provided.
        cost_surrogate_callback : CostSurrogateCallback | None, defaults to None
            A pre-configured cost surrogate callback. If provided, ``cost_model`` and
            ``cost_formula`` are ignored.
        xi : float, defaults to 0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        if cost_surrogate_callback is None:
            cost_model = CostAwareFacade.get_cost_model(scenario, cost_model=cost_model, cost_formula=cost_formula)
            cost_surrogate_callback = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

        return CostAwareAcquisitionFunction(
            acquisition_function=EI(xi=xi),
            cost_surrogate_callback=cost_surrogate_callback,
        )

    @staticmethod
    def get_initial_design(  # type: ignore
        scenario: Scenario,
        *,
        cost_model: AbstractModel | None = None,
        cost_formula: Callable | None = None,
        initial_budget: float,
        candidate_pool_size: int = 1000,
        n_bootstrap_points: int = 1,
        additional_configs: list[Configuration] | None = None,
    ) -> CostAwareInitialDesign:
        """Returns a cost-aware initial design.

        The initial design selects configurations that are diverse and cheap, staying within
        a given initial budget.

        Parameters
        ----------
        scenario : Scenario
        cost_model : AbstractModel | None, defaults to None
            A model for predicting configuration costs. Mutually exclusive with ``cost_formula``.
        cost_formula : Callable | None, defaults to None
            A callable that calculates the cost of a configuration. Mutually exclusive with
            ``cost_model``.
        initial_budget : float
            Resource budget allocated for the initial design phase.
        candidate_pool_size : int, defaults to 1000
            Number of candidate configurations to generate in each elimination round.
        n_bootstrap_points : int, defaults to 1
            Number of random configurations to sample before the elimination process.
        additional_configs : list[Configuration] | None, defaults to None
            Adds additional configurations to the initial design.
        """
        cost_model = CostAwareFacade.get_cost_model(scenario, cost_model=cost_model, cost_formula=cost_formula)

        return CostAwareInitialDesign(
            scenario=scenario,
            cost_model=cost_model,
            initial_budget=initial_budget,
            candidate_pool_size=candidate_pool_size,
            n_bootstrap_points=n_bootstrap_points,
        )

    @staticmethod
    def _wrap_target_function(target_function: Callable) -> Callable:
        """Wraps a target function that returns ``{"performance": ..., "cost": ...}``
        into the SMAC-expected format ``(cost, additional_info)``.

        Parameters
        ----------
        target_function : Callable
            The original target function returning a dict with ``"performance"``
            and ``"cost"`` keys.

        Returns
        -------
        Callable
            The wrapped function returning ``(performance, {"resource_cost": cost})``.
        """

        @wraps(target_function)
        def wrapper(config: Configuration, **kwargs: Any) -> tuple[float, dict[str, float]]:
            result = target_function(config, **kwargs)
            performance, cost = result["performance"], result["cost"]
            additional_info = {"resource_cost": cost}

            return performance, additional_info

        return wrapper
