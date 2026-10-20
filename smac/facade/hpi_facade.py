from __future__ import annotations

from smac.acquisition.maximizer.hpi_random_search import HPIRandomSearch
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade,
)
from smac.main.config_selector import ConfigSelector
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.multi_objective.parego import ParEGO
from smac.scenario import Scenario

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class HPIFacade(HyperparameterOptimizationFacade):
    """Facade for HPI-ParEGO (Theodorakopoulos et al., 2026).

    Builds on the ``HyperparameterOptimizationFacade`` but replaces the acquisition maximizer by
    ``HPIRandomSearch``, which uses HyperSHAP to dynamically restrict the search to the currently
    most important hyperparameters. For multiple objectives, ``ParEGO`` is used as the multi-objective
    algorithm. The ``HPIRandomSearch`` is a sorted random search.

    Note
    ----
    Requires the optional ``hypershap`` dependency: ``pip install smac[hpi]``.
    """

    @staticmethod
    def get_acquisition_maximizer(  # type: ignore
        scenario: Scenario,
        *,
        challengers: int = 5000,
        threshold: float | list[float] | None = None,
        fixing_strategy: str = "incumbent",
        random_prob: float = 0.1,
    ) -> HPIRandomSearch:
        """Returns ``HPIRandomSearch`` as acquisition maximizer.

        Parameters
        ----------
        scenario : Scenario
        challengers : int, defaults to 5000
            Number of challengers.
        threshold : float | list[float] | None, defaults to None
            See ``HPIRandomSearch``. ``None`` reduces the configuration space during the middle third of the trials.
        fixing_strategy : str, defaults to "incumbent"
            See ``HPIRandomSearch``.
        random_prob : float, defaults to 0.1
            See ``HPIRandomSearch``.
        """
        return HPIRandomSearch(
            scenario.configspace,
            challengers=challengers,
            seed=scenario.seed,
            n_trials=scenario.n_trials,
            threshold=threshold,
            fixing_strategy=fixing_strategy,
            random_prob=random_prob,
        )

    @staticmethod
    def get_multi_objective_algorithm(  # type: ignore
        scenario: Scenario,
        *,
        reweigh: int = 5,
    ) -> ParEGO | MeanAggregationStrategy:
        """Returns ``ParEGO`` for multi-objective optimization, and the mean aggregation otherwise.

        Parameters
        ----------
        scenario : Scenario
        reweigh : int, defaults to 5
            Resample the ParEGO scalarization weights only every ``reweigh``-th iteration, to give the optimizer
            time to exploit a scalarization before resampling.
        """
        if scenario.count_objectives() == 1:
            return MeanAggregationStrategy(scenario=scenario)

        return ParEGO(scenario, reweigh=reweigh)

    @staticmethod
    def get_config_selector(
        scenario: Scenario,
        *,
        retrain_after: int | None = 2,
        retrain_wallclock_ratio: int | None = None,
        retries: int = 16,
    ) -> ConfigSelector:
        """Returns the configuration selector. The surrogate model is retrained after every two configurations by
        default so that the hyperparameter importance estimates stay up to date.

        Parameters
        ----------
        scenario : Scenario
        retrain_after : int | None, defaults to 2
        retrain_wallclock_ratio : int | None, defaults to None
        """
        return ConfigSelector(
            scenario,
            retrain_after=retrain_after,
            retrain_wallclock_ratio=retrain_wallclock_ratio,
            max_new_config_tries=retries,
        )
