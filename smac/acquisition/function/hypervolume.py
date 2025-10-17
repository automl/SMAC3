from __future__ import annotations

from typing import Any

import numpy as np
import pygmo
from ConfigSpace import Configuration

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.runhistory import RunHistory
from smac.runhistory.encoder import AbstractRunHistoryEncoder
from smac.utils.logging import get_logger
from smac.utils.multi_objective import normalize_costs

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractHVI(AbstractAcquisitionFunction):
    def __init__(self) -> None:
        """Computes for a given x the predicted hypervolume improvement as
        acquisition value.
        """
        super(AbstractHVI, self).__init__()
        self._required_updates = ("model",)
        self._reference_point = None
        self._objective_bounds = None

        self._runhistory: RunHistory | None = None
        self._runhistory_encoder: AbstractRunHistoryEncoder | None = None

        self._population_hv: float | None = None
        self._population_costs: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Return name of the acquisition function."""
        return "Abstract Hypervolume Improvement"

    def _update(self, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        incumbents: list[Configuration]
            List of incumbent configurations to compute the predicted improvement over.
        runhistory : RunHistory
            Needed to dynamically obtain the objective bounds used for normalisation.
        runhistory_encoder : AbstractRunHistoryEncoder
            Needed to dynamically obtain the objective bounds used for normalisation.
        """
        super(AbstractHVI, self)._update(**kwargs)

        incumbents: list[Configuration] = kwargs.get("incumbents", None)
        if incumbents is None:
            raise ValueError("Incumbents are not passed properly.")
        if len(incumbents) == 0:
            raise ValueError(
                "No incumbents here. Did the intensifier properly update the incumbents in the runhistory?"
            )

        self._runhistory = kwargs.get("runhistory")
        self._runhistory_encoder = kwargs.get("runhistory_encoder")
        assert self._runhistory is not None, "Did you update the AF with the runhistory?"
        assert self._runhistory_encoder is not None, "Did you update the AF with the runhistory encoder?"

        objective_bounds = np.array(self._runhistory.objective_bounds)
        self._objective_bounds = self._runhistory_encoder.transform_response_values(objective_bounds)
        self._reference_point = [1.1] * len(self._objective_bounds)  # type: ignore[arg-type,assignment]

    def get_hypervolume(self, points: np.ndarray) -> float:
        """
        Compute the hypervolume

        Parameters
        ----------
        points : np.ndarray
            A 2d numpy array. 1st dimension is an entity and the 2nd dimension are the costs
        reference_point : list

        Return
        ------
        hypervolume: float
        """
        # Normalize the objectives here to give equal attention to the objectives when computing the HV
        points = [normalize_costs(p, self._objective_bounds) for p in points]
        hv = pygmo.hypervolume(points)
        return hv.compute(self._reference_point)

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the PHVI values and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected HV Improvement of X
        """
        assert self.model is not None, "Did you update the AF with the model?"
        assert self._population_costs is not None
        assert self._population_hv is not None

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        # TODO non-dominated sorting of costs. Compute EHVI only until the EHVI is not expected to improve anymore.
        # Option 1: Supplement missing instances of population with acq. function to get predicted performance over
        # all instances. Idea is this prevents optimizing for the initial instances which get it stuck in local optima
        # Option 2: Only on instances of population
        # Option 3: EVHI per instance and aggregate afterwards
        mean, var_ = self.model.predict_marginalized(X)  # Expected to be not normalized

        phvi = np.zeros(len(X))
        for i, indiv in enumerate(mean):
            points = list(self._population_costs) + [indiv]
            hv = self.get_hypervolume(points)
            phvi[i] = hv - self._population_hv

        return phvi.reshape(-1, 1)


class PHVI(AbstractHVI):
    def __init__(self) -> None:
        super(PHVI, self).__init__()

    @property
    def name(self) -> str:
        """Return name of the acquisition function."""
        return "Predicted Hypervolume Improvement"

    def _update(self, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        Parameters
        ----------
        incumbents: list[Configuration]
            List of incumbent configurations to compute the predicted improvement over.
        runhistory : RunHistory
            Needed to dynamically obtain the objective bounds used for normalisation.
        runhistory_encoder : AbstractRunHistoryEncoder
            Needed to dynamically obtain the objective bounds used for normalisation.
        """
        super(PHVI, self)._update(**kwargs)
        assert self.model is not None, "Did you update the AF with the model?"
        incumbents: list[Configuration] = kwargs.get("incumbents", None)

        # Update PHVI
        # Prediction all
        population_configs = incumbents
        population_X = np.array([config.get_array() for config in population_configs])
        population_costs, _ = self.model.predict_marginalized(population_X)

        # Compute HV
        population_hv = self.get_hypervolume(population_costs)

        self._population_costs = population_costs
        self._population_hv = population_hv

        logger.info(f"New population HV: {population_hv}")

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the PHVI values and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Predicted HV Improvement of X
        """
        assert self.model is not None, "Did you update the AF with the model?"
        assert self._population_costs is not None
        assert self._population_hv is not None

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        mean, _ = self.model.predict_marginalized(X)  # Expected to be not normalized
        phvi = np.zeros(len(X))
        for i, indiv in enumerate(mean):
            points = list(self._population_costs) + [indiv]
            hv = self.get_hypervolume(points)
            phvi[i] = hv - self._population_hv

        return phvi.reshape(-1, 1)
