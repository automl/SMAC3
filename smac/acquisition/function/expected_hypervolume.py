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

# import torch
# from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
# from botorch.models.model import Model
# from botorch.utils.multi_objective.box_decompositions.non_dominated import (
#     NondominatedPartitioning,
# )

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)

# class _PosteriorProxy(object):
#     def __init__(self) -> None:
#         self.mean: Tensor = []
#         self.variance: Tensor = []

# class _ModelProxy(Model, ABC):
#     def __init__(self, model: AbstractModel, objective_bounds: list[tuple[float, float]]):
#         super(_ModelProxy).__init__()
#         self.model = model
#         self._objective_bounds = objective_bounds
#
#     def posterior(self, X: Tensor, **kwargs: Any) -> _PosteriorProxy:
#         """Docstring
#         X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
#         feature space, `q` is the number of points considered jointly,
#         and `b` is the batch dimension.
#
#
#         A `Posterior` object, representing a batch of `b` joint distributions
#         over `q` points and `m` outputs each.
#         """
#         assert X.shape[1] == 1
#         X = X.reshape([X.shape[0], -1]).numpy()  # 3D -> 2D
#
#         # predict
#         # start_time = time.time()
#         # print(f"Start predicting ")
#         mean, var_ = self.model.predict_marginalized(X)
#         normalized_mean = np.array([normalize_costs(m, self._objective_bounds) for m in mean])
#         scale = normalized_mean / mean
#         var_ *= scale  # Scale variance accordingly
#         mean = normalized_mean
#         # print(f"Done in {time.time() - start_time}s")
#         post = _PosteriorProxy()
#         post.mean = torch.asarray(mean).reshape(X.shape[0], 1, -1)  # 2D -> 3D
#         post.variance = torch.asarray(var_).reshape(X.shape[0], 1, -1)  # 2D -> 3D
#
#         return post


class AbstractHVI(AbstractAcquisitionFunction):
    def __init__(self):
        """Computes for a given x the predicted hypervolume improvement as
        acquisition value.
        """
        super(AbstractHVI, self).__init__()
        self._required_updates = ("model",)
        self._reference_point = None
        self._objective_bounds = None

        self._runhistory: RunHistory | None = None
        self._runhistory_encoder: AbstractRunHistoryEncoder | None = None

    @property
    def runhistory(self) -> RunHistory:
        """Return the runhistory."""
        return self._runhistory

    @runhistory.setter
    def runhistory(self, runhistory: RunHistory):
        self._runhistory = runhistory

    @property
    def runhistory_encoder(self) -> AbstractRunHistoryEncoder:
        """Return the runhistory encoder."""
        return self._runhistory_encoder

    @runhistory_encoder.setter
    def runhistory_encoder(self, runhistory_encoder: AbstractRunHistoryEncoder):
        self._runhistory_encoder = runhistory_encoder

    @property
    def name(self) -> str:
        """Return name of the acquisition function."""
        return "Abstract Hypervolume Improvement"

    def _update(self, **kwargs: Any) -> None:
        super(AbstractHVI, self)._update(**kwargs)

        incumbents: list[Configuration] = kwargs.get("incumbents", None)
        if incumbents is None:
            raise ValueError("Incumbents are not passed properly.")
        if len(incumbents) == 0:
            raise ValueError(
                "No incumbents here. Did the intensifier properly update the incumbents in the runhistory?"
            )

        objective_bounds = np.array(self.runhistory.objective_bounds)
        self._objective_bounds = self.runhistory_encoder.transform_response_values(objective_bounds)
        self._reference_point = [1.1] * len(self._objective_bounds)

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
            points = list(self.population_costs) + [indiv]
            hv = self.get_hypervolume(points)
            phvi[i] = hv - self.population_hv

        # if len(X) == 10000:
        #     for op in ["max", "min", "mean", "median"]:
        #         val = getattr(np, op)(phvi)
        #         print(f"{op:6} - {val}")
        #     time.sleep(1.5)

        return phvi.reshape(-1, 1)


# class EHVI(AbstractHVI):
#     def __init__(self):
#         super(EHVI, self).__init__()
#         self._ehvi: ExpectedHypervolumeImprovement | None = None
#
#     @property
#     def name(self) -> str:
#         return "Expected Hypervolume Improvement"
#
#     def _update(self, **kwargs: Any) -> None:
#         super(EHVI, self)._update(**kwargs)
#         incumbents: list[Configuration] = kwargs.get("incumbents", None)
#
#         # Update EHVI
#         # Prediction all
#         population_configs = incumbents
#         population_X = np.array([config.get_array() for config in population_configs])
#         population_costs, _ = self.model.predict_marginalized(population_X)
#         # Normalize the objectives here to give equal attention to the objectives when computing the HV
#         population_costs = [normalize_costs(p, self._objective_bounds) for p in population_costs]
#
#         # BOtorch EHVI implementation
#         bomodel = _ModelProxy(self.model, self._objective_bounds)
#         # ref_point = pygmo.hypervolume(population_costs).refpoint(
#         #     offset=1
#         # )  # TODO get proper reference points from user/cutoffs
#         ref_point = [1.1] * len(self._objective_bounds)
#         # ref_point = torch.asarray(ref_point)
#         # TODO partition from all runs instead of only population?
#         # TODO NondominatedPartitioning and ExpectedHypervolumeImprovement seem no too difficult to implement natively
#         # TODO pass along RNG
#         # Transfrom the objective space to cells based on the population
#         partitioning = NondominatedPartitioning(torch.asarray(ref_point), torch.asarray(population_costs))
#         self._ehvi = ExpectedHypervolumeImprovement(bomodel, ref_point, partitioning)
#
#     def _compute(self, X: np.ndarray) -> np.ndarray:
#         """Computes the EHVI values and its derivatives.
#
#         Parameters
#         ----------
#         X: np.ndarray(N, D), The input points where the acquisition function
#             should be evaluated. The dimensionality of X is (N, D), with N as
#             the number of points to evaluate at and D is the number of
#             dimensions of one X.
#
#         Returns
#         -------
#         np.ndarray(N,1)
#             Expected HV Improvement of X
#         """
#         if self._ehvi is None:
#             raise ValueError(f"The expected hypervolume improvement is not defined yet. Call self.update.")
#
#         if len(X.shape) == 1:
#             X = X[:, np.newaxis]
#
#         # m, var_ = self.model.predict_marginalized_over_instances(X)
#         # Find a way to propagate the variance into the HV
#         boX = torch.asarray(X).reshape(X.shape[0], 1, -1)  # 2D -> #3D
#         improvements = self._ehvi(boX).numpy().reshape(-1, 1)  # TODO here are the expected hv improvements computed.
#         return improvements
#
#         # TODO non-dominated sorting of costs. Compute EHVI only until the EHVI is not expected to improve anymore.
#         # Option 1: Supplement missing instances of population with acq. function to get predicted performance over
#         # all instances. Idea is this prevents optimizing for the initial instances which get it stuck in local optima
#         # Option 2: Only on instances of population
#         # Option 3: EVHI per instance and aggregate afterwards
#         # ehvi = np.zeros(len(X))
#         # for i, indiv in enumerate(m):
#         #     ehvi[i] = self.get_hypervolume(population_costs + [indiv]) - population_hv
#         #
#         # return ehvi.reshape(-1, 1)


class PHVI(AbstractHVI):
    def __init__(self):
        super(PHVI, self).__init__()
        self.population_hv = None
        self.population_costs = None

    @property
    def name(self) -> str:
        """Return name of the acquisition function."""
        return "Predicted Hypervolume Improvement"

    def _update(self, **kwargs: Any) -> None:
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

        self.population_costs = population_costs
        self.population_hv = population_hv

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

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        mean, _ = self.model.predict_marginalized(X)  # Expected to be not normalized
        phvi = np.zeros(len(X))
        for i, indiv in enumerate(mean):
            points = list(self.population_costs) + [indiv]
            hv = self.get_hypervolume(points)
            phvi[i] = hv - self.population_hv

        return phvi.reshape(-1, 1)
