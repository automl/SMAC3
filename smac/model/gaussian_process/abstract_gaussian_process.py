from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import sklearn.gaussian_process
from ConfigSpace import ConfigurationSpace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, KernelOperator

from smac.model.abstract_model import AbstractModel
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior
from smac.model.gaussian_process.priors.tophat_prior import SoftTopHatPrior, TophatPrior
from smac.model.surrogate_transformer import SurrogateTransformer

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class AbstractGaussianProcess(AbstractModel):
    """Abstract base class for all Gaussian process models.

    Parameters
    ----------
    configspace : ConfigurationSpace
    kernel : Kernel
        Kernel which is used for the Gaussian process.
    instance_features : dict[str, list[int | float]] | None, defaults to None
        Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
        on which the model is trained on.
    pca_components : float, defaults to 7
        Number of components to keep when using PCA to reduce dimensionality of instance features.
    seed : int
    normalize_y : bool, defaults to True
        Zero mean unit variance normalization of the output values.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        kernel: Kernel,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
        normalize_y: bool = True,
    ):
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

        self._kernel = kernel
        self._gp = self._get_gaussian_process()
        self.transformer = self.build_transformer(normalize_y)

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"kernel": self._kernel.meta})

        return meta

    def build_transformer(self, normalize_y: bool = False) -> SurrogateTransformer:  # noqa: D102
        return SurrogateTransformer(
            n_hps=self._n_hps,
            n_features=self._n_features,
            instance_features=self._instance_features,
            impute_inactive=self._impute_inactive,
            normalize_y=normalize_y,
            pca_components=self._pca_components,
        )

    @abstractmethod
    def _get_gaussian_process(self) -> GaussianProcessRegressor:
        """Generates a Gaussian process."""
        raise NotImplementedError()

    def _get_all_priors(
        self,
        add_bound_priors: bool = True,
        add_soft_bounds: bool = False,
    ) -> list[list[AbstractPrior]]:
        """Returns all priors."""
        # Obtain a list of all priors for each tunable hyperparameter of the kernel
        all_priors = []
        to_visit = []
        to_visit.append(self._gp.kernel.k1)
        to_visit.append(self._gp.kernel.k2)

        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param, KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                continue
            elif isinstance(current_param, Kernel):
                hps = current_param.hyperparameters
                assert len(hps) == 1

                hp = hps[0]
                if hp.fixed:
                    continue

                bounds = hps[0].bounds
                for i in range(hps[0].n_elements):
                    priors_for_hp = []

                    if current_param.prior is not None:
                        priors_for_hp.append(current_param.prior)

                    if add_bound_priors:
                        if add_soft_bounds:
                            priors_for_hp.append(
                                SoftTopHatPrior(
                                    lower_bound=bounds[i][0],
                                    upper_bound=bounds[i][1],
                                    seed=self._rng.randint(0, 2**20),
                                    exponent=2,
                                )
                            )
                        else:
                            priors_for_hp.append(
                                TophatPrior(
                                    lower_bound=bounds[i][0],
                                    upper_bound=bounds[i][1],
                                    seed=self._rng.randint(0, 2**20),
                                )
                            )
                    all_priors.append(priors_for_hp)

        return all_priors

    def _set_has_conditions(self) -> None:
        """Sets `has_conditions` on `current_param`."""
        has_conditions = len(self._configspace.conditions) > 0
        to_visit = []
        to_visit.append(self._kernel)

        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param, sklearn.gaussian_process.kernels.KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                current_param.has_conditions = has_conditions
            elif isinstance(current_param, sklearn.gaussian_process.kernels.Kernel):
                current_param.has_conditions = has_conditions
            else:
                raise ValueError(current_param)

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        """Imputes inactives."""
        X = X.copy()
        X[~np.isfinite(X)] = -1

        return X
