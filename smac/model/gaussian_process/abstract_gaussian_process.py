from __future__ import annotations

import numpy as np
import sklearn.gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, KernelOperator

import smac.model.gaussian_process.priors
from ConfigSpace import ConfigurationSpace
from smac.model.abstract_model import AbstractModel
from smac.model.gaussian_process.priors.prior import Prior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class AbstractGaussianProcess(AbstractModel):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        kernel: Kernel,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ):
        """Abstract base class for all Gaussian process models."""
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

        self.kernel = kernel
        self.gp = self._get_gp()

    def _get_gp(self) -> GaussianProcessRegressor:
        """Returns the Gaussian process."""
        raise NotImplementedError()

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean unit standard deviation.

        Parameters
        ----------
        y : np.ndarray
            Targets for the Gaussian process

        Returns
        -------
        np.ndarray
        """
        self.mean_y_ = np.mean(y)
        self.std_y_ = np.std(y)
        if self.std_y_ == 0:
            self.std_y_ = 1
        return (y - self.mean_y_) / self.std_y_

    def _untransform_y(
        self,
        y: np.ndarray,
        var: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Transform zeromean unit standard deviation data into the regular space.

        This function should be used after a prediction with the Gaussian process which was
        trained on normalized data.

        Parameters
        ----------
        y : np.ndarray
            Normalized data.
        var : np.ndarray (optional)
            Normalized variance

        Returns
        -------
        np.ndarray on tuple[np.ndarray, np.ndarray]
        """
        y = y * self.std_y_ + self.mean_y_
        if var is not None:
            var = var * self.std_y_**2
            return y, var  # type: ignore
        return y

    def _get_all_priors(
        self,
        add_bound_priors: bool = True,
        add_soft_bounds: bool = False,
    ) -> list[list[Prior]]:
        """Returns all priors."""
        # Obtain a list of all priors for each tunable hyperparameter of the kernel
        all_priors = []
        to_visit = []
        to_visit.append(self.gp.kernel.k1)
        to_visit.append(self.gp.kernel.k2)
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
                                smac.model.gaussian_process.priors.SoftTopHatPrior(
                                    lower_bound=bounds[i][0],
                                    upper_bound=bounds[i][1],
                                    seed=self.rng.randint(0, 2**20),
                                    exponent=2,
                                )
                            )
                        else:
                            priors_for_hp.append(
                                smac.model.gaussian_process.priors.TophatPrior(
                                    lower_bound=bounds[i][0],
                                    upper_bound=bounds[i][1],
                                    seed=self.rng.randint(0, 2**20),
                                )
                            )
                    all_priors.append(priors_for_hp)
        return all_priors

    def _set_has_conditions(self) -> None:
        """Sets `has_conditions` on `current_param`."""
        has_conditions = len(self.configspace.get_conditions()) > 0
        to_visit = []
        to_visit.append(self.kernel)
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
