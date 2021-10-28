import typing
from collections import OrderedDict

import numpy as np
from scipy import optimize

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import Kernel
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from botorch.optim.numpy_converter import module_to_array, set_params_with_array
from botorch.optim.utils import _scipy_objective_and_grad

from smac.configspace import ConfigurationSpace
from smac.utils.constants import VERY_SMALL_NUMBER
from smac.epm.base_gp import BaseModel

gpytorch.settings.debug.off()


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, base_covar_kernel, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = base_covar_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GaussianProcessGPyTorch(BaseModel):
    def __init__(self,
                 configspace: ConfigurationSpace,
                 types: typing.List[int],
                 bounds: typing.List[typing.Tuple[float, float]],
                 seed: int,
                 kernel: Kernel,
                 likelihood: typing.Optional[FixedNoiseGaussianLikelihood] = None,
                 normalize_y: bool = True,
                 n_opt_restarts: int = 10,
                 instance_features: typing.Optional[np.ndarray] = None,
                 pca_components: typing.Optional[int] = None,
                 ):
        """
        A Gaussian Process written with GPyTorch, its interface is written to be compatible with partial sparse gaussian
        process

        Parameters
        ----------
        configspace: ConfigurationSpace
            Configuration space
        types : List[int]
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass [3, 0]. Note that we count starting from 0.
        bounds : List[Tuple[float, float]]
            bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan) for categorical dims
        bounds_cont: np.ndarray
            bounds of continuous hyperparameters
        bounds_cat:  typing.List[typing.List[typing.Tuple]],
            bounds of categorical hyperparameters, need to be flattened, e.g. all the possible categorical needs to
            be listed
        seed : int
            Model seed.
        kernel : Kernel
            Specifies the kernel that is used for all Gaussian Process
        likelihood: typing.Optional[FixedNoiseGaussianLikelihood] = None,
            Gaussian Likelihood (or noise)
        normalize_y : bool
            Zero mean unit variance normalization of the output values
        n_opt_restart : int
            Number of restarts for GP hyperparameter optimization
        instance_features : np.ndarray (I, K)
            Contains the K dimensional instance features of the I different instances
        pca_components : float
            Number of components to keep when using PCA to reduce dimensionality of instance features. Requires to
            set n_feats (> pca_dims).
        """
        super(GaussianProcessGPyTorch, self).__init__(configspace,
                                                      types,
                                                      bounds,
                                                      seed,
                                                      kernel,
                                                      instance_features,
                                                      pca_components,
                                                      )
        self.kernel = kernel
        if likelihood is None:
            noise_prior = HorseshoePrior(0.1)
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=Interval(np.exp(-25), np.exp(2), transform=None)
            ).double()
        self.likelihood = likelihood

        self.cat_dims = np.where(np.array(types) != 0)[0]
        self.cont_dims = np.where(np.array(types) == 0)[0]

        self.normalize_y = normalize_y
        self.n_opt_restarts = n_opt_restarts

        self.hypers = np.empty((0,))
        self.property_dict = OrderedDict()
        self.is_trained = False
        self._n_ll_evals = 0

        self.num_points = 0

    def _train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool = True) -> 'GaussianProcess':
        """
        Computes the Cholesky decomposition of the covariance of X and
        estimates the GP hyperparameters by optimizing the marginal
        loglikelihood. The prior mean of the GP is set to the empirical
        mean of X.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters of the kernel are used.
        """

        X = self._impute_inactive(X)
        if len(y.shape) == 1:
            self.n_objectives_ = 1
        else:
            self.n_objectives_ = y.shape[1]
        if self.n_objectives_ == 1:
            y = y.flatten()

        if self.normalize_y:
            y = self._normalize_y(y)

        n_tries = 10
        for i in range(n_tries):
            try:
                self.gp = self._get_gp(X, y)
                break
            except np.linalg.LinAlgError as e:
                if i == n_tries:
                    raise e
                # Assume that the last entry of theta is the noise
                # theta = np.exp(self.kernel.theta)
                # theta[-1] += 1
                # self.kernel.theta = np.log(theta)

        if do_optimize:
            # self._all_priors = self._get_all_priors(add_bound_priors=False)
            self.hypers = self._optimize()
            self.gp = set_params_with_array(self.gp, self.hypers, self.property_dict)
        else:
            self.hypers, self.property_dict, _ = module_to_array(module=self.gp)
        self.is_trained = True
        return self

    def _get_gp(self,
                X: typing.Optional[np.ndarray] = None,
                y: typing.Optional[np.ndarray] = None) -> typing.Optional[ExactMarginalLogLikelihood]:
        """
        Get the GP model with the given X and y values, as GPyTorch requires the input data to initialize a new
        model, we also pass X and y here

        Parameters
        -------
        X: typing.Optional[np.ndarray]
            input feature vectors
        y: typing.Optional[np.ndarray]
            input observations
        Returns
        -------
        mll : typing.Optional[ExactMarginalLogLikelihood]
            a GPyTorch model
        """

        if X is None:
            return None

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        self.gp_model = ExactGPModel(X, y, likelihood=self.likelihood, base_covar_kernel=self.kernel).double()

        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        mll.double()
        return mll

    def _optimize(self) -> np.ndarray:
        """
        Optimizes the marginal log likelihood and returns the best found
        hyperparameter configuration theta.

        Returns
        -------
        theta : np.ndarray(H)
            Hyperparameter vector that maximizes the marginal log likelihood
        """
        x0, property_dict, bounds = module_to_array(module=self.gp)

        bounds = np.asarray(bounds).transpose().tolist()
        # x0 = x0.astype(np.float64)

        self.property_dict = property_dict

        p0 = [x0]

        while len(p0) < self.n_opt_restarts:
            try:
                self.gp.pyro_sample_from_prior()
                sample, _, _ = module_to_array(module=self.gp)
                p0.append(sample.astype(np.float64))
            except Exception as e:
                continue

        self.gp_model.train()
        self.likelihood.train()

        theta_star = x0
        f_opt_star = np.inf
        for i, start_point in enumerate(p0):
            try:
                theta, f_opt, _ = optimize.fmin_l_bfgs_b(
                    _scipy_objective_and_grad,
                    start_point,
                    args=(self.gp, property_dict),
                    bounds=bounds,
                )
            except RuntimeError as e:
                self.logger.warning(f"Fail to optimize as an Error occurs: {e}")
                continue
            if f_opt < f_opt_star:
                f_opt_star = f_opt
                theta_star = theta
        return theta_star

    def _predict(self, X_test: np.ndarray,
                 cov_return_type: typing.Optional[str] = 'diagonal_cov') \
            -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        cov_return_type: typing.Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,) or np.array(N, N) or None
            predictive variance or standard deviation

        """
        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        X_test = torch.from_numpy(self._impute_inactive(X_test))
        self.likelihood.eval()
        self.gp_model.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.gp_model(X_test))

            mu = observed_pred.mean.numpy()
            if cov_return_type is None:
                var = None

                if self.normalize_y:
                    mu = self._untransform_y(mu)

            else:
                if cov_return_type != 'full_cov':
                    var = observed_pred.stddev.numpy()
                    var = var ** 2  # since we get standard deviation for faster computation
                else:
                    # output full covariance
                    var = observed_pred.covariance_matrix().numpy()

                # Clip negative variances and set them to the smallest
                # positive float value
                var = np.clip(var, VERY_SMALL_NUMBER, np.inf)

                if self.normalize_y:
                    mu, var = self._untransform_y(mu, var)

                if cov_return_type == 'diagonal_std':
                    var = np.sqrt(var)  # converting variance to std deviation if specified

        return mu, var

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        """
        Samples F function values from the current posterior at the N
        specified test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        n_funcs: int
            Number of function values that are drawn at each test point.

        Returns
        ----------
        function_samples: np.array(F, N)
            The F function values drawn at the N test points.
        """

        if not self.is_trained:
            raise Exception('Model has to be trained first!')
        self.likelihood.eval()
        self.gp_model.eval()

        X_test = torch.from_numpy(self._impute_inactive(X_test))
        with torch.no_grad():
            funcs = self.likelihood(self.gp_model(X_test)).sample(torch.Size([n_funcs])).t().cpu().detach().numpy()

        if self.normalize_y:
            funcs = self._untransform_y(funcs)

        if len(funcs.shape) == 1:
            return funcs[None, :]
        else:
            return funcs
