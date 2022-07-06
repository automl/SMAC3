import typing

from collections import OrderedDict

import gpytorch
import numpy as np
import torch
from botorch.optim.numpy_converter import module_to_array, set_params_with_array
from botorch.optim.utils import _get_extra_mll_args
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.utils.errors import NanError
from scipy import optimize
from scipy.stats.qmc import LatinHypercube

from smac.configspace import ConfigurationSpace
from smac.epm.gp.gpytorch import ExactGPModel, GPyTorchGaussianProcess
from smac.epm.gp.kernels.boing import FITCKernel, FITCMean
from smac.epm.utils import check_subspace_points

gpytorch.settings.debug.off()


class AugmentedLocalGaussianProcess(ExactGP):
    def __init__(
        self,
        X_in: torch.Tensor,
        y_in: torch.Tensor,
        X_out: torch.Tensor,
        y_out: torch.Tensor,
        likelihood: GaussianLikelihood,
        base_covar_kernel: Kernel,
    ):
        """
        An Augmented Local GP, it is dense inside a given subregion and the impact of all other
        points are approximated by a sparse GP.

        Parameters
        ----------
        X_in: torch.Tensor (N_in, D),
            feature vector of the points inside the subregion
        y_in: torch.Tensor (N_in, 1),
            observation inside the subregion
        X_out: torch.Tensor (N_out, D),
            feature vector  of the points outside the subregion
        y_out:torch.Tensor (N_out, 1),
            observation inside the subregion
        likelihood: GaussianLikelihood,
            likelihood of the GP (noise)
        base_covar_kernel: Kernel,
            Covariance Kernel
        X_inducing: torch.tensor,
            position of the inducing points
        """
        X_in = X_in.unsqueeze(-1) if X_in.ndimension() == 1 else X_in
        X_out = X_out.unsqueeze(-1) if X_out.ndimension() == 1 else X_out
        assert X_in.shape[-1] == X_out.shape[-1]

        super(AugmentedLocalGaussianProcess, self).__init__(X_in, y_in, likelihood)

        self._mean_module = ZeroMean()
        self.base_covar = base_covar_kernel

        self.X_out = X_out
        self.y_out = y_out
        self.augmented = False

    def set_augment_module(self, X_inducing: torch.Tensor) -> None:
        """
        Set an augmention module, which will be used later for inference

        Parameters
        ----------
        X_inducing: torch.Tensor(N_inducing, D)
            inducing points, it needs to have the same number of dimension as X_in
        """
        X_inducing = X_inducing.unsqueeze(-1) if X_inducing.ndimension() == 1 else X_inducing
        # assert X_inducing.shape[-1] == self.X_out.shape[-1]
        self.covar_module = FITCKernel(
            self.base_covar, X_inducing=X_inducing, X_out=self.X_out, y_out=self.y_out, likelihood=self.likelihood
        )
        self.mean_module = FITCMean(covar_module=self.covar_module)
        self.augmented = True

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Compute the prior values, if optimize_kernel_hps is set True in the training phases, this model degenerates to
         a vanilla GP model with ZeroMean and base_covar as covariance matrix, otherwise we apply partial sparse GP
         mean and kernels here.
        """
        if not self.augmented:
            # we only optimize for kernel hyperparameters
            covar_x = self.base_covar(x)
            mean_x = self._mean_module(x)
        else:
            covar_x = self.covar_module(x)
            mean_x = self.mean_module(x)
        return MultivariateNormal(mean_x, covar_x)


class VariationalGPModel(gpytorch.models.ApproximateGP):
    """
    A variational GP to compute the position of the inducing points.
    We only compute the position of the inducing points w.r.t. the continuous dimensions
    """

    def __init__(self, kernel: Kernel, X_inducing: torch.Tensor):
        """
        Initialize a Variational GP
        we set the lower bound and upper bounds of indcuing points for numerical hyperparmaters between 0 and 1,
        that is, we constraint the indcuing points to lay inside the subregion.

        Parameters
        ----------
        kernel: Kernel
            kernel of the variational GP, its hyperparameter needs to be fixed when it is by PSGP
        X_inducing: torch.tensor (N_inducing, D)
            inducing points
        """
        variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(X_inducing.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, X_inducing, variational_distribution, learn_inducing_locations=True
        )
        super(VariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

        shape_X_inducing = X_inducing.shape
        lower_X_inducing = torch.zeros([shape_X_inducing[-1]]).repeat(shape_X_inducing[0])
        upper_X_inducing = torch.ones([shape_X_inducing[-1]]).repeat(shape_X_inducing[0])

        self.variational_strategy.register_constraint(
            param_name="inducing_points",
            constraint=Interval(lower_X_inducing, upper_X_inducing, transform=None),
        )
        self.double()

        for p_name, t in self.named_hyperparameters():
            if p_name != "variational_strategy.inducing_points":
                t.requires_grad = False

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Pass compute the posterior mean and varaince given input X

        Parameters
        ----------
        x: torch.Tensor
            Input data
        Returns
        -------
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, cont_only=True)
        return MultivariateNormal(mean_x, covar_x)


class GloballyAugmentedLocalGP(GPyTorchGaussianProcess):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        types: typing.List[int],
        bounds: typing.List[typing.Tuple[float, float]],
        bounds_cont: np.ndarray,
        bounds_cat: typing.List[typing.Tuple],
        seed: int,
        kernel: Kernel,
        num_inducing_points: int = 2,
        likelihood: typing.Optional[GaussianLikelihood] = None,
        normalize_y: bool = True,
        n_opt_restarts: int = 10,
        instance_features: typing.Optional[np.ndarray] = None,
        pca_components: typing.Optional[int] = None,
    ):
        """
        Local GP with global augmentation. It is composed of two models: an Exact GP to descirbe the data
        distribution inside a subregion and an Approximate GP to approxiamte the data distribution outside a
        subregion.

        The GP hyperparameterŝ are obtained by optimizing the marginal log likelihood and optimize with botorch
        We train a PSGP in two stages:
        In the first stage, we only train the kernel hyperparameter and thus deactivate the gradient w.r.t the position
        of hte inducing points.
        In the second stage, we use the kernel hyperparameter acquired in the first stage to initialize a new
        variational gaussian process and only optimize its inducing points position with natural gradients.
        Finally we update the position of the indcuing points and use it for evaluating

        Parameters
        ----------
        bounds_cont: np.ndarray(N_cont, 2),
            bounds of the continuous hyperparameters, store as [[0,1] * N_cont]
        bounds_cat: typing.List[typing.Tuple],
            bounds of categorical hyperparameters
        kernel : gpytorch kernel object
            Specifies the kernel that is used for all Gaussian Process
        num_inducing_points: int
            Number of inducing points
        likelihood: Optional[GaussianLikelihood]
            Likelihood values
        normalize_y : bool
            Zero mean unit variance normalization of the output values, when the model is a partial sparse GP model,
        """
        super(GloballyAugmentedLocalGP, self).__init__(
            configspace=configspace,
            types=types,
            bounds=bounds,
            seed=seed,
            kernel=kernel,
            likelihood=likelihood,
            normalize_y=normalize_y,
            n_opt_restarts=n_opt_restarts,
            instance_features=instance_features,
            pca_components=pca_components,
        )
        self.cont_dims = np.where(np.array(types) == 0)[0]
        self.cat_dims = np.where(np.array(types) != 0)[0]
        self.bounds_cont = bounds_cont
        self.bounds_cat = bounds_cat
        self.num_inducing_points = num_inducing_points

    def update_attribute(self, **kwargs: typing.Dict) -> None:
        """We update the class attribute (for instance, number of inducing points)"""
        for key in kwargs:
            if not hasattr(self, key):
                raise AttributeError(f"{self.__class__.__name__} has no attribute named {key}")
            setattr(self, key, kwargs[key])

    def _train(
        self, X: np.ndarray, y: np.ndarray, do_optimize: bool = True
    ) -> typing.Union[AugmentedLocalGaussianProcess, GPyTorchGaussianProcess]:
        """
        Update the hyperparameters of the partial sparse kernel. Depending on the number of inputs inside and
        outside the subregion, we initalize a  PartialSparseGaussianProcess or a GaussianProcessGPyTorch

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features., N = N_in + N_out
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

        ss_data_indices = check_subspace_points(
            X,
            cont_dims=self.cont_dims,
            cat_dims=self.cat_dims,
            bounds_cont=self.bounds_cont,
            bounds_cat=self.bounds_cat,
            expand_bound=True,
        )

        if np.sum(ss_data_indices) > np.shape(y)[0] - self.num_inducing_points:
            # we initialize a vanilla GaussianProcessGPyTorch
            if self.normalize_y:
                y = self._normalize_y(y)
            self.num_points = np.shape(y)[0]
            get_gp_kwargs = {"X_in": X, "y_in": y, "X_out": None, "y_out": None}
        else:
            # we initialize a PartialSparseGaussianProcess object
            X_in = X[ss_data_indices]
            y_in = y[ss_data_indices]
            X_out = X[~ss_data_indices]
            y_out = y[~ss_data_indices]
            self.num_points = np.shape(y_in)[0]
            if self.normalize_y:
                y_in = self._normalize_y(y_in)
                y_out = (y_out - self.mean_y_) / self.std_y_
            get_gp_kwargs = {"X_in": X_in, "y_in": y_in, "X_out": X_out, "y_out": y_out}

        n_tries = 10

        for i in range(n_tries):
            try:
                self.gp = self._get_gp(**get_gp_kwargs)
                break
            except Exception as e:
                if i == n_tries - 1:
                    raise RuntimeError(f"Fails to initialize a GP model, {e}")

        if do_optimize:
            self.hypers = self._optimize()
            self.gp = set_params_with_array(self.gp, self.hypers, self.property_dict)
            if isinstance(self.gp.model, AugmentedLocalGaussianProcess):
                # we optimize the position of the inducing points and thus needs to deactivate the gradient of kernel
                # hyperparameters
                lhd = LatinHypercube(d=X.shape[-1], seed=self.rng.randint(0, 1000000))

                inducing_points = torch.from_numpy(lhd.random(n=self.num_inducing_points))

                kernel = self.gp.model.base_covar
                var_gp = VariationalGPModel(kernel, X_inducing=inducing_points)

                X_out_ = torch.from_numpy(X_out)
                y_out_ = torch.from_numpy(y_out)

                variational_ngd_optimizer = gpytorch.optim.NGD(
                    var_gp.variational_parameters(), num_data=y_out_.size(0), lr=0.1
                )

                var_gp.train()
                likelihood = GaussianLikelihood().double()
                likelihood.train()

                mll_func = gpytorch.mlls.PredictiveLogLikelihood

                var_mll = mll_func(likelihood, var_gp, num_data=y_out_.size(0))

                for t in var_gp.variational_parameters():
                    t.requires_grad = False

                x0, property_dict, bounds = module_to_array(module=var_mll)
                for t in var_gp.variational_parameters():
                    t.requires_grad = True
                bounds = np.asarray(bounds).transpose().tolist()

                start_points = [x0]

                inducing_idx = 0

                inducing_size = X_out.shape[-1] * self.num_inducing_points
                for p_name, attrs in property_dict.items():
                    if p_name != "model.variational_strategy.inducing_points":
                        # Construct the new tensor
                        if len(attrs.shape) == 0:  # deal with scalar tensors
                            inducing_idx = inducing_idx + 1
                        else:
                            inducing_idx = inducing_idx + np.prod(attrs.shape)
                    else:
                        break
                while len(start_points) < 3:
                    new_start_point = np.random.rand(*x0.shape)
                    new_inducing_points = torch.from_numpy(lhd.random(n=self.num_inducing_points)).flatten()
                    new_start_point[inducing_idx : inducing_idx + inducing_size] = new_inducing_points
                    start_points.append(new_start_point)

                def sci_opi_wrapper(
                    x: np.ndarray,
                    mll: gpytorch.module,
                    property_dict: typing.Dict,
                    train_inputs: torch.Tensor,
                    train_targets: torch.Tensor,
                ) -> typing.Tuple[float, np.ndarray]:
                    """
                    A modification of from botorch.optim.utils._scipy_objective_and_grad, the key difference is that
                    we do an additional natural gradient update before computing the gradient values
                    Parameters
                    ----------
                    x: np.ndarray
                        optimizer input
                    mll: gpytorch.module
                        a gpytorch module whose hyperparameters are defined by x
                    property_dict: typing.Dict
                        a dict describing how x is mapped to initialize mll
                    train_inputs: torch.Tensor (N_input, D)
                        input points of the GP model
                    train_targets: torch.Tensor (N_input, 1)
                        target value of the GP model
                    Returns
                    ----------
                    loss: np.ndarray
                        loss value
                    grad: np.ndarray
                        gradient w.r.t. the inputs
                    ----------
                    """
                    # A modification of from botorch.optim.utils._scipy_objective_and_grad:
                    # https://botorch.org/api/_modules/botorch/optim/utils.html
                    # The key difference is that we do an additional natural gradient update here
                    variational_ngd_optimizer.zero_grad()

                    mll = set_params_with_array(mll, x, property_dict)
                    mll.zero_grad()
                    try:  # catch linear algebra errors in gpytorch
                        output = mll.model(train_inputs)
                        args = [output, train_targets] + _get_extra_mll_args(mll)
                        loss = -mll(*args).sum()
                    except RuntimeError as e:
                        if isinstance(e, NanError) or "singular" in e.args[0]:
                            return float("nan"), np.full_like(x, "nan")
                        else:
                            raise e  # pragma: nocover
                    loss.backward()
                    variational_ngd_optimizer.step()
                    param_dict = OrderedDict(mll.named_parameters())
                    grad = []
                    for p_name in property_dict:
                        t = param_dict[p_name].grad
                        if t is None:
                            # this deals with parameters that do not affect the loss
                            grad.append(np.zeros(property_dict[p_name].shape.numel()))
                        else:
                            grad.append(t.detach().view(-1).cpu().double().clone().numpy())
                    mll.zero_grad()
                    return loss.item(), np.concatenate(grad)

                theta_star = x0
                f_opt_star = np.inf
                for start_point in start_points:
                    try:
                        theta, f_opt, res_dict = optimize.fmin_l_bfgs_b(
                            sci_opi_wrapper,
                            start_point,
                            args=(var_mll, property_dict, X_out_, y_out_),
                            bounds=bounds,
                            maxiter=50,
                        )
                        if f_opt < f_opt_star:
                            f_opt_star = f_opt
                            theta_star = theta
                    except Exception as e:
                        self.logger.warning(f"An exception {e} occurs during the optimizaiton")

                start_idx = 0
                # modification on botorch.optim.numpy_converter.set_params_with_array as we only need to extract the
                # positions of inducing points
                for p_name, attrs in property_dict.items():
                    if p_name != "model.variational_strategy.inducing_points":
                        # Construct the new tensor
                        if len(attrs.shape) == 0:  # deal with scalar tensors
                            start_idx = start_idx + 1
                        else:
                            start_idx = start_idx + np.prod(attrs.shape)
                    else:
                        end_idx = start_idx + np.prod(attrs.shape)
                        X_inducing = torch.tensor(
                            theta_star[start_idx:end_idx], dtype=attrs.dtype, device=attrs.device
                        ).view(*attrs.shape)
                        break
                # set inducing points for covariance module here
                self.gp_model.set_augment_module(X_inducing)
        else:
            self.hypers, self.property_dict, _ = module_to_array(module=self.gp)

        self.is_trained = True
        return self

    def _get_gp(
        self,
        X_in: typing.Optional[np.ndarray] = None,
        y_in: typing.Optional[np.ndarray] = None,
        X_out: typing.Optional[np.ndarray] = None,
        y_out: typing.Optional[np.ndarray] = None,
    ) -> typing.Optional[ExactMarginalLogLikelihood]:
        """
        Construction a new GP model based on the inputs
        If both in and out are None: return an empty models
        If only in_x and in_y are given: return a vanilla GP model
        If in_x, in_y, out_x, out_y are given: return a partial sparse GP model.

        Parameters
        ----------
        X_in: Optional[np.ndarray (N_in, D)]
            Input data points inside the subregion. The dimensionality of X_in is (N_in, D),
            with N_in as the number of points inside the subregion and D is the number of features. If it is not given,
            this function will return None to be compatible with the implementation of its parent class
        y_in: Optional[np.ndarray (N_in,)]
            The corresponding target values inside the subregion.
        X_out: Optional[np.ndarray (N_out, D).
            Input data points ouside the subregion. The dimensionality of X_out is (N_out, D), if it is not given, this
        function will return a vanilla Gaussian Process
        y_out: typing.Optional[np.ndarray (N_out)]
            The corresponding target values outside the subregion.

        Returns
        -------
        mll: ExactMarginalLogLikelihood
            a gp module
        """
        if X_in is None:
            return None

        X_in = torch.from_numpy(X_in)
        y_in = torch.from_numpy(y_in)
        if X_out is None:
            self.gp_model = ExactGPModel(X_in, y_in, likelihood=self.likelihood, base_covar_kernel=self.kernel).double()
        else:
            X_out = torch.from_numpy(X_out)
            y_out = torch.from_numpy(y_out)

            self.gp_model = AugmentedLocalGaussianProcess(
                X_in, y_in, X_out, y_out, likelihood=self.likelihood, base_covar_kernel=self.kernel  # type:ignore
            ).double()
        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        mll.double()
        return mll
