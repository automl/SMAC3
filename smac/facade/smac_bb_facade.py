import typing

import numpy as np

from smac.epm.base_gp import BaseModel
from smac.epm.gaussian_process_mcmc import GaussianProcess, GaussianProcessMCMC
from smac.epm.gp_base_prior import HorseshoePrior, LognormalPrior
from smac.epm.gp_kernels import ConstantKernel, HammingKernel, Matern, WhiteKernel
from smac.epm.util_funcs import get_rng, get_types
from smac.facade.smac_ac_facade import SMAC4AC
from smac.initial_design.sobol_design import SobolDesign
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class SMAC4BB(SMAC4AC):
    """Facade to use SMAC for Black-Box optimization using a GP.

    see smac.facade.smac_Facade for API
    This facade overwrites options available via the SMAC facade

    Hyperparameters are chosen according to the best configuration for Gaussian process maximum likelihood found in
    "Towards Assessing the Impact of Bayesian Optimization's Own Hyperparameters" by Lindauer et al., presented at the
    DSO workshop 2019 (https://arxiv.org/abs/1908.06674).

    Changes are:

    * Instead of having an initial design of size 10*D as suggested by Jones et al. 1998 (actually, they suggested
      10*D+1), we use an initial design of 8*D.
    * More restrictive lower and upper bounds on the length scale for the Matern and Hamming Kernel than the ones
      suggested by Klein et al. 2017 in the RoBO package. In practice, they are ``np.exp(-6.754111155189306)``
      instead of ``np.exp(-10)`` for the lower bound and ``np.exp(0.0858637988771976)`` instead of
      ``np.exp(2)`` for the upper bound.
    * The initial design is set to be a Sobol grid
    * The random fraction is set to ``0.08447232371720552``, it was ``0.0`` before.

    See Also
    --------
    :class:`~smac.facade.smac_ac_facade.SMAC4AC` for documentation of parameters.

    Attributes
    ----------
    logger
    stats : Stats
    solver : SMBO
    runhistory : RunHistory
        List with information about previous runs
    trajectory : list
        List of all incumbents
    """

    def __init__(self, model_type: str = "gp_mcmc", **kwargs: typing.Any):
        scenario = kwargs["scenario"]

        if len(scenario.cs.get_hyperparameters()) <= 21201:
            kwargs["initial_design"] = kwargs.get("initial_design", SobolDesign)
        else:
            raise ValueError(
                'The default initial design "Sobol sequence" can only handle up to 21201 dimensions. '
                'Please use a different initial design, such as "the Latin Hypercube design".',
            )
        kwargs["runhistory2epm"] = kwargs.get("runhistory2epm", RunHistory2EPM4Cost)

        init_kwargs = kwargs.get("initial_design_kwargs", dict()) or dict()
        init_kwargs["n_configs_x_params"] = init_kwargs.get("n_configs_x_params", 8)
        init_kwargs["max_config_fracs"] = init_kwargs.get("max_config_fracs", 0.25)
        kwargs["initial_design_kwargs"] = init_kwargs

        if kwargs.get("model") is None:

            model_kwargs = kwargs.get("model_kwargs", dict()) or dict()

            _, rng = get_rng(
                rng=kwargs.get("rng", None),
                run_id=kwargs.get("run_id", None),
                logger=None,
            )

            types, bounds = get_types(kwargs["scenario"].cs, instance_features=None)

            cov_amp = ConstantKernel(
                2.0,
                constant_value_bounds=(np.exp(-10), np.exp(2)),
                prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
            )

            cont_dims = np.where(np.array(types) == 0)[0]
            cat_dims = np.where(np.array(types) != 0)[0]

            if len(cont_dims) > 0:
                exp_kernel = Matern(
                    np.ones([len(cont_dims)]),
                    [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cont_dims))],
                    nu=2.5,
                    operate_on=cont_dims,
                )

            if len(cat_dims) > 0:
                ham_kernel = HammingKernel(
                    np.ones([len(cat_dims)]),
                    [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cat_dims))],
                    operate_on=cat_dims,
                )

            assert (len(cont_dims) + len(cat_dims)) == len(scenario.cs.get_hyperparameters())

            noise_kernel = WhiteKernel(
                noise_level=1e-8,
                noise_level_bounds=(np.exp(-25), np.exp(2)),
                prior=HorseshoePrior(scale=0.1, rng=rng),
            )

            if len(cont_dims) > 0 and len(cat_dims) > 0:
                # both
                kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
            elif len(cont_dims) > 0 and len(cat_dims) == 0:
                # only cont
                kernel = cov_amp * exp_kernel + noise_kernel
            elif len(cont_dims) == 0 and len(cat_dims) > 0:
                # only cont
                kernel = cov_amp * ham_kernel + noise_kernel
            else:
                raise ValueError()

            if model_type == "gp":
                model_class = GaussianProcess  # type: typing.Type[BaseModel]
                kwargs["model"] = model_class
                model_kwargs["kernel"] = kernel
                model_kwargs["normalize_y"] = True
                model_kwargs["seed"] = rng.randint(0, 2**20)
            elif model_type == "gp_mcmc":
                model_class = GaussianProcessMCMC
                kwargs["model"] = model_class
                kwargs["integrate_acquisition_function"] = True

                model_kwargs["kernel"] = kernel

                n_mcmc_walkers = 3 * len(kernel.theta)
                if n_mcmc_walkers % 2 == 1:
                    n_mcmc_walkers += 1
                model_kwargs["n_mcmc_walkers"] = n_mcmc_walkers
                model_kwargs["chain_length"] = 250
                model_kwargs["burnin_steps"] = 250
                model_kwargs["normalize_y"] = True
                model_kwargs["seed"] = rng.randint(0, 2**20)
            else:
                raise ValueError("Unknown model type %s" % model_type)
            kwargs["model_kwargs"] = model_kwargs

        if kwargs.get("random_configuration_chooser") is None:
            random_config_chooser_kwargs = (
                kwargs.get(
                    "random_configuration_chooser_kwargs",
                    dict(),
                )
                or dict()
            )
            random_config_chooser_kwargs["prob"] = random_config_chooser_kwargs.get("prob", 0.08447232371720552)
            kwargs["random_configuration_chooser_kwargs"] = random_config_chooser_kwargs

        if kwargs.get("acquisition_function_optimizer") is None:
            acquisition_function_optimizer_kwargs = (
                kwargs.get(
                    "acquisition_function_optimizer_kwargs",
                    dict(),
                )
                or dict()
            )
            acquisition_function_optimizer_kwargs["n_sls_iterations"] = 10
            kwargs["acquisition_function_optimizer_kwargs"] = acquisition_function_optimizer_kwargs

        # only 1 configuration per SMBO iteration
        intensifier_kwargs = kwargs.get("intensifier_kwargs", dict()) or dict()
        intensifier_kwargs["min_chall"] = 1
        kwargs["intensifier_kwargs"] = intensifier_kwargs
        scenario.intensification_percentage = 1e-10

        super().__init__(**kwargs)

        if self.solver.scenario.n_features > 0:
            raise NotImplementedError("BOGP cannot handle instances")

        self.logger.info(self.__class__)

        self.solver.scenario.acq_opt_challengers = 1000  # type: ignore[attr-defined] # noqa F821
        # activate predict incumbent
        self.solver.epm_chooser.predict_x_best = True
