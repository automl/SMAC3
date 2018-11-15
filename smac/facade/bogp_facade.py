import numpy as np
import george

from smac.facade.smac_facade import SMAC
from smac.epm.gp_default_priors import DefaultPrior
from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC, GaussianProcess
from smac.utils.util_funcs import get_types, get_rng


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class BOGP(SMAC):
    """
    Facade to use BORF default mode

    see smac.facade.smac_Facade for API
    This facade overwrites option available via the SMAC facade

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

    def __init__(self, model_type='gp_mcmc', **kwargs):
        """
        Constructor
        see ~smac.facade.smac_facade for documentation
        """
        scenario = kwargs['scenario']
        if scenario.initial_incumbent not in ['LHD', 'FACTORIAL', 'SOBOL']:
            scenario.initial_incumbent = 'SOBOL'

        if scenario.transform_y is 'NONE':
            scenario.transform_y = "LOGS"

        if kwargs.get('model') is None:
            _, rng = get_rng(rng=kwargs.get("rng", None), run_id=kwargs.get("run_id", None), logger=None)

            cov_amp = 2
            types, bounds = get_types(kwargs['scenario'].cs, instance_features=None)
            n_dims = len(types)

            initial_ls = np.ones([n_dims])
            exp_kernel = george.kernels.Matern52Kernel(initial_ls, ndim=n_dims)
            kernel = cov_amp * exp_kernel

            prior = DefaultPrior(len(kernel) + 1, rng=rng)

            n_hypers = 3 * len(kernel)
            if n_hypers % 2 == 1:
                n_hypers += 1

            if model_type == "gp":
                model = GaussianProcess(
                    types=types,
                    bounds=bounds,
                    kernel=kernel,
                    prior=prior,
                    rng=rng,
                    normalize_output=True,
                    normalize_input=True,
                )
            elif model_type == "gp_mcmc":
                model = GaussianProcessMCMC(
                    types=types,
                    bounds=bounds,
                    kernel=kernel,
                    prior=prior,
                    n_hypers=n_hypers,
                    chain_length=200,
                    burnin_steps=100,
                    normalize_input=True,
                    normalize_output=True,
                    rng=rng,
                )
            kwargs['model'] = model
        super().__init__(**kwargs)

        if self.solver.scenario.n_features > 0:
            raise NotImplementedError("BOGP cannot handle instances")

        self.logger.info(self.__class__)

        self.solver.random_configuration_chooser.prob = 0.0

        # only 1 configuration per SMBO iteration
        self.solver.scenario.intensification_percentage = 1e-10
        self.solver.intensifier.min_chall = 1

        # better improve acqusition function optimization
        # 1. increase number of sls iterations
        self.solver.acq_optimizer.n_sls_iterations = 100
        # 2. more randomly sampled configurations
        self.solver.scenario.acq_opt_challengers = 1000

        # activate predict incumbent
        self.solver.predict_incumbent = True
