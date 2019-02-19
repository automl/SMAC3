import numpy as np
import george

from smac.facade.smac_facade import SMAC
from smac.epm.gp_default_priors import DefaultPrior
from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC, GaussianProcess
from smac.utils.util_funcs import get_types, get_rng
from smac.initial_design.sobol_design import SobolDesign
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogScaledCost


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

        kwargs['initial_design'] = kwargs.get('initial_design', SobolDesign)
        kwargs['runhistory2epm'] = kwargs.get('runhistory2epm', RunHistory2EPM4LogScaledCost)

        init_kwargs = kwargs.get('initial_design_kwargs', dict())
        init_kwargs['n_configs_x_params'] = init_kwargs.get('n_configs_x_params', 10)
        init_kwargs['max_config_fracs'] = init_kwargs.get('max_config_fracs', 0.25)
        kwargs['initial_design_kwargs'] = init_kwargs

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
                model_class = GaussianProcess
                kwargs['model'] = model_class
                model_kwargs = kwargs.get('model_kwargs', dict())
                model_kwargs['kernel'] = kernel
                model_kwargs['prior'] = prior
                model_kwargs['normalize_input'] = True
                model_kwargs['normalize_output'] = True
                model_kwargs['normalize_input'] = True
                model_kwargs['seed'] = rng.randint(0, 2 ** 20)
            elif model_type == "gp_mcmc":
                model_class = GaussianProcessMCMC
                kwargs['model'] = model_class
                model_kwargs = kwargs.get('model_kwargs', dict())
                model_kwargs['kernel'] = kernel
                model_kwargs['prior'] = prior
                model_kwargs['n_hypers'] = n_hypers
                model_kwargs['chain_length'] = 200
                model_kwargs['burnin_steps'] = 100
                model_kwargs['normalize_input'] = True
                model_kwargs['normalize_output'] = True
                model_kwargs['seed'] = rng.randint(0, 2**20)
            kwargs['model_kwargs'] = model_kwargs

        super().__init__(**kwargs)

        if self.solver.scenario.n_features > 0:
            raise NotImplementedError("BOGP cannot handle instances")

        self.logger.info(self.__class__)

        # assumes random chooser for random configs
        random_config_chooser_kwargs = kwargs.get('random_configuration_chooser_kwargs', dict())
        random_config_chooser_kwargs['prob'] = random_config_chooser_kwargs.get('prob', 0.0)
        kwargs['random_configuration_chooser_kwargs'] = random_config_chooser_kwargs

        # only 1 configuration per SMBO iteration
        intensifier_kwargs = kwargs.get('intensifier_kwargs', dict())
        intensifier_kwargs['min_chall'] = 1
        kwargs['intensifier_kwargs'] = intensifier_kwargs
        scenario.intensification_percentage = 1e-10

        # better improve acqusition function optimization
        # 1. increase number of sls iterations
        self.solver.acq_optimizer.n_sls_iterations = 100
        # 2. more randomly sampled configurations
        self.solver.scenario.acq_opt_challengers = 1000

        # activate predict incumbent
        self.solver.predict_incumbent = True
