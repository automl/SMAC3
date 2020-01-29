import logging
import typing
from collections import OrderedDict

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import NumericalHyperparameter, \
    Constant, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters
import numpy as np

from smac.utils.io.traj_logging import TrajLogger

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2019, AutoML"
__license__ = "3-clause BSD"


class InitialDesign:
    """Base class for initial design strategies that evaluates multiple configurations

    Attributes
    ----------
    cs : ConfigurationSpace
    configs : typing.List[Configuration]
        List of configurations to be evaluated
    """

    def __init__(self,
                 cs: ConfigurationSpace,
                 rng: np.random.RandomState,
                 traj_logger: TrajLogger,
                 ta_run_limit: int,
                 configs: typing.Optional[typing.List[Configuration]] = None,
                 n_configs_x_params: typing.Optional[int] = 10,
                 max_config_fracs: float = 0.25,
                 init_budget: typing.Optional[int] = None,
                 ):
        """Constructor

        Parameters
        ---------
        cs: ConfigurationSpace
            configuration space object
        rng: np.random.RandomState
            Random state
        traj_logger: TrajLogger
            Trajectory logging to add new incumbents found by the initial
            design.
        ta_run_limit: int
            Number of iterations allowed for the target algorithm
        configs: typing.Optional[typing.List[Configuration]]
            List of initial configurations. Disables the arguments ``n_configs_x_params`` if given.
            Either this, or ``n_configs_x_params`` or ``init_budget`` must be provided.
        n_configs_x_params: int
            how many configurations will be used at most in the initial design (X*D). Either
            this, or ``init_budget`` or ``configs`` must be provided. Disables the argument
            ``n_configs_x_params`` if given.
        max_config_fracs: float
            use at most X*budget in the initial design. Not active if a time limit is given.
        init_budget : int, optional
            Maximal initial budget (disables the arguments ``n_configs_x_params`` and ``configs``
            if both are given). Either this, or ``n_configs_x_params`` or ``configs`` must be
            provided.
        """

        self.cs = cs
        self.rng = rng
        self.traj_logger = traj_logger
        self.configs = configs

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        n_params = len(self.cs.get_hyperparameters())
        if init_budget is not None:
            self.init_budget = init_budget
            if n_configs_x_params is not None:
                self.logger.debug(
                    'Ignoring argument `n_configs_x_params` (value %d).',
                    n_configs_x_params,
                )
        elif configs is not None:
            self.init_budget = len(configs)
        elif n_configs_x_params is not None:
            self.init_budget = int(max(1, min(n_configs_x_params * n_params,
                                              (max_config_fracs * ta_run_limit))))
        else:
            raise ValueError('Need to provide either argument `init_budget`, `configs` or '
                             '`n_configs_x_params`, but provided none of them.')
        if self.init_budget > ta_run_limit:
            raise ValueError(
                'Initial budget %d cannot be higher than the run limit %d.'
                % (self.init_budget, ta_run_limit)
            )
        self.logger.info("Running initial design for %d configurations" % self.init_budget)

    def select_configurations(self) -> typing.List[Configuration]:

        if self.configs is None:
            self.configs = self._select_configurations()

        for config in self.configs:
            if config.origin is None:
                config.origin = 'Initial design'

        # add this incumbent right away to have an entry to time point 0
        self.traj_logger.add_entry(train_perf=2**31,
                                   incumbent_id=1,
                                   incumbent=self.configs[0])

        # removing duplicates
        # (Reference: https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists)
        self.configs = list(OrderedDict.fromkeys(self.configs))
        return self.configs

    def _select_configurations(self) -> typing.List[Configuration]:
        raise NotImplementedError

    def _transform_continuous_designs(self,
                                      design: np.ndarray,
                                      origin: str,
                                      cs: ConfigurationSpace) -> typing.List[Configuration]:

        params = cs.get_hyperparameters()
        for idx, param in enumerate(params):
            if isinstance(param, NumericalHyperparameter):
                continue
            elif isinstance(param, Constant):
                # add a vector with zeros
                design_ = np.zeros(np.array(design.shape) + np.array((0, 1)))
                design_[:, :idx] = design[:, :idx]
                design_[:, idx + 1:] = design[:, idx:]
                design = design_
            elif isinstance(param, CategoricalHyperparameter):
                v_design = design[:, idx]
                v_design[v_design == 1] = 1 - 10**-10
                design[:, idx] = np.array(v_design * len(param.choices), dtype=np.int)
            elif isinstance(param, OrdinalHyperparameter):
                v_design = design[:, idx]
                v_design[v_design == 1] = 1 - 10**-10
                design[:, idx] = np.array(v_design * len(param.sequence), dtype=np.int)
            else:
                raise ValueError("Hyperparameter not supported in LHD")

        self.logger.debug("Initial Design")
        configs = []
        for vector in design:
            conf = deactivate_inactive_hyperparameters(configuration=None,
                                                       configuration_space=cs,
                                                       vector=vector)
            conf.origin = origin
            configs.append(conf)
            self.logger.debug(conf)

        self.logger.debug("Size of initial design: %d" % (len(configs)))

        return configs
