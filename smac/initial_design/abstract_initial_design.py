from __future__ import annotations

from typing import Any
from abc import abstractmethod

from collections import OrderedDict

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.util import ForbiddenValueError, deactivate_inactive_hyperparameters

from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractInitialDesign:
    """Base class for initial design strategies that evaluates multiple configurations.

    Parameters
    ----------
    configspace: ConfigurationSpace
        configuration space object
    rng: np.random.RandomState
        Random state
    n_runs: int
        Number of iterations allowed for the target algorithm
    configs: list[Configuration] | None
        List of initial configurations. Disables the arguments ``n_configs_per_hyperparameter`` if given.
        Either this, or ``n_configs_per_hyperparameter`` or ``n_configs`` must be provided.
    n_configs_per_hyperparameter: int, defaults to 10
        how many configurations will be used at most in the initial design (X*D). Either
        this, or ``n_configs`` or ``configs`` must be provided. Disables the argument
        ``n_configs_per_hyperparameter`` if given.
    max_config_ratio: float, defaults to 0.25
        use at most X*budget in the initial design. Not active if a time limit is given.
    n_configs : int, optional
        Maximal initial budget (disables the arguments ``n_configs_per_hyperparameter`` and ``configs``
        if both are given). Either this, or ``n_configs_per_hyperparameter`` or ``configs`` must be
        provided.
    seed : int | None, default to None.
        Random seed. If None, will use the seed from the scenario.

    Attributes
    ----------
    configspace : ConfigurationSpace
    configs : list[Configuration]
        List of configurations to be evaluated
    n_configs : int
        Number of configurations to be evaluated. It is dynamically computed.
    seed
    rng
    """

    def __init__(
        self,
        scenario: Scenario,
        configs: list[Configuration] | None = None,
        n_configs: int | None = None,
        n_configs_per_hyperparameter: int | None = 10,
        max_config_ratio: float = 0.25,
        seed: int | None = None,
    ):
        self.configspace = scenario.configspace

        if seed is None:
            seed = scenario.seed

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.configs = configs
        self.n_configs_per_hyperparameter = n_configs_per_hyperparameter

        n_params = len(self.configspace.get_hyperparameters())
        if configs is not None:
            logger.info("Ignoring `n_configs` and `n_configs_per_hyperparameter` since `configs` is given.")
            self.n_configs = len(configs)
        elif n_configs is not None:
            logger.info("Ignoring `configs` and `n_configs_per_hyperparameter` since `n_configs` is given.")
            self.n_configs = n_configs
        elif n_configs_per_hyperparameter is not None:
            logger.info("Ignoring `configs` and `n_configs` since `n_configs_per_hyperparameter` is given.")
            self.n_configs = int(
                max(1, min(n_configs_per_hyperparameter * n_params, (max_config_ratio * scenario.n_trials)))
            )
        else:
            raise ValueError(
                "Need to provide either argument `configs`, `n_configs` or "
                "`n_configs_per_hyperparameter`, but provided none of them."
            )

        if self.n_configs > scenario.n_trials:
            raise ValueError(
                f"Initial budget {self.n_configs} cannot be higher than the number of trials {scenario.n_trials}."
            )

    @abstractmethod
    def _select_configurations(self) -> list[Configuration]:
        """Selects the initial configurations. Depending on the implementation
        of the initial_design."""
        raise NotImplementedError

    def _transform_continuous_designs(
        self, design: np.ndarray, origin: str, configspace: ConfigurationSpace
    ) -> list[Configuration]:
        """Transforms the continuous designs into a discrete list of configurations.

        Parameters
        ----------
        design : np.ndarray
            Array of hyperparameters originating from the initial design strategy.
            See e.g. scipy.qmc.LatinHypercube for details.
        origin : str | None, defaults to None
            Label for a configuration where it originated from.
        configspace : ConfigurationSpace
        """

        params = configspace.get_hyperparameters()
        for idx, param in enumerate(params):
            if isinstance(param, NumericalHyperparameter):
                continue
            elif isinstance(param, Constant):
                design_ = np.zeros(np.array(design.shape) + np.array((0, 1)))
                design_[:, :idx] = design[:, :idx]
                design_[:, idx + 1 :] = design[:, idx:]
                design = design_
            elif isinstance(param, CategoricalHyperparameter):
                v_design = design[:, idx]
                v_design[v_design == 1] = 1 - 10**-10
                design[:, idx] = np.array(v_design * len(param.choices), dtype=int)
            elif isinstance(param, OrdinalHyperparameter):
                v_design = design[:, idx]
                v_design[v_design == 1] = 1 - 10**-10
                design[:, idx] = np.array(v_design * len(param.sequence), dtype=int)
            else:
                raise ValueError("Hyperparameter not supported in LHD.")

        logger.debug("Initial Design")
        configs = []
        for vector in design:
            try:
                conf = deactivate_inactive_hyperparameters(
                    configuration=None, configuration_space=configspace, vector=vector
                )
            except ForbiddenValueError:
                continue
            conf.origin = origin
            configs.append(conf)
            logger.debug(conf)

        logger.debug("Size of initial design: %d" % (len(configs)))

        return configs

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""

        configs = None
        if self.configs is not None:
            configs = [config.get_dictionary() for config in self.configs]

        return {
            "name": self.__class__.__name__,
            "n_configs": self.n_configs,
            "seed": self.seed,
            "configs": configs,
            "n_configs_per_hyperparameter": self.n_configs_per_hyperparameter,
        }

    def select_configurations(self) -> list[Configuration]:
        """Selects the initial configurations."""
        logger.info(f"Retrieving {self.n_configs} configurations for the initial design.")
        if self.n_configs == 0:
            return []
        if self.configs is None:
            self.configs = self._select_configurations()

        for config in self.configs:
            if config.origin is None:
                config.origin = "Initial design"

        # Removing duplicates
        # (Reference: https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists)
        self.configs = list(OrderedDict.fromkeys(self.configs))
        return self.configs
