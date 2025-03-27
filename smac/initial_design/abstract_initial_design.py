from __future__ import annotations

from abc import abstractmethod
from typing import Any

from collections import OrderedDict

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    IntegerHyperparameter,
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
    scenario : Scenario
    n_configs : int | None, defaults to None
        Number of initial configurations (disables the arguments ``n_configs_per_hyperparameter``).
    n_configs_per_hyperparameter: int, defaults to 10
        Number of initial configurations per hyperparameter. For example, if my configuration space covers five
        hyperparameters and ``n_configs_per_hyperparameter`` is set to 10, then 50 initial configurations will be
        samples.
    max_ratio: float, defaults to 0.25
        Use at most ``scenario.n_trials`` * ``max_ratio`` number of configurations in the initial design.
        Additional configurations are not affected by this parameter.
    additional_configs: list[Configuration], defaults to []
        Adds additional configurations to the initial design.
    seed : int | None, default to None
    """

    def __init__(
        self,
        scenario: Scenario,
        n_configs: int | None = None,
        n_configs_per_hyperparameter: int | None = 10,
        max_ratio: float = 0.25,
        additional_configs: list[Configuration] = None,
        seed: int | None = None,
    ):
        self._configspace = scenario.configspace

        if seed is None:
            seed = scenario.seed

        self.use_default_config = scenario.use_default_config

        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._n_configs_per_hyperparameter = n_configs_per_hyperparameter

        # make sure that additional configs is not a mutable default value
        # this avoids issues
        if additional_configs is None:
            additional_configs = []

        if self.use_default_config:
            default_config = self._configspace.get_default_configuration()
            default_config.origin = "Initial Design: Default configuration"
            additional_configs.append(default_config)

        self._additional_configs = additional_configs

        n_params = len(self._configspace.get_hyperparameters())
        if n_configs is not None:
            logger.info("Using `n_configs` and ignoring `n_configs_per_hyperparameter`.")
            self._n_configs = n_configs
        elif n_configs_per_hyperparameter is not None:
            self._n_configs = n_configs_per_hyperparameter * n_params
        else:
            raise ValueError(
                "Need to provide either argument `n_configs` or "
                "`n_configs_per_hyperparameter` but provided none of them."
            )

        # If the number of configurations is too large, we reduce it
        _n_configs = int(max(1, min(self._n_configs, (max_ratio * scenario.n_trials))))
        if self._n_configs != _n_configs:
            logger.info(
                f"Reducing the number of initial configurations from {self._n_configs} to "
                f"{_n_configs} (max_ratio == {max_ratio})."
            )
            self._n_configs = _n_configs

        # We allow no configs if we have additional configs
        if n_configs is not None and n_configs == 0 and len(additional_configs) > 0:
            self._n_configs = 0

        if self._n_configs + len(additional_configs) > scenario.n_trials:
            raise ValueError(
                f"Initial budget {self._n_configs} cannot be higher than the number of trials {scenario.n_trials}."
            )

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "n_configs": self._n_configs,
            "n_configs_per_hyperparameter": self._n_configs_per_hyperparameter,
            "additional_configs": [c.get_dictionary() for c in self._additional_configs],
            "seed": self._seed,
        }

    def select_configurations(self) -> list[Configuration]:
        """Selects the initial configurations. Internally, `_select_configurations` is called,
        which has to be implemented by the child class.

        Returns
        -------
        configs : list[Configuration]
            Configurations from the child class.
        """
        configs: list[Configuration] = []

        if self._n_configs == 0:
            logger.info("No initial configurations are used.")
        else:
            configs += self._select_configurations()

        # Adding additional configs
        configs += self._additional_configs

        for config in configs:
            if config.origin is None:
                config.origin = "Initial design"

        # Removing duplicates
        # (Reference: https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists)
        configs = list(OrderedDict.fromkeys(configs))
        logger.info(
            f"Using {len(configs) - len(self._additional_configs)} initial design configurations "
            f"and {len(self._additional_configs)} additional configurations."
        )

        return configs

    @abstractmethod
    def _select_configurations(self) -> list[Configuration]:
        """Selects the initial configurations, depending on the implementation of the initial design."""
        raise NotImplementedError

    def _transform_continuous_designs(
        self, design: np.ndarray, origin: str, configspace: ConfigurationSpace
    ) -> list[Configuration]:
        """Transforms the continuous designs into a discrete list of configurations.

        Parameters
        ----------
        design : np.ndarray
            Array of hyperparameters originating from the initial design strategy.
        origin : str | None, defaults to None
            Label for a configuration where it originated from.
        configspace : ConfigurationSpace

        Returns
        -------
        configs : list[Configuration]
            Continuous transformed configs.
        """
        params = configspace.get_hyperparameters()
        for idx, param in enumerate(params):

            if isinstance(param, IntegerHyperparameter):
                design[:, idx] = param._inverse_transform(param._transform(design[:, idx]))
            elif isinstance(param, NumericalHyperparameter):
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
                raise ValueError("Hyperparameter not supported when transforming a continuous design.")

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

        return configs
