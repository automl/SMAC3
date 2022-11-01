from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator
import numpy as np
from smac.config_selector.config_selector import ConfigSelector

from smac.runhistory import TrialInfo
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractIntensifier:
    def __init__(self, scenario: Scenario, seed: int | None = None):
        self._scenario = scenario
        self._config_selector: Iterator[ConfigSelector] | None = None
        self._runhistory: RunHistory | None = None

        if seed is None:
            seed = self._scenario.seed

        self._rng = np.random.RandomState(seed)

        # Internal variables
        self._tf_seeds: list[int] = []
        self._tf_instances: list[str | None] = []
        self._tf_budgets: list[float | None] = []

    @property
    def config_selector(self) -> Iterator[ConfigSelector]:
        assert self._config_selector is not None
        return self._config_selector

    @config_selector.setter
    def config_selector(self, config_selector: ConfigSelector) -> None:
        # Set it global
        self._config_selector = iter(config_selector)
        self._runhistory = config_selector._runhistory

        # Validate runhistory: Are seeds/instances/budgets used?
        # Add seed/instance/budget to the cache
        for k in self.runhistory.keys():
            if self.uses_seeds:
                if k.seed is None:
                    raise ValueError("Trial contains no seed information but intensifier expects seeds to be used.")

                if k.seed not in self._tf_seeds:
                    self._tf_seeds.append(k.seed)

            if self.uses_instances:
                # if k.instance is None:
                #    raise ValueError(
                #        "Trial contains no instance information, but intensifier expects instances to be used."
                #    )

                if self._scenario.instances is None and k.instance is not None:
                    raise ValueError(
                        "Scenario does not specify any instances but found instance information in runhistory."
                    )

                if self._scenario.instances is not None and k.instance not in self._scenario.instances:
                    raise ValueError(
                        "Instance information in runhistory is not part of the defined instances in scenario."
                    )

                if k.instance not in self._tf_instances:
                    self._tf_instances.append(k.instance)

            if self.uses_budgets:
                if k.budget is None:
                    raise ValueError("Trial contains no budget information but intensifier expects budgets to be used.")

                if k.budget not in self._tf_budgets:
                    self._tf_budgets.append(k.budget)

        # Add all other instances to ``_tf_instances``
        # Behind idea: Prioritize instances that are found in the runhistory
        if (instances := self._scenario.instances) is not None:
            for inst in instances:
                if inst not in self._tf_instances:
                    self._tf_instances.append(inst)

        if len(self._tf_instances) == 0:
            self._tf_instances = [None]

        if len(self._tf_budgets) == 0:
            self._tf_budgets = [None]

    @property
    def runhistory(self) -> RunHistory:
        assert self._runhistory is not None
        return self._runhistory

    @property
    @abstractmethod
    def uses_seeds(self) -> bool:
        """If the intensifier needs to make use of seeds."""
        raise NotImplementedError

    @property
    @abstractmethod
    def uses_budgets(self) -> bool:
        """If the intensifier needs to make use of budgets."""
        raise NotImplementedError

    @property
    @abstractmethod
    def uses_instances(self) -> bool:
        """If the intensifier needs to make use of instances."""
        raise NotImplementedError

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    @abstractmethod
    def __iter__(self) -> Iterator[TrialInfo]:
        """Main loop of the intensifier. This method always returns a TrialInfo object, although the intensifier
        algorithm may need to wait for the result of the trial. Please refer to a specific
        intensifier to get more information.
        """
        raise NotImplementedError
