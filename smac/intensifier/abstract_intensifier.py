from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator

import dataclasses
import json
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration

import smac
from smac.callback import Callback
from smac.main.config_selector import ConfigSelector
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import (
    InstanceSeedBudgetKey,
    TrajectoryItem,
    TrialValue,
)
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash, print_config_changes
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractIntensifier:
    def __init__(
        self,
        scenario: Scenario,
        max_incumbents: int = 20,
        seed: int | None = None,
    ):
        self._scenario = scenario
        self._config_selector: ConfigSelector | None = None
        self._config_generator: Iterator[ConfigSelector] | None = None
        self._runhistory: RunHistory | None = None

        if seed is None:
            seed = self._scenario.seed

        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # Internal variables
        self._tf_seeds: list[int] = []
        self._tf_instances: list[str | None] = []
        self._tf_budgets: list[float | None] = []

        # Incumbent variables
        self._max_incumbents = max_incumbents

        # Reset
        self.reset()

    def reset(self) -> None:
        """Resets the intensifier."""
        self._incumbents: list[Configuration] = []
        self._incumbents_changed = 0
        self._rejected_config_ids: list[int] = []
        self._trajectory: list[TrajectoryItem] = []

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "max_incumbents": self._max_incumbents,
            "seed": self._seed,
        }

    @property
    def runhistory(self) -> RunHistory:
        """Runhistory of the intensifier."""
        assert self._runhistory is not None
        return self._runhistory

    @runhistory.setter
    def runhistory(self, runhistory: RunHistory) -> None:
        """Sets the runhistory and fills ``self._tf_seeds`` and ``self._tf_instances``. Moreover, the incumbents
        are updated.
        """
        self._runhistory = runhistory

        # Validate runhistory: Are seeds/instances/budgets used?
        # Add seed/instance/budget to the cache
        for k in runhistory.keys():
            if self.uses_seeds:
                if k.seed is None:
                    raise ValueError("Trial contains no seed information but intensifier expects seeds to be used.")

                if k.seed not in self._tf_seeds:
                    self._tf_seeds.append(k.seed)

            if self.uses_instances:
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

        # Update our incumbents here
        for config in self._runhistory.get_configs():
            self.update_incumbents(config)

    @property
    def config_generator(self) -> Iterator[ConfigSelector]:
        """Based on the configuration selector, an iterator is returned that generates configurations."""
        assert self._config_generator is not None
        return self._config_generator

    @property
    def config_selector(self) -> ConfigSelector:
        """The configuration selector for the intensifier."""
        assert self._config_selector is not None
        return self._config_selector

    @config_selector.setter
    def config_selector(self, config_selector: ConfigSelector) -> None:
        # Set it global
        self._config_selector = config_selector
        self._config_generator = iter(config_selector)

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
    def incumbents_changed(self) -> int:
        return self._incumbents_changed

    def get_incumbent(self) -> Configuration | None:
        """Returns the current incumbent in a single-objective setting."""
        if self._scenario.count_objectives() > 1:
            raise ValueError("Cannot get a single incumbent for multi-objective optimization.")

        if len(self._incumbents) == 0:
            return None

        assert len(self._incumbents) == 1
        return self._incumbents[0]

    def get_incumbents(self, sort_by: str | None = None) -> list[Configuration]:
        """Returns the incumbents (points on the pareto front) of the runhistory as copy. In case of a single-objective
        optimization, only one incumbent (if is) is returned.

        Returns
        -------
        configs : list[Configuration]
            The configs of the Pareto front.
        sort_by : str, defaults to None
            Sort the trials by ``cost`` (lowest cost first) or ``num_trials`` (config with lowest number of trials
            first).
        """
        rh = self.runhistory

        if sort_by == "cost":
            return list(sorted(self._incumbents, key=lambda config: rh._cost_per_config[rh._config_ids[config]]))
        elif sort_by == "num_trials":
            return list(sorted(self._incumbents, key=lambda config: len(rh.get_trials(config))))
        elif sort_by is None:
            return list(self._incumbents)
        else:
            raise ValueError(f"Unknown sort_by value: {sort_by}.")

    def get_incumbent_instances(self) -> list[InstanceSeedBudgetKey]:
        """Find the lowest intersection of instances for all incumbents."""
        incumbents = self.get_incumbents()

        if len(incumbents) > 0:
            # We want to calculate the smallest set of trials that is used by all incumbents
            # Reason: We can not fairly compare otherwise
            incumbent_instances = [self.runhistory.get_instances(incumbent) for incumbent in incumbents]
            instances = list(set.intersection(*map(set, incumbent_instances)))  # type: ignore

            return instances  # type: ignore

        return []

    def get_incumbent_instance_differences(self) -> list[InstanceSeedBudgetKey]:
        """There are situations in which incumbents are evaluated on more trials than others. This method returns the
        instances which are not part of the lowest intersection of instances for all incumbents.
        """
        incumbents = self.get_incumbents()

        if len(incumbents) > 0:
            # We want to calculate the differences so that we can evaluate the other incumbents on the same instances
            incumbent_instances = [self.runhistory.get_instances(incumbent) for incumbent in incumbents]

            if len(incumbent_instances) <= 1:
                return []

            instances = list(set.difference(*map(set, incumbent_instances)))  # type: ignore

            if len(instances) == 0:
                return []

            return instances  # type: ignore

        return []

    def get_rejected_configs(self) -> list[Configuration]:
        """Returns rejected configurations when racing against the incumbent failed."""
        configs = []
        for rejected_config_id in self._rejected_config_ids:
            configs.append(self.runhistory._ids_config[rejected_config_id])

        return configs

    def get_callback(self) -> Callback:
        """The intensifier makes use of a callback to efficiently update the incumbent based on the runhistory
        (every time new information are available). Moreover, incorporating the callback here allows developers
        more options in the future.
        """

        class RunHistoryCallback(Callback):
            def __init__(self, intensifier: AbstractIntensifier):
                self.intensifier = intensifier

            def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> None:
                self.intensifier.update_incumbents(info.config)

        return RunHistoryCallback(self)

    def update_incumbents(self, config: Configuration) -> None:
        """Updates the incumbents. This method is called everytime a trial is added to the runhistory. Since only
        the affected config and the current incumbents are used, this method is very efficient. Furthermore, a
        configuration is only considered incumbent if it has a better performance on all incumbent instances.
        """
        rh = self.runhistory

        # What happens if a config was rejected but it appears again? Give it another try since it
        # already was evaluated? YES!

        # Associated trials and id
        config_instances = rh.get_instances(config)
        config_id = rh._config_ids[config]
        config_hash = get_config_hash(config)

        # We skip updating incumbents if no instances are available
        # Note: This is especially the case if trials of a config are still running
        # because if trials are running, the runhistory does not update the trials in the fast data structure
        if len(config_instances) == 0:
            logger.debug(f"No instances evaluated for config {config_hash}. Updating incumbents is skipped.")
            return

        # Now we get the incumbents and see which trials have been used
        incumbents = self.get_incumbents()
        incumbent_ids = [rh._config_ids[c] for c in incumbents]
        incumbent_instances = self.get_incumbent_instances()

        # Save for later
        previous_incumbents = incumbents.copy()
        previous_incumbent_ids = incumbent_ids.copy()

        # Little sanity check here for consistency
        if len(incumbents) > 0:
            assert incumbent_instances is not None
            assert len(incumbent_instances) > 0

        # If there are no incumbents at all, we just use the new config as new incumbent
        # Problem: We can add running incumbents
        if len(incumbents) == 0:  # incumbent_instances is None and len(incumbents) == 0:
            self._incumbents = [config]
            self._incumbents_changed += 1
            self._trajectory.append(TrajectoryItem(config_ids=[config_id], finished_trials=rh.finished))
            logger.info(f"Added config {config_hash} as new incumbent because there are no incumbents yet.")
            logger.debug("Updated trajectory.")

            # Nothing else to do
            return

        assert incumbent_instances is not None

        # Now we have to check if the new config has been evaluated on the same trials as the incumbents
        if not all([trial in config_instances for trial in incumbent_instances]):
            # We can not tell if the new config is better/worse than the incumbents because it has not been
            # evaluated on the necessary trials
            logger.debug(
                f"Could not compare config {config_hash} with incumbents because it's evaluated on "
                f"{len(config_instances)}/{len(incumbent_instances)} trials only."
            )

            # The config has to go to a queue now as it is a challenger and a potential incumbent
            return
        else:
            # If all instances are available and the config is incumbent and even evaluated on more trials
            # then there's nothing we can do
            if config in incumbents and len(config_instances) > len(incumbent_instances):
                logger.debug(
                    "Config is already an incumbent but can not be compared to other incumbents because "
                    "the others are missing trials."
                )
                return

        # Now we get the costs for the trials of the config
        average_costs = []

        # Add config to incumbents so that we compare only the new config and existing incumbents
        if config not in incumbents:
            incumbents.append(config)
            incumbent_ids.append(config_id)

        for incumbent in incumbents:
            # Since we use multiple seeds, we have to average them to get only one cost value pair for each
            # configuration
            # However, we only want to consider the config trials
            # Average cost is a list of floats (one for each objective)
            average_cost = rh.average_cost(incumbent, config_instances, normalize=False)
            average_costs += [average_cost]

        # Let's work with a numpy array for efficiency
        costs = np.vstack(average_costs)

        # The following code is an efficient pareto front implementation
        is_efficient = np.arange(costs.shape[0])
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

        new_incumbents = [incumbents[i] for i in is_efficient]
        new_incumbent_ids = [rh._config_ids[c] for c in new_incumbents]

        if len(previous_incumbents) == len(new_incumbents):
            if previous_incumbents == new_incumbents:
                # No changes in the incumbents
                self._remove_rejected_config(config_id)
                logger.debug("Karpador setzt Platscher ein.")
                return
            else:
                # In this case, we have to determine which config replaced which incumbent and reject it
                removed_incumbent_id = list(set(previous_incumbent_ids) - set(new_incumbent_ids))[0]
                removed_incumbent_hash = get_config_hash(rh.get_config(removed_incumbent_id))
                self._add_rejected_config(removed_incumbent_id)

                if removed_incumbent_id == config_id:
                    logger.debug(
                        f"Rejected config {config_hash} because it is not better than the incumbents on "
                        f"{len(config_instances)} instances."
                    )
                else:
                    self._remove_rejected_config(config_id)
                    logger.info(
                        f"Added config {config_hash} and rejected config {removed_incumbent_hash} because "
                        f"it is not better than the incumbents on {len(config_instances)} instances:"
                    )
                    print_config_changes(config, rh.get_config(removed_incumbent_id), logger=logger)
        elif len(previous_incumbents) < len(new_incumbents):
            # Config becomes a new incumbent; nothing is rejected in this case
            self._remove_rejected_config(config_id)
            logger.info(
                f"Config {config_hash} is a new incumbent. " f"Total number of incumbents: {len(new_incumbents)}."
            )
        else:
            # There might be situations that the incumbents might be removed because of updated cost information of
            # config
            for incumbent in previous_incumbents:
                if incumbent not in new_incumbents:
                    self._add_rejected_config(incumbent)
                    logger.debug(
                        f"Removed incumbent {get_config_hash(incumbent)} because of the updated costs from config "
                        f"{config_hash}."
                    )

        # Cut incumbents: We only want to keep a specific number of incumbents
        # Approach: Do it randomly for now; task for a future phd student ;)
        if len(new_incumbents) > self._max_incumbents:
            idx = self._rng.randint(0, len(new_incumbents))
            del new_incumbents[idx]
            del new_incumbent_ids[idx]

            logger.info(f"Removed one incumbent randomly because more than {self._max_incumbents} are available.")

        self._incumbents = new_incumbents
        self._incumbents_changed += 1
        self._trajectory.append(TrajectoryItem(config_ids=new_incumbent_ids, finished_trials=rh.finished))
        logger.debug("Updated trajectory.")

    @abstractmethod
    def __iter__(self) -> Iterator[TrialInfo]:
        """Main loop of the intensifier. This method always returns a TrialInfo object, although the intensifier
        algorithm may need to wait for the result of the trial. Please refer to a specific
        intensifier to get more information.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trials_of_interest(
        self,
        config: Configuration,
        *,
        validate: bool = False,
        seed: int | None = None,
    ) -> list[TrialInfo]:
        """Returns a list of trials of interest for a given configuration.

        Warning
        -------
        The passed seed is only used for validation.

        Parameters
        ----------
        config : Configuration
            The config of interest,
        validate : bool, defaults to False
            Whether to get validation trials or training trials. The only difference lays in different seeds.
        seed : int | None, defaults to None
            The seed used for the validation trials.

        Returns
        -------
        trials : list[TrialInfo]
            Trials of the config of interest.
        """
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        """The current state of the intensifier. Used to restore the state of the intensifier when continuing a run."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Sets the state of the intensifier. Used to restore the state of the intensifier when continuing a run."""
        pass

    def save(self, filename: str | Path) -> None:
        """Saves the current state of the intensifier. In addition to the state (retrieved by ``get_state``), this
        method also saves the incumbents and trajectory.
        """
        if isinstance(filename, str):
            filename = Path(filename)

        assert str(filename).endswith(".json")
        filename.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "incumbent_ids": [self.runhistory.get_config_id(config) for config in self._incumbents],
            "rejected_config_ids": self._rejected_config_ids,
            "incumbents_changed": self._incumbents_changed,
            "trajectory": [dataclasses.asdict(item) for item in self._trajectory],
            "state": self.get_state(),
        }

        with open(filename, "w") as fp:
            json.dump(data, fp, indent=2)

    def load(self, filename: str | Path) -> None:
        """Loads the latest state of the intensifier including the incumbents and trajectory."""
        if isinstance(filename, str):
            filename = Path(filename)

        try:
            with open(filename) as fp:
                data = json.load(fp)
        except Exception as e:
            logger.warning(
                f"Encountered exception {e} while reading runhistory from {filename}. Not adding any trials!"
            )
            return

        self._incumbents = [self.runhistory.get_config(config_id) for config_id in data["incumbent_ids"]]
        self._incumbents_changed = data["incumbents_changed"]
        self._rejected_config_ids = data["rejected_config_ids"]
        self._trajectory = [TrajectoryItem(**item) for item in data["trajectory"]]
        self.set_state(data["state"])

    def print_config_changes(
        self,
        incumbent: Configuration | None,
        challenger: Configuration | None,
    ) -> None:
        """Compares two configurations and prints the differences."""
        if incumbent is None or challenger is None:
            return

        params = sorted([(param, incumbent[param], challenger[param]) for param in challenger.keys()])
        for param in params:
            if param[1] != param[2]:
                logger.info("--- %s: %r -> %r" % param)
            else:
                logger.debug("--- %s Remains unchanged: %r", param[0], param[1])

    def _add_rejected_config(self, config: Configuration | int) -> None:
        if isinstance(config, Configuration):
            config_id = self.runhistory.get_config_id(config)
        else:
            config_id = config

        if config_id not in self._rejected_config_ids:
            self._rejected_config_ids.append(config_id)

    def _remove_rejected_config(self, config: Configuration | int) -> None:
        if isinstance(config, Configuration):
            config_id = self.runhistory.get_config_id(config)
        else:
            config_id = config

        if config_id in self._rejected_config_ids:
            self._rejected_config_ids.remove(config_id)
