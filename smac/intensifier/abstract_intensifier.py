from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Iterator

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration

import smac
from smac.callback.callback import Callback
from smac.constants import MAXINT
from smac.main.config_selector import ConfigSelector
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import (
    InstanceSeedBudgetKey,
    InstanceSeedKey,
    TrajectoryItem,
    TrialValue,
)
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash, print_config_changes
from smac.utils.logging import get_logger
from smac.utils.pareto_front import calculate_pareto_front, sort_by_crowding_distance

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractIntensifier:
    """Abstract implementation of an intensifier supporting multi-fidelity, multi-objective, and multi-threading.
    The abstract intensifier keeps track of the incumbent, which is updated everytime the runhistory changes.

    Parameters
    ----------
    n_seeds : int | None, defaults to None
        How many seeds to use for each instance. It is used in the abstract intensifier to determine validation trials.
    max_config_calls : int, defaults to None
        Maximum number of configuration evaluations. Basically, how many instance-seed keys should be max evaluated
        for a configuration. It is used in the abstract intensifier to determine validation trials.
    max_incumbents : int, defaults to 10
        How many incumbents to keep track of in the case of multi-objective.
    seed : int, defaults to None
        Internal seed used for random events like shuffle seeds.
    """

    def __init__(
        self,
        scenario: Scenario,
        n_seeds: int | None = None,
        max_config_calls: int | None = None,
        max_incumbents: int = 10,
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
        self._n_seeds = n_seeds
        self._max_config_calls = max_config_calls
        self._max_incumbents = max_incumbents
        self._used_walltime_func: Callable | None = None

        # Reset everything
        self.reset()

    def reset(self) -> None:
        """Reset the internal variables of the intensifier."""
        self._tf_seeds: list[int] = []
        self._tf_instances: list[str | None] = []
        self._tf_budgets: list[float | None] = []
        self._instance_seed_keys: list[InstanceSeedKey] | None = None
        self._instance_seed_keys_validation: list[InstanceSeedKey] | None = None

        # Incumbent variables
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
    def trajectory(self) -> list[TrajectoryItem]:
        """Returns the trajectory (changes of incumbents) of the optimization run."""
        return self._trajectory

    @property
    def runhistory(self) -> RunHistory:
        """Runhistory of the intensifier."""
        assert self._runhistory is not None
        return self._runhistory

    @runhistory.setter
    def runhistory(self, runhistory: RunHistory) -> None:
        """Sets the runhistory."""
        self._runhistory = runhistory

    @property
    def used_walltime(self) -> float:
        """Returns used wallclock time."""
        if self._used_walltime_func is None:
            return 0.0

        return self._used_walltime_func()

    @used_walltime.setter
    def used_walltime(self, func: Callable) -> None:
        """Sets the used wallclock time."""
        self._used_walltime_func = func

    def __post_init__(self) -> None:
        """Fills ``self._tf_seeds`` and ``self._tf_instances``. Moreover, the incumbents are updated."""
        rh = self.runhistory

        # Validate runhistory: Are seeds/instances/budgets used?
        # Add seed/instance/budget to the cache
        for k in rh.keys():
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
        for config in rh.get_configs():
            self.update_incumbents(config)

    @property
    def config_generator(self) -> Iterator[Configuration]:
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
        """How often the incumbents have changed."""
        return self._incumbents_changed

    def get_instance_seed_keys_of_interest(
        self,
        *,
        validate: bool = False,
        seed: int | None = None,
    ) -> list[InstanceSeedKey]:
        """Returns a list of instance-seed keys. Considers seeds and instances from the
        runhistory (``self._tf_seeds`` and ``self._tf_instances``). If no seeds or instances were found, new
        seeds and instances are generated based on the global intensifier seed.

        Warning
        -------
        The passed seed is only used for validation. For training, the global intensifier seed is used.

        Parameters
        ----------
        validate : bool, defaults to False
            Whether to get validation trials or training trials. The only difference lies in different seeds.
        seed : int | None, defaults to None
            The seed used for the validation trials.

        Returns
        -------
        instance_seed_keys : list[InstanceSeedKey]
            Instance-seed keys of interest.
        """
        if self._runhistory is None:
            raise RuntimeError("Please set the runhistory before calling this method.")

        if len(self._tf_instances) == 0:
            raise RuntimeError("Please call __post_init__ before calling this method.")

        if seed is None:
            seed = 0

        # We cache the instance-seed keys for efficiency and consistency reasons
        if (self._instance_seed_keys is None and not validate) or (
            self._instance_seed_keys_validation is None and validate
        ):
            instance_seed_keys: list[InstanceSeedKey] = []
            if validate:
                rng = np.random.RandomState(seed)
            else:
                rng = self._rng

            i = 0
            while True:
                found_enough_configs = (
                    self._max_config_calls is not None and len(instance_seed_keys) >= self._max_config_calls
                )
                used_enough_seeds = self._n_seeds is not None and i >= self._n_seeds

                if found_enough_configs or used_enough_seeds:
                    break

                if validate:
                    next_seed = int(rng.randint(low=0, high=MAXINT, size=1)[0])
                else:
                    try:
                        next_seed = self._tf_seeds[i]
                        logger.info(f"Added existing seed {next_seed} from runhistory to the intensifier.")
                    except IndexError:
                        # Use global random generator for a new seed and mark it so it will be reused for another config
                        next_seed = int(rng.randint(low=0, high=MAXINT, size=1)[0])

                        # This line here is really important because we don't want to add the same seed twice
                        if next_seed in self._tf_seeds:
                            continue

                        self._tf_seeds.append(next_seed)
                        logger.debug(f"Added a new random seed {next_seed} to the intensifier.")

                # If no instances are used, tf_instances includes None
                for instance in self._tf_instances:
                    instance_seed_keys.append(InstanceSeedKey(instance, next_seed))

                # Only use one seed in deterministic case
                if self._scenario.deterministic:
                    logger.info("Using only one seed for deterministic scenario.")
                    break

                # Seed counter
                i += 1

            # Now we cut so that we only have max_config_calls instance_seed_keys
            # We favor instances over seeds here: That makes sure we always work with the same instance/seed pairs
            if self._max_config_calls is not None:
                if len(instance_seed_keys) > self._max_config_calls:
                    instance_seed_keys = instance_seed_keys[: self._max_config_calls]
                    logger.info(f"Cut instance-seed keys to {self._max_config_calls} entries.")

            # Set it globally
            if not validate:
                self._instance_seed_keys = instance_seed_keys
            else:
                self._instance_seed_keys_validation = instance_seed_keys

        if not validate:
            assert self._instance_seed_keys is not None
            instance_seed_keys = self._instance_seed_keys
        else:
            assert self._instance_seed_keys_validation is not None
            instance_seed_keys = self._instance_seed_keys_validation

        return instance_seed_keys.copy()

    def get_trials_of_interest(
        self,
        config: Configuration,
        *,
        validate: bool = False,
        seed: int | None = None,
    ) -> list[TrialInfo]:
        """Returns the trials of interest for a given configuration.
        Expands the keys from ``get_instance_seed_keys_of_interest`` with the config.
        """
        is_keys = self.get_instance_seed_keys_of_interest(validate=validate, seed=seed)

        trials = []
        for key in is_keys:
            trials.append(TrialInfo(config=config, instance=key.instance, seed=key.seed))

        return trials

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
            return list(sorted(self._incumbents, key=lambda config: rh._cost_per_config[rh.get_config_id(config)]))
        elif sort_by == "num_trials":
            return list(sorted(self._incumbents, key=lambda config: len(rh.get_trials(config))))
        elif sort_by is None:
            return list(self._incumbents)
        else:
            raise ValueError(f"Unknown sort_by value: {sort_by}.")

    def get_instance_seed_budget_keys(
        self, config: Configuration, compare: bool = False
    ) -> list[InstanceSeedBudgetKey]:
        """Returns the instance-seed-budget keys for a given configuration. This method is *used for
        updating the incumbents* and might differ for different intensifiers. For example, if incumbents should only
        be compared on the highest observed budgets.
        """
        return self.runhistory.get_instance_seed_budget_keys(config, highest_observed_budget_only=False)

    def get_incumbent_instance_seed_budget_keys(self, compare: bool = False) -> list[InstanceSeedBudgetKey]:
        """Find the lowest intersection of instance-seed-budget keys for all incumbents."""
        incumbents = self.get_incumbents()

        if len(incumbents) > 0:
            # We want to calculate the smallest set of trials that is used by all incumbents
            # Reason: We can not fairly compare otherwise
            incumbent_isb_keys = [self.get_instance_seed_budget_keys(incumbent, compare) for incumbent in incumbents]
            instances = list(set.intersection(*map(set, incumbent_isb_keys)))  # type: ignore

            return instances  # type: ignore

        return []

    def get_incumbent_instance_seed_budget_key_differences(self, compare: bool = False) -> list[InstanceSeedBudgetKey]:
        """There are situations in which incumbents are evaluated on more trials than others. This method returns the
        instances that are not part of the lowest intersection of instances for all incumbents.
        """
        incumbents = self.get_incumbents()

        if len(incumbents) > 0:
            # We want to calculate the differences so that we can evaluate the other incumbents on the same instances
            incumbent_isb_keys = [self.get_instance_seed_budget_keys(incumbent, compare) for incumbent in incumbents]

            if len(incumbent_isb_keys) <= 1:
                return []

            # Compute the actual differences
            intersection_isb_keys = set.intersection(*map(set, incumbent_isb_keys))  # type: ignore
            union_isb_keys = set.union(*map(set, incumbent_isb_keys))  # type: ignore
            incumbent_isb_keys = list(union_isb_keys - intersection_isb_keys)  # type: ignore

            if len(incumbent_isb_keys) == 0:
                return []

            return incumbent_isb_keys  # type: ignore

        return []

    def get_rejected_configs(self) -> list[Configuration]:
        """Returns rejected configurations when racing against the incumbent failed."""
        configs = []
        for rejected_config_id in self._rejected_config_ids:
            configs.append(self.runhistory._ids_config[rejected_config_id])

        return configs

    def get_callback(self) -> Callback:
        """The intensifier makes use of a callback to efficiently update the incumbent based on the runhistory
        (every time new information is available). Moreover, incorporating the callback here allows developers
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

        Crucially, if there is no incumbent (at the start) then, the first configuration assumes
        incumbent status. For the next configuration, we need to check if the configuration
        is better on all instances that have been evaluated for the incumbent. If this is the
        case, then we can replace the incumbent. Otherwise, a) we need to requeue the config to
        obtain the missing instance-seed-budget combination or b) mark this configuration as
        inferior ("rejected") to not consider it again. The comparison behaviour is controlled by
        self.get_instance_seed_budget_keys() and self.get_incumbent_instance_seed_budget_keys().

        Notably, this method is written to support both multi-fidelity and multi-objective
        optimization. While the get_instance_seed_budget_keys() method and
        self.get_incumbent_instance_seed_budget_keys() are used for the multi-fidelity behaviour,
        calculate_pareto_front() is used as a hard coded way to support multi-objective
        optimization, including the single objective as special case. calculate_pareto_front()
        is called on the set of all (in case of MO) incumbents amended with the challenger
        configuration, provided it has a sufficient overlap in seed-instance-budget combinations.

        Lastly, if we have a self._max_incumbents and the pareto front provides more than this
        specified amount, we cut the incumbents using crowding distance.
        """
        rh = self.runhistory

        # What happens if a config was rejected, but it appears again? Give it another try even if it
        # has already been evaluated? Yes!

        # Associated trials and id
        config_isb_keys = self.get_instance_seed_budget_keys(config)
        config_id = rh.get_config_id(config)
        config_hash = get_config_hash(config)

        # We skip updating incumbents if no instances are available
        # Note: This is especially the case if trials of a config are still running
        # because if trials are running, the runhistory does not update the trials in the fast data structure
        if len(config_isb_keys) == 0:
            logger.debug(f"No relevant instances evaluated for config {config_hash}. Updating incumbents is skipped.")
            return

        # Now we get the incumbents and see which trials have been used
        incumbents = self.get_incumbents()
        incumbent_ids = [rh.get_config_id(c) for c in incumbents]
        # Find the lowest intersection of instance-seed-budget keys for all incumbents.
        incumbent_isb_keys = self.get_incumbent_instance_seed_budget_keys()

        # Save for later
        previous_incumbents = incumbents.copy()
        previous_incumbent_ids = incumbent_ids.copy()

        # Little sanity check here for consistency
        if len(incumbents) > 0:
            assert incumbent_isb_keys is not None
            assert len(incumbent_isb_keys) > 0

        # If there are no incumbents at all, we just use the new config as new incumbent
        # Problem: We can add running incumbents
        if len(incumbents) == 0:  # incumbent_isb_keys is None and len(incumbents) == 0:
            logger.info(f"Added config {config_hash} as new incumbent because there are no incumbents yet.")
            self._update_trajectory([config])

            # Nothing else to do
            return

        # Comparison keys
        # This one is a bit tricky: We would have problems if we compare with budgets because we might have different
        # scenarios (depending on the incumbent selection specified in Successive Halving).
        # 1) Any budget/highest observed budget: We want to get rid of the budgets because if we know it is calculated
        # on the same instance-seed already then we are ready to go. Imagine we would check for the same budgets,
        # then the configs can not be compared although the user does not care on which budgets configurations have
        # been evaluated.
        # 2) Highest budget: We only want to compare the configs if they are evaluated on the highest budget.
        # Here we do actually care about the budgets. Please see the ``get_instance_seed_budget_keys`` method from
        # Successive Halving to get more information.
        # Noitce: compare=True only takes effect when subclass implemented it. -- e.g. in SH it
        # will remove the budgets from the keys.
        config_isb_comparison_keys = self.get_instance_seed_budget_keys(config, compare=True)
        # Find the lowest intersection of instance-seed-budget keys for all incumbents.
        config_incumbent_isb_comparison_keys = self.get_incumbent_instance_seed_budget_keys(compare=True)

        # Now we have to check if the new config has been evaluated on the same keys as the incumbents
        if not all([key in config_isb_comparison_keys for key in config_incumbent_isb_comparison_keys]):
            # We can not tell if the new config is better/worse than the incumbents because it has not been
            # evaluated on the necessary trials
            logger.debug(
                f"Could not compare config {config_hash} with incumbents because it's evaluated on "
                f"different trials."
            )

            # The config has to go to a queue now as it is a challenger and a potential incumbent
            return
        else:
            # If all instances are available and the config is incumbent and even evaluated on more trials
            # then there's nothing we can do
            if config in incumbents and len(config_isb_keys) > len(incumbent_isb_keys):
                logger.debug(
                    "Config is already an incumbent but can not be compared to other incumbents because "
                    "the others are missing trials."
                )
                return

        # Add config to incumbents so that we compare only the new config and existing incumbents
        if config not in incumbents:
            incumbents.append(config)
            incumbent_ids.append(config_id)

        # Now we get all instance-seed-budget keys for each incumbent (they might be different when using budgets)
        all_incumbent_isb_keys = []
        for incumbent in incumbents:
            all_incumbent_isb_keys.append(self.get_instance_seed_budget_keys(incumbent))

        # We compare the incumbents now and only return the ones on the pareto front
        new_incumbents = calculate_pareto_front(rh, incumbents, all_incumbent_isb_keys)
        new_incumbent_ids = [rh.get_config_id(c) for c in new_incumbents]

        if len(previous_incumbents) == len(new_incumbents):
            if previous_incumbents == new_incumbents:
                # No changes in the incumbents, we need this clause because we can't use set difference then
                if config_id in new_incumbent_ids:
                    self._remove_rejected_config(config_id)
                else:
                    # config worse than incumbents and thus rejected
                    self._add_rejected_config(config_id)
                return
            else:
                # In this case, we have to determine which config replaced which incumbent and reject it
                removed_incumbent_id = list(set(previous_incumbent_ids) - set(new_incumbent_ids))[0]
                removed_incumbent_hash = get_config_hash(rh.get_config(removed_incumbent_id))
                self._add_rejected_config(removed_incumbent_id)

                if removed_incumbent_id == config_id:
                    logger.debug(
                        f"Rejected config {config_hash} because it is not better than the incumbents on "
                        f"{len(config_isb_keys)} instances."
                    )
                else:
                    self._remove_rejected_config(config_id)
                    logger.info(
                        f"Added config {config_hash} and rejected config {removed_incumbent_hash} as incumbent because "
                        f"it is not better than the incumbents on {len(config_isb_keys)} instances:"
                    )
                    print_config_changes(rh.get_config(removed_incumbent_id), config, logger=logger)
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
        # We use the crowding distance for that
        if len(new_incumbents) > self._max_incumbents:
            new_incumbents = sort_by_crowding_distance(rh, new_incumbents, all_incumbent_isb_keys)
            new_incumbents = new_incumbents[: self._max_incumbents]

            # or random?
            # idx = self._rng.randint(0, len(new_incumbents))
            # del new_incumbents[idx]
            # del new_incumbent_ids[idx]

            logger.info(
                f"Removed one incumbent using crowding distance because more than {self._max_incumbents} are "
                "available."
            )

        self._update_trajectory(new_incumbents)

    @abstractmethod
    def __iter__(self) -> Iterator[TrialInfo]:
        """Main loop of the intensifier. This method always returns a TrialInfo object, although the intensifier
        algorithm may need to wait for the result of the trial. Please refer to a specific
        intensifier to get more information.
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

        # We reset the intensifier and then reset the runhistory
        self.reset()
        if self._runhistory is not None:
            self.runhistory = self._runhistory

        self._incumbents = [self.runhistory.get_config(config_id) for config_id in data["incumbent_ids"]]
        self._incumbents_changed = data["incumbents_changed"]
        self._rejected_config_ids = data["rejected_config_ids"]
        self._trajectory = [TrajectoryItem(**item) for item in data["trajectory"]]
        self.set_state(data["state"])

    def _update_trajectory(self, configs: list[Configuration]) -> None:
        rh = self.runhistory
        config_ids = [rh.get_config_id(c) for c in configs]
        costs = [rh.average_cost(c, normalize=False) for c in configs]

        self._incumbents = configs
        self._incumbents_changed += 1
        self._trajectory.append(
            TrajectoryItem(
                config_ids=config_ids,
                costs=costs,
                trial=rh.finished,
                walltime=self.used_walltime,
            )
        )
        logger.debug("Updated trajectory.")

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

    def _reorder_instance_seed_keys(
        self,
        instance_seed_keys: list[InstanceSeedKey],
        *,
        seed: int | None = None,
    ) -> list[InstanceSeedKey]:
        """Shuffles the instance-seed keys by groups (first all instances, then all seeds). The following is done:
        - Group by seeds
        - Shuffle instances in the group of seeds
        - Attach groups together
        """
        if seed is None:
            rng = self._rng
        else:
            rng = np.random.RandomState(seed)

        groups = defaultdict(list)
        for key in instance_seed_keys:
            groups[key.seed].append(key)
            assert key.seed in self._tf_seeds

        # Shuffle groups + attach groups together
        shuffled_keys: list[InstanceSeedKey] = []
        for seed in self._tf_seeds:
            if seed in groups and len(groups[seed]) > 0:
                # Shuffle pairs in the group and add to shuffled pairs
                shuffled = rng.choice(groups[seed], size=len(groups[seed]), replace=False)  # type: ignore
                shuffled_keys += [pair for pair in shuffled]  # type: ignore

        # Small sanity check
        assert len(shuffled_keys) == len(instance_seed_keys)

        return shuffled_keys
